from __future__ import annotations

import ctypes
import os
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from attendance_cv.api_dispatcher import BackendDispatcher
from attendance_cv.config import load_config
from attendance_cv.database import AttendanceDB
from attendance_cv.face_engine import FaceEngine
from attendance_cv.matcher import FaceMatcher
from attendance_cv.vision_utils import (
    IdentityVote,
    crossing_direction,
    stable_identity,
)


@dataclass
class FramePacket:
    frame: np.ndarray
    capture_time: float      # time.monotonic() — used for duration calculations
    t_capture_wall: float    # time.time()      — absolute wall-clock at cap.read() return
    frame_id: int


@dataclass
class FaceRenderInfo:
    bbox: np.ndarray
    employee_id: str | None
    score: float
    face_id: int


@dataclass
class DetectionState:
    faces: list[FaceRenderInfo] = field(default_factory=list)
    inference_time_ms: float = 0.0
    inference_fps: float = 0.0
    timestamp: float = 0.0
    # Pipeline latency timestamps (all time.monotonic())
    capture_time: float = 0.0        # when RTSPReader captured the frame
    dequeue_time: float = 0.0        # when inference thread dequeued the frame
    inference_done_time: float = 0.0 # when inference + matching finished
    t_capture_wall: float = 0.0      # time.time() at cap.read() — absolute wall clock


class RTSPReader:
    """Threaded RTSP Capture with HW acceleration enabled & bounded single-frame buffer."""

    def __init__(self, source: int | str, reconnect_delay: float = 3.0) -> None:
        self.source = source
        self.reconnect_delay = reconnect_delay
        self._latest_packet: FramePacket | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame_count = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            cap = cv.VideoCapture(self.source, cv.CAP_FFMPEG)
            cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            try:
                cap.set(cv.CAP_PROP_HW_ACCELERATION, cv.VIDEO_ACCELERATION_ANY)
            except AttributeError:
                pass

            if not cap.isOpened():
                print("Camera unavailable. Reconnecting...")
                time.sleep(self.reconnect_delay)
                continue

            print("RTSP Capture Stream Opened.")

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                capture_time    = time.monotonic()  # for duration math
                t_capture_wall  = time.time()       # absolute wall-clock at first frame touch
                if not ok or frame is None:
                    print("Stream read failed. Reconnecting...")
                    break

                self._frame_count += 1
                packet = FramePacket(
                    frame=frame,
                    capture_time=capture_time,
                    t_capture_wall=t_capture_wall,
                    frame_id=self._frame_count,
                )
                with self._lock:
                    self._latest_packet = packet

            cap.release()
            if not self._stop_event.is_set():
                time.sleep(self.reconnect_delay)

    def get_latest_frame(self) -> FramePacket | None:
        with self._lock:
            return self._latest_packet

    def is_ready(self) -> bool:
        with self._lock:
            return self._latest_packet is not None

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def passes_liveness(allow_insecure: bool) -> bool:
    return allow_insecure


def match_face_to_tracks(
    face_bbox: np.ndarray,
    previous_tracks: dict[int, tuple[float, float]],
    max_dist: float = 120.0,
) -> int:
    """Simple centroid tracker for face boxes across frames."""
    x1, y1, x2, y2 = face_bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    best_id = None
    min_d = float("inf")

    for track_id, (pcx, pcy) in previous_tracks.items():
        dist = np.hypot(cx - pcx, cy - pcy)
        if dist < min_d and dist <= max_dist:
            min_d = dist
            best_id = track_id

    if best_id is None:
        best_id = int(time.monotonic() * 1000) % 100000

    return best_id


class AsyncInferenceWorker:
    """Streamlined Pure-Face Inference Worker: Runs SCRFD + ArcFace directly
    without YOLO person detection overhead for maximum FPS."""

    def __init__(self, config) -> None:
        self.config = config
        self.face_engine = FaceEngine(config.face)
        self.matcher = FaceMatcher(
            config.attendance.embedding_file,
            config.face,
        )
        self.database = AttendanceDB(
            config.attendance.event_database,
            config.attendance.duplicate_cooldown_seconds,
        )
        self.backend_dispatcher = BackendDispatcher(
            api_url=config.backend.api_url,
            api_key=config.backend.api_key,
            enabled=config.backend.enabled,
            timeout=config.backend.timeout,
        )

        self.track_votes = defaultdict(
            lambda: deque(maxlen=config.face.vote_window)
        )
        self.confirmed_identities: dict[int, tuple[str, float]] = {}
        self.previous_sides: dict[int, int] = {}
        self.last_seen: dict[int, float] = {}
        self.face_centroids: dict[int, tuple[float, float]] = {}

        self._input_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        self._state_lock = threading.Lock()
        self.current_state = DetectionState()

        self._stop_event = threading.Event()
        self._fps_counter = 0
        self._fps_timer = time.monotonic()
        self._last_fps = 0.0

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_frame(self, packet: FramePacket) -> None:
        try:
            self._input_queue.put_nowait(packet)
        except queue.Full:
            try:
                self._input_queue.get_nowait()
                self._input_queue.put_nowait(packet)
            except (queue.Empty, queue.Full):
                pass

    def get_detection_state(self) -> DetectionState:
        with self._state_lock:
            return self.current_state

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                packet = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t_start = time.monotonic()
            dequeue_time = t_start  # frame just came off the queue
            frame = packet.frame
            frame_height, frame_width = frame.shape[:2]
            line_y = int(frame_height * self.config.attendance.line_y_ratio)

            # 1. InsightFace SCRFD Detection & Face Matching
            detected_faces = []
            new_centroids = {}

            for face in self.face_engine.detect(frame):
                if not face.quality_ok:
                    continue

                track_id = match_face_to_tracks(face.bbox, self.face_centroids)
                fx1, fy1, fx2, fy2 = face.bbox
                fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
                new_centroids[track_id] = (fcx, fcy)

                if not passes_liveness(
                    self.config.attendance.allow_insecure_no_liveness
                ):
                    self.track_votes[track_id].append(IdentityVote(None, 0.0))
                    continue

                match = self.matcher.match(face.embedding)
                self.track_votes[track_id].append(
                    IdentityVote(match.employee_id, match.score)
                )

                employee_id, score = stable_identity(
                    self.track_votes[track_id],
                    self.config.face.votes_required,
                )

                if employee_id:
                    self.confirmed_identities[track_id] = (
                        employee_id,
                        score,
                    )

                    # Opsi B: Langsung memicu CHECK-IN saat wajah dikenali stabil
                    event_id = self.database.create_pending_event(
                        camera_id=self.config.camera.camera_id,
                        track_id=track_id,
                        employee_id=employee_id,
                        direction="ENTER",
                        event_type="CHECK_IN",
                        similarity=score,
                    )

                    if event_id is not None:
                        print(
                            f"\n>>> [CHECK-IN TRIGGERED] Event #{event_id}: {employee_id} "
                            f"(score={score:.3f}) | Status: PENDING_CONFIRMATION"
                        )
                        # Asynchronous dispatch ke Backend API
                        self.backend_dispatcher.dispatch_checkin(
                            employee_id=employee_id,
                            similarity=score,
                            camera_id=self.config.camera.camera_id,
                        )

                self.last_seen[track_id] = time.monotonic()

                display_name, display_score = self.confirmed_identities.get(
                    track_id, (match.employee_id, match.score)
                )
                detected_faces.append(
                    FaceRenderInfo(
                        bbox=face.bbox,
                        employee_id=display_name,
                        score=display_score if display_name else match.score,
                        face_id=track_id,
                    )
                )

            self.face_centroids = new_centroids

            # Clean stale tracking votes
            now = time.monotonic()
            stale_ids = [
                tid for tid, seen_at in self.last_seen.items() if now - seen_at > 10
            ]
            for tid in stale_ids:
                self.track_votes.pop(tid, None)
                self.confirmed_identities.pop(tid, None)
                self.previous_sides.pop(tid, None)
                self.last_seen.pop(tid, None)
                self.face_centroids.pop(tid, None)

            t_end = time.monotonic()
            inference_dur_ms = (t_end - t_start) * 1000.0

            self._fps_counter += 1
            if t_end - self._fps_timer >= 1.0:
                self._last_fps = self._fps_counter / (t_end - self._fps_timer)
                self._fps_counter = 0
                self._fps_timer = t_end

            with self._state_lock:
                self.current_state = DetectionState(
                    faces=detected_faces,
                    inference_time_ms=inference_dur_ms,
                    inference_fps=self._last_fps,
                    timestamp=t_end,
                    capture_time=packet.capture_time,
                    dequeue_time=dequeue_time,
                    inference_done_time=t_end,
                    t_capture_wall=packet.t_capture_wall,
                )

    def stop(self) -> None:
        self._stop_event.set()
        self.backend_dispatcher.stop()
        self._thread.join(timeout=5)


def main() -> None:
    if os.name == "nt":
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass

    # CPU Core Thread Optimization
    num_cores = os.cpu_count() or 4
    cv.setNumThreads(num_cores)

    config = load_config()

    reader = RTSPReader(
        parse_source(config.camera.source),
        reconnect_delay=config.camera.reconnect_delay_seconds,
    )

    print("Waiting for first frame from RTSP stream...")
    while not reader.is_ready():
        time.sleep(0.05)
    print(f"Stream connected. Initializing Fast Face Inference Pipeline (CPU Threads: {num_cores}).")

    worker = AsyncInferenceWorker(config)

    window_initialized = False
    display_fps_counter = 0
    display_fps_timer = time.monotonic()
    display_fps = 0.0
    last_printed_state_ts = 0.0

    try:
        while True:
            packet = reader.get_latest_frame()
            if packet is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()


            skip_ratio = max(1, config.camera.process_every_n_frames)
            if packet.frame_id % skip_ratio == 0:
                worker.submit_frame(packet)

            state = worker.get_detection_state()

            display_fps_counter += 1
            if now - display_fps_timer >= 1.0:
                display_fps = display_fps_counter / (now - display_fps_timer)
                display_fps_counter = 0
                display_fps_timer = now

            if config.camera.show_preview:
                # Zero-copy frame assignment (draw directly on preview buffer without full RAM copy)
                frame = packet.frame
                frame_h, frame_w = frame.shape[:2]

                # Pass 1 — deduplicate by face_id (1 entry per tracked face).
                seen_face_ids: dict[int, FaceRenderInfo] = {}
                for face_info in state.faces:
                    fid = face_info.face_id
                    existing = seen_face_ids.get(fid)
                    if existing is None or face_info.score > existing.score:
                        seen_face_ids[fid] = face_info

                # Pass 2 — deduplicate identified faces by employee_id (class).
                # If two tracks both match "gibran", keep only the highest-scoring one.
                # UNKNOWN faces are all kept (each is a different physical person).
                best_per_identity: dict[str, FaceRenderInfo] = {}
                unknowns: list[FaceRenderInfo] = []
                for face_info in seen_face_ids.values():
                    if face_info.employee_id:
                        existing = best_per_identity.get(face_info.employee_id)
                        if existing is None or face_info.score > existing.score:
                            best_per_identity[face_info.employee_id] = face_info
                    else:
                        unknowns.append(face_info)

                faces_to_render = list(best_per_identity.values()) + unknowns

                # Render Face Bounding Box & Identity Label — 1 label per detected person
                for face_info in faces_to_render:
                    fx1, fy1, fx2, fy2 = face_info.bbox.astype(int)
                    label = face_info.employee_id or "UNKNOWN"
                    score = face_info.score

                    if label != "UNKNOWN":
                        text = f"[CHECK-IN] {label} ({score:.2f})"
                        box_color = (0, 255, 0)
                    else:
                        text = f"UNKNOWN ({score:.2f})"
                        box_color = (0, 215, 255)

                    # Thin bounding box (thickness 1 for crisp look on Substream)
                    cv.rectangle(frame, (fx1, fy1), (fx2, fy2), box_color, 1)

                    # Compact text badge with dark background
                    font_scale = 0.38
                    (tw, th), _ = cv.getTextSize(
                        text, cv.FONT_HERSHEY_SIMPLEX, font_scale, 1
                    )
                    ty = max(th + 6, fy1 - 4)
                    
                    # Background pill behind label for clear readability
                    cv.rectangle(
                        frame,
                        (fx1, ty - th - 3),
                        (fx1 + tw + 6, ty + 2),
                        (0, 0, 0),
                        -1,
                    )
                    cv.putText(
                        frame,
                        text,
                        (fx1 + 3, ty - 1),
                        cv.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        box_color,
                        1,
                        cv.LINE_AA,
                    )

                # On-Screen HUD Performance Panel (Compact & Sleek)
                # Compute true E2E latency from state timestamps (inference-frame reference).
                # This is stable and accurate: state.capture_time is stamped at cap.read()
                # in the reader thread; render_time is stamped immediately before imshow.
                render_time = time.monotonic()
                if state.capture_time > 0:
                    from datetime import datetime
                    queue_wait_ms  = (state.dequeue_time        - state.capture_time)        * 1000.0
                    inference_ms   = (state.inference_done_time - state.dequeue_time)        * 1000.0
                    render_wait_ms = (render_time               - state.inference_done_time) * 1000.0
                    total_e2e_ms   = (render_time               - state.capture_time)        * 1000.0
                    wall_ts        = datetime.fromtimestamp(state.t_capture_wall).strftime("%H:%M:%S.%f")[:-3]
                    
                    if state.timestamp != last_printed_state_ts:
                        print(
                            f"[{wall_ts}] "
                            f"cap.read->queue: {queue_wait_ms:.1f}ms | "
                            f"Inference: {inference_ms:.1f}ms | "
                            f"Render wait: {render_wait_ms:.1f}ms | "
                            f"E2E: {total_e2e_ms:.1f}ms"
                        )
                        last_printed_state_ts = state.timestamp
                else:
                    total_e2e_ms = 0.0

                # Compact HUD Panel
                hud_w, hud_h = 270, 64
                cv.rectangle(frame, (8, 8), (8 + hud_w, 8 + hud_h), (15, 15, 15), -1)
                cv.rectangle(frame, (8, 8), (8 + hud_w, 8 + hud_h), (80, 80, 80), 1)

                cv.putText(
                    frame, f"LIVE DISPLAY FPS : {display_fps:.1f} FPS",
                    (16, 24), cv.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 0), 1, cv.LINE_AA,
                )
                cv.putText(
                    frame, f"INFERENCE SPEED  : {state.inference_time_ms:.1f} ms ({state.inference_fps:.1f} FPS)",
                    (16, 42), cv.FONT_HERSHEY_SIMPLEX, 0.36, (0, 230, 255), 1, cv.LINE_AA,
                )
                cv.putText(
                    frame, f"CAP->DISPLAY LAG : {total_e2e_ms:.1f} ms",
                    (16, 60), cv.FONT_HERSHEY_SIMPLEX, 0.36, (0, 230, 255), 1, cv.LINE_AA,
                )

                if not window_initialized:
                    cv.namedWindow("SCRFD Attendance Starter", cv.WINDOW_NORMAL)
                    cv.resizeWindow("SCRFD Attendance Starter", frame_w, frame_h)
                    window_initialized = True

                cv.imshow("SCRFD Attendance Starter", frame)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    break


    finally:
        worker.stop()
        reader.stop()
        cv.destroyAllWindows()
        if os.name == "nt":
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass


if __name__ == "__main__":
    main()
