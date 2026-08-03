from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np
from ultralytics import YOLO

from attendance_cv.config import load_config
from attendance_cv.database import AttendanceDB
from attendance_cv.face_engine import FaceEngine
from attendance_cv.matcher import FaceMatcher
from attendance_cv.vision_utils import (
    IdentityVote,
    associate_face_to_track,
    crossing_direction,
    stable_identity,
)


@dataclass
class FramePacket:
    frame: np.ndarray
    capture_time: float
    frame_id: int


@dataclass
class DetectionState:
    people: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)
    faces: list[tuple[np.ndarray, int, float]] = field(default_factory=list)
    confirmed_identities: dict[int, tuple[str, float]] = field(default_factory=dict)
    inference_time_ms: float = 0.0
    inference_fps: float = 0.0
    timestamp: float = 0.0


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
            # Enable hardware acceleration if available (FFmpeg / D3D11 / CUDA)
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

            print("RTSP Capture Stream Opened (Hardware Acceleration Attempted).")

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                capture_time = time.monotonic()
                if not ok or frame is None:
                    print("Stream read failed. Reconnecting...")
                    break

                self._frame_count += 1
                packet = FramePacket(
                    frame=frame,
                    capture_time=capture_time,
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


def pad_box(
    x1: int, y1: int, x2: int, y2: int,
    pad: int,
    frame_w: int, frame_h: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(frame_w, x2 + pad),
        min(frame_h, y2 + pad),
    )


def passes_liveness(allow_insecure: bool) -> bool:
    return allow_insecure


class AsyncInferenceWorker:
    """Decoupled Inference Worker running YOLO + InsightFace asynchronously
    without blocking the live video display loop."""

    def __init__(self, config) -> None:
        self.config = config
        self.person_model = YOLO(config.person.model_path)
        self.face_engine = FaceEngine(config.face)
        self.matcher = FaceMatcher(
            config.attendance.embedding_file,
            config.face,
        )
        self.database = AttendanceDB(
            config.attendance.event_database,
            config.attendance.duplicate_cooldown_seconds,
        )

        self.track_votes = defaultdict(
            lambda: deque(maxlen=config.face.vote_window)
        )
        self.confirmed_identities: dict[int, tuple[str, float]] = {}
        self.previous_sides: dict[int, int] = {}
        self.last_seen: dict[int, float] = {}

        # Bounded queue (maxsize=1): drops stale frames when worker is busy
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
        """Submit frame to bounded queue. Discard old frame if queue is full."""
        try:
            self._input_queue.put_nowait(packet)
        except queue.Full:
            # Drop stale frame — queue always contains ONLY the freshest frame!
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
            frame = packet.frame

            # 1. YOLO Person Detection & Tracking
            result = self.person_model.track(
                frame,
                persist=True,
                classes=[0],
                conf=self.config.person.confidence,
                tracker=self.config.person.tracker,
                verbose=False,
            )[0]

            people: dict[int, tuple[int, int, int, int]] = {}
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                ids = result.boxes.id.int().cpu().tolist()
                people = {
                    int(track_id): tuple(map(int, box))
                    for track_id, box in zip(ids, boxes)
                }

            # 2. InsightFace SCRFD Detection & Matching
            detected_faces = []
            for face in self.face_engine.detect(frame):
                if not face.quality_ok:
                    continue

                track_id = associate_face_to_track(face, people)
                if track_id is None:
                    continue

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

                self.last_seen[track_id] = time.monotonic()
                detected_faces.append((face.bbox, track_id, match.score))

            # 3. Crossing Line Events
            frame_height, frame_width = frame.shape[:2]
            line_y = int(frame_height * self.config.attendance.line_y_ratio)

            pad = getattr(self.config.person, "bbox_padding", 0)
            people_padded = {
                tid: pad_box(*box, pad, frame_width, frame_height)
                for tid, box in people.items()
            }

            for track_id, (x1, y1, x2, y2) in people_padded.items():
                center_y = (y1 + y2) // 2
                current_side = 1 if center_y >= line_y else -1
                previous_side = self.previous_sides.get(track_id)

                if previous_side is not None:
                    direction = crossing_direction(
                        previous_side,
                        current_side,
                        self.config.attendance.inside_is_below_line,
                    )

                    if direction:
                        identity = self.confirmed_identities.get(track_id)
                        if identity:
                            employee_id, score = identity
                            event_type = (
                                "CHECK_IN"
                                if direction == "ENTER"
                                else "EXIT_SELECTION"
                            )

                            event_id = self.database.create_pending_event(
                                camera_id=self.config.camera.camera_id,
                                track_id=track_id,
                                employee_id=employee_id,
                                direction=direction,
                                event_type=event_type,
                                similarity=score,
                            )

                            if event_id is not None:
                                print(
                                    f"EVENT {event_id}: "
                                    f"{employee_id} {event_type} "
                                    f"score={score:.3f} | Status: PENDING_CONFIRMATION"
                                )

                self.previous_sides[track_id] = current_side
                self.last_seen[track_id] = time.monotonic()

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

            t_end = time.monotonic()
            inference_dur_ms = (t_end - t_start) * 1000.0

            self._fps_counter += 1
            if t_end - self._fps_timer >= 1.0:
                self._last_fps = self._fps_counter / (t_end - self._fps_timer)
                self._fps_counter = 0
                self._fps_timer = t_end

            # Update shared state thread-safely
            with self._state_lock:
                self.current_state = DetectionState(
                    people=people_padded,
                    faces=detected_faces,
                    confirmed_identities=dict(self.confirmed_identities),
                    inference_time_ms=inference_dur_ms,
                    inference_fps=self._last_fps,
                    timestamp=t_end,
                )

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)


def main() -> None:
    config = load_config()

    reader = RTSPReader(
        parse_source(config.camera.source),
        reconnect_delay=config.camera.reconnect_delay_seconds,
    )

    print("Waiting for first frame from RTSP stream...")
    while not reader.is_ready():
        time.sleep(0.05)
    print("Stream connected. Initializing Async Inference Pipeline.")

    worker = AsyncInferenceWorker(config)

    window_initialized = False
    display_fps_counter = 0
    display_fps_timer = time.monotonic()
    display_fps = 0.0

    try:
        while True:
            packet = reader.get_latest_frame()
            if packet is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()

            # Submit frame asynchronously to worker
            worker.submit_frame(packet)

            # Get latest available detection results (asynchronous overlay)
            state = worker.get_detection_state()

            # Calculate Latency Instrumentation Metrics
            capture_to_display_latency_ms = (now - packet.capture_time) * 1000.0

            display_fps_counter += 1
            if now - display_fps_timer >= 1.0:
                display_fps = display_fps_counter / (now - display_fps_timer)
                display_fps_counter = 0
                display_fps_timer = now

            if config.camera.show_preview:
                frame = packet.frame.copy()
                frame_h, frame_w = frame.shape[:2]

                # 1. Overlay Person Bounding Boxes & Labels
                for track_id, (x1, y1, x2, y2) in state.people.items():
                    label, score = state.confirmed_identities.get(
                        track_id, ("UNKNOWN", 0.0)
                    )
                    person_color = (0, 255, 0) if label != "UNKNOWN" else (0, 215, 255)

                    cv.rectangle(frame, (x1, y1), (x2, y2), person_color, 2)

                    line1 = f"PERSON #{track_id}"
                    line2 = f"ID: {label} ({score:.2f})"

                    cv.putText(
                        frame, line1, (x1 + 4, max(25, y1 - 22)),
                        cv.FONT_HERSHEY_SIMPLEX, 0.55, person_color, 2,
                    )
                    cv.putText(
                        frame, line2, (x1 + 4, max(42, y1 - 4)),
                        cv.FONT_HERSHEY_SIMPLEX, 0.55, person_color, 2,
                    )

                # 2. Overlay Face Bounding Boxes
                face_color = (255, 0, 255)  # Magenta
                for f_bbox, tid, f_score in state.faces:
                    fx1, fy1, fx2, fy2 = f_bbox.astype(int)
                    cv.rectangle(frame, (fx1, fy1), (fx2, fy2), face_color, 2)
                    cv.putText(
                        frame, f"FACE ({f_score:.2f})", (fx1, max(15, fy1 - 6)),
                        cv.FONT_HERSHEY_SIMPLEX, 0.45, face_color, 1,
                    )

                # 3. Draw On-Screen Latency Instrumentation Panel
                hud_bg_color = (0, 0, 0)
                hud_text_color = (0, 255, 255)  # Yellow
                cv.rectangle(frame, (10, 10), (450, 110), hud_bg_color, -1)
                cv.rectangle(frame, (10, 10), (450, 110), (100, 100, 100), 1)

                cv.putText(
                    frame, f"LIVE DISPLAY FPS : {display_fps:.1f} FPS (Target: 30)",
                    (20, 32), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                )
                cv.putText(
                    frame, f"INFERENCE SPEED  : {state.inference_time_ms:.1f} ms ({state.inference_fps:.1f} FPS)",
                    (20, 56), cv.FONT_HERSHEY_SIMPLEX, 0.5, hud_text_color, 1,
                )
                cv.putText(
                    frame, f"CAP->DISPLAY LAG : {capture_to_display_latency_ms:.1f} ms",
                    (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,
                )
                cv.putText(
                    frame, f"PIPELINE MODE    : Async Decoupled (Zero-Lag)",
                    (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                )

                if not window_initialized:
                    cv.namedWindow("SCRFD Attendance Starter", cv.WINDOW_NORMAL)
                    cv.resizeWindow("SCRFD Attendance Starter", frame_w, frame_h)
                    window_initialized = True

                cv.imshow("SCRFD Attendance Starter", frame)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

            # Cap display loop frequency to ~60 FPS
            time.sleep(0.015)

    finally:
        worker.stop()
        reader.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
