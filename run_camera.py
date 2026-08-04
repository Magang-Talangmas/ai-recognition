from __future__ import annotations

import os
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from attendance_cv.config import load_config
from attendance_cv.database import AttendanceDB
from attendance_cv.face_engine import FaceEngine, FaceResult
from attendance_cv.matcher import FaceMatcher
from attendance_cv.vision_utils import (
    IdentityVote,
    crossing_direction,
    stable_identity,
)


@dataclass
class FramePacket:
    frame: np.ndarray
    capture_time: float
    t_capture_wall: float
    frame_id: int


class RTSPReader:
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
                time.sleep(self.reconnect_delay)
                continue

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                capture_time = time.monotonic()
                if not ok or frame is None:
                    break

                self._frame_count += 1
                packet = FramePacket(
                    frame=frame,
                    capture_time=capture_time,
                    t_capture_wall=time.time(),
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


def match_face_to_tracks(
    face_bbox: np.ndarray,
    previous_tracks: dict[int, tuple[float, float]],
    max_dist: float = 120.0,
) -> int:
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
    def __init__(self, config) -> None:
        self.config = config
        self.face_engine = FaceEngine(config.face)
        self.matcher = FaceMatcher(config.attendance.embedding_file, config.face)
        self.database = AttendanceDB(
            config.attendance.event_database,
            config.attendance.duplicate_cooldown_seconds,
        )
        
        self._input_queue = queue.Queue(maxsize=1)
        self._output_queue = queue.Queue()
        self._stop_event = threading.Event()
        
        self.track_votes = defaultdict(lambda: deque(maxlen=config.face.vote_window))
        self.confirmed_identities = {}
        self.face_centroids = {}
        
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

    def get_latest_results(self):
        result = None
        while True:
            try:
                result = self._output_queue.get_nowait()
            except queue.Empty:
                break
        return result

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                packet = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t_start = time.monotonic()
            
            # Run FULL inference in background (detect + recognize)
            # This takes 1.5 seconds but doesn't block the UI loop!
            faces = self.face_engine.detect(packet.frame)
            
            new_centroids = {}
            processed_faces = []

            for face in faces:
                if not face.quality_ok:
                    continue

                track_id = match_face_to_tracks(face.bbox, self.face_centroids)
                fx1, fy1, fx2, fy2 = face.bbox
                fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
                new_centroids[track_id] = (fcx, fcy)

                match = self.matcher.match(face.embedding)
                self.track_votes[track_id].append(IdentityVote(match.employee_id, match.score))
                
                employee_id, score = stable_identity(self.track_votes[track_id], self.config.face.votes_required)
                if employee_id:
                    if track_id not in self.confirmed_identities:
                        # Log check-in upon first confident recognition
                        event_id = self.database.create_pending_event(
                            camera_id=self.config.camera.camera_id,
                            track_id=track_id,
                            employee_id=employee_id,
                            direction="ENTER",
                            event_type="CHECK_IN",
                            similarity=score,
                        )
                        if event_id is not None:
                            print(f"EVENT {event_id}: {employee_id} CHECK_IN score={score:.3f} | Status: PENDING_CONFIRMATION")
                    self.confirmed_identities[track_id] = (employee_id, score)

                emp_name, emp_score = self.confirmed_identities.get(track_id, (None, 0.0))
                
                processed_faces.append({
                    'track_id': track_id,
                    'bbox': [int(fx1), int(fy1), int(fx2), int(fy2)],
                    'name': emp_name,
                    'score': emp_score if emp_name else match.score
                })

            self.face_centroids = new_centroids
            self._output_queue.put((packet.capture_time, processed_faces))

            t_end = time.monotonic()
            self._fps_counter += 1
            if t_end - self._fps_timer >= 1.0:
                self._last_fps = self._fps_counter / (t_end - self._fps_timer)
                self._fps_counter = 0
                self._fps_timer = t_end

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)


class FastFaceTracker:
    """Super lightweight UI-thread tracker using template matching."""
    def __init__(self):
        self.faces = [] 
        self.last_gray = None

    def update_from_ai(self, frame_gray, faces_from_ai):
        self.faces = []
        self.last_gray = frame_gray
        for face in faces_from_ai:
            x1, y1, x2, y2 = face['bbox']
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_gray.shape[1], x2), min(frame_gray.shape[0], y2)
            
            template = frame_gray[y1:y2, x1:x2]
            if template.size == 0 or template.shape[0] < 5 or template.shape[1] < 5:
                continue
                
            self.faces.append({
                'track_id': face['track_id'],
                'bbox': [x1, y1, x2, y2],
                'template': template,
                'name': face['name'],
                'score': face['score']
            })

    def track(self, frame_gray):
        if self.last_gray is None:
            return

        for face in self.faces:
            x1, y1, x2, y2 = face['bbox']
            template = face['template']
            h, w = template.shape
            
            pad = 40
            sx1, sy1 = max(0, x1 - pad), max(0, y1 - pad)
            sx2, sy2 = min(frame_gray.shape[1], x2 + pad), min(frame_gray.shape[0], y2 + pad)
            
            search_region = frame_gray[sy1:sy2, sx1:sx2]
            if search_region.shape[0] < h or search_region.shape[1] < w:
                continue
                
            res = cv.matchTemplate(search_region, template, cv.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv.minMaxLoc(res)
            
            if max_val > 0.4:
                nx1 = sx1 + max_loc[0]
                ny1 = sy1 + max_loc[1]
                face['bbox'] = [nx1, ny1, nx1 + w, ny1 + h]
                
                # Update template slightly for smooth drift
                face['template'] = frame_gray[ny1:ny1+h, nx1:nx1+w]

        self.last_gray = frame_gray


def main() -> None:
    config = load_config("config.yaml")
    source = parse_source(config.camera.source)
    reader = RTSPReader(source=source, reconnect_delay=config.camera.reconnect_delay_seconds)

    print("Waiting for first frame from RTSP stream...")
    while not reader.is_ready():
        time.sleep(0.1)

    print("Stream connected. Initializing Fast Template Tracking Pipeline.")
    worker = AsyncInferenceWorker(config)
    tracker = FastFaceTracker()

    display_fps_counter = 0
    display_fps_timer = time.monotonic()
    display_fps = 0.0
    window_initialized = False
    
    last_ai_update = 0.0

    try:
        while True:
            packet = reader.get_latest_frame()
            if packet is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            frame = packet.frame.copy()  # Use copy to avoid queue corruption
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Check if background worker finished a heavy inference task
            ai_result = worker.get_latest_results()
            if ai_result is not None:
                capture_time, faces = ai_result
                if capture_time > last_ai_update:
                    tracker.update_from_ai(frame_gray, faces)
                    last_ai_update = capture_time
            else:
                # Track the faces using lightweight matchTemplate
                tracker.track(frame_gray)

            # Submit new frame to worker
            skip_ratio = max(1, config.camera.process_every_n_frames)
            if packet.frame_id % skip_ratio == 0:
                worker.submit_frame(packet)

            # Render
            display_fps_counter += 1
            if now - display_fps_timer >= 1.0:
                display_fps = display_fps_counter / (now - display_fps_timer)
                display_fps_counter = 0
                display_fps_timer = now

            if config.camera.show_preview:
                # Draw Tracked Boxes
                for face in tracker.faces:
                    fx1, fy1, fx2, fy2 = face['bbox']
                    name = face['name']
                    score = face['score']
                    
                    if name:
                        cv.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                        cv.putText(frame, f"{name} ({score:.2f})", (fx1, max(20, fy1 - 8)), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    else:
                        cv.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 215, 255), 2)
                        cv.putText(frame, "UNKNOWN", (fx1, max(20, fy1 - 8)), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)

                # HUD
                cv.rectangle(frame, (10, 10), (450, 80), (0, 0, 0), -1)
                cv.rectangle(frame, (10, 10), (450, 80), (100, 100, 100), 1)
                cv.putText(frame, f"LIVE TRACKING FPS: {display_fps:.1f} FPS", (20, 32), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.putText(frame, f"ARCFACE SPEED    : {worker._last_fps:.1f} FPS", (20, 56), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                if not window_initialized:
                    cv.namedWindow("AI Attendance", cv.WINDOW_NORMAL)
                    cv.resizeWindow("AI Attendance", frame.shape[1], frame.shape[0])
                    window_initialized = True

                cv.imshow("AI Attendance", frame)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        reader.stop()
        worker.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
