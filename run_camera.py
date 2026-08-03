from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

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

# Set FFmpeg options for zero-latency real-time RTSP capture
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|max_delay;500000|buffer_size;1024000|flags;low_delay"
)


class FreshRTSPStream:
    """Threaded RTSP Stream reader that continuously flushes stale frames
    to ensure zero-latency real-time stream processing."""

    def __init__(self, source: int | str) -> None:
        self.source = source
        self.capture = cv.VideoCapture(source)
        self.capture.set(cv.CAP_PROP_BUFFERSIZE, 1)
        self.latest_frame: np.ndarray | None = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self) -> None:
        while not self.stopped:
            ok, frame = self.capture.read()
            if not ok:
                self.stopped = True
                break
            with self.lock:
                self.latest_frame = frame

    def is_opened(self) -> bool:
        return self.capture.isOpened() and not self.stopped

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def release(self) -> None:
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.capture.release()


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def passes_liveness(allow_insecure: bool) -> bool:
    # TODO: replace with tested PAD/liveness model.
    return allow_insecure


def main() -> None:
    config = load_config()

    person_model = YOLO(config.person.model_path)
    face_engine = FaceEngine(config.face)
    matcher = FaceMatcher(
        config.attendance.embedding_file,
        config.face,
    )
    database = AttendanceDB(
        config.attendance.event_database,
        config.attendance.duplicate_cooldown_seconds,
    )

    track_votes = defaultdict(
        lambda: deque(maxlen=config.face.vote_window)
    )
    confirmed_identities: dict[int, tuple[str, float]] = {}
    previous_sides: dict[int, int] = {}
    last_seen: dict[int, float] = {}

    frame_number = 0

    window_name = "SCRFD Attendance Starter"
    if config.camera.show_preview:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_name, 1280, 720)

    while True:
        source_val = parse_source(config.camera.source)
        stream = FreshRTSPStream(source_val)

        # Wait up to 5 seconds for initial frame
        start_wait = time.time()
        while not stream.is_opened() and time.time() - start_wait < 5.0:
            time.sleep(0.1)

        if not stream.is_opened():
            print("Camera unavailable. Reconnecting...")
            stream.release()
            time.sleep(config.camera.reconnect_delay_seconds)
            continue

        print("Camera stream connected (Real-Time Mode).")

        while stream.is_opened():
            ok, frame = stream.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            frame_number += 1
            if frame_number % config.camera.process_every_n_frames != 0:
                continue

            result = person_model.track(
                frame,
                persist=True,
                classes=[0],
                conf=config.person.confidence,
                tracker=config.person.tracker,
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

            face_boxes_to_draw: list[tuple[tuple[int, int, int, int], str, float]] = []

            for face in face_engine.detect(frame):
                if not face.quality_ok:
                    continue

                fx1, fy1, fx2, fy2 = map(int, face.bbox)
                track_id = associate_face_to_track(face, people)
                if track_id is None:
                    continue

                if not passes_liveness(
                    config.attendance.allow_insecure_no_liveness
                ):
                    track_votes[track_id].append(
                        IdentityVote(None, 0.0)
                    )
                    continue

                match = matcher.match(face.embedding)
                track_votes[track_id].append(
                    IdentityVote(match.employee_id, match.score)
                )

                employee_id, score = stable_identity(
                    track_votes[track_id],
                    config.face.votes_required,
                )

                if employee_id:
                    confirmed_identities[track_id] = (
                        employee_id,
                        score,
                    )
                elif match.employee_id:
                    confirmed_identities[track_id] = (
                        match.employee_id,
                        match.score,
                    )

                label, f_score = confirmed_identities.get(
                    track_id, ("UNKNOWN", 0.0)
                )
                face_boxes_to_draw.append(((fx1, fy1, fx2, fy2), label, f_score))

                last_seen[track_id] = time.monotonic()

            frame_height, frame_width = frame.shape[:2]
            line_y = int(
                frame_height * config.attendance.line_y_ratio
            )

            # 1. Draw Face Bounding Boxes (BBox 2 - Green / Coral)
            if config.camera.show_preview:
                for (fx1, fy1, fx2, fy2), f_label, f_score in face_boxes_to_draw:
                    face_color = (0, 255, 127) if f_label != "UNKNOWN" else (0, 165, 255)
                    cv.rectangle(
                        frame,
                        (fx1, fy1),
                        (fx2, fy2),
                        face_color,
                        2,
                    )
                    cv.putText(
                        frame,
                        f"Face: {f_label} ({f_score:.2f})",
                        (fx1, max(15, fy1 - 5)),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        face_color,
                        1,
                        cv.LINE_AA,
                    )

            for track_id, (x1, y1, x2, y2) in people.items():
                center_y = (y1 + y2) // 2
                current_side = 1 if center_y >= line_y else -1
                previous_side = previous_sides.get(track_id)

                if previous_side is not None:
                    direction = crossing_direction(
                        previous_side,
                        current_side,
                        config.attendance.inside_is_below_line,
                    )

                    if direction:
                        identity = confirmed_identities.get(track_id)
                        if identity:
                            employee_id, score = identity
                            event_type = (
                                "CHECK_IN"
                                if direction == "ENTER"
                                else "EXIT_SELECTION"
                            )

                            event_id = database.create_pending_event(
                                camera_id=config.camera.camera_id,
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
                                    f"score={score:.3f}"
                                )
                                print(
                                    "Status: PENDING_CONFIRMATION"
                                )

                previous_sides[track_id] = current_side
                last_seen[track_id] = time.monotonic()

                # 2. Draw Person Bounding Boxes (BBox 1 - Cyan / Gold)
                if config.camera.show_preview:
                    label, score = confirmed_identities.get(
                        track_id,
                        ("UNKNOWN", 0.0),
                    )
                    person_color = (255, 200, 0) if label != "UNKNOWN" else (220, 220, 220)
                    cv.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        person_color,
                        2,
                    )
                    cv.putText(
                        frame,
                        f"Person #{track_id}: {label}",
                        (x1, max(20, y1 - 8)),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        person_color,
                        2,
                        cv.LINE_AA,
                    )

            now = time.monotonic()
            stale_ids = [
                track_id
                for track_id, seen_at in last_seen.items()
                if now - seen_at > 10
            ]
            for track_id in stale_ids:
                track_votes.pop(track_id, None)
                confirmed_identities.pop(track_id, None)
                previous_sides.pop(track_id, None)
                last_seen.pop(track_id, None)

            if config.camera.show_preview:
                preview_w = 1152
                preview_h = int(frame_height * (preview_w / frame_width))
                preview_frame = cv.resize(frame, (preview_w, preview_h))
                cv.imshow(window_name, preview_frame)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    stream.release()
                    cv.destroyAllWindows()
                    return

        stream.release()
        time.sleep(config.camera.reconnect_delay_seconds)


if __name__ == "__main__":
    main()
