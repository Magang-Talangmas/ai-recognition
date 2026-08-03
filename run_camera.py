from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

import cv2 as cv
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


class RTSPReader:
    """Background thread that continuously grabs the latest frame from an
    RTSP stream, discarding buffered/stale frames to eliminate delay."""

    def __init__(self, source: int | str, reconnect_delay: float = 3.0) -> None:
        self.source = source
        self.reconnect_delay = reconnect_delay
        self._frame: cv.typing.MatLike | None = None
        self._ok: bool = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            cap = cv.VideoCapture(self.source)
            cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print("Camera unavailable. Reconnecting...")
                time.sleep(self.reconnect_delay)
                continue
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    print("Stream read failed. Reconnecting...")
                    break
                with self._lock:
                    self._frame = frame
                    self._ok = True
            cap.release()
            if not self._stop_event.is_set():
                time.sleep(self.reconnect_delay)

    def read(self) -> tuple[bool, cv.typing.MatLike | None]:
        """Return the latest frame without blocking."""
        with self._lock:
            return self._ok, self._frame.copy() if self._frame is not None else None

    def is_ready(self) -> bool:
        with self._lock:
            return self._ok

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
    """Expand a bounding box by `pad` pixels on each side, clamped to frame."""
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(frame_w, x2 + pad),
        min(frame_h, y2 + pad),
    )


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
    window_initialized = False

    reader = RTSPReader(
        parse_source(config.camera.source),
        reconnect_delay=config.camera.reconnect_delay_seconds,
    )

    print("Waiting for first frame from RTSP stream...")
    while not reader.is_ready():
        time.sleep(0.1)
    print("Stream connected. Starting detection.")

    try:
        while True:
            ok, frame = reader.read()
            if not ok or frame is None:
                time.sleep(0.05)
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

            for face in face_engine.detect(frame):
                if not face.quality_ok:
                    continue

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

                last_seen[track_id] = time.monotonic()

            frame_height, frame_width = frame.shape[:2]
            line_y = int(
                frame_height * config.attendance.line_y_ratio
            )

            # Expand all person boxes by configured padding
            pad = getattr(config.person, "bbox_padding", 40)
            people = {
                tid: pad_box(*box, pad, frame_width, frame_height)
                for tid, box in people.items()
            }

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

                if config.camera.show_preview:
                    label, score = confirmed_identities.get(
                        track_id,
                        ("UNKNOWN", 0.0),
                    )
                    cv.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        2,
                    )
                    cv.putText(
                        frame,
                        f"{track_id}: {label} {score:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
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
                cv.line(
                    frame,
                    (0, line_y),
                    (frame_width, line_y),
                    (255, 255, 255),
                    2,
                )

                if not window_initialized:
                    cv.namedWindow("SCRFD Attendance Starter", cv.WINDOW_NORMAL)
                    cv.resizeWindow("SCRFD Attendance Starter", frame_width, frame_height)
                    window_initialized = True

                cv.imshow("SCRFD Attendance Starter", frame)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    cv.destroyAllWindows()
                    reader.stop()
                    return

    finally:
        reader.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
