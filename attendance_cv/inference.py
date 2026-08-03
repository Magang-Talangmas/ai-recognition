"""
attendance_cv/inference.py

Core real-time inference pipeline — NO torch/ultralytics dependency.

Pipeline per frame:
  1. RtspReader   → latest BGR frame via OpenCV VideoCapture (background thread)
  2. YoloOnnx     → person bounding boxes via onnxruntime (no torch)
  3. IoUTracker   → stable track IDs across frames (replaces ByteTrack)
  4. FaceEngine   → SCRFD face detection (InsightFace / onnxruntime)
  5. FaceMatcher  → ArcFace 512-d cosine similarity → employee identity
  6. Vote window  → stable_identity() confirms after N consistent votes
  7. Crossing     → detect line crossing → create SQLite attendance event
  8. EventBus     → broadcast new event JSON to all WebSocket clients
  9. OpenCV       → annotate frame, imencode JPEG → MJPEG stream
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Optional

import cv2 as cv
import numpy as np

from attendance_cv.config import AppConfig
from attendance_cv.database import AttendanceDB
from attendance_cv.event_bus import EventBus
from attendance_cv.face_engine import FaceEngine
from attendance_cv.matcher import FaceMatcher
from attendance_cv.rtsp_reader import RtspReader
from attendance_cv.tracker import IoUTracker
from attendance_cv.vision_utils import (
    IdentityVote,
    associate_face_to_track,
    crossing_direction,
    stable_identity,
)
from attendance_cv.yolo_onnx import YoloOnnx

# ── colour palette (BGR) ──────────────────────────────────────────────────────
_CLR_KNOWN   = (50,  220, 100)   # green  — confirmed identity
_CLR_UNKNOWN = (180, 180, 180)   # grey   — unrecognised person
_CLR_LINE    = (0,   230, 255)   # cyan   — crossing line
_CLR_FACE    = (80,  160, 255)   # blue   — SCRFD face box


def passes_liveness(allow_insecure: bool) -> bool:
    # TODO: replace with a real PAD/liveness model before production.
    return allow_insecure


# ── Main entry point ─────────────────────────────────────────────────────────

def run_inference(
    config: AppConfig,
    database: AttendanceDB,
    bus: Optional[EventBus] = None,
) -> None:
    """
    Blocking inference loop. Intended to run in a dedicated thread.

    Parameters
    ----------
    config   : loaded AppConfig
    database : shared AttendanceDB instance
    bus      : optional EventBus — if None, WebSocket push and MJPEG are skipped
    """

    # ── Model loading (onnxruntime only, no torch) ────────────────────────────
    print("[Inference] Loading YOLOv10 ONNX …")
    detector = YoloOnnx(
        model_path=config.person.model_path,
        confidence=config.person.confidence,
        providers=config.face.providers,
    )

    print("[Inference] Loading SCRFD + ArcFace (InsightFace buffalo_l) …")
    face_engine = FaceEngine(config.face)

    print("[Inference] Loading face matcher …")
    matcher = FaceMatcher(config.attendance.embedding_file, config.face)

    # ── Per-track state ───────────────────────────────────────────────────────
    tracker = IoUTracker(iou_threshold=0.30, max_age=30)

    track_votes: dict[int, deque[IdentityVote]] = defaultdict(
        lambda: deque(maxlen=config.face.vote_window)
    )
    confirmed_identities: dict[int, tuple[str, float]] = {}
    previous_sides: dict[int, int] = {}
    last_seen: dict[int, float] = {}

    # ── RTSP reader (OpenCV-backed background thread) ─────────────────────────
    reader = RtspReader(
        url=config.camera.source,
        reconnect_delay=float(config.camera.reconnect_delay_seconds),
    ).start()

    print(f"[Inference] Waiting for first frame from: {config.camera.source}")

    frame_count = 0

    try:
        while True:
            frame = reader.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            if frame_count % config.camera.process_every_n_frames != 0:
                time.sleep(0.001)
                continue

            # ── 1. Person detection (YOLOv10 ONNX) ───────────────────────────
            raw_dets = detector.detect(frame, class_id=0)
            det_boxes = [d.xyxy for d in raw_dets]

            # ── 2. IoU tracking → stable track IDs ───────────────────────────
            people: dict[int, tuple[int, int, int, int]] = tracker.update(det_boxes)

            # ── 3. Face detection (SCRFD) + 4. Recognition (ArcFace) ─────────
            detected_faces = face_engine.detect(frame)

            for face in detected_faces:
                if not face.quality_ok:
                    continue

                track_id = associate_face_to_track(face, people)
                if track_id is None:
                    continue

                if not passes_liveness(config.attendance.allow_insecure_no_liveness):
                    track_votes[track_id].append(IdentityVote(None, 0.0))
                    continue

                # 5. Embedding match → 6. Voting
                match = matcher.match(face.embedding)
                track_votes[track_id].append(
                    IdentityVote(match.employee_id, match.score)
                )

                employee_id, score = stable_identity(
                    track_votes[track_id],
                    config.face.votes_required,
                )
                if employee_id:
                    confirmed_identities[track_id] = (employee_id, score)

                    # Direct CHECK_IN trigger (no crossing line needed)
                    event_id = database.create_pending_event(
                        camera_id=config.camera.camera_id,
                        track_id=track_id,
                        employee_id=employee_id,
                        direction="IN",
                        event_type="CHECK_IN",
                        similarity=score,
                    )
                    if event_id is not None:
                        print(
                            f"[Inference] CHECK_IN EVENT {event_id}: "
                            f"{employee_id} (score={score:.3f})"
                        )
                        # Push to WebSocket clients
                        if bus is not None:
                            bus.publish_event({
                                "event_id": event_id,
                                "employee_id": employee_id,
                                "event_type": "CHECK_IN",
                                "direction": "IN",
                                "similarity": round(score, 4),
                                "status": "PENDING_CONFIRMATION",
                                "camera_id": config.camera.camera_id,
                            })

                last_seen[track_id] = time.monotonic()

            # ── Stale track cleanup ───────────────────────────────────────────
            now = time.monotonic()
            stale = [
                tid for tid, seen_at in last_seen.items()
                if now - seen_at > 10
            ]
            for tid in stale:
                track_votes.pop(tid, None)
                confirmed_identities.pop(tid, None)
                last_seen.pop(tid, None)

            # ── 7. OpenCV annotation ──────────────────────────────────────────
            _annotate_frame(frame, people, confirmed_identities, detected_faces)

            if config.camera.show_preview:
                win_title = "Talangmas Attendance — Q to quit"
                cv.namedWindow(win_title, cv.WINDOW_NORMAL)
                cv.imshow(win_title, frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

            # MJPEG publish for /stream/mjpeg endpoint
            if bus is not None:
                enc_ok, jpeg_buf = cv.imencode(
                    ".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 72]
                )
                if enc_ok:
                    bus.publish_frame(jpeg_buf.tobytes())

    finally:
        reader.stop()
        if config.camera.show_preview:
            cv.destroyAllWindows()
        print("[Inference] Loop exited.")


# ── OpenCV annotation ─────────────────────────────────────────────────────────

def _annotate_frame(
    frame: np.ndarray,
    people: dict[int, tuple[int, int, int, int]],
    confirmed: dict[int, tuple[str, float]],
    faces: list,
) -> None:
    """Draw boxes, labels, face detections — all via OpenCV."""
    for track_id, (x1, y1, x2, y2) in people.items():
        label, score = confirmed.get(track_id, ("UNKNOWN", 0.0))
        color = _CLR_KNOWN if label != "UNKNOWN" else _CLR_UNKNOWN

        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"#{track_id} {label} ({score:.2f})" if label != "UNKNOWN" else f"#{track_id} {label}"
        (tw, th), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        ty = max(th + baseline + 4, y1 - 4)
        cv.rectangle(frame, (x1, ty - th - baseline - 4), (x1 + tw + 6, ty + 2), color, cv.FILLED)
        cv.putText(frame, text, (x1 + 3, ty - baseline),
                   cv.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv.LINE_AA)

    # SCRFD face boxes
    for face in faces:
        fx1, fy1, fx2, fy2 = face.bbox.astype(int)
        cv.rectangle(frame, (fx1, fy1), (fx2, fy2), _CLR_FACE, 1)
        cv.putText(frame, f"det:{face.detection_score:.2f}",
                   (fx1, max(12, fy1 - 4)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.40, _CLR_FACE, 1, cv.LINE_AA)

    # Timestamp & info
    cv.putText(frame, time.strftime("%Y-%m-%d  %H:%M:%S"),
               (8, frame.shape[0] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv.LINE_AA)
