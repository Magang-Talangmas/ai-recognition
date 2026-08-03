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

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import cv2 as cv
import numpy as np

from attendance_cv.config import AppConfig
from attendance_cv.database import AttendanceDB
from attendance_cv.event_bus import EventBus
from attendance_cv.face_engine import FaceEngine, FaceResult
from attendance_cv.matcher import FaceMatcher
from attendance_cv.rtsp_reader import RtspReader
from attendance_cv.tracker import IoUTracker
from attendance_cv.vision_utils import (
    IdentityVote,
    associate_face_to_track,
    stable_identity,
)
from attendance_cv.yolo_onnx import YoloOnnx

# ── colour palette (BGR) ──────────────────────────────────────────────────────
_CLR_KNOWN   = (50,  220, 100)   # green  — confirmed identity
_CLR_UNKNOWN = (180, 180, 180)   # grey   — unrecognised person
_CLR_FACE    = (80,  160, 255)   # blue   — SCRFD face box


def passes_liveness(allow_insecure: bool) -> bool:
    # TODO: replace with a real PAD/liveness model before production.
    return allow_insecure


@dataclass
class SharedDetectionState:
    """Thread-safe snapshot shared between background AI worker and display stream."""
    people: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)
    confirmed: dict[int, tuple[str, float]] = field(default_factory=dict)
    faces: list[FaceResult] = field(default_factory=list)
    ai_fps: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


def _ai_worker_loop(
    config: AppConfig,
    database: AttendanceDB,
    bus: Optional[EventBus],
    reader: RtspReader,
    shared_state: SharedDetectionState,
    stop_event: threading.Event,
) -> None:
    """Background AI inference worker: processes YOLO + SCRFD + ArcFace without blocking video stream."""
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

    tracker = IoUTracker(iou_threshold=0.30, max_age=30)
    track_votes: dict[int, deque[IdentityVote]] = defaultdict(
        lambda: deque(maxlen=config.face.vote_window)
    )
    confirmed_identities: dict[int, tuple[str, float]] = {}
    last_seen: dict[int, float] = {}

    fps_count = 0
    fps_start = time.monotonic()
    current_ai_fps = 0.0

    print("[Inference] AI background worker started.")

    while not stop_event.is_set():
        frame = reader.get_frame()
        if frame is None:
            time.sleep(0.02)
            continue

        loop_start = time.monotonic()

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

                # Direct CHECK_IN trigger
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
            if now - seen_at > 8
        ]
        for tid in stale:
            track_votes.pop(tid, None)
            confirmed_identities.pop(tid, None)
            last_seen.pop(tid, None)

        # ── Calculate AI FPS ──────────────────────────────────────────────
        fps_count += 1
        elapsed = time.monotonic() - fps_start
        if elapsed >= 1.0:
            current_ai_fps = fps_count / elapsed
            fps_count = 0
            fps_start = time.monotonic()

        # ── Atomic update of shared state ─────────────────────────────────
        with shared_state.lock:
            shared_state.people = people
            shared_state.confirmed = confirmed_identities.copy()
            shared_state.faces = detected_faces
            shared_state.ai_fps = current_ai_fps

        # Yield a few milliseconds to CPU
        time.sleep(0.005)

    print("[Inference] AI background worker stopped.")


# ── Main entry point (Display / Stream loop @ 25-30 FPS) ─────────────────────

def run_inference(
    config: AppConfig,
    database: AttendanceDB,
    bus: Optional[EventBus] = None,
) -> None:
    """
    Decoupled real-time display and streaming loop.
    Runs at camera native FPS (smooth 25-30 FPS) while AI executes in background.
    """
    shared_state = SharedDetectionState()
    stop_event = threading.Event()

    reader = RtspReader(
        url=config.camera.source,
        reconnect_delay=float(config.camera.reconnect_delay_seconds),
    ).start()

    print(f"[Inference] Connecting to stream: {config.camera.source}")

    # Launch background AI worker thread
    ai_thread = threading.Thread(
        target=_ai_worker_loop,
        args=(config, database, bus, reader, shared_state, stop_event),
        daemon=True,
        name="ai-worker",
    )
    ai_thread.start()

    display_fps_count = 0
    display_fps_start = time.monotonic()
    current_display_fps = 0.0

    if config.camera.show_preview:
        win_title = "Talangmas Attendance — Q to quit"
        cv.namedWindow(win_title, cv.WINDOW_NORMAL)

    try:
        while True:
            frame = reader.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            # Calculate Display FPS
            display_fps_count += 1
            elapsed = time.monotonic() - display_fps_start
            if elapsed >= 1.0:
                current_display_fps = display_fps_count / elapsed
                display_fps_count = 0
                display_fps_start = time.monotonic()

            # Retrieve latest detection snapshot from background AI
            with shared_state.lock:
                people = shared_state.people
                confirmed = shared_state.confirmed
                faces = shared_state.faces
                ai_fps = shared_state.ai_fps

            # Render overlay annotations
            _annotate_frame(
                frame=frame,
                people=people,
                confirmed=confirmed,
                faces=faces,
                display_fps=current_display_fps,
                ai_fps=ai_fps,
            )

            # Local OpenCV window
            if config.camera.show_preview:
                cv.imshow("Talangmas Attendance — Q to quit", frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

            # MJPEG publish for /stream/mjpeg endpoint
            if bus is not None:
                enc_ok, jpeg_buf = cv.imencode(
                    ".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 75]
                )
                if enc_ok:
                    bus.publish_frame(jpeg_buf.tobytes())

            # Maintain natural camera pace (~30 FPS display)
            time.sleep(0.01)

    finally:
        stop_event.set()
        ai_thread.join(timeout=3)
        reader.stop()
        if config.camera.show_preview:
            cv.destroyAllWindows()
        print("[Inference] Loop exited.")


# ── OpenCV annotation ─────────────────────────────────────────────────────────

def _annotate_frame(
    frame: np.ndarray,
    people: dict[int, tuple[int, int, int, int]],
    confirmed: dict[int, tuple[str, float]],
    faces: list[FaceResult],
    display_fps: float = 0.0,
    ai_fps: float = 0.0,
) -> None:
    """Draw boxes, labels, face detections, and real-time HUD stats."""
    # 1. Person bounding boxes
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

    # 2. SCRFD face boxes
    for face in faces:
        fx1, fy1, fx2, fy2 = face.bbox.astype(int)
        cv.rectangle(frame, (fx1, fy1), (fx2, fy2), _CLR_FACE, 1)
        cv.putText(frame, f"det:{face.detection_score:.2f}",
                   (fx1, max(12, fy1 - 4)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.40, _CLR_FACE, 1, cv.LINE_AA)

    # 3. Status HUD: FPS + Timestamp
    hud_text = f"STREAM: {display_fps:.1f} FPS | AI: {ai_fps:.1f} FPS | {time.strftime('%Y-%m-%d %H:%M:%S')}"
    (hw, hh), hbase = cv.getTextSize(hud_text, cv.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv.rectangle(frame, (6, frame.shape[0] - hh - hbase - 10), (hw + 16, frame.shape[0] - 4), (0, 0, 0), cv.FILLED)
    cv.putText(frame, hud_text,
               (10, frame.shape[0] - 8),
               cv.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1, cv.LINE_AA)
