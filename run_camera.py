from __future__ import annotations

import os
import queue
import threading
import time

import cv2 as cv
import numpy as np
from ultralytics import YOLO

from attendance_cv.config import load_config
from attendance_cv.face_engine import FaceEngine
from attendance_cv.matcher import FaceMatcher

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


def associate_face_to_person_box(
    face_bbox: np.ndarray,
    people_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    fx1, fy1, fx2, fy2 = face_bbox
    center_x = (fx1 + fx2) / 2
    center_y = (fy1 + fy2) / 2
    for px1, py1, px2, py2 in people_boxes:
        if px1 <= center_x <= px2 and py1 <= center_y <= py2:
            return (px1, py1, px2, py2)
    return None


class AsyncAIWorker:
    """Asynchronous AI Worker Thread (Pure Detection Mode - No Tracking / No Database)
    Processes YOLOv10 prediction and SCRFD face recognition in parallel."""

    def __init__(
        self,
        person_model: YOLO,
        face_engine: FaceEngine,
        matcher: FaceMatcher,
        config: any,
    ) -> None:
        self.person_model = person_model
        self.face_engine = face_engine
        self.matcher = matcher
        self.config = config

        self.input_queue = queue.Queue(maxsize=1)
        self.stopped = False
        self.lock = threading.Lock()

        self.latest_people: list[tuple[int, int, int, int]] = []
        self.latest_face_boxes: list[
            tuple[tuple[int, int, int, int], str, float]
        ] = []
        self.latest_person_identities: dict[
            tuple[int, int, int, int], tuple[str, float]
        ] = {}
        self.batch_size = getattr(config.face, "batch_size", 32)

        # FPS & Inference Latency Metrics
        self.ai_fps = 0.0
        self.ai_latency_ms = 0.0
        self._ai_frame_count = 0
        self._ai_start_time = time.time()

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def process_async(self, frame: np.ndarray) -> None:
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.input_queue.put_nowait(frame)
        except queue.Full:
            pass

    def get_latest_overlays(
        self,
    ) -> tuple[
        list[tuple[int, int, int, int]],
        list[tuple[tuple[int, int, int, int], str, float]],
        dict[tuple[int, int, int, int], tuple[str, float]],
    ]:
        with self.lock:
            return (
                self.latest_people.copy(),
                self.latest_face_boxes.copy(),
                self.latest_person_identities.copy(),
            )

    def get_stats(self) -> tuple[float, float]:
        with self.lock:
            return self.ai_fps, self.ai_latency_ms

    def _run(self) -> None:
        while not self.stopped:
            try:
                frame = self.input_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            t_start = time.time()

            # 1. Pure YOLOv10 Person Detection (No tracking overhead)
            result = self.person_model.predict(
                frame,
                classes=[0],
                conf=self.config.person.confidence,
                verbose=False,
            )[0]

            people_boxes: list[tuple[int, int, int, int]] = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                people_boxes = [tuple(map(int, box)) for box in boxes]

            # 2. SCRFD Face Detection & Batch Matcher
            detected_faces = self.face_engine.detect(frame)
            face_boxes_to_draw: list[
                tuple[tuple[int, int, int, int], str, float]
            ] = []
            person_identities: dict[
                tuple[int, int, int, int], tuple[str, float]
            ] = {}

            if detected_faces:
                valid_faces = [f for f in detected_faces if f.quality_ok]
                if valid_faces:
                    for b_start in range(
                        0, len(valid_faces), self.batch_size
                    ):
                        batch_chunk = valid_faces[
                            b_start : b_start + self.batch_size
                        ]
                        batch_embeddings = [f.embedding for f in batch_chunk]
                        batch_matches = self.matcher.match_batch(
                            batch_embeddings
                        )

                        for face, match in zip(batch_chunk, batch_matches):
                            fx1, fy1, fx2, fy2 = map(int, face.bbox)
                            label = (
                                match.employee_id
                                if match.employee_id
                                else "UNKNOWN"
                            )
                            score = match.score
                            face_boxes_to_draw.append(
                                ((fx1, fy1, fx2, fy2), label, score)
                            )

                            p_box = associate_face_to_person_box(
                                face.bbox, people_boxes
                            )
                            if p_box:
                                person_identities[p_box] = (label, score)

            t_end = time.monotonic()
            latency = (t_end - t_start) * 1000.0

            self._ai_frame_count += 1
            if t_end - self._ai_start_time >= 1.0:
                calc_fps = self._ai_frame_count / (t_end - self._ai_start_time)
                self._ai_frame_count = 0
                self._ai_start_time = t_end
                with self.lock:
                    self.ai_fps = calc_fps
                    self.ai_latency_ms = latency
            else:
                with self.lock:
                    self.ai_latency_ms = latency

            with self.lock:
                self.latest_people = people_boxes
                self.latest_face_boxes = face_boxes_to_draw
                self.latest_person_identities = person_identities

    def stop(self) -> None:
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> None:
    config = load_config()

    print("=== Pure AI Visual Recognition Pipeline (No Tracking / No DB Purpose) ===")
    print("1. YOLOv10: Pure Person Object Detection (.predict)")
    print("2. SCRFD: Face Detection & Batch Size 32 Matrix Matcher")
    print("3. Renderer: High-FPS Decoupled Preview Stream")

    person_model = YOLO(config.person.model_path)
    face_engine = FaceEngine(config.face)
    matcher = FaceMatcher(
        config.attendance.embedding_file,
        config.face,
    )

    window_name = "SCRFD Attendance Starter"
    if config.camera.show_preview:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_name, 1280, 720)

    while True:
        source_val = parse_source(config.camera.source)
        stream = FreshRTSPStream(source_val)

        start_wait = time.time()
        while not stream.is_opened() and time.time() - start_wait < 5.0:
            time.sleep(0.1)

        if not stream.is_opened():
            print("Camera unavailable. Reconnecting...")
            stream.release()
            time.sleep(config.camera.reconnect_delay_seconds)
            continue

        print("Camera stream connected (Pure Visual Mode).")

        ai_worker = AsyncAIWorker(person_model, face_engine, matcher, config)

        render_fps = 0.0
        render_frame_count = 0
        render_start_time = time.time()

        while stream.is_opened():
            ok, frame = stream.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            # FPS calculation for UI rendering stream
            render_frame_count += 1
            now = time.time()
            if now - render_start_time >= 1.0:
                render_fps = render_frame_count / (now - render_start_time)
                render_frame_count = 0
                render_start_time = now

            # Queue frame asynchronously to background AI worker
            ai_worker.process_async(frame)

            # Get latest overlays & AI stats
            people_boxes, face_boxes, person_identities = (
                ai_worker.get_latest_overlays()
            )
            ai_fps, ai_ms = ai_worker.get_stats()

            # Render smooth 30-60 FPS video stream
            if config.camera.show_preview:
                # 1. Render Face Bounding Boxes (Green / Coral)
                for (fx1, fy1, fx2, fy2), f_label, f_score in face_boxes:
                    face_color = (
                        (0, 255, 127)
                        if f_label != "UNKNOWN"
                        else (0, 165, 255)
                    )
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

                # 2. Render Person Bounding Boxes (Cyan / Gold)
                for p_box in people_boxes:
                    x1, y1, x2, y2 = p_box
                    label, score = person_identities.get(
                        p_box, ("UNKNOWN", 0.0)
                    )
                    person_color = (
                        (255, 200, 0)
                        if label != "UNKNOWN"
                        else (220, 220, 220)
                    )
                    cv.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        person_color,
                        2,
                    )
                    cv.putText(
                        frame,
                        f"Person: {label}",
                        (x1, max(20, y1 - 8)),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        person_color,
                        2,
                        cv.LINE_AA,
                    )

                # 3. Render Real-Time FPS HUD Overlay Badge
                cv.rectangle(frame, (15, 15), (290, 75), (20, 20, 20), cv.FILLED)
                cv.rectangle(frame, (15, 15), (290, 75), (0, 255, 127), 1)

                cv.putText(
                    frame,
                    f"STREAM FPS : {render_fps:.1f} (Native Smooth)",
                    (25, 38),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 127),
                    1,
                    cv.LINE_AA,
                )
                cv.putText(
                    frame,
                    f"AI INFERENCE: {ai_fps:.1f} FPS ({ai_ms:.0f} ms)",
                    (25, 60),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA,
                )

                preview_w = 1152
                preview_h = int(
                    frame.shape[0] * (preview_w / frame.shape[1])
                )
                preview_frame = cv.resize(frame, (preview_w, preview_h))
                cv.imshow(window_name, preview_frame)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    ai_worker.stop()
                    stream.release()
                    cv.destroyAllWindows()
                    return

        ai_worker.stop()
        stream.release()
        time.sleep(config.camera.reconnect_delay_seconds)


if __name__ == "__main__":
    main()
