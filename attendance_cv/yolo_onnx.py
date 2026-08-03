"""
attendance_cv/yolo_onnx.py

Pure onnxruntime + OpenCV person detector.
No torch / ultralytics dependency — works regardless of AppLocker/WDAC policies.

Supports both model output formats:
  YOLOv10 export : (1, 300, 6)   — NMS-free [x1,y1,x2,y2,conf,cls]
  YOLOv8  export : (1, 84, 8400) — needs transpose + cv2.dnn.NMSBoxes

Preprocessing uses OpenCV (cv2.resize, cvtColor) for zero extra deps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2 as cv
import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    """One detected person bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: int = field(default=-1)

    @property
    def box_xywh(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


class YoloOnnx:
    """
    Lightweight YOLOv8 / YOLOv10 person detector via onnxruntime.

    Usage::

        detector = YoloOnnx("models/yolov10n.onnx", confidence=0.40)
        detections = detector.detect(frame)   # list[Detection]
    """

    INPUT_SIZE = 640

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.40,
        nms_iou: float = 0.45,
        providers: list[str] | None = None,
    ) -> None:
        self.confidence = confidence
        self.nms_iou = nms_iou

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        # Auto-detect output format from ONNX graph metadata
        out0 = self.session.get_outputs()[0]
        shape = out0.shape  # e.g. [1, 300, 6] or [1, 84, 8400]
        self._is_v10_format = (
            len(shape) == 3
            and isinstance(shape[2], int)
            and shape[2] == 6
        )
        print(
            f"[YoloOnnx] Loaded {model_path} | "
            f"format={'YOLOv10-NMSfree' if self._is_v10_format else 'YOLOv8-NMS'} | "
            f"output shape={shape}"
        )

    # ── Public ───────────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        class_id: int = 0,  # 0 = person (COCO)
    ) -> list[Detection]:
        """Run inference on a BGR OpenCV frame, return person detections."""
        orig_h, orig_w = frame.shape[:2]
        blob = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        raw = outputs[0]

        if self._is_v10_format:
            return self._postprocess_v10(raw, orig_w, orig_h, class_id)
        return self._postprocess_v8(raw, orig_w, orig_h, class_id)

    # ── Preprocessing (OpenCV) ────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        OpenCV-based preprocessing:
          BGR → RGB → resize 640×640 → normalise /255 → CHW → add batch dim
        """
        resized = cv.resize(frame, (self.INPUT_SIZE, self.INPUT_SIZE))
        rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))          # HWC → CHW
        return np.expand_dims(blob, axis=0)            # → NCHW

    # ── Postprocessing ────────────────────────────────────────────────────────

    def _postprocess_v10(
        self,
        raw: np.ndarray,
        orig_w: int,
        orig_h: int,
        class_id: int,
    ) -> list[Detection]:
        """
        YOLOv10 NMS-free output: shape (1, 300, 6).
        Each row: [x1, y1, x2, y2, confidence, class_id]  — already in input coords.
        """
        sx = orig_w / self.INPUT_SIZE
        sy = orig_h / self.INPUT_SIZE
        detections: list[Detection] = []

        for det in raw[0]:
            x1, y1, x2, y2, conf, cls = det
            if float(conf) < self.confidence:
                continue
            if int(cls) != class_id:
                continue
            detections.append(
                Detection(
                    x1=max(0, int(x1 * sx)),
                    y1=max(0, int(y1 * sy)),
                    x2=min(orig_w, int(x2 * sx)),
                    y2=min(orig_h, int(y2 * sy)),
                    confidence=float(conf),
                )
            )
        return detections

    def _postprocess_v8(
        self,
        raw: np.ndarray,
        orig_w: int,
        orig_h: int,
        class_id: int,
    ) -> list[Detection]:
        """
        YOLOv8 output: shape (1, 84, 8400).
        Transpose → (8400, 84) → filter class & conf → OpenCV NMS.
        """
        sx = orig_w / self.INPUT_SIZE
        sy = orig_h / self.INPUT_SIZE

        preds = raw[0].T  # (8400, 84)
        boxes_xywh: list[list[int]] = []
        scores: list[float] = []

        for pred in preds:
            cx, cy, bw, bh = pred[:4]
            class_scores = pred[4:]
            cls = int(np.argmax(class_scores))
            conf = float(class_scores[cls])

            if conf < self.confidence or cls != class_id:
                continue

            x1 = int((cx - bw / 2) * sx)
            y1 = int((cy - bh / 2) * sy)
            w  = int(bw * sx)
            h  = int(bh * sy)
            boxes_xywh.append([x1, y1, w, h])
            scores.append(conf)

        if not boxes_xywh:
            return []

        # OpenCV NMS (no torch required)
        indices = cv.dnn.NMSBoxes(
            boxes_xywh, scores, self.confidence, self.nms_iou
        )
        detections: list[Detection] = []
        for i in (indices.flatten() if hasattr(indices, "flatten") else indices):
            x, y, w, h = boxes_xywh[int(i)]
            detections.append(
                Detection(
                    x1=max(0, x),
                    y1=max(0, y),
                    x2=min(orig_w, x + w),
                    y2=min(orig_h, y + h),
                    confidence=scores[int(i)],
                )
            )
        return detections
