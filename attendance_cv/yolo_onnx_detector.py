"""
yolo_onnx_detector.py — YOLOv10/v8 inference via ONNX Runtime (DirectML / CPU).

Replaces PyTorch/Ultralytics runtime with a pure ONNX Runtime session so that
GPU acceleration works via DmlExecutionProvider on Windows (no CUDA required).

Supports:
  - YOLOv10  output shape (1, N, 6) — [x1, y1, x2, y2, conf, cls_id]  (NMS built-in)
  - YOLOv8   output shape (1, 84, N) — [cx, cy, w, h, cls0..cls79]     (manual NMS)

Usage:
    from attendance_cv.yolo_onnx_detector import YoloOnnxDetector
    detector = YoloOnnxDetector(
        "models/yolov10n.onnx",
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    persons = detector.detect(frame, conf_threshold=0.4)
    for p in persons:
        cv.rectangle(frame, (p.x1, p.y1), (p.x2, p.y2), (0, 255, 0), 2)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PersonBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class YoloOnnxDetector:
    """
    Lightweight YOLOv10/v8 detector powered by ONNX Runtime.

    Parameters
    ----------
    model_path     : Path to the exported .onnx file.
    providers      : ONNX Runtime execution providers in priority order.
                     Default: ["DmlExecutionProvider", "CPUExecutionProvider"]
    input_size     : (height, width) — must match export imgsz (default 640x640).
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None,
        input_size: tuple[int, int] = (640, 640),
    ) -> None:
        if providers is None:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]

        model_path = str(model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"[YoloOnnxDetector] Model not found: {model_path}")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers
        )

        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_h, self.input_w = input_size

        # Detect output format: YOLOv10 vs YOLOv8
        # YOLOv10: (1, N, 6) — last dim == 6
        # YOLOv8:  (1, 84, N) — second dim == 84
        out_shape = self.session.get_outputs()[0].shape
        self._is_yolov10 = (
            len(out_shape) == 3
            and isinstance(out_shape[2], int)
            and out_shape[2] == 6
        )

        active = self.session.get_providers()
        print(
            f"[YoloOnnxDetector] Loaded '{Path(model_path).name}' "
            f"| Format: {'YOLOv10' if self._is_yolov10 else 'YOLOv8'} "
            f"| Providers: {active}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.40,
        nms_threshold: float  = 0.45,
        person_only: bool     = True,
    ) -> list[PersonBox]:
        """
        Run inference on a BGR frame and return filtered detections.

        Parameters
        ----------
        frame          : BGR numpy array (H, W, 3)
        conf_threshold : minimum confidence to keep a detection
        nms_threshold  : IoU threshold for NMS (YOLOv8 only; YOLOv10 has built-in NMS)
        person_only    : if True, keep only class_id == 0 (person)

        Returns
        -------
        List of PersonBox(x1, y1, x2, y2, confidence, class_id) in original frame coords
        """
        orig_h, orig_w = frame.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(frame)

        outputs = self.session.run([self.output_name], {self.input_name: blob})
        raw = outputs[0]  # (1, 300, 6) or (1, 84, N)

        if self._is_yolov10:
            detections = self._parse_yolov10(raw, conf_threshold)
        else:
            detections = self._parse_yolov8(raw, conf_threshold, nms_threshold)

        # Scale back to original frame coordinates
        results: list[PersonBox] = []
        for det in detections:
            if person_only and det.class_id != self.PERSON_CLASS_ID:
                continue

            x1 = int(max(0, (det.x1 - pad_x) / scale))
            y1 = int(max(0, (det.y1 - pad_y) / scale))
            x2 = int(min(orig_w, (det.x2 - pad_x) / scale))
            y2 = int(min(orig_h, (det.y2 - pad_y) / scale))

            if x2 > x1 and y2 > y1:
                results.append(PersonBox(x1, y1, x2, y2, det.confidence, det.class_id))

        return results

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float]:
        """
        Letterbox resize → BGR→RGB → normalize [0,1] → NCHW float32 blob.
        Returns (blob, scale, pad_x, pad_y)
        """
        orig_h, orig_w = frame.shape[:2]
        scale = min(self.input_w / orig_w, self.input_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (self.input_w - new_w) / 2
        pad_y = (self.input_h - new_h) / 2

        resized = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        canvas  = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        x0 = int(pad_x)
        y0 = int(pad_y)
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

        # BGR → RGB, normalize [0,1], add batch dim → (1, 3, H, W)
        rgb  = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[np.newaxis])

        return blob, scale, pad_x, pad_y

    # ------------------------------------------------------------------
    # Output parsers
    # ------------------------------------------------------------------

    def _parse_yolov10(
        self,
        raw: np.ndarray,
        conf_threshold: float,
    ) -> list[PersonBox]:
        """
        YOLOv10 output: (1, N, 6) — [x1, y1, x2, y2, confidence, class_id]
        NMS is handled inside the model; only filter by confidence here.
        """
        preds = raw[0]  # (N, 6)
        mask  = preds[:, 4] >= conf_threshold
        preds = preds[mask]

        results = []
        for det in preds:
            x1, y1, x2, y2, conf, cls = det
            results.append(PersonBox(
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                confidence=float(conf), class_id=int(cls),
            ))
        return results

    def _parse_yolov8(
        self,
        raw: np.ndarray,
        conf_threshold: float,
        nms_threshold: float,
    ) -> list[PersonBox]:
        """
        YOLOv8 output: (1, 84, N) — [cx, cy, w, h, cls0..cls79]
        Applies NMS manually via cv2.dnn.NMSBoxes.
        """
        preds = raw[0].T  # (N, 84)

        cx, cy, w, h  = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        class_scores  = preds[:, 4:]
        class_ids     = class_scores.argmax(axis=1)
        confidences   = class_scores.max(axis=1)

        mask        = confidences >= conf_threshold
        cx, cy      = cx[mask], cy[mask]
        w, h        = w[mask],  h[mask]
        class_ids   = class_ids[mask]
        confidences = confidences[mask]

        if len(confidences) == 0:
            return []

        x1s = (cx - w / 2).astype(int)
        y1s = (cy - h / 2).astype(int)
        ws  = w.astype(int)
        hs  = h.astype(int)

        boxes_for_nms = np.stack([x1s, y1s, ws, hs], axis=1).tolist()
        indices = cv.dnn.NMSBoxes(
            boxes_for_nms, confidences.tolist(), conf_threshold, nms_threshold
        )

        results = []
        flat_indices = indices.flatten() if len(indices) else []
        for i in flat_indices:
            results.append(PersonBox(
                x1=x1s[i], y1=y1s[i],
                x2=x1s[i] + ws[i], y2=y1s[i] + hs[i],
                confidence=float(confidences[i]),
                class_id=int(class_ids[i]),
            ))
        return results
