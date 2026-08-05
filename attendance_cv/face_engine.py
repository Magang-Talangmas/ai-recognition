from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from insightface.app import FaceAnalysis

from attendance_cv.config import FaceConfig


@dataclass(slots=True)
class FaceResult:
    bbox: np.ndarray
    embedding: np.ndarray | None
    quality_ok: bool
    blur_score: float
    detection_score: float
    kps: np.ndarray | None = None
    _insight_face: object = None  # Holds the insightface Face object for later recognition


class FaceEngine:
    def __init__(self, config: FaceConfig) -> None:
        cv.setNumThreads(os.cpu_count() or 4)
        self.config = config
        self.app = FaceAnalysis(
            name=config.model_pack,
            root=config.model_root,
            allowed_modules=["detection", "recognition"],
            providers=config.providers,
        )
        use_cuda = "CUDAExecutionProvider" in config.providers
        self.app.prepare(
            ctx_id=0 if use_cuda else -1,
            det_thresh=config.detection_threshold,
            det_size=(config.detection_size, config.detection_size),
        )

        # ----------------------------------------------------------------
        # OpenVINO backend — replace onnxruntime sessions after prepare()
        # ----------------------------------------------------------------
        if config.backend == "openvino":
            from attendance_cv.ov_backend import patch_app_with_openvino
            # IR models live alongside the ONNX pack, in a _ov sibling dir
            onnx_dir  = Path(config.model_root) / "models" / config.model_pack
            ir_dir    = Path(config.model_root) / "models" / (config.model_pack + "_ov")
            status    = patch_app_with_openvino(
                self.app,
                ir_model_dir=ir_dir,
                device=config.openvino_device,
            )
            ok = sum(1 for s in status.values() if s != "skipped")
            print(
                f"[FaceEngine] OpenVINO backend active — "
                f"{ok}/{len(status)} models patched on device={config.openvino_device}"
            )
        else:
            print(
                f"[FaceEngine] Using onnxruntime backend "
                f"(providers={config.providers})"
            )


    def detect_only(self, frame: np.ndarray) -> list[FaceResult]:
        """Runs ONLY the extremely fast SCRFD detection."""
        output: list[FaceResult] = []
        bboxes, kpss = self.app.det_model.detect(frame, max_num=0, metric='default')
        if bboxes.shape[0] == 0:
            return output
            
        from insightface.app.common import Face
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i] if kpss is not None else None
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            
            x1, y1, x2, y2 = bbox.astype(int)
            crop = frame[
                max(0, y1):max(0, y2),
                max(0, x1):max(0, x2),
            ]
            blur = self._blur_score(crop)
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            
            quality_ok = (
                width >= self.config.min_face_size
                and height >= self.config.min_face_size
                and blur >= self.config.blur_threshold
            )
            
            output.append(
                FaceResult(
                    bbox=bbox,
                    embedding=None,
                    quality_ok=quality_ok,
                    blur_score=blur,
                    detection_score=float(det_score),
                    kps=kps,
                    _insight_face=face,
                )
            )
        return output

    def recognize_only(self, frame: np.ndarray, faces: list[FaceResult]) -> None:
        """Runs the heavy ArcFace recognition on previously detected faces. Updates the FaceResult in-place."""
        for res in faces:
            if not res.quality_ok or res._insight_face is None:
                continue
            
            for taskname, model in self.app.models.items():
                if taskname == 'detection':
                    continue
                model.get(frame, res._insight_face)
                
            res.embedding = np.asarray(
                res._insight_face.normed_embedding,
                dtype=np.float32,
            )

    def detect(self, frame: np.ndarray) -> list[FaceResult]:
        """Legacy synchronous method that runs both detection and recognition."""
        faces = self.detect_only(frame)
        self.recognize_only(frame, faces)
        return faces

    def build_template(
        self,
        images: list[np.ndarray],
    ) -> tuple[np.ndarray, int]:
        embeddings: list[np.ndarray] = []

        for image in images:
            quality_faces = [
                face
                for face in self.detect(image)
                if face.quality_ok
            ]

            if len(quality_faces) == 0:
                continue

            if len(quality_faces) == 1:
                # Ideal case: exactly one quality face
                best = quality_faces[0]
            else:
                # Multiple quality faces: pick the largest by bounding box area
                best = max(
                    quality_faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )

            embeddings.append(best.embedding)

        if len(embeddings) < 2:
            raise ValueError(
                "At least 2 valid photos with a detectable quality face are required"
            )

        template = np.mean(np.stack(embeddings), axis=0)
        template /= max(float(np.linalg.norm(template)), 1e-12)

        return template.astype(np.float32), len(embeddings)

    @staticmethod
    def _blur_score(image: np.ndarray) -> float:
        if image.size == 0:
            return 0.0
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Use single-precision CV_32F float Laplacian for faster vector unit operations (AVX2)
        return float(cv.Laplacian(gray, cv.CV_32F).var())
