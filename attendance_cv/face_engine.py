from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np
from insightface.app import FaceAnalysis

from attendance_cv.config import FaceConfig


@dataclass(slots=True)
class FaceResult:
    bbox: np.ndarray
    embedding: np.ndarray
    quality_ok: bool
    blur_score: float
    detection_score: float


class FaceEngine:
    def __init__(self, config: FaceConfig) -> None:
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

    def detect(self, frame: np.ndarray) -> list[FaceResult]:
        output: list[FaceResult] = []

        for face in self.app.get(frame):
            bbox = np.asarray(face.bbox, dtype=np.float32)
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

            embedding = np.asarray(
                face.normed_embedding,
                dtype=np.float32,
            )

            output.append(
                FaceResult(
                    bbox=bbox,
                    embedding=embedding,
                    quality_ok=quality_ok,
                    blur_score=blur,
                    detection_score=float(face.det_score),
                )
            )

        return output

    def build_template(
        self,
        images: list[np.ndarray],
    ) -> tuple[np.ndarray, int]:
        embeddings: list[np.ndarray] = []

        for image in images:
            valid = [
                face
                for face in self.detect(image)
                if face.quality_ok
            ]
            if valid:
                main_face = max(
                    valid,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                embeddings.append(main_face.embedding)

        if not embeddings:
            raise ValueError(
                "No valid quality face was detected in any of the photos"
            )

        template = np.mean(np.stack(embeddings), axis=0)
        template /= max(float(np.linalg.norm(template)), 1e-12)

        return template.astype(np.float32), len(embeddings)

    @staticmethod
    def _blur_score(image: np.ndarray) -> float:
        if image.size == 0:
            return 0.0
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return float(cv.Laplacian(gray, cv.CV_64F).var())
