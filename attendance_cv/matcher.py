from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from attendance_cv.config import FaceConfig


@dataclass(slots=True)
class MatchResult:
    employee_id: str | None
    score: float
    margin: float


class FaceMatcher:
    def __init__(self, embedding_file: str, config: FaceConfig) -> None:
        self.config = config
        path = Path(embedding_file)
        if not path.exists():
            raise FileNotFoundError(
                f"Embedding file not found: {path}. Run enroll_faces.py first."
            )

        data = np.load(path)
        self.employee_ids = data["employee_ids"].astype(str)
        self.templates = data["templates"].astype(np.float32)

        norms = np.linalg.norm(self.templates, axis=1, keepdims=True)
        self.templates = self.templates / np.maximum(norms, 1e-12)

    def match(self, embedding: np.ndarray) -> MatchResult:
        vector = embedding.astype(np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)

        scores = self.templates @ vector
        order = np.argsort(scores)[::-1]

        best_idx = int(order[0])
        best_score = float(scores[best_idx])
        second_score = float(scores[order[1]]) if len(order) > 1 else -1.0
        margin = best_score - second_score

        if best_score < self.config.match_threshold:
            return MatchResult(None, best_score, margin)

        if margin < self.config.match_margin:
            return MatchResult(None, best_score, margin)

        return MatchResult(
            str(self.employee_ids[best_idx]),
            best_score,
            margin,
        )

    def match_batch(self, embeddings: list[np.ndarray]) -> list[MatchResult]:
        if not embeddings:
            return []
        
        matrix = np.stack(embeddings).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= np.maximum(norms, 1e-12)

        # Batch matrix multiplication: (N_templates, D) @ (D, Batch_Size) -> (N_templates, Batch_Size)
        all_scores = self.templates @ matrix.T
        results: list[MatchResult] = []

        for b in range(all_scores.shape[1]):
            scores = all_scores[:, b]
            order = np.argsort(scores)[::-1]
            best_idx = int(order[0])
            best_score = float(scores[best_idx])
            second_score = float(scores[order[1]]) if len(order) > 1 else -1.0
            margin = best_score - second_score

            if best_score < self.config.match_threshold or margin < self.config.match_margin:
                results.append(MatchResult(None, best_score, margin))
            else:
                results.append(MatchResult(str(self.employee_ids[best_idx]), best_score, margin))

        return results
