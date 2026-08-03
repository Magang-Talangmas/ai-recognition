"""
attendance_cv/tracker.py

Minimal IoU-based multi-object tracker — replaces ByteTrack (ultralytics).

Algorithm:
  1. Build IoU matrix between existing tracks and new detections.
  2. Greedy highest-IoU-first matching (fast, good enough for ≤30 people).
  3. Unmatched tracks age up; removed after max_age frames.
  4. Unmatched detections become new tracks with fresh IDs.

Track IDs are monotonically increasing integers, never reused.
"""
from __future__ import annotations

import numpy as np


class Track:
    _next_id: int = 1

    __slots__ = ("id", "box", "hits", "time_since_update")

    def __init__(self, box: tuple[int, int, int, int]) -> None:
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.hits = 1
        self.time_since_update = 0

    def update(self, box: tuple[int, int, int, int]) -> None:
        self.box = box
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self) -> None:
        self.time_since_update += 1


class IoUTracker:
    """
    Greedy IoU tracker.

    Usage::

        tracker = IoUTracker()
        # each frame:
        people = tracker.update([(x1,y1,x2,y2), ...])
        # → dict[track_id, (x1,y1,x2,y2)]
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_age: int = 30,
        min_hits: int = 1,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[Track] = []

    def update(
        self,
        detections: list[tuple[int, int, int, int]],
    ) -> dict[int, tuple[int, int, int, int]]:
        """
        Parameters
        ----------
        detections : list of (x1, y1, x2, y2)

        Returns
        -------
        dict mapping stable track_id → (x1, y1, x2, y2)
        Only tracks with hits >= min_hits are returned.
        """
        if not detections:
            for t in self.tracks:
                t.mark_missed()
            self._prune()
            return self._visible()

        if not self.tracks:
            for det in detections:
                self.tracks.append(Track(det))
            self._prune()
            return self._visible()

        # ── IoU matrix ─────────────────────────────────────────────────────
        iou_mat = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                iou_mat[ti, di] = _iou(track.box, det)

        # ── Greedy match (highest IoU first) ───────────────────────────────
        matched_t: set[int] = set()
        matched_d: set[int] = set()

        order = np.argsort(-iou_mat, axis=None)
        for flat in order:
            ti = int(flat) // len(detections)
            di = int(flat) % len(detections)
            if iou_mat[ti, di] < self.iou_threshold:
                break
            if ti in matched_t or di in matched_d:
                continue
            self.tracks[ti].update(detections[di])
            matched_t.add(ti)
            matched_d.add(di)

        for ti, track in enumerate(self.tracks):
            if ti not in matched_t:
                track.mark_missed()

        for di, det in enumerate(detections):
            if di not in matched_d:
                self.tracks.append(Track(det))

        self._prune()
        return self._visible()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _prune(self) -> None:
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

    def _visible(self) -> dict[int, tuple[int, int, int, int]]:
        return {
            t.id: t.box
            for t in self.tracks
            if t.hits >= self.min_hits and t.time_since_update == 0
        }


def _iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter) / max(union, 1e-6)
