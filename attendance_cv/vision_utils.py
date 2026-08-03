from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from attendance_cv.face_engine import FaceResult


@dataclass(slots=True)
class IdentityVote:
    employee_id: str | None
    score: float


def associate_face_to_track(
    face: FaceResult,
    people: dict[int, tuple[int, int, int, int]],
) -> int | None:
    x1, y1, x2, y2 = face.bbox
    center_x = float((x1 + x2) / 2)
    center_y = float((y1 + y2) / 2)

    candidates: list[tuple[int, int]] = []
    for track_id, (px1, py1, px2, py2) in people.items():
        if px1 <= center_x <= px2 and py1 <= center_y <= py2:
            area = max(1, (px2 - px1) * (py2 - py1))
            candidates.append((area, track_id))

    return min(candidates)[1] if candidates else None


def stable_identity(
    votes: deque[IdentityVote],
    required_votes: int,
) -> tuple[str | None, float]:
    valid = [vote for vote in votes if vote.employee_id]
    if not valid:
        return None, 0.0

    counts = Counter(vote.employee_id for vote in valid)
    employee_id, count = counts.most_common(1)[0]

    if count < required_votes:
        return None, 0.0

    scores = [
        vote.score
        for vote in valid
        if vote.employee_id == employee_id
    ]
    return employee_id, max(scores)


def crossing_direction(
    previous_side: int,
    current_side: int,
    inside_is_below_line: bool,
) -> str | None:
    if previous_side == current_side:
        return None

    entered_inside = (
        current_side == 1
        if inside_is_below_line
        else current_side == -1
    )
    return "ENTER" if entered_inside else "EXIT"
