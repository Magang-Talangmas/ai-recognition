from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class CameraConfig:
    source: str
    camera_id: str
    process_every_n_frames: int
    reconnect_delay_seconds: int
    show_preview: bool


@dataclass(slots=True)
class PersonConfig:
    model_path: str
    confidence: float


@dataclass(slots=True)
class FaceConfig:
    model_pack: str
    model_root: str
    providers: list[str]
    detection_size: int
    detection_threshold: float
    min_face_size: int
    blur_threshold: float
    match_threshold: float
    match_margin: float
    vote_window: int
    votes_required: int


@dataclass(slots=True)
class AttendanceConfig:
    line_y_ratio: float
    inside_is_below_line: bool
    duplicate_cooldown_seconds: int
    event_database: str
    embedding_file: str
    allow_insecure_no_liveness: bool


@dataclass(slots=True)
class ApiConfig:
    host: str
    port: int


@dataclass(slots=True)
class AppConfig:
    camera: CameraConfig
    person: PersonConfig
    face: FaceConfig
    attendance: AttendanceConfig
    api: ApiConfig


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    data: dict[str, Any]
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    api_data = data.get("api", {"host": "0.0.0.0", "port": 8000})

    return AppConfig(
        camera=CameraConfig(**data["camera"]),
        person=PersonConfig(**data["person_detection"]),
        face=FaceConfig(**data["face"]),
        attendance=AttendanceConfig(**data["attendance"]),
        api=ApiConfig(**api_data),
    )
