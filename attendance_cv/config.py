import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _load_env(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip("'\""))


def _resolve_env_str(val: Any) -> Any:
    if isinstance(val, str):
        if val.startswith("${") and val.endswith("}"):
            var_name = val[2:-1]
            return os.getenv(var_name, "0")
        return os.getenv(val, val)
    return val


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
    tracker: str
    half_precision: bool = True


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
    batch_size: int = 32
    provider_options: dict[str, Any] = None


@dataclass(slots=True)
class AttendanceConfig:
    line_y_ratio: float
    inside_is_below_line: bool
    duplicate_cooldown_seconds: int
    event_database: str
    embedding_file: str
    allow_insecure_no_liveness: bool


@dataclass(slots=True)
class AppConfig:
    camera: CameraConfig
    person: PersonConfig
    face: FaceConfig
    attendance: AttendanceConfig


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    _load_env()

    data: dict[str, Any]
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    camera_data = data["camera"].copy()
    camera_data["source"] = str(_resolve_env_str(camera_data["source"]))

    return AppConfig(
        camera=CameraConfig(**camera_data),
        person=PersonConfig(**data["person_detection"]),
        face=FaceConfig(**data["face"]),
        attendance=AttendanceConfig(**data["attendance"]),
    )
