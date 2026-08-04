from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os
import re

import yaml


def _load_dotenv(env_path: Path = Path(".env")) -> None:
    """Load key=value pairs from .env file into os.environ (if not already set)."""
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key:
                # Expand any references to previously loaded variables
                value = _expand_env(value)
                os.environ[key] = value


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} placeholders in string values."""
    if isinstance(value, str):
        return re.sub(
            r"\$\{([^}]+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            value,
        )
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value



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
    bbox_padding: int = 0


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
    # OpenVINO inference backend ("onnxruntime" | "openvino")
    backend: str = "onnxruntime"
    # OpenVINO device: AUTO, CPU, GPU.0, GPU.1 …
    openvino_device: str = "AUTO"


@dataclass(slots=True)
class AttendanceConfig:
    line_y_ratio: float
    inside_is_below_line: bool
    duplicate_cooldown_seconds: int
    event_database: str
    embedding_file: str
    allow_insecure_no_liveness: bool


@dataclass(slots=True)
class BackendConfig:
    api_url: str = "http://localhost:3000/api/v1/attendance"
    api_key: str = "your-ml-api-key-change-in-production"
    enabled: bool = True
    timeout: float = 4.0


@dataclass(slots=True)
class AppConfig:
    camera: CameraConfig
    person: PersonConfig
    face: FaceConfig
    attendance: AttendanceConfig
    backend: BackendConfig = field(default_factory=BackendConfig)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    _load_dotenv()
    data: dict[str, Any]
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    data = _expand_env(data)

    backend_data = data.get("backend", {})

    return AppConfig(
        camera=CameraConfig(**data["camera"]),
        person=PersonConfig(**data["person_detection"]),
        face=FaceConfig(**data["face"]),
        attendance=AttendanceConfig(**data["attendance"]),
        backend=BackendConfig(**backend_data),
    )
