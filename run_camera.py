"""
run_camera.py — Standalone camera runner (no API server).

Runs only the inference pipeline without starting the FastAPI server.
Useful for quick local testing or when the API is not needed.

For the full production setup (inference + REST API + WebSocket), use:
  python main.py

Usage:
  python run_camera.py
  python run_camera.py --config custom.yaml
"""
from __future__ import annotations

import argparse
import warnings

# Suppress harmless FutureWarning from insightface.utils.face_align
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

from attendance_cv.config import load_config
from attendance_cv.database import AttendanceDB
from attendance_cv.inference import run_inference


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Talangmas — standalone RTSP inference (no API server)"
    )
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    database = AttendanceDB(
        config.attendance.event_database,
        config.attendance.duplicate_cooldown_seconds,
    )

    print("[run_camera] Starting inference-only mode (no API server).")
    print("[run_camera] Use  python main.py  to also start the REST/WS API.")
    print("[run_camera] Press Ctrl+C to stop.")

    try:
        # bus=None → no WebSocket push, no MJPEG publish
        run_inference(config, database, bus=None)
    except KeyboardInterrupt:
        print("\n[run_camera] Stopped by user.")


if __name__ == "__main__":
    main()
