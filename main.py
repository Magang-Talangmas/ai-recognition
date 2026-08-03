"""
main.py — Talangmas Attendance AI — Combined entry point

Starts two components in parallel:
  1. Inference thread  — reads RTSP via OpenCV, runs YOLOv10 + SCRFD,
                         writes events to SQLite, publishes to EventBus
  2. FastAPI/uvicorn   — REST API + WebSocket + MJPEG stream

Usage
-----
  python main.py                          # default host 0.0.0.0:8000
  python main.py --port 9000              # custom port
  python main.py --host 127.0.0.1        # local-only
  python main.py --config custom.yaml    # alternate config file

Endpoints
---------
  GET  /health                — liveness probe
  GET  /docs                  — Swagger UI
  GET  /events                — list attendance events
  POST /events/{id}/respond   — confirm / reject / checkout …
  WS   /ws/events             — real-time WebSocket push
  GET  /stream/mjpeg          — MJPEG annotated camera preview
  GET  /stream/preview        — browser preview page
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import warnings

# Force UTF-8 output on Windows to avoid UnicodeEncodeError with cp1252
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Suppress harmless FutureWarning from insightface.utils.face_align
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

import uvicorn

from attendance_cv.api_server import create_app
from attendance_cv.config import load_config
from attendance_cv.database import AttendanceDB
from attendance_cv.event_bus import EventBus
from attendance_cv.inference import run_inference


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Talangmas Attendance AI - RTSP + YOLOv10 + SCRFD server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=None, help="Override API host from config")
    parser.add_argument("--port", type=int, default=None, help="Override API port from config")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    host = args.host or config.api.host
    port = args.port or config.api.port

    # Shared objects
    bus = EventBus()
    database = AttendanceDB(
        config.attendance.event_database,
        config.attendance.duplicate_cooldown_seconds,
    )

    # FastAPI app
    app = create_app(database, bus)

    # Inference thread (daemon -> exits when main thread exits)
    inference_thread = threading.Thread(
        target=run_inference,
        args=(config, database, bus),
        daemon=True,
        name="inference",
    )
    inference_thread.start()

    # Uvicorn server (runs in the asyncio event loop of the main thread)
    uv_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )
    server = uvicorn.Server(uv_config)

    async def _serve() -> None:
        # Give the event loop reference to EventBus so the inference
        # thread can safely push events into asyncio queues.
        bus.set_loop(asyncio.get_running_loop())
        await server.serve()

    print(
        f"\n"
        f"  +-------------------------------------------------+\n"
        f"  |  Talangmas Attendance AI                        |\n"
        f"  |  RTSP  : {config.camera.source:<37} |\n"
        f"  |  API   : http://{host}:{port:<28} |\n"
        f"  |  Docs  : http://{host}:{port}/docs{'':<22} |\n"
        f"  |  Stream: http://{host}:{port}/stream/preview{'':<13} |\n"
        f"  +-------------------------------------------------+\n"
    )

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt - shutting down ...")
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
