"""
attendance_cv/api_server.py

FastAPI application exposing:
  REST  GET  /health                — liveness probe
  REST  GET  /events                — list attendance events (filterable)
  REST  POST /events/{id}/respond   — employee / admin action on an event
  WS        /ws/events              — real-time push of new events (JSON)
  HTTP  GET  /stream/mjpeg          — MJPEG preview of the annotated camera feed

CORS is open (*) so Flutter / React Native / web clients can connect freely.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from attendance_cv.database import AttendanceDB
from attendance_cv.event_bus import EventBus


# ── Request / Response schemas ────────────────────────────────────────────────


class RespondPayload(BaseModel):
    """
    Action the employee or admin sends to resolve a PENDING_CONFIRMATION event.

    Valid actions:
      confirm          → CHECK_IN
      break            → START_BREAK
      temporary_exit   → TEMPORARY_EXIT
      checkout         → CHECK_OUT
      return_break     → RETURN_FROM_BREAK
      return_temporary → RETURN_FROM_TEMPORARY_EXIT
      reject           → marks event UNRESOLVED_RECOGNITION (not me)
    """

    action: str


# ── App factory ───────────────────────────────────────────────────────────────


def create_app(db: AttendanceDB, bus: EventBus) -> FastAPI:
    app = FastAPI(
        title="Talangmas Attendance AI",
        description=(
            "Real-time attendance system powered by "
            "**YOLOv10** (person detection) + **SCRFD + ArcFace** (face recognition). "
            "Connects to CCTV via RTSP and exposes events over REST and WebSocket."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ────────────────────────────────────────────────────────────────

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        """Quick liveness check for load-balancers / mobile apps."""
        return {"status": "ok", "service": "talangmas-attendance-ai"}

    # ── Events ───────────────────────────────────────────────────────────────

    @app.get("/events", tags=["events"])
    def list_events(
        limit: int = 50,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return recent attendance events.

        - **limit**: max number of rows (default 50)
        - **status**: filter by status string, e.g. `PENDING_CONFIRMATION`
        """
        rows = db.list_events(limit=limit)
        result = [dict(row) for row in rows]
        if status:
            result = [r for r in result if r.get("status") == status]
        return result

    @app.post("/events/{event_id}/respond", tags=["events"])
    def respond_event(
        event_id: int,
        payload: RespondPayload,
    ) -> dict[str, Any]:
        """
        Resolve a pending attendance event.

        Called by the **mobile app** when the employee confirms/rejects,
        or by the **web admin** for manual correction.
        """
        try:
            db.respond(event_id, payload.action)
            return {
                "status": "ok",
                "event_id": event_id,
                "action": payload.action,
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ── WebSocket real-time stream ─────────────────────────────────────────────

    @app.websocket("/ws/events")
    async def websocket_events(ws: WebSocket) -> None:
        """
        WebSocket endpoint that pushes new attendance events in real-time.

        Connect with any WS client:
          ws://localhost:8000/ws/events

        Each message is a JSON object::

          {
            "event_id": 42,
            "employee_id": "sabrinaAskaAmalina",
            "event_type": "CHECK_IN",
            "direction": "ENTER",
            "similarity": 0.8731,
            "status": "PENDING_CONFIRMATION",
            "camera_id": "main-entrance"
          }
        """
        await ws.accept()
        queue = bus.subscribe()
        try:
            while True:
                # Wait up to 30 s for an event; send a keepalive ping if idle
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=25.0)
                    await ws.send_text(json.dumps(event))
                except asyncio.TimeoutError:
                    # Send a heartbeat so the connection doesn't time out
                    await ws.send_text(json.dumps({"type": "ping"}))
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            bus.unsubscribe(queue)

    # ── MJPEG preview stream ──────────────────────────────────────────────────

    @app.get("/stream/mjpeg", tags=["stream"])
    def mjpeg_stream() -> StreamingResponse:
        """
        MJPEG video stream of the annotated camera feed.

        Open in a browser or an `<img>` tag::

          <img src="http://localhost:8000/stream/mjpeg" />

        Frames are JPEG-encoded by OpenCV (quality 72) and pushed at
        up to 25 fps.  No direct RTSP exposure — the raw CCTV stream
        is never forwarded to the web client.
        """

        async def _generate():
            while True:
                jpeg = bus.get_latest_frame_jpeg()
                if jpeg is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                # ~25 fps cap; actual rate limited by inference speed
                await asyncio.sleep(0.04)

        return StreamingResponse(
            _generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/stream/preview", tags=["stream"], response_class=HTMLResponse)
    def preview_page() -> str:
        """Minimal HTML page embedding the MJPEG stream — open in any browser."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Talangmas CCTV Preview</title>
  <style>
    body { margin: 0; background: #111; display: flex;
           flex-direction: column; align-items: center;
           justify-content: center; min-height: 100vh; }
    h1   { color: #eee; font-family: sans-serif; margin-bottom: 12px; }
    img  { max-width: 95vw; border-radius: 8px;
           box-shadow: 0 4px 24px #0008; }
  </style>
</head>
<body>
  <h1>Talangmas Attendance — CCTV Preview</h1>
  <img src="/stream/mjpeg" alt="CCTV feed" />
</body>
</html>
"""

    return app
