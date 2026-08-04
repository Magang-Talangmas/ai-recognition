from __future__ import annotations

import json
import logging
import queue
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("attendance_cv.api_dispatcher")


class BackendDispatcher:
    """
    Asynchronous, non-blocking HTTP dispatcher to forward attendance events
    from AI Recognition Engine to Backend API (POST /api/v1/attendance).

    Uses a background worker queue & daemon thread so video processing and camera
    display FPS are never blocked by network latency.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:3000/api/v1/attendance",
        api_key: str = "your-ml-api-key-change-in-production",
        enabled: bool = True,
        timeout: float = 4.0,
        max_queue_size: int = 50,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.enabled = enabled
        self.timeout = timeout

        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="BackendDispatcherWorker",
            daemon=True,
        )
        if self.enabled:
            self._worker_thread.start()

    def dispatch_checkin(
        self,
        employee_id: str,
        similarity: float,
        camera_id: str = "main-entrance",
        event_id: str | None = None,
        detected_at: str | None = None,
    ) -> None:
        """
        Enqueue a CHECK_IN attendance event to be dispatched asynchronously to the Backend.
        """
        if not self.enabled:
            return

        if event_id is None:
            event_id = str(uuid.uuid4())

        if detected_at is None:
            detected_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        payload = {
            "event_id": event_id,
            "employee_id": employee_id,
            "event_type": "CHECK_IN",
            "similarity": round(similarity, 4),
            "detected_at": detected_at,
            "camera_id": camera_id,
        }

        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            print(f"[WARN] BackendDispatcher queue full. Dropping event for {employee_id}")

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if payload is None:  # Sentinel value for shutdown
                break

            self._send_payload(payload)
            self._queue.task_done()

    def _send_payload(self, payload: dict[str, Any]) -> None:
        emp_id = payload.get("employee_id")
        event_type = payload.get("event_type")
        event_id = payload.get("event_id")

        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.api_url,
                data=data_bytes,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "User-Agent": "AI-Recognition-Engine/1.0",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                status_code = response.getcode()
                response_body = response.read().decode("utf-8")
                print(
                    f"[BE DISPATCH SUCCESS] {emp_id} | {event_type} | "
                    f"HTTP {status_code} | event_id={event_id}"
                )

        except urllib.error.HTTPError as e:
            error_msg = e.read().decode("utf-8", errors="replace")
            print(
                f"[BE DISPATCH HTTP {e.code}] {emp_id} | {event_type} | "
                f"Error: {error_msg}"
            )
        except urllib.error.URLError as e:
            print(
                f"[BE DISPATCH CONN FAILED] {emp_id} | {event_type} | "
                f"Cannot connect to {self.api_url} ({e.reason})"
            )
        except Exception as e:
            print(
                f"[BE DISPATCH ERROR] {emp_id} | {event_type} | "
                f"Unexpected error: {e}"
            )

    def stop(self) -> None:
        """Gracefully stop dispatcher thread."""
        if not self.enabled:
            return
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker_thread.join(timeout=2.0)
