"""
attendance_cv/event_bus.py

Thread-safe in-process pub/sub bridge between the synchronous
inference/camera thread and the asynchronous FastAPI WebSocket handlers.

Design:
  - The inference thread calls publish_event() / publish_frame() (sync).
  - Each WebSocket handler subscribes to get an asyncio.Queue.
  - call_soon_threadsafe delivers events into the asyncio event loop
    without locking or polling.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any


class EventBus:
    """
    Pub/sub bus for bridging sync inference → async WebSocket clients
    and for sharing the latest MJPEG frame with the streaming endpoint.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        self._sub_lock = threading.Lock()

        # Latest MJPEG-encoded frame (set by inference, read by API)
        self._latest_jpeg: bytes | None = None
        self._frame_lock = threading.Lock()

    # ── Setup ────────────────────────────────────────────────────────────────

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Must be called once from the asyncio thread (inside an async function)
        before any publish_event() calls arrive from the inference thread.
        """
        self._loop = loop

    # ── Attendance event pub/sub ──────────────────────────────────────────────

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """
        Register a new subscriber. Returns an asyncio.Queue that will receive
        event dicts published via publish_event(). Call from async context.
        """
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        """Deregister a subscriber (e.g. when a WebSocket disconnects)."""
        with self._sub_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def publish_event(self, payload: dict[str, Any]) -> None:
        """
        Thread-safe. Call from the sync inference thread to push an event
        to all connected WebSocket clients.
        """
        if self._loop is None or not self._loop.is_running():
            return
        with self._sub_lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            self._loop.call_soon_threadsafe(_put_nowait_safe, q, payload)

    # ── MJPEG frame sharing ───────────────────────────────────────────────────

    def publish_frame(self, jpeg_bytes: bytes) -> None:
        """Thread-safe. Store the latest annotated JPEG for MJPEG streaming."""
        with self._frame_lock:
            self._latest_jpeg = jpeg_bytes

    def get_latest_frame_jpeg(self) -> bytes | None:
        """Thread-safe. Return the most recent JPEG frame, or None."""
        with self._frame_lock:
            return self._latest_jpeg


# ── helpers ──────────────────────────────────────────────────────────────────


def _put_nowait_safe(q: asyncio.Queue, item: Any) -> None:
    """Drop the item silently if the queue is full (slow consumer)."""
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        pass
