"""
attendance_cv/rtsp_reader.py

Non-blocking RTSP/webcam frame reader using a background thread and OpenCV.

The reader always keeps the LATEST decoded frame in memory.
The inference loop calls get_frame() and always receives a fresh image,
eliminating the built-up capture buffer delay that causes "lag" in
synchronous cv2.VideoCapture.read() loops.

Key OpenCV settings for near-zero latency RTSP:
  - CAP_PROP_BUFFERSIZE = 1   → discard old frames, OS keeps only 1 buffered
  - CAP_FFMPEG backend        → hardware-accelerated decode via FFmpeg
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

import cv2 as cv
import numpy as np

# Force TCP transport and zero-buffering for low-latency RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"


class RtspReader:
    """
    Background-thread RTSP/webcam reader with auto-reconnect.

    Usage::

        reader = RtspReader("rtsp://admin:pass@192.168.77.209:554").start()
        frame = reader.get_frame()   # always the latest frame, or None
        reader.stop()
    """

    def __init__(
        self,
        url: str,
        reconnect_delay: float = 3.0,
    ) -> None:
        self.url = url
        self.reconnect_delay = reconnect_delay

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._connected = threading.Event()

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="rtsp-reader",
        )

    # ── public ──────────────────────────────────────────────────────────────

    def start(self) -> "RtspReader":
        """Start the background reader thread and return self for chaining."""
        self._thread.start()
        return self

    def stop(self) -> None:
        """Signal the reader thread to stop and wait for it to exit."""
        self._stop_event.set()
        self._thread.join(timeout=6)

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Return a copy of the latest decoded frame, or None if the stream
        has not yet delivered the first frame.
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ── internal ────────────────────────────────────────────────────────────

    def _open_capture(self) -> cv.VideoCapture:
        """
        Open the video source with OpenCV.

        For RTSP URLs we use the FFmpeg backend and set buffer size to 1
        so the OS discards all but the most recent frame — this is the
        key trick for near-zero capture latency.
        """
        is_rtsp = self.url.lower().startswith("rtsp://")
        backend = cv.CAP_FFMPEG if is_rtsp else cv.CAP_ANY

        cap = cv.VideoCapture(self.url, backend)

        if is_rtsp:
            # Minimise OS-level frame buffer → always decode the freshest
            # frame, never a frame that arrived 300 ms ago.
            cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        return cap

    def _run(self) -> None:
        while not self._stop_event.is_set():
            print(f"[RtspReader] Connecting → {self.url}")
            cap = self._open_capture()

            if not cap.isOpened():
                print(
                    f"[RtspReader] Cannot open stream. "
                    f"Retry in {self.reconnect_delay}s …"
                )
                self._connected.clear()
                time.sleep(self.reconnect_delay)
                continue

            print("[RtspReader] ✓ Stream connected.")
            self._connected.set()

            consecutive_failures = 0

            while not self._stop_event.is_set():
                ok, frame = cap.read()

                if not ok:
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        print("[RtspReader] Too many read failures. Reconnecting …")
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0

                with self._lock:
                    self._frame = frame

            cap.release()
            self._connected.clear()

            if not self._stop_event.is_set():
                print(f"[RtspReader] Reconnecting in {self.reconnect_delay}s …")
                time.sleep(self.reconnect_delay)

        print("[RtspReader] Stopped.")
