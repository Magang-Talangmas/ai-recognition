"""
ipc_consumer.py — Windows Named Pipe consumer for the C++ Ingestion Engine.

Reads RoiHeader + JPEG crop payloads sent by the C++ attendance_engine binary
and feeds them into the Python SCRFD+ArcFace recognition pipeline.

Wire format (matches IpcPublisher.hpp RoiHeader):
    [uint32  total_len  ]   — payload body size (bytes after this field)
    [uint32  track_id   ]   — C++ CentroidTracker ID
    [uint32  frame_id   ]   — raw frame counter from C++ ingestion thread
    [float32 confidence ]   — YOLO detection confidence
    [int32   orig_x     ]   — crop top-left X in original frame
    [int32   orig_y     ]   — crop top-left Y in original frame
    [int32   orig_w     ]   — crop width
    [int32   orig_h     ]   — crop height
    [uint32  jpeg_len   ]   — bytes of JPEG data
    [uint8   jpeg_data  ]   — JPEG-encoded person crop image

Usage (standalone test):
    python -m attendance_cv.ipc_consumer

Usage (integrated):
    from attendance_cv.ipc_consumer import IpcConsumer, RoiPayload
    consumer = IpcConsumer()
    consumer.connect()
    for roi in consumer:
        frame = roi.decode_frame()
        # run SCRFD + ArcFace on frame ...
"""
from __future__ import annotations

import struct
import threading
from dataclasses import dataclass
from typing import Iterator

import cv2 as cv
import numpy as np

# ── Constants matching IpcPublisher.hpp ─────────────────────────────────────
PIPE_NAME = r"\\.\pipe\attendance_engine"

# RoiHeader struct layout: I I I f i i i i I  (all little-endian)
# Fields: total_len, track_id, frame_id, confidence, orig_x, orig_y, orig_w, orig_h, jpeg_len
HEADER_FMT    = "<IIIfiiiiI"
HEADER_SIZE   = struct.calcsize(HEADER_FMT)   # 36 bytes


@dataclass
class RoiPayload:
    track_id:   int
    frame_id:   int
    confidence: float
    orig_x:     int
    orig_y:     int
    orig_w:     int
    orig_h:     int
    jpeg_data:  bytes   # raw JPEG bytes

    def decode_frame(self) -> np.ndarray:
        """Decode JPEG bytes to BGR numpy array."""
        arr = np.frombuffer(self.jpeg_data, dtype=np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode JPEG crop from C++ engine")
        return img

    @property
    def crop_rect(self) -> tuple[int, int, int, int]:
        """Returns (x, y, w, h) in original frame coordinates."""
        return (self.orig_x, self.orig_y, self.orig_w, self.orig_h)


class IpcConsumer:
    """
    Reads RoiPayload objects from the Windows Named Pipe written by the C++
    attendance_engine binary.

    The C++ binary is the PIPE SERVER; this Python class is the CLIENT.
    Always start the C++ binary before calling connect().

    Thread-safety: IpcConsumer is NOT thread-safe. Use one instance per thread,
    or protect access externally.
    """

    def __init__(self, pipe_name: str = PIPE_NAME) -> None:
        self.pipe_name = pipe_name
        self._handle   = None
        self._connected = False

    # ──────────────────────────────────────────────────────────────────────
    # Connection
    # ──────────────────────────────────────────────────────────────────────

    def connect(self, timeout_ms: int = 30_000) -> bool:
        """
        Connect to the C++ named pipe server.
        Blocks until connected or timeout expires.
        Returns True on success.
        """
        import win32pipe   # type: ignore[import]
        import win32file   # type: ignore[import]
        import pywintypes
        import time

        print(f"[IpcConsumer] Connecting to pipe: {self.pipe_name} ...")
        start_time = time.time()
        timeout_sec = timeout_ms / 1000.0

        while time.time() - start_time < timeout_sec:
            try:
                # WaitNamedPipe only waits if the pipe exists but is busy.
                # If it doesn't exist yet, it throws pywintypes.error (ERROR_FILE_NOT_FOUND)
                win32pipe.WaitNamedPipe(self.pipe_name, 1000)
                
                self._handle = win32file.CreateFile(
                    self.pipe_name,
                    win32file.GENERIC_READ,
                    0,
                    None,
                    win32file.OPEN_EXISTING,
                    0,
                    None
                )
                print("[IpcConsumer] Successfully connected to C++ engine!")
                self._connected = True
                return True
                
            except pywintypes.error as e:
                # 2 = ERROR_FILE_NOT_FOUND
                if e.winerror == 2:
                    time.sleep(1.0)
                    continue
                else:
                    print(f"[IpcConsumer] Connect error: {e}")
                    time.sleep(1.0)
                    continue
            except Exception as e:
                print(f"[IpcConsumer] Unexpected error: {e}")
                time.sleep(1.0)
                continue

        print("[IpcConsumer] Connection timed out.")
        return False

    def disconnect(self) -> None:
        if self._handle:
            import win32file  # type: ignore[import]
            win32file.CloseHandle(self._handle)
            self._handle = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ──────────────────────────────────────────────────────────────────────
    # Reading
    # ──────────────────────────────────────────────────────────────────────

    def read_one(self) -> RoiPayload | None:
        """
        Blocking read of one RoiPayload from the pipe.
        Returns None on disconnection or error.
        """
        import win32file  # type: ignore[import]

        if not self._connected or self._handle is None:
            return None

        try:
            # Read total_len prefix (4 bytes)
            _, raw_len = win32file.ReadFile(self._handle, 4)
            (total_len,) = struct.unpack("<I", raw_len)

            # Read remaining header + jpeg in one shot
            body_size = total_len
            _, body   = win32file.ReadFile(self._handle, body_size)

            # Ensure we got all bytes (pipe may fragment)
            while len(body) < body_size:
                _, chunk = win32file.ReadFile(self._handle, body_size - len(body))
                body += chunk

        except Exception as e:
            print(f"[IpcConsumer] Read error: {e}")
            self._connected = False
            return None

        # Parse header fields from body (skip total_len — already consumed)
        # body starts at: track_id, frame_id, confidence, orig_x, orig_y, orig_w, orig_h, jpeg_len
        hdr_body_fmt  = "<IIfiiiiI"
        hdr_body_size = struct.calcsize(hdr_body_fmt)
        (
            track_id, frame_id, confidence,
            orig_x, orig_y, orig_w, orig_h,
            jpeg_len,
        ) = struct.unpack_from(hdr_body_fmt, body, 0)

        jpeg_data = body[hdr_body_size : hdr_body_size + jpeg_len]

        return RoiPayload(
            track_id=track_id,
            frame_id=frame_id,
            confidence=confidence,
            orig_x=orig_x,
            orig_y=orig_y,
            orig_w=orig_w,
            orig_h=orig_h,
            jpeg_data=bytes(jpeg_data),
        )

    def __iter__(self) -> Iterator[RoiPayload]:
        """Iterate over incoming ROI payloads until disconnected."""
        while self._connected:
            roi = self.read_one()
            if roi is None:
                break
            yield roi

    def __enter__(self) -> "IpcConsumer":
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()


# ── Standalone smoke test ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== IpcConsumer standalone test ===")
    print("Make sure the C++ attendance_engine binary is running first.")
    consumer = IpcConsumer()
    if not consumer.connect(timeout_ms=30_000):
        print("Could not connect to C++ engine. Is it running?")
        raise SystemExit(1)

    count = 0
    for roi in consumer:
        frame = roi.decode_frame()
        count += 1
        print(
            f"[{count:05d}] track={roi.track_id:04d} frame={roi.frame_id} "
            f"conf={roi.confidence:.2f} crop={frame.shape[:2]} "
            f"@({roi.orig_x},{roi.orig_y})"
        )
        cv.imshow(f"Track {roi.track_id}", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    consumer.disconnect()
    cv.destroyAllWindows()
    print(f"Received {count} ROI payloads total.")
