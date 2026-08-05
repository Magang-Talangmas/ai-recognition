"""
High-Performance Zero-Latency Async RTSP Substream Feeder for Talangmas AI-Recognition (C Edition).
Features:
  - Threaded real-time RTSP frame grabber with 0-buffer lag (drops stale frames).
  - Asynchronous / multi-threaded SCRFD + ArcFace (OpenVINO) face detection.
  - Sliced RTSP substream URL configuration support.
  - Dedicated binary pipe isolation (immune to stdout text contamination).
"""

import sys
import os
import time
import struct
import threading
import numpy as np

# Duplicate real binary stdout fd before any library imports or prints
REAL_STDOUT_FD = os.dup(1)

# Redirect standard stdout (fd 1) to stderr (fd 2) to prevent any print/log corruption
os.dup2(2, 1)
sys.stdout = sys.stderr

# Set binary mode on Windows for binary stream descriptor
if sys.platform == "win32":
    import msvcrt
    msvcrt.setmode(REAL_STDOUT_FD, os.O_BINARY)

binary_stream = os.fdopen(REAL_STDOUT_FD, "wb", buffering=0)

MAGIC = 0x53414D54  # 'TMAS' in little endian

class ZeroLatencyRTSPCapture:
    """Threaded RTSP frame grabber that eliminates buffer accumulation and lag."""
    def __init__(self, source):
        import cv2
        self.cv2 = cv2
        self.source = int(source) if isinstance(source, str) and source.isdigit() else source
        
        # Smooth low-latency flags for FFmpeg RTSP backend
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|probesize;32768|analyzeduration;100000"
        )
        
        self.cap = None
        self.latest_frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.running = True
        self.is_connected = False
        
        self._connect()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _connect(self):
        try:
            if self.cap is not None:
                self.cap.release()
            
            is_rtsp = isinstance(self.source, str) and self.source.startswith("rtsp")
            backend = self.cv2.CAP_FFMPEG if is_rtsp else self.cv2.CAP_ANY
            self.cap = self.cv2.VideoCapture(self.source, backend)
            self.cap.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.cap.isOpened():
                self.is_connected = True
                sys.stderr.write(f"[Feeder] Substream connected: {self.source}\n")
            else:
                self.is_connected = False
                sys.stderr.write(f"[Feeder] Warning: Connection failed to {self.source}\n")
        except Exception as e:
            self.is_connected = False
            sys.stderr.write(f"[Feeder] Connection error: {e}\n")

    def _capture_loop(self):
        while self.running:
            if not self.is_connected or self.cap is None or not self.cap.isOpened():
                time.sleep(1.0)
                self._connect()
                continue
            
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
                    self.frame_id += 1
            else:
                time.sleep(0.005)

    def read_fresh(self, last_id=-1):
        with self.lock:
            if self.latest_frame is not None and self.frame_id != last_id:
                return True, self.latest_frame.copy(), self.frame_id
            return False, None, last_id

    def release(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()

class AsyncFaceDetector:
    """Decoupled background face detector to maintain 30+ FPS stream throughput."""
    def __init__(self, face_engine):
        self.engine = face_engine
        self.pending_frame = None
        self.latest_faces = []
        self.lock = threading.Lock()
        self.running = True
        
        if self.engine is not None:
            self.thread = threading.Thread(target=self._detect_loop, daemon=True)
            self.thread.start()

    def submit_frame(self, frame):
        with self.lock:
            self.pending_frame = frame

    def get_faces(self):
        with self.lock:
            return list(self.latest_faces)

    def _detect_loop(self):
        while self.running:
            frame_to_process = None
            with self.lock:
                if self.pending_frame is not None:
                    frame_to_process = self.pending_frame
                    self.pending_frame = None
            
            if frame_to_process is not None and self.engine is not None:
                try:
                    detected = self.engine.detect(frame_to_process)
                    with self.lock:
                        self.latest_faces = detected
                except Exception:
                    pass
            else:
                time.sleep(0.005)

def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    target_width = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    target_height = int(sys.argv[3]) if len(sys.argv) > 3 else 480

    try:
        import cv2
    except ImportError:
        sys.stderr.write("[Feeder] Error: cv2 not found.\n")
        sys.exit(1)

    # Initialize Face Engine (SCRFD + ArcFace / OpenVINO)
    face_engine = None
    try:
        candidate_paths = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ai-recognition")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai-recognition")),
            "D:/kamera/ai-recognition"
        ]
        for p in candidate_paths:
            if os.path.exists(p) and p not in sys.path:
                sys.path.insert(0, p)

        from attendance_cv.config import load_config
        from attendance_cv.face_engine import FaceEngine
        
        cfg_candidates = ["config.yaml", "../config.yaml", "d:/kamera/camera-c/config.yaml", "d:/kamera/ai-recognition/config.yaml"]
        cfg_file = next((c for c in cfg_candidates if os.path.exists(c)), "config.yaml")
        cfg = load_config(cfg_file)
        face_engine = FaceEngine(cfg.face)
        sys.stderr.write(f"[Feeder] High-speed FaceEngine (SCRFD + ArcFace / OpenVINO) ready ({cfg_file})\n")
    except Exception as e:
        sys.stderr.write(f"[Feeder] FaceEngine notice: {e}\n")

    capture = ZeroLatencyRTSPCapture(source)
    detector = AsyncFaceDetector(face_engine)

    frame_counter = 0
    last_frame_id = -1

    while True:
        ret, frame, frame_id = capture.read_fresh(last_frame_id)
        if not ret or frame is None:
            time.sleep(0.002)
            continue

        last_frame_id = frame_id
        frame_counter += 1

        if frame.shape[1] != target_width or frame.shape[0] != target_height:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        # Trigger background face detection every 2-3 frames for ultra-smooth responsiveness
        if frame_counter % 2 == 0:
            detector.submit_frame(frame)

        faces = detector.get_faces()
        num_faces = min(len(faces), 16)
        
        # Build packet header: uint32 magic, uint32 width, uint32 height, uint32 num_faces
        header = struct.pack("<IIII", MAGIC, target_width, target_height, num_faces)
        binary_stream.write(header)

        # Write detected faces metadata & ArcFace embeddings
        for i in range(num_faces):
            f = faces[i]
            bbox = f.bbox
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            det_score = float(f.detection_score)
            blur_score = float(f.blur_score)
            
            lm_x = [x1 + (x2 - x1) * 0.3, x1 + (x2 - x1) * 0.7, x1 + (x2 - x1) * 0.5, x1 + (x2 - x1) * 0.35, x1 + (x2 - x1) * 0.65]
            lm_y = [y1 + (y2 - y1) * 0.35, y1 + (y2 - y1) * 0.35, y1 + (y2 - y1) * 0.55, y1 + (y2 - y1) * 0.75, y1 + (y2 - y1) * 0.75]
            
            face_meta = struct.pack("<ffffff5f5f", x1, y1, x2, y2, det_score, blur_score, *lm_x, *lm_y)
            binary_stream.write(face_meta)

            # 512-dim ArcFace embedding vector
            emb = f.embedding.astype(np.float32)
            if len(emb) == 512:
                binary_stream.write(emb.tobytes())
            else:
                binary_stream.write(b'\x00' * (512 * 4))

        # Write raw uncompressed BGR image bytes to C renderer
        binary_stream.write(frame.tobytes())
        binary_stream.flush()

    capture.release()

if __name__ == "__main__":
    main()
