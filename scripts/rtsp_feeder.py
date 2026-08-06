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
from datetime import datetime
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
        
        # Aggressive low-latency flags for FFmpeg RTSP backend
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|probesize;32|analyzeduration;0"
        )
        
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.is_connected = False
        
        self._connect()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _connect(self):
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            
            is_rtsp = isinstance(self.source, str) and self.source.startswith("rtsp")
            backend = self.cv2.CAP_FFMPEG if is_rtsp else self.cv2.CAP_ANY
            self.cap = self.cv2.VideoCapture(self.source, backend)
            self.cap.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.cap.isOpened():
                self.is_connected = True
                sys.stderr.write(f"[Feeder] Video source connected: {self.source}\n")
            else:
                self.is_connected = False
                sys.stderr.write(f"[Feeder] Warning: Connection failed to {self.source}\n")
        except Exception as e:
            self.is_connected = False
            sys.stderr.write(f"[Feeder] Connection error: {e}\n")

    def _capture_loop(self):
        backoff = 0.5
        while self.running:
            if not self.is_connected or self.cap is None or not self.cap.isOpened():
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 5.0)
                self._connect()
                continue
            
            # Drain internal buffers with grab() to guarantee zero-latency fresh frames
            try:
                grabbed = self.cap.grab()
                if grabbed:
                    backoff = 0.5
                    ret, frame = self.cap.retrieve()
                    if ret and frame is not None:
                        with self.lock:
                            self.latest_frame = frame
                else:
                    time.sleep(0.005)
                    if self.cap is None or not self.cap.isOpened():
                        self.is_connected = False
            except Exception:
                self.is_connected = False
                time.sleep(0.05)

    def read_fresh(self):
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def release(self):
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

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

# Global stream buffers and state for Web and Mobile HTTP clients
latest_jpeg_frame = None
jpeg_lock = threading.Lock()
jpeg_cond = threading.Condition(jpeg_lock)
stream_stats = {
    "fps": 0.0,
    "faces_detected": 0,
    "active_names": [],
    "camera_source": "",
    "resolution": ""
}

# In-memory cache for employee profile thumbnails
profile_cache = {}

def get_profile_thumbnail(person_id, size=(75, 75)):
    if not person_id or person_id == "UNKNOWN":
        return None
    if person_id in profile_cache:
        return profile_cache[person_id]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    enroll_dirs = [
        os.path.join(base_dir, "data", "enroll", person_id),
        os.path.join(base_dir, "data", "enroll", person_id.replace(" ", "")),
        os.path.join(base_dir, "data", "enroll", person_id.lower()),
    ]
    img = None
    import cv2
    for d in enroll_dirs:
        if os.path.isdir(d):
            for fname in os.listdir(d):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    fpath = os.path.join(d, fname)
                    img = cv2.imread(fpath)
                    if img is not None:
                        break
        if img is not None:
            break

    if img is not None:
        h, w = img.shape[:2]
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        cropped = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
        resized = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)
        profile_cache[person_id] = resized
        return resized
    else:
        profile_cache[person_id] = None
        return None

def format_display_name(raw_name):
    if not raw_name or raw_name == "UNKNOWN":
        return "Unknown"
    if " " in raw_name:
        return raw_name
    import re
    s = re.sub(r'([A-Z])', r' \1', raw_name).strip()
    return s.title()

from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class MJPEGStreamHandler(BaseHTTPRequestHandler):
    """Serve Live MJPEG stream, snapshot JPEG, and telemetry status."""
    def log_message(self, format, *args):
        return  # Suppress noisy HTTP request logging

    def do_GET(self):
        global latest_jpeg_frame, stream_stats

        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.end_headers()

            last_frame_bytes = None
            while True:
                with jpeg_cond:
                    if latest_jpeg_frame is last_frame_bytes or latest_jpeg_frame is None:
                        jpeg_cond.wait(timeout=0.1)
                    frame_bytes = latest_jpeg_frame

                if frame_bytes is not None and frame_bytes is not last_frame_bytes:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame_bytes)))
                        self.end_headers()
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                        last_frame_bytes = frame_bytes
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                        break

        elif self.path == "/snapshot":
            with jpeg_lock:
                frame_bytes = latest_jpeg_frame
            if frame_bytes is not None:
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame_bytes)))
                self.end_headers()
                self.wfile.write(frame_bytes)
            else:
                self.send_response(503)
                self.end_headers()

        elif self.path == "/status":
            import json
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(stream_stats).encode("utf-8"))

        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Talangmas AI Recognition - Camera Stream (C Edition)</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #090d16;
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 16px;
        }}
        .window {{
            background: #111827;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #1f293d;
            max-width: 1000px;
            width: 100%;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
        }}
        .titlebar {{
            background: #0f172a;
            padding: 10px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #1e293b;
        }}
        .titlebar-left {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 14px;
            font-weight: 600;
            color: #e2e8f0;
        }}
        .dot {{ width: 10px; height: 10px; border-radius: 50%; background: #10b981; display: inline-block; box-shadow: 0 0 8px #10b981; }}
        .badge {{ background: #1e293b; color: #38bdf8; border: 1px solid #334155; padding: 3px 8px; border-radius: 6px; font-size: 12px; font-weight: 500; }}
        .video-container {{
            width: 100%;
            background: #000;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        img {{ width: 100%; height: auto; display: block; }}
        .footer {{
            padding: 10px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: #64748b;
            background: #0f172a;
            border-top: 1px solid #1e293b;
        }}
    </style>
</head>
<body>
    <div class="window">
        <div class="titlebar">
            <div class="titlebar-left">
                <span class="dot"></span>
                <span>Talangmas AI Recognition - Camera Stream (C Edition)</span>
            </div>
            <span class="badge">OpenVINO FP16 Engine</span>
        </div>
        <div class="video-container">
            <img src="/stream" alt="Live AI Camera Feed" />
        </div>
        <div class="footer">
            <span>Unified Live Feed &bull; Port {stream_stats.get('port', 8088)}</span>
            <span>Talangmas AI Surveillance Gateway</span>
        </div>
    </div>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))

def main():
    global latest_jpeg_frame, stream_stats

    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python rtsp_feeder.py <source> [width] [height] [stream_port]\n")
        sys.exit(1)

    source = sys.argv[1]
    target_width = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    target_height = int(sys.argv[3]) if len(sys.argv) > 3 else 480
    stream_port = int(sys.argv[4]) if len(sys.argv) > 4 else int(os.environ.get("STREAM_PORT", 8088))
    stream_stats["port"] = stream_port

    try:
        import cv2
    except ImportError:
        sys.stderr.write("[Feeder] Error: cv2 not found.\n")
        sys.exit(1)

    # Initialize Face Engine (MTCNN + OpenVINO FP16)
    face_engine = None
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)

        from src.openvino_mtcnn import OpenVINOMTCNN
        from src.openvino_face_engine import OpenVINOFaceEngine, FaceMatcher

        class StandaloneOpenVINOEngine:
            def __init__(self, device="CPU"):
                self.detector = OpenVINOMTCNN(device=device)
                self.feature_engine = OpenVINOFaceEngine(device=device)
                npz_db = os.path.join(base_dir, "data", "face_embeddings_openvino.npz")
                self.matcher = FaceMatcher(db_path=npz_db if os.path.exists(npz_db) else None, similarity_threshold=0.35)

            def detect(self, frame):
                boxes, landmarks = self.detector.detect(frame)
                if len(boxes) == 0:
                    return []

                results = []
                for i in range(len(boxes)):
                    box = boxes[i]
                    lm = landmarks[i]
                    aligned = self.detector.align_face(frame, lm, target_size=(160, 160))
                    emb = self.feature_engine.extract_embedding(aligned)

                    pid, name, sim, is_match = self.matcher.match(emb)

                    class FaceObj:
                        pass
                    f = FaceObj()
                    f.bbox = box[:4]
                    f.detection_score = float(box[4])
                    f.blur_score = 50.0
                    f.landmarks = lm
                    f.embedding = emb
                    f.person_id = pid
                    f.person_name = name if is_match else None
                    f.similarity = sim
                    results.append(f)
                return results

        face_engine = StandaloneOpenVINOEngine(device="CPU")
        sys.stderr.write("[Feeder] Standalone OpenVINO MTCNN + FaceEngine FP16 initialized successfully.\n")
    except Exception as e:
        sys.stderr.write(f"[Feeder] FaceEngine error: {e}\n")

    # Start HTTP Streaming Server for Frontend & Mobile
    try:
        http_server = ThreadedHTTPServer(("0.0.0.0", stream_port), MJPEGStreamHandler)
        http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
        http_thread.start()
        sys.stderr.write(f"[Stream Server] Live HTTP MJPEG Stream active on http://0.0.0.0:{stream_port}/stream\n")
    except Exception as e:
        sys.stderr.write(f"[Stream Server] Warning starting HTTP server: {e}\n")

    is_piped = not os.isatty(REAL_STDOUT_FD)

    capture = ZeroLatencyRTSPCapture(source)
    detector = AsyncFaceDetector(face_engine)

    stream_stats["camera_source"] = str(source)
    stream_stats["resolution"] = f"{target_width + 340}x{target_height}"

    frame_counter = 0
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0.0

    shm = None

    while True:
        ret, frame = capture.read_fresh()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        frame_counter += 1
        fps_counter += 1

        if frame.shape[1] != target_width or frame.shape[0] != target_height:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        # Trigger background face detection every 2 frames
        if frame_counter % 2 == 0:
            detector.submit_frame(frame)

        faces = detector.get_faces()
        num_faces = min(len(faces), 16)

        # Calculate FPS
        now = time.time()
        if now - fps_start_time >= 1.0:
            current_fps = fps_counter / (now - fps_start_time)
            fps_counter = 0
            fps_start_time = now
            stream_stats["fps"] = round(current_fps, 1)
            stream_stats["faces_detected"] = num_faces
            stream_stats["active_names"] = [
                getattr(f, 'person_name', 'Unknown') for f in faces if getattr(f, 'person_name', None)
            ]
        
        # Write binary stream to C engine if pipe is attached
        if is_piped:
            try:
                # Build packet header: uint32 magic, uint32 width, uint32 height, uint32 num_faces
                header = struct.pack("<IIII", MAGIC, target_width, target_height, num_faces)
                binary_stream.write(header)

                # Write detected faces metadata & ArcFace embeddings to C Engine
                for i in range(num_faces):
                    f = faces[i]
                    bbox = f.bbox
                    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    det_score = float(f.detection_score)
                    blur_score = float(f.blur_score)
                    
                    if hasattr(f, 'landmarks') and f.landmarks is not None and len(f.landmarks) == 5:
                        lm_x = [float(f.landmarks[k][0]) for k in range(5)]
                        lm_y = [float(f.landmarks[k][1]) for k in range(5)]
                    else:
                        lm_x = [x1 + (x2 - x1) * 0.3, x1 + (x2 - x1) * 0.7, x1 + (x2 - x1) * 0.5, x1 + (x2 - x1) * 0.35, x1 + (x2 - x1) * 0.65]
                        lm_y = [y1 + (y2 - y1) * 0.35, y1 + (y2 - y1) * 0.35, y1 + (y2 - y1) * 0.55, y1 + (y2 - y1) * 0.75, y1 + (y2 - y1) * 0.75]
                    
                    face_meta = struct.pack("<ffffff5f5f", x1, y1, x2, y2, det_score, blur_score, *lm_x, *lm_y)
                    binary_stream.write(face_meta)

                    emb = f.embedding.astype(np.float32)
                    if len(emb) == 512:
                        binary_stream.write(emb.tobytes())
                    else:
                        binary_stream.write(b'\x00' * (512 * 4))

                # Write raw uncompressed BGR image bytes to C renderer
                binary_stream.write(frame.tobytes())
                binary_stream.flush()
            except (BrokenPipeError, OSError):
                is_piped = False

        # =========================================================================
        # Direct Synchronization with Native C Desktop Window via Shared Memory
        # =========================================================================
        frame_to_serve = None
        if shm is None:
            try:
                import mmap
                shm = mmap.mmap(-1, 16 * 1024 * 1024, tagname="Local\\TMAS_PREVIEW_SHM", access=mmap.ACCESS_READ)
            except Exception:
                shm = None

        if shm is not None:
            try:
                shm.seek(0)
                hdr = shm.read(24)
                if len(hdr) == 24:
                    magic, shm_w, shm_h, shm_c, seq = struct.unpack("<IIIIQ", hdr)
                    if magic == 0x53414D54 and 0 < shm_w <= 3840 and 0 < shm_h <= 2160 and shm_c == 3:
                        raw_bytes = shm.read(shm_w * shm_h * 3)
                        if len(raw_bytes) == shm_w * shm_h * 3:
                            frame_to_serve = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((shm_h, shm_w, 3))
            except Exception:
                shm = None

        # Fallback to camera frame if C preview is initializing
        if frame_to_serve is None:
            frame_to_serve = frame

        # Encode composite frame to JPEG for HTTP clients (100% pixel-synced with Desktop GUI)
        ret_enc, jpeg_bytes = cv2.imencode(".jpg", frame_to_serve, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret_enc:
            with jpeg_cond:
                latest_jpeg_frame = jpeg_bytes.tobytes()
                jpeg_cond.notify_all()

        # Brief yield to keep CPU healthy while maintaining high frame rate
        time.sleep(0.005)

    capture.release()

if __name__ == "__main__":
    main()
