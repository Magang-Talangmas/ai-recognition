"""
Ultra Low-Latency Face Recognition & Streaming Engine (MTCNN + OpenVINO FP16)
Optimized for maximum FPS and zero cumulative latency drift.
"""

import os
import sys
import time
import argparse
import threading
import queue
import json
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np
import requests
import yaml

from src.openvino_mtcnn import OpenVINOMTCNN
from src.openvino_face_engine import OpenVINOFaceEngine, FaceMatcher


def load_env(env_path=".env"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, env_path)
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                # Expand nested ${VAR} if present
                for ek, ev in os.environ.items():
                    v = v.replace(f"${{{ek}}}", ev)
                os.environ[k] = v


def load_config(config_path="config.yaml"):
    load_env(".env")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, config_path)
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            raw_text = f.read()
            
        # Parse environment variables in yaml
        for k, v in os.environ.items():
            raw_text = raw_text.replace(f"${{{k}}}", v)
        try:
            return yaml.safe_load(raw_text) or {}
        except Exception as e:
            print(f"[Config] Warning loading {config_path}: {e}")
    return {}


class ThreadedRTSPCapture:
    """
    Decoupled RTSP Frame Grabber.
    Ensures zero latency drift by continuously draining the RTSP socket buffer
    and only holding the single most recent frame.
    """
    def __init__(self, src):
        self.src = src
        self.cap = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread = None
        self.fps_in = 0.0
        self.last_grab_time = time.time()
        self.frame_count = 0
        self._init_capture()

    def _init_capture(self):
        # Convert integer strings to camera indices
        source = int(self.src) if str(self.src).isdigit() else self.src
        
        # Configure FFmpeg RTSP options for lowest latency
        if isinstance(source, str) and source.startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"
            
        self.cap = cv2.VideoCapture(source)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[RTSPCapture] Successfully opened source: {source}")
        else:
            print(f"[RTSPCapture] Warning: Unable to open source: {source}")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        return self

    def _reader_loop(self):
        last_t = time.time()
        frames = 0
        
        while self.running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(1.0)
                self._init_capture()
                continue
                
            grabbed = self.cap.grab()
            if not grabbed:
                time.sleep(0.01)
                continue
                
            ret, frame = self.cap.retrieve()
            if ret and frame is not None:
                with self.frame_lock:
                    self.latest_frame = frame
                    
                frames += 1
                now = time.time()
                if now - last_t >= 1.0:
                    self.fps_in = frames / (now - last_t)
                    frames = 0
                    last_t = now
            else:
                time.sleep(0.01)

    def read(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


class StreamEngine:
    """
    Main Face Recognition & Streaming Pipeline.
    """
    def __init__(self, config=None, db_path="data/face_embeddings_openvino.npz"):
        self.config = config or {}
        
        # Paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(base_dir, db_path) if not os.path.isabs(db_path) else db_path
        
        # Recognition Settings
        face_cfg = self.config.get("face", {})
        self.match_threshold = float(face_cfg.get("match_threshold", 0.60))
        self.device = face_cfg.get("openvino_device", "CPU")
        
        # Backend Settings
        backend_cfg = self.config.get("backend", {})
        self.backend_url = backend_cfg.get("api_url", "http://localhost:3000/api/v1/attendance")
        self.backend_key = backend_cfg.get("api_key", "your-ml-api-key-change-in-production")
        self.camera_id = self.config.get("camera", {}).get("camera_id", "main-entrance")
        
        # Cooldown management for duplicate attendance events
        self.cooldown_seconds = float(self.config.get("attendance", {}).get("duplicate_cooldown_seconds", 10.0))
        self.last_attendance_time = {}
        
        # Initialize Models
        print(f"[Engine] Initializing MTCNN & Face Engine on {self.device} (FP16)...")
        self.detector = OpenVINOMTCNN(device=self.device)
        self.feature_engine = OpenVINOFaceEngine(device=self.device)
        self.matcher = FaceMatcher(self.db_path, similarity_threshold=self.match_threshold)
        
        # Metrics
        self.fps_ai = 0.0
        self.latency_ai_ms = 0.0
        self.recent_events = []
        
        # Attendance Dispatch Queue
        self.event_queue = queue.Queue(maxsize=100)
        self.dispatch_thread = threading.Thread(target=self._event_dispatcher_loop, daemon=True)
        self.dispatch_thread.start()

    def process_frame(self, frame):
        """
        Runs asynchronous MTCNN detection + landmark alignment + feature matching.
        """
        t0 = time.perf_counter()
        h_orig, w_orig = frame.shape[:2]
        
        # Dynamic downsampling for ultra fast MTCNN detection on high-res frames
        target_det_w = 640
        if w_orig > target_det_w:
            scale = target_det_w / float(w_orig)
            det_img = cv2.resize(frame, (target_det_w, int(h_orig * scale)))
        else:
            scale = 1.0
            det_img = frame
            
        boxes, landmarks = self.detector.detect(det_img)
        
        detections = []
        if len(boxes) > 0:
            # Scale coordinates back to original frame
            if scale != 1.0:
                boxes[:, :4] /= scale
                landmarks /= scale
                
            for i in range(len(boxes)):
                box = boxes[i]
                lm = landmarks[i]
                
                # Align face with similarity transformation
                aligned_face = self.detector.align_face(frame, lm, target_size=(160, 160))
                
                # Extract FP16 embedding & Match
                emb = self.feature_engine.extract_embedding(aligned_face)
                emp_id, emp_name, similarity, is_match = self.matcher.match(emb)
                
                detections.append({
                    "box": box[:4].astype(int),
                    "score": float(box[4]),
                    "landmarks": lm.astype(int),
                    "employee_id": emp_id,
                    "employee_name": emp_name,
                    "similarity": similarity,
                    "is_match": is_match
                })
                
                # Check for attendance event trigger
                if is_match:
                    self._trigger_attendance(emp_id, emp_name, similarity)
                    
        t1 = time.perf_counter()
        self.latency_ai_ms = (t1 - t0) * 1000.0
        return detections

    def _trigger_attendance(self, emp_id, emp_name, similarity):
        now = time.time()
        last_seen = self.last_attendance_time.get(emp_id, 0.0)
        
        if (now - last_seen) >= self.cooldown_seconds:
            self.last_attendance_time[emp_id] = now
            dt_str = datetime.now(timezone.utc).isoformat()
            
            event = {
                "employee_id": emp_id,
                "employee_name": emp_name,
                "event_type": "CHECK_IN",
                "similarity": round(float(similarity), 4),
                "detected_at": dt_str,
                "camera_id": self.camera_id
            }
            
            self.recent_events.insert(0, event)
            if len(self.recent_events) > 15:
                self.recent_events.pop()
                
            print(f"\n[ATTENDANCE EVENT] Recognized '{emp_name}' (ID: {emp_id}, Sim: {similarity:.2%}) -> Queuing dispatch")
            try:
                self.event_queue.put_nowait(event)
            except queue.Full:
                pass

    def _event_dispatcher_loop(self):
        while True:
            event = self.event_queue.get()
            try:
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": self.backend_key
                }
                payload = {
                    "employee_id": event["employee_id"],
                    "event_type": event["event_type"],
                    "similarity": event["similarity"],
                    "detected_at": event["detected_at"],
                    "camera_id": event["camera_id"]
                }
                resp = requests.post(self.backend_url, json=payload, headers=headers, timeout=3.0)
                if resp.status_code in [200, 201]:
                    print(f"  [+] Backend dispatched successfully: {event['employee_name']}")
                else:
                    print(f"  [-] Backend returned {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"  [-] Failed to reach backend API ({self.backend_url}): {e}")
            finally:
                self.event_queue.task_done()

    def draw_overlays(self, frame, detections, fps_cap, fps_ai):
        """
        Draw sleek real-time bounding boxes, landmark points, and HUD on frame.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # 1. Draw Detections
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            
            is_match = det["is_match"]
            color = (46, 204, 113) if is_match else (52, 152, 219)  # Green vs Orange
            
            # Corner bounding box style
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw 5 landmarks
            for pt in det["landmarks"]:
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 3, (241, 196, 15), -1)  # Yellow-cyan dots
                
            # Text label
            label = f"{det['employee_name']} ({det['similarity']:.0%})" if is_match else f"Unknown ({det['similarity']:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            
            cv2.rectangle(overlay, (x1, max(0, y1 - th - 8)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 5, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            
        # 2. Top-Left HUD Card
        hud_bg = (20, 24, 30)
        cv2.rectangle(overlay, (15, 15), (320, 95), hud_bg, -1)
        cv2.rectangle(overlay, (15, 15), (320, 95), (60, 70, 85), 1)
        
        cv2.putText(overlay, f"CAPTURE FPS : {fps_cap:4.1f} fps", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 128), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"AI INFER FPS: {fps_ai:4.1f} fps ({self.latency_ai_ms:.1f}ms)", (25, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"DEVICE: {self.device} (FP16) | CAM: {self.camera_id}", (25, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 200), 1, cv2.LINE_AA)
        
        return overlay


# Global frame sharing for HTTP MJPEG Server
latest_jpeg = None
jpeg_lock = threading.Lock()
global_engine = None


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress HTTP access logging in terminal
        
    def do_GET(self):
        global latest_jpeg, jpeg_lock, global_engine
        
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            
            while True:
                with jpeg_lock:
                    frame_bytes = latest_jpeg
                    
                if frame_bytes is not None:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", len(frame_bytes))
                        self.end_headers()
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(0.03)
                
        elif self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            stats = {
                "fps_ai": global_engine.fps_ai if global_engine else 0.0,
                "latency_ai_ms": global_engine.latency_ai_ms if global_engine else 0.0,
                "device": global_engine.device if global_engine else "CPU",
                "enrolled_count": len(global_engine.matcher.employee_ids) if global_engine else 0,
                "recent_events": global_engine.recent_events if global_engine else []
            }
            self.wfile.write(json.dumps(stats).encode("utf-8"))
            
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head>
    <title>AI Attendance Monitor - MTCNN + OpenVINO FP16</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 15px; margin-bottom: 20px; }
        h1 { margin: 0; font-size: 22px; font-weight: 700; color: #38bdf8; }
        .badge { background: #0284c7; color: white; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        .video-card { background: #1e293b; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); border: 1px solid #334155; }
        .video-feed { width: 100%; display: block; }
        .panel { background: #1e293b; border-radius: 12px; padding: 18px; border: 1px solid #334155; height: fit-content; }
        h3 { margin-top: 0; font-size: 15px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
        .event-item { padding: 10px 12px; background: #0f172a; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #10b981; }
        .event-name { font-weight: 600; color: #f8fafc; font-size: 14px; }
        .event-meta { font-size: 12px; color: #64748b; margin-top: 2px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Talangmas AI Attendance Engine</h1>
        <span class="badge">MTCNN + OpenVINO FP16</span>
    </div>
    <div class="grid">
        <div class="video-card">
            <img class="video-feed" src="/stream" alt="Live Camera Stream">
        </div>
        <div class="panel">
            <h3>Recent Check-ins</h3>
            <div id="events-list">
                <p style="color: #64748b; font-size: 13px;">Waiting for face recognition events...</p>
            </div>
        </div>
    </div>
    <script>
        async function updateEvents() {
            try {
                const res = await fetch('/status');
                const data = await res.json();
                const list = document.getElementById('events-list');
                if (data.recent_events && data.recent_events.length > 0) {
                    list.innerHTML = data.recent_events.map(e => `
                        <div class="event-item">
                            <div class="event-name">${e.employee_name}</div>
                            <div class="event-meta">${e.detected_at.substring(11, 19)} | Confidence: ${(e.similarity * 100).toFixed(1)}%</div>
                        </div>
                    `).join('');
                }
            } catch (err) {}
        }
        setInterval(updateEvents, 1000);
    </script>
</body>
</html>
"""
            self.wfile.write(html.encode("utf-8"))


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def run_pipeline(args):
    global latest_jpeg, jpeg_lock, global_engine
    
    # 1. Load config
    config = load_config(args.config)
    
    # 2. Determine camera source
    cam_source = args.source or os.environ.get("CAMERA_SOURCE") or config.get("camera", {}).get("source", "0")
    if cam_source.startswith("${") and cam_source.endswith("}"):
        cam_source = "0"
        
    print("\n=================================================================")
    print("      TALANGMAS AI ATTENDANCE ENGINE (MTCNN + OPENVINO FP16)     ")
    print("=================================================================")
    print(f"[*] Camera Source: {cam_source}")
    print(f"[*] Database Path: {args.db}")
    print(f"[*] Web Server   : http://0.0.0.0:{args.port}/")
    print("=================================================================\n")
    
    # 3. Initialize AI Engine
    engine = StreamEngine(config=config, db_path=args.db)
    global_engine = engine
    
    # 4. Start HTTP Streaming Server
    server = ThreadedHTTPServer(("0.0.0.0", args.port), MJPEGHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[HTTP] Live MJPEG Stream available at: http://localhost:{args.port}/stream\n")
    
    # 5. Start RTSP Reader
    grabber = ThreadedRTSPCapture(cam_source).start()
    
    # Main Processing Loop
    last_fps_time = time.time()
    frames_processed = 0
    
    try:
        while True:
            ret, frame = grabber.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
                
            # Process AI detection & recognition
            detections = engine.process_frame(frame)
            
            # FPS tracking
            frames_processed += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                engine.fps_ai = frames_processed / (now - last_fps_time)
                frames_processed = 0
                last_fps_time = now
                
            # Draw overlays
            annotated = engine.draw_overlays(frame, detections, grabber.fps_in, engine.fps_ai)
            
            # Compress to JPEG for Web Stream
            ret_enc, jpeg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret_enc:
                with jpeg_lock:
                    latest_jpeg = jpeg_buf.tobytes()
                    
            if args.gui:
                cv2.imshow("Talangmas AI Face Attendance (MTCNN + OpenVINO FP16)", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        print("\n[Engine] Stopping gracefully...")
    finally:
        grabber.stop()
        if args.gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Talangmas MTCNN + OpenVINO FP16 Streaming Engine")
    parser.add_argument("--source", type=str, default=None, help="RTSP URL, video path, or camera device index")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--db", type=str, default="data/face_embeddings_openvino.npz", help="Path to face embeddings database")
    parser.add_argument("--port", type=int, default=8088, help="HTTP MJPEG stream port")
    parser.add_argument("--gui", action="store_true", help="Open local OpenCV GUI preview window")
    args = parser.parse_args()
    
    run_pipeline(args)
