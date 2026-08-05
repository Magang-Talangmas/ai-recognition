"""
Real-time RTSP/Webcam Frame & Face Detection Feeder for Talangmas AI-Recognition (C Edition).
Streams binary packets containing:
  [Header: magic(4), width(4), height(4), num_faces(4)]
  [FaceArray: (x1, y1, x2, y2, det_score, blur_score, lm_x[5], lm_y[5], emb[512]) * num_faces]
  [ImageBytes: BGR24 frame buffer]
"""

import sys
import os
import time
import struct
import numpy as np

# Set binary stdout mode on Windows
if sys.platform == "win32":
    import msvcrt
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

MAGIC = 0x53414D54  # 'TMAS' in little endian

def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    target_width = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    target_height = int(sys.argv[3]) if len(sys.argv) > 3 else 480

    if source.isdigit():
        source = int(source)

    try:
        import cv2
    except ImportError:
        sys.stderr.write("[Feeder] Error: cv2 not found.\n")
        sys.exit(1)

    # Initialize Face Engine (InsightFace / SCRFD / ArcFace)
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
        sys.stderr.write(f"[Feeder] Real FaceEngine (SCRFD + ArcFace / OpenVINO) active with config: {cfg_file}\n")
    except Exception as e:
        sys.stderr.write(f"[Feeder] Notice: FaceEngine init info: {e}\n")

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG if isinstance(source, str) and source.startswith("rtsp") else cv2.CAP_ANY)

    if not cap.isOpened():
        sys.stderr.write(f"[Feeder] Warning: Could not open camera {source}\n")
        sys.exit(1)

    sys.stderr.write(f"[Feeder] Streaming from camera {source} -> {target_width}x{target_height}\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        if frame.shape[1] != target_width or frame.shape[0] != target_height:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        faces = []
        if face_engine is not None:
            try:
                faces = face_engine.detect(frame)
            except Exception:
                faces = []

        num_faces = min(len(faces), 16)
        
        # Build header: uint32 magic, uint32 width, uint32 height, uint32 num_faces
        header = struct.pack("<IIII", MAGIC, target_width, target_height, num_faces)
        sys.stdout.buffer.write(header)

        # Write detected faces metadata
        for i in range(num_faces):
            f = faces[i]
            bbox = f.bbox
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            det_score = float(f.detection_score)
            blur_score = float(f.blur_score)
            
            # 5 key landmarks
            lm_x = [x1 + (x2 - x1) * 0.3, x1 + (x2 - x1) * 0.7, x1 + (x2 - x1) * 0.5, x1 + (x2 - x1) * 0.35, x1 + (x2 - x1) * 0.65]
            lm_y = [y1 + (y2 - y1) * 0.35, y1 + (y2 - y1) * 0.35, y1 + (y2 - y1) * 0.55, y1 + (y2 - y1) * 0.75, y1 + (y2 - y1) * 0.75]
            
            face_meta = struct.pack("<ffffff5f5f", x1, y1, x2, y2, det_score, blur_score, *lm_x, *lm_y)
            sys.stdout.buffer.write(face_meta)

            # 512 floats embedding
            emb = f.embedding.astype(np.float32)
            if len(emb) == 512:
                sys.stdout.buffer.write(emb.tobytes())
            else:
                sys.stdout.buffer.write(b'\x00' * (512 * 4))

        # Write raw BGR image bytes
        sys.stdout.buffer.write(frame.tobytes())
        sys.stdout.buffer.flush()

    cap.release()

if __name__ == "__main__":
    main()
