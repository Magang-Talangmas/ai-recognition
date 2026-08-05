"""
Real-time RTSP/Webcam Frame & Face Detection Feeder for Talangmas AI-Recognition (C Edition).
Guarantees clean binary stream via dedicated binary stdout descriptor.
"""

import sys
import os
import time
import struct
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
        sys.stderr.write(f"[Feeder] Real FaceEngine (SCRFD + ArcFace / OpenVINO) initialized with {cfg_file}\n")
    except Exception as e:
        sys.stderr.write(f"[Feeder] FaceEngine load info: {e}\n")

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG if isinstance(source, str) and source.startswith("rtsp") else cv2.CAP_ANY)

    if not cap.isOpened():
        sys.stderr.write(f"[Feeder] Error: Could not open camera source: {source}\n")
        sys.exit(1)

    sys.stderr.write(f"[Feeder] Successfully connected to camera: {source}\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.005)
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
        binary_stream.write(header)

        # Write detected faces metadata
        for i in range(num_faces):
            f = faces[i]
            bbox = f.bbox
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            det_score = float(f.detection_score)
            blur_score = float(f.blur_score)
            
            # Key landmarks
            lm_x = [x1 + (x2 - x1) * 0.3, x1 + (x2 - x1) * 0.7, x1 + (x2 - x1) * 0.5, x1 + (x2 - x1) * 0.35, x1 + (x2 - x1) * 0.65]
            lm_y = [y1 + (y2 - y1) * 0.35, y1 + (y2 - y1) * 0.35, y1 + (y2 - y1) * 0.55, y1 + (y2 - y1) * 0.75, y1 + (y2 - y1) * 0.75]
            
            face_meta = struct.pack("<ffffff5f5f", x1, y1, x2, y2, det_score, blur_score, *lm_x, *lm_y)
            binary_stream.write(face_meta)

            # 512 floats embedding
            emb = f.embedding.astype(np.float32)
            if len(emb) == 512:
                binary_stream.write(emb.tobytes())
            else:
                binary_stream.write(b'\x00' * (512 * 4))

        # Write raw BGR image bytes
        binary_stream.write(frame.tobytes())
        binary_stream.flush()

    cap.release()

if __name__ == "__main__":
    main()
