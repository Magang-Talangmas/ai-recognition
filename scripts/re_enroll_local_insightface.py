import os
import cv2
import struct
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

def enroll_local():
    print("Initializing InsightFace (buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    target_enroll = Path("data/enroll")
    target_bin = Path("data/embeddings.bin")
    
    emp_ids = []
    templates = []
    
    for emp_dir in sorted(target_enroll.iterdir()):
        if not emp_dir.is_dir(): continue
        
        emp_vectors = []
        for p in emp_dir.glob("*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                img = cv2.imread(str(p))
                if img is None: continue
                faces = app.get(img)
                if faces and len(faces) > 0:
                    best_face = max(faces, key=lambda f: f.det_score)
                    emp_vectors.append(best_face.embedding)
        
        if emp_vectors:
            mean_vec = np.mean(emp_vectors, axis=0)
            mean_vec = mean_vec / np.linalg.norm(mean_vec)
            emp_ids.append(emp_dir.name)
            templates.append(mean_vec)
            print(f"Enrolled {emp_dir.name} ({len(emp_vectors)} photos)")

    if emp_ids:
        target_bin.parent.mkdir(parents=True, exist_ok=True)
        with open(target_bin, "wb") as f:
            f.write(b"FACES1\x00\x00")
            f.write(struct.pack("<ii", len(emp_ids), 512))
            for i in range(len(emp_ids)):
                emp_id_bytes = emp_ids[i].encode("utf-8")[:127].ljust(128, b"\x00")
                f.write(emp_id_bytes)
                vec = templates[i].astype(np.float32)
                f.write(vec.tobytes())
        print(f"Saved {len(emp_ids)} employees to {target_bin}")

if __name__ == "__main__":
    enroll_local()
