"""
Local Face Enrollment Utility (MTCNN + OpenVINO FP16)
Processes all employee reference photos from data/enroll/ without cloud dependency.
"""

import os
import sys
import glob
import re

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np
from src.openvino_mtcnn import OpenVINOMTCNN
from src.openvino_face_engine import OpenVINOFaceEngine, FaceMatcher


def format_display_name(folder_name):
    """Convert camelCase / PascalCase or snake_case to Title Case Name"""
    s = re.sub(r'([A-Z])', r' \1', folder_name)
    s = s.replace('_', ' ').replace('-', ' ').strip()
    return s.title()


def enroll_local_dataset(enroll_dir="data/enroll", output_db="data/face_embeddings_openvino.npz"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(enroll_dir):
        enroll_dir = os.path.join(base_dir, enroll_dir)
    if not os.path.isabs(output_db):
        output_db = os.path.join(base_dir, output_db)
        
    print("=================================================================")
    print("     LOCAL FACE ENROLLMENT UTILITY (MTCNN + OPENVINO FP16)       ")
    print("=================================================================")
    print(f"[*] Reading dataset from: {enroll_dir}")
    print(f"[*] Target database     : {output_db}\n")
    
    if not os.path.exists(enroll_dir):
        print(f"[Error] Dataset directory not found: {enroll_dir}")
        return False
        
    # Initialize OpenVINO MTCNN and Feature Engine
    print("[1/3] Initializing OpenVINO FP16 Detector & Feature Extractor...")
    detector = OpenVINOMTCNN()
    feature_engine = OpenVINOFaceEngine()
    matcher = FaceMatcher(similarity_threshold=0.60)
    
    person_dirs = [d for d in os.listdir(enroll_dir) if os.path.isdir(os.path.join(enroll_dir, d))]
    print(f"[2/3] Found {len(person_dirs)} employee profiles in local directory:\n")
    
    total_enrolled = 0
    total_photos_processed = 0
    
    for idx, p_dir in enumerate(sorted(person_dirs), 1):
        person_path = os.path.join(enroll_dir, p_dir)
        person_id = p_dir
        display_name = format_display_name(p_dir)
        
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_files.extend(glob.glob(os.path.join(person_path, ext)))
            image_files.extend(glob.glob(os.path.join(person_path, ext.upper())))
            
        print(f"  [{idx}/{len(person_dirs)}] Enrolling: {display_name} (ID: {person_id}) - {len(image_files)} photos")
        
        person_embeddings = []
        
        for img_p in image_files:
            img = cv2.imread(img_p)
            if img is None:
                continue
            total_photos_processed += 1
            
            # For ultra high-res photos (e.g. 3000x4000), resize to standard workable dimension (max dim 1200)
            h, w = img.shape[:2]
            max_dim = max(h, w)
            if max_dim > 1200:
                scale = 1200.0 / float(max_dim)
                work_img = cv2.resize(img, (int(w * scale), int(h * scale)))
            else:
                scale = 1.0
                work_img = img
                
            # Detect face & landmarks with MTCNN
            boxes, landmarks = detector.detect(work_img)
            
            # If not detected, try multiple pyramid scales (0.5, 0.75, 1.5)
            if len(boxes) == 0:
                for alt_scale in [0.5, 0.75, 1.25]:
                    scaled = cv2.resize(work_img, (int(work_img.shape[1]*alt_scale), int(work_img.shape[0]*alt_scale)))
                    boxes, landmarks = detector.detect(scaled)
                    if len(boxes) > 0:
                        landmarks = landmarks / alt_scale
                        break
                        
            if len(boxes) == 0:
                print(f"      [Warning] No face detected in: {os.path.basename(img_p)}")
                continue
                
            # Scale landmarks back to original image
            if scale != 1.0:
                landmarks = landmarks / scale
                
            # Pick the largest / highest confidence face
            best_face_idx = 0
            if len(boxes) > 1:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best_face_idx = int(np.argmax(areas))
                
            aligned = detector.align_face(img, landmarks[best_face_idx], target_size=(160, 160))
            emb = feature_engine.extract_embedding(aligned)
            person_embeddings.append(emb)
            
        if len(person_embeddings) == 0:
            print(f"      [!] FAILED: Could not extract face embeddings for {display_name}")
            continue
            
        # Compute multi-shot average template & L2 normalize
        avg_template = np.mean(person_embeddings, axis=0)
        norm = np.linalg.norm(avg_template)
        if norm > 1e-6:
            avg_template = avg_template / norm
            
        matcher.add_template(person_id, display_name, avg_template)
        print(f"      [OK] Successfully enrolled {len(person_embeddings)} reference samples.")
        total_enrolled += 1
        
    print(f"\n[3/3] Saving embedding database...")
    matcher.save_database(output_db)

    # Also save binary format (data/embeddings.bin) for C native matcher
    bin_path = os.path.join(base_dir, "data", "embeddings.bin")
    try:
        import struct
        with open(bin_path, "wb") as f:
            f.write(b"FACES1\x00\x00")
            count = len(matcher.employee_ids)
            dim = 512
            f.write(struct.pack("<ii", count, dim))
            for emp_id, tmpl in zip(matcher.employee_ids, matcher.templates):
                id_bytes = emp_id.encode("utf-8")[:127].ljust(128, b"\x00")
                f.write(id_bytes)
                f.write(tmpl.astype(np.float32).tobytes())
        print(f" Native C database saved to: {bin_path}")
    except Exception as e:
        print(f" [Warning] Could not export binary embeddings.bin: {e}")
    
    print("\n=================================================================")
    print(f" [SUCCESS] Enrolled {total_enrolled}/{len(person_dirs)} employees ({total_photos_processed} photos)")
    print(f" NPZ DB : {output_db}")
    print(f" Bin DB : {bin_path}")
    print("=================================================================")
    return True


if __name__ == "__main__":
    enroll_local_dataset()
