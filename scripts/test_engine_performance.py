"""
Benchmark & Validation script for MTCNN + OpenVINO FP16 Pipeline
"""

import os
import sys
import time
import glob

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np
from src.openvino_mtcnn import OpenVINOMTCNN
from src.openvino_face_engine import OpenVINOFaceEngine, FaceMatcher


def benchmark_pipeline():
    print("=================================================================")
    print("      BENCHMARKING MTCNN + OPENVINO FP16 PIPELINE                ")
    print("=================================================================\n")
    
    # 1. Initialize Engines
    t0 = time.perf_counter()
    detector = OpenVINOMTCNN()
    feature_engine = OpenVINOFaceEngine()
    matcher = FaceMatcher("data/face_embeddings_openvino.npz", similarity_threshold=0.60)
    t1 = time.perf_counter()
    print(f"[Init] Engines compiled & loaded in {(t1-t0)*1000:.1f}ms\n")
    
    # 2. Collect sample images
    images = glob.glob("data/enroll/*/*.jpg") + glob.glob("data/enroll/*/*.png")
    if not images:
        print("[Error] No sample images found in data/enroll")
        return
        
    print(f"[*] Testing with {min(10, len(images))} sample employee images...")
    
    latencies_detect = []
    latencies_align = []
    latencies_extract = []
    latencies_match = []
    accuracies = []
    
    for img_path in images[:10]:
        img = cv2.imread(img_path)
        expected_person = os.path.basename(os.path.dirname(img_path))
        
        # Resize to 704x480 standard substream size for realistic camera benchmark
        h, w = img.shape[:2]
        target_w = 704
        target_h = int(h * (target_w / w))
        frame = cv2.resize(img, (target_w, target_h))
        
        # Benchmark Detection
        t_start = time.perf_counter()
        boxes, landmarks = detector.detect(frame)
        t_det = time.perf_counter()
        latencies_detect.append((t_det - t_start) * 1000.0)
        
        if len(boxes) > 0:
            # Benchmark Alignment
            t_al_start = time.perf_counter()
            aligned = detector.align_face(frame, landmarks[0], target_size=(160, 160))
            t_al_end = time.perf_counter()
            latencies_align.append((t_al_end - t_al_start) * 1000.0)
            
            # Benchmark Extraction
            t_ext_start = time.perf_counter()
            emb = feature_engine.extract_embedding(aligned)
            t_ext_end = time.perf_counter()
            latencies_extract.append((t_ext_end - t_ext_start) * 1000.0)
            
            # Benchmark Matching
            t_m_start = time.perf_counter()
            emp_id, emp_name, sim, is_match = matcher.match(emb)
            t_m_end = time.perf_counter()
            latencies_match.append((t_m_end - t_m_start) * 1000.0)
            
            is_correct = (emp_id == expected_person)
            accuracies.append(is_correct)
            
            print(f"  Sample: {os.path.basename(img_path)[:20]:20s} -> Match: {emp_name:25s} (Sim: {sim:.1%}) [Correct: {is_correct}]")
            
    print("\n-----------------------------------------------------------------")
    print("                    PERFORMANCE SUMMARY                          ")
    print("-----------------------------------------------------------------")
    avg_det = np.mean(latencies_detect) if latencies_detect else 0
    avg_align = np.mean(latencies_align) if latencies_align else 0
    avg_ext = np.mean(latencies_extract) if latencies_extract else 0
    avg_match = np.mean(latencies_match) if latencies_match else 0
    total_pipeline = avg_det + avg_align + avg_ext + avg_match
    throughput_fps = 1000.0 / total_pipeline if total_pipeline > 0 else 0
    accuracy_rate = np.mean(accuracies) * 100.0 if accuracies else 0
    
    print(f"  - MTCNN Detection Latency  : {avg_det:6.2f} ms")
    print(f"  - Landmark Alignment       : {avg_align:6.2f} ms")
    print(f"  - FP16 Feature Extraction  : {avg_ext:6.2f} ms")
    print(f"  - Vector Cosine Matching   : {avg_match:6.2f} ms")
    print(f"  - Total End-to-End Latency : {total_pipeline:6.2f} ms")
    print(f"  - Theoretical Max FPS      : {throughput_fps:6.1f} FPS")
    print(f"  - Top-1 Match Accuracy     : {accuracy_rate:6.1f} %")
    print("=================================================================\n")


if __name__ == "__main__":
    benchmark_pipeline()
