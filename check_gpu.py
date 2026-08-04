"""
GPU Verification Script for ONNX Runtime, PyTorch, and InsightFace
"""
import sys

def check_gpu_status():
    print("=" * 50)
    print(" SYSTEM GPU CHECK ")
    print("=" * 50)

    # 1. Check ONNX Runtime (Used by InsightFace)
    print("\n[1] Checking ONNX Runtime (InsightFace backend)...")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  Available Providers: {providers}")
        
        if "CUDAExecutionProvider" in providers:
            print("  STATUS: GPU (NVIDIA CUDA) is ACTIVE & AVAILABLE!")
        elif "DmlExecutionProvider" in providers:
            print("  STATUS: GPU (DirectML - Windows DirectX 12) is ACTIVE & AVAILABLE!")
        elif "ROCMExecutionProvider" in providers:
            print("  STATUS: GPU (AMD ROCm) is ACTIVE & AVAILABLE!")
        else:
            print("  STATUS: CPU ONLY (No GPU provider registered in ONNX Runtime)")
    except Exception as e:
        print(f"  Error checking ONNX Runtime: {e}")

    # 2. Check PyTorch (Used by Ultralytics YOLO)
    print("\n[2] Checking PyTorch (Ultralytics YOLO backend)...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"  PyTorch CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"  GPU Device Count: {torch.cuda.device_count()}")
            print(f"  GPU Device Name: {torch.cuda.get_device_name(0)}")
            print("  STATUS: GPU (CUDA) is READY for PyTorch / YOLO!")
        else:
            print("  STATUS: CPU ONLY (PyTorch CUDA support not installed or GPU not detected)")
    except Exception as e:
        print(f"  Error checking PyTorch: {e}")

    # 3. Check Active FaceEngine Config
    print("\n[3] Checking Project config.yaml settings...")
    try:
        from attendance_cv.config import load_config
        config = load_config()
        print(f"  Face Config Providers: {config.face.providers}")
    except Exception as e:
        print(f"  Could not read config.yaml: {e}")

    print("\n" + "=" * 50)

if __name__ == "__main__":
    check_gpu_status()
