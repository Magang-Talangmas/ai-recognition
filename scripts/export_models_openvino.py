import os
import sys
import torch
import numpy as np
import openvino as ov
from facenet_pytorch import MTCNN, InceptionResnetV1

def export_all_models(output_dir=None):
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "models", "openvino_fp16")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Exporting OpenVINO FP16 models to: {output_dir}")
    
    # 1. Initialize MTCNN and InceptionResnetV1
    print("[1/4] Initializing PyTorch models...")
    mtcnn = MTCNN(keep_all=True, device='cpu')
    pnet = mtcnn.pnet.eval()
    rnet = mtcnn.rnet.eval()
    onet = mtcnn.onet.eval()
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    # 2. Export P-Net (Dynamic H, W)
    print("[2/4] Converting P-Net (Dynamic shape) to OpenVINO FP16 IR...")
    dummy_pnet = torch.randn(1, 3, 240, 320, dtype=torch.float32)
    pnet_ov = ov.convert_model(
        pnet,
        example_input=dummy_pnet,
        input=[(1, 3, ov.Dimension.dynamic(), ov.Dimension.dynamic())]
    )
    pnet_path = os.path.join(output_dir, "pnet.xml")
    ov.save_model(pnet_ov, pnet_path, compress_to_fp16=True)
    print(f"  [+] Saved P-Net: {pnet_path}")
    
    # 3. Export R-Net (Batch, 3, 24, 24)
    print("[3/4] Converting R-Net to OpenVINO FP16 IR...")
    dummy_rnet = torch.randn(1, 3, 24, 24, dtype=torch.float32)
    rnet_ov = ov.convert_model(
        rnet,
        example_input=dummy_rnet,
        input=[(ov.Dimension.dynamic(), 3, 24, 24)]
    )
    rnet_path = os.path.join(output_dir, "rnet.xml")
    ov.save_model(rnet_ov, rnet_path, compress_to_fp16=True)
    print(f"  [+] Saved R-Net: {rnet_path}")
    
    # 4. Export O-Net (Batch, 3, 48, 48)
    print("[4/4] Converting O-Net to OpenVINO FP16 IR...")
    dummy_onet = torch.randn(1, 3, 48, 48, dtype=torch.float32)
    onet_ov = ov.convert_model(
        onet,
        example_input=dummy_onet,
        input=[(ov.Dimension.dynamic(), 3, 48, 48)]
    )
    onet_path = os.path.join(output_dir, "onet.xml")
    ov.save_model(onet_ov, onet_path, compress_to_fp16=True)
    print(f"  [+] Saved O-Net: {onet_path}")
    
    # 5. Export InceptionResnetV1 (Batch, 3, 160, 160)
    print("[5/4] Converting InceptionResnetV1 (512-dim embedding) to OpenVINO FP16 IR...")
    dummy_resnet = torch.randn(1, 3, 160, 160, dtype=torch.float32)
    resnet_ov = ov.convert_model(
        resnet,
        example_input=dummy_resnet,
        input=[(ov.Dimension.dynamic(), 3, 160, 160)]
    )
    resnet_path = os.path.join(output_dir, "facenet_vggface2.xml")
    ov.save_model(resnet_ov, resnet_path, compress_to_fp16=True)
    print(f"  [+] Saved InceptionResnetV1: {resnet_path}")
    
    print("\n=======================================================")
    print(" [SUCCESS] All MTCNN and FaceNet models converted to FP16 IR!")
    print("=======================================================")

if __name__ == "__main__":
    export_all_models()
