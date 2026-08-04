"""
export_to_openvino.py — One-time script to convert InsightFace ONNX models to
OpenVINO IR format (.xml / .bin) with static input shapes.

Models converted:
  det_10g.onnx    → det_10g.xml  (SCRFD, static 1×3×640×640)
  w600k_r50.onnx  → w600k_r50.xml (ArcFace, static 1×3×112×112)

Output directory: .insightface/models/buffalo_l_ov/

Usage:
    python export_to_openvino.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import openvino as ov
    from openvino import Core, convert_model, save_model
    from openvino.preprocess import PrePostProcessor
except ImportError:
    print("ERROR: openvino is not installed. Run: pip install openvino")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BUFFALO_L_ONNX_DIR = Path(".insightface/models/buffalo_l")
OV_IR_DIR          = Path(".insightface/models/buffalo_l_ov")

MODELS_TO_CONVERT = [
    {
        "name":        "SCRFD Face Detector",
        "onnx":        "det_10g.onnx",
        "input_name":  "input.1",
        "input_shape": [1, 3, 320, 320],   # turun dari 640: ~2-3x speedup pada CPU
        "description": "SCRFD-10G — static 320×320",
    },
    {
        "name":        "ArcFace Recognition",
        "onnx":        "w600k_r50.onnx",
        "input_name":  "input.1",
        "input_shape": [1, 3, 112, 112],
        "description": "ResNet-50 ArcFace — static 112×112",
    },
]


# ---------------------------------------------------------------------------
# Device report
# ---------------------------------------------------------------------------

def report_devices(core: Core) -> None:
    print("\n" + "=" * 60)
    print("  OpenVINO Available Devices")
    print("=" * 60)
    for device in core.available_devices:
        try:
            name = core.get_property(device, "FULL_DEVICE_NAME")
        except Exception:
            name = device
        print(f"  [{device}]  {name}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------

def convert_one(
    core: Core,
    cfg: dict,
    out_dir: Path,
) -> Path | None:
    onnx_path = BUFFALO_L_ONNX_DIR / cfg["onnx"]
    out_xml   = out_dir / (Path(cfg["onnx"]).stem + ".xml")

    if not onnx_path.exists():
        print(f"  [SKIP]  {cfg['onnx']} not found at {onnx_path}")
        return None

    if out_xml.exists():
        print(f"  [SKIP]  {out_xml.name} already exists — delete to re-convert")
        return out_xml

    print(f"\n  Converting: {cfg['name']} ({cfg['description']})")
    print(f"    Source : {onnx_path}")
    print(f"    Output : {out_xml}")

    # Build reshape dict keyed by input name
    reshape = {cfg["input_name"]: cfg["input_shape"]}

    try:
        # convert_model() reads ONNX and returns an ov.Model
        model = convert_model(str(onnx_path))

        # Apply static shape -- avoids dynamic dispatch overhead at runtime
        model.reshape(reshape)

        save_model(model, str(out_xml), compress_to_fp16=False)
        print(f"    Done  OK ({out_xml.stat().st_size / 1e6:.1f} MB XML + BIN pair)")
        return out_xml

    except Exception as exc:
        print(f"    ERROR: {exc}")
        return None


# ---------------------------------------------------------------------------
# Verification — quick inference smoke test on each converted model
# ---------------------------------------------------------------------------

def verify_model(core: Core, xml_path: Path, input_shape: list[int], device: str = "CPU") -> bool:
    import numpy as np
    try:
        model    = core.read_model(str(xml_path))
        compiled = core.compile_model(model, device)
        dummy    = {compiled.inputs[0]: np.zeros(input_shape, dtype=np.float32)}
        result   = compiled(dummy)
        out_shapes = [list(v.shape) for v in result.values()]
        print(f"    Verify OK on {device} -- outputs: {out_shapes}")
        return True
    except Exception as exc:
        print(f"    Verify FAILED: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    core = Core()
    report_devices(core)

    OV_IR_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Converting ONNX -> OpenVINO IR")
    print("=" * 60)

    converted: list[tuple[Path, list[int]]] = []
    for cfg in MODELS_TO_CONVERT:
        xml_path = convert_one(core, cfg, OV_IR_DIR)
        if xml_path:
            converted.append((xml_path, cfg["input_shape"]))

    if not converted:
        print("\nNo models were converted.")
        return

    print("\n" + "=" * 60)
    print("  Smoke-Testing Converted Models on CPU")
    print("=" * 60)
    all_ok = True
    for xml_path, shape in converted:
        print(f"\n  {xml_path.name}")
        ok = verify_model(core, xml_path, shape, device="CPU")
        all_ok = all_ok and ok

    print("\n" + "=" * 60)
    if all_ok:
        print("  ALL MODELS CONVERTED AND VERIFIED SUCCESSFULLY")
        print(f"  IR files are in: {OV_IR_DIR.resolve()}")
        print("\n  Next step: run_camera.py will use backend: openvino")
    else:
        print("  WARNING: Some models failed verification. Check errors above.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
