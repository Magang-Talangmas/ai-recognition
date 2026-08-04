"""
OpenVINO Backend for InsightFace.

Provides:
  - OVSession : drop-in replacement for onnxruntime.InferenceSession,
                implementing the subset of the API that InsightFace calls.
  - patch_app_with_openvino() : replaces every model's .session in a
                FaceAnalysis app with an OVSession, keeping all
                InsightFace pre/post-processing logic intact.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Force UTF-8 on Windows so OpenVINO's internal log messages (which contain
# Unicode arrows/symbols) don't crash when printed to a cp1252 terminal.
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import openvino as ov
    from openvino import Core, CompiledModel, Model
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "openvino package is not installed. Run: pip install openvino"
    ) from exc


# ---------------------------------------------------------------------------
# Lightweight NodeInfo shim — InsightFace calls .name on session inputs/outputs
# ---------------------------------------------------------------------------

class _NodeInfo:
    """Minimal shim that mimics onnxruntime NodeArg (has a .name attribute)."""

    def __init__(self, name: str, shape: list | None = None) -> None:
        self.name = name
        self.shape = shape or []

    def __repr__(self) -> str:  # pragma: no cover
        return f"_NodeInfo(name={self.name!r})"


# ---------------------------------------------------------------------------
# OVSession — mirrors onnxruntime.InferenceSession interface
# ---------------------------------------------------------------------------

class OVSession:
    """
    OpenVINO-powered drop-in replacement for onnxruntime.InferenceSession.

    InsightFace models use three methods on their session object:
        session.get_inputs()         → list of NodeArg with .name
        session.get_outputs()        → list of NodeArg with .name
        session.run(None, feed_dict) → list[np.ndarray]
    """

    def __init__(
        self,
        model_path: str | Path,
        core: Core,
        device: str = "AUTO",
        input_shapes: dict[str, list[int]] | None = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"OpenVINO model not found: {model_path}")

        # Read model (supports .onnx, .xml/.bin)
        model: Model = core.read_model(str(model_path))

        # Apply static input shape overrides
        if input_shapes:
            reshape_map: dict[str | int, list[int]] = {}
            for inp in model.inputs:
                name = inp.get_any_name()
                if name in input_shapes:
                    reshape_map[name] = input_shapes[name]
            if reshape_map:
                model.reshape(reshape_map)

        self._compiled: CompiledModel = core.compile_model(model, device)


        # Cache input / output descriptors once
        self._inputs = [
            _NodeInfo(
                inp.get_any_name(),
                list(inp.get_partial_shape().get_min_shape()),
            )
            for inp in self._compiled.inputs
        ]
        self._outputs = [
            _NodeInfo(out.get_any_name())
            for out in self._compiled.outputs
        ]

        # Map from all output names/aliases → compiled output tensor for fast lookup
        self._out_map: dict[str, Any] = {}
        for idx, out in enumerate(self._compiled.outputs):
            try:
                for name in out.names:
                    self._out_map[name] = out
            except Exception:
                pass
            try:
                self._out_map[out.get_any_name()] = out
            except Exception:
                pass
            self._out_map[str(idx)] = out

    # ------------------------------------------------------------------
    # onnxruntime-compatible API
    # ------------------------------------------------------------------

    def get_inputs(self) -> list[_NodeInfo]:
        return self._inputs

    def get_outputs(self) -> list[_NodeInfo]:
        return self._outputs

    def set_providers(self, providers: list[str], provider_options: list | None = None) -> None:
        """No-op stub — InsightFace calls this during prepare(); OpenVINO device
        is fixed at compile time so there is nothing to change here."""
        pass

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        """
        Run inference. When output_names is a non-empty list of strings the
        outputs are returned in that name order (InsightFace SCRFD passes
        self.output_names explicitly). When output_names is None all outputs
        are returned in declaration order.
        """
        result = self._compiled(input_feed)

        if output_names:
            outs = []
            for n in output_names:
                if n in self._out_map:
                    outs.append(np.array(result[self._out_map[n]]))
                else:
                    found = False
                    for out in self._compiled.outputs:
                        if n in out.names:
                            outs.append(np.array(result[out]))
                            found = True
                            break
                    if not found:
                        raise KeyError(f"Output name '{n}' not found in OpenVINO outputs: {list(self._out_map.keys())}")
            return outs

        # Return in output declaration order
        return [np.array(result[out]) for out in self._compiled.outputs]


# ---------------------------------------------------------------------------
# Public helper — patch a fully-prepared FaceAnalysis app
# ---------------------------------------------------------------------------

def _guess_shape_override(model_obj: Any) -> dict[str, list[int]] | None:
    """Attempt to determine correct static input shape for a model dynamically."""
    task = getattr(model_obj, "taskname", None)
    model_file = str(getattr(model_obj, "model_file", "")).lower()

    # SCRFD Detection model
    if task == "detection" or "det" in model_file:
        input_size = getattr(model_obj, "input_size", None)
        if input_size is not None and len(input_size) == 2:
            return {"input.1": [1, 3, int(input_size[1]), int(input_size[0])]}
        return {"input.1": [1, 3, 640, 640]}

    # ArcFace Recognition model (always 112x112)
    if task == "recognition" or "w600k" in model_file:
        return {"input.1": [1, 3, 112, 112]}

    return None


def patch_app_with_openvino(
    app: Any,
    ir_model_dir: str | Path,
    device: str = "AUTO",
) -> dict[str, str]:
    """
    Replace every model's .session in a prepared FaceAnalysis app with an
    OVSession that runs on OpenVINO.

    Parameters
    ----------
    app          : FaceAnalysis instance (already .prepare()-d)
    ir_model_dir : directory that contains the exported .xml IR models
                   (produced by export_to_openvino.py).
                   If an .xml file is missing for a model, falls back
                   to the original .onnx file.
    device       : OpenVINO device string, e.g. "AUTO", "CPU", "GPU"

    Returns
    -------
    dict  model_name → "ov_ir" | "ov_onnx" | "skipped"
    """
    core = Core()
    ir_dir = Path(ir_model_dir)
    status: dict[str, str] = {}

    for model_name, model_obj in app.models.items():
        # Locate best available model file (IR preferred, fall back to ONNX)
        onnx_path = Path(getattr(model_obj, "model_file", ""))
        ir_path = ir_dir / (onnx_path.stem + ".xml")

        if ir_path.exists():
            load_path = ir_path
            kind = "ov_ir"
        elif onnx_path.exists():
            load_path = onnx_path
            kind = "ov_onnx"
        else:
            print(f"[OVBackend] WARNING: Cannot locate model file for '{model_name}', skipping.")
            status[model_name] = "skipped"
            continue

        shape_override = _guess_shape_override(model_obj)

        try:
            session = OVSession(
                model_path=load_path,
                core=core,
                device=device,
                input_shapes=shape_override,
            )
            model_obj.session = session
            print(
                f"[OVBackend] '{model_name}' -> {load_path.name} "
                f"on {device} ({kind})"
            )
            status[model_name] = kind
        except Exception as exc:
            print(f"[OVBackend] ERROR patching '{model_name}': {exc}. Keeping original session.")
            status[model_name] = "skipped"

    return status


def get_available_devices() -> list[str]:
    """Return the list of OpenVINO available device names on this machine."""
    return Core().available_devices
