from .inference import predictions
from .model_loader import DEVICE
from .detection import detect
from .purification import purify


def run_pipeline(registry, pixels, threshold=None, use_purification=True, clean_reference=None):
    result = {}
    try:
        result["detection"] = detect(registry, pixels, threshold)
    except RuntimeError as error:
        result["detection_error"] = str(error)
    final_input = pixels
    if use_purification:
        reconstruction = purify(registry, pixels, clean_reference)
        final_input = reconstruction["image"]
        result["purification"] = {"image": final_input, "mae": reconstruction["mae"], "mse": reconstruction["mse"], "latency_ms": reconstruction["latency_ms"]}
    if "protected" not in registry.models:
        raise RuntimeError("Protected classifier unavailable.")
    if clean_reference is not None:
        result["protected_clean"] = predictions(registry.models["protected"], clean_reference)
    result["protected"] = predictions(registry.models["protected"], final_input)
    result["input_prediction"] = predictions(registry.models["protected"], pixels)
    return result
