import torch
from .inference import normalized
from .model_loader import MEAN, STD


def detect(registry, pixels, threshold=None):
    if "detector" not in registry.models:
        raise RuntimeError("Adversarial detector unavailable: its trained checkpoint is missing.")
    with torch.inference_mode():
        probability = torch.sigmoid(registry.models["detector"](normalized(pixels))).item()
    cutoff = registry.threshold if threshold is None else threshold
    return {"probability": probability, "threshold": cutoff, "classification": "adversarial" if probability >= cutoff else "clean"}
