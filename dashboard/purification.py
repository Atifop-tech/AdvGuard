import torch


def purify(registry, pixels, reference=None):
    if "purifier" not in registry.models:
        raise RuntimeError("Purification CNN unavailable: checkpoint failed to load.")
    started = __import__("time").perf_counter()
    with torch.inference_mode():
        output = registry.models["purifier"](pixels).clamp(0, 1)
    result = {"image": output, "latency_ms": (__import__("time").perf_counter() - started) * 1000}
    if reference is not None:
        delta = output - reference
        result.update({"mae": float(delta.abs().mean()), "mse": float(delta.square().mean())})
    return result
