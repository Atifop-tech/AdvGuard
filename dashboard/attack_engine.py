import torch
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.cw import cw_l2_attack
from .inference import normalized, predictions
from .model_loader import DEVICE, MEAN, STD


def generate(registry, pixels, attack, settings):
    victim = registry.models["victim"]
    clean_norm = normalized(pixels)
    with torch.no_grad():
        label = victim(clean_norm).argmax(1)
    attack = attack.upper()
    kwargs = {"mean": MEAN, "std": STD}
    if attack == "FGSM":
        adv_norm = fgsm_attack(victim, clean_norm, label, epsilon=settings["epsilon"], **kwargs)
    elif attack == "PGD":
        adv_norm = pgd_attack(victim, clean_norm, label, epsilon=settings["epsilon"], alpha=settings["alpha"], steps=settings["steps"], random_start=settings["random_start"], **kwargs)
    elif attack in ("C&W", "CW"):
        adv_norm = cw_l2_attack(victim, clean_norm, label, MEAN, STD, c=settings["c"], kappa=settings["kappa"], steps=settings["steps"], learning_rate=settings["learning_rate"])
    else:
        raise ValueError("Attack must be FGSM, PGD, or C&W")
    adv_pixels = (adv_norm * STD + MEAN).clamp(0, 1)
    before = predictions(victim, pixels)
    after = predictions(victim, adv_pixels)
    perturbation = adv_pixels - pixels
    abs_delta = perturbation.abs()
    heat = (abs_delta.amax(dim=1, keepdim=True) * 8).expand(-1, 3, -1, -1)
    return {"attack": attack, "parameters": settings, "original": pixels,
            "adversarial": adv_pixels, "before": before, "after": after,
            "success": after["prediction"] != before["prediction"],
            "linf": float(abs_delta.max()), "l2": float(torch.linalg.vector_norm(perturbation)),
            "heatmap": heat.clamp(0, 1)}
