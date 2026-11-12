from pathlib import Path
import torch

from model import create_resnet18_cifar10
from detector.detector_model import AdversarialDetector
from purification.model import PurificationNet

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616), device=DEVICE).view(1, 3, 1, 1)
CLASSES = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]


def _load_classifier(path: Path):
    model = create_resnet18_cifar10()
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if next(iter(state)).startswith("module."):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.threshold = 0.26
        self.errors = {}
        self.checkpoints = {
            "victim": ("Victim ResNet-18", "model/resnet18_cifar10.pth", "CIFAR-10 classifier"),
            "protected_baseline": ("Protected ResNet-18 baseline", "model/protected_resnet18_baseline.pth", "Baseline protected classifier checkpoint"),
            "protected": ("Protected ResNet-18", "model/protected_resnet18_continued_best.pth", "Adversarially protected classifier"),
            "purifier": ("Purification CNN", "model/purifier_best.pth", "Residual image denoiser"),
            "detector": ("Adversarial detector", "model/adversarial_detector_final.pth", "Binary adversarial detector"),
        }
        for key, (name, relative, role) in self.checkpoints.items():
            path = ROOT / relative
            if not path.is_file():
                self.errors[key] = "Checkpoint not found"
                continue
            try:
                if key in ("victim", "protected", "protected_baseline"):
                    model = _load_classifier(path)
                elif key == "purifier":
                    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
                    model = PurificationNet(int(checkpoint.get("channels", 48)), int(checkpoint.get("blocks", 4)))
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model.to(DEVICE).eval()
                else:
                    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
                    model = AdversarialDetector()
                    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
                    self.threshold = float(checkpoint.get("threshold", self.threshold))
                    model.to(DEVICE).eval()
                self.models[key] = model
            except Exception as error:
                self.errors[key] = f"Load failed: {error}"

    def status(self):
        details = {
            "victim": ("ResNet-18 · CIFAR-10 stem", "Classifier"),
            "protected_baseline": ("ResNet-18 · CIFAR-10 stem", "Protected classifier baseline"),
            "protected": ("ResNet-18 · CIFAR-10 stem", "Protected classifier"),
            "purifier": ("Residual CNN · 48 channels · 4 blocks", "Purification"),
            "detector": ("CNN · 3 convolution stages · binary logit", "Detection"),
        }
        return {
            "device": str(DEVICE),
            "models": [{"id": key, "name": name, "checkpoint": relative, "role": role,
                        "architecture": details[key][0], "kind": details[key][1],
                        "available": key in self.models, "status": self.errors.get(key, "Loaded")}
                       for key, (name, relative, role) in self.checkpoints.items()],
            "detector_threshold": self.threshold,
        }
