import base64
import io
import time
import torch
from PIL import Image, UnidentifiedImageError
from .model_loader import CLASSES, DEVICE, MEAN, STD


def decode_image(value: str):
    try:
        encoded = value.split(",", 1)[1] if "," in value else value
        payload = base64.b64decode(encoded, validate=True)
        if len(payload) > 10 * 1024 * 1024:
            raise ValueError("Image must be 10 MB or smaller.")
        image = Image.open(io.BytesIO(payload))
        if image.width * image.height > 40_000_000:
            raise ValueError("Image dimensions are too large to process safely.")
        image = image.convert("RGB")
    except (ValueError, UnidentifiedImageError, OSError) as error:
        raise ValueError("Could not read image. Upload a valid PNG, JPEG, or WebP.") from error
    return image, torch.tensor(list(image.resize((32, 32)).getdata()), dtype=torch.float32).view(32, 32, 3).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0


def normalized(pixels):
    return (pixels - MEAN) / STD


def predictions(model, pixels):
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(normalized(pixels))
        probs = torch.softmax(logits, dim=1)[0]
        values, indices = probs.topk(5)
    return {"prediction": CLASSES[int(indices[0])], "confidence": float(values[0]),
            "top5": [{"class": CLASSES[int(i)], "confidence": float(v)} for v, i in zip(values, indices)],
            "latency_ms": (time.perf_counter() - started) * 1000}


def encode_tensor(pixels):
    array = (pixels.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    image = Image.fromarray(array)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
