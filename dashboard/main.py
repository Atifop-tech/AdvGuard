from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
import asyncio
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .model_loader import ModelRegistry
from .inference import decode_image, predictions, encode_tensor
from .attack_engine import generate
from .detection import detect
from .purification import purify
from .pipeline import run_pipeline
from .schemas import ExperimentRequest, ClassifyRequest

registry = ModelRegistry()
history = []
lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app):
    yield

app = FastAPI(title="AdvGuard Defence Intelligence Lab", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])

def require(key):
    if key not in registry.models:
        reason = registry.errors.get(key, "model not available")
        raise HTTPException(503, f"{key.title()} unavailable: {reason}")

def image_or_400(encoded):
    try: return decode_image(encoded)
    except ValueError as error: raise HTTPException(400, str(error)) from error

@app.get("/api/health")
def health(): return {"status": "ok", "device": str(registry.status()["device"])}

@app.get("/api/models")
def models(): return registry.status()

@app.post("/api/classify")
async def classify(request: ClassifyRequest):
    require(request.model_id); _, pixels = image_or_400(request.image)
    return predictions(registry.models[request.model_id], pixels)

@app.post("/api/attack")
async def attack(request: ExperimentRequest):
    require("victim"); _, pixels = image_or_400(request.image)
    settings = {"epsilon": request.epsilon, "alpha": request.alpha, "steps": request.steps, "random_start": request.random_start, "c": request.c, "kappa": request.kappa, "learning_rate": request.learning_rate}
    async with lock:
        result = await asyncio.to_thread(generate, registry, pixels, request.attack, settings)
        return await save_result(result, pixels, request)

@app.post("/api/detect")
async def detection(request: ClassifyRequest):
    require("detector"); _, pixels = image_or_400(request.image)
    try: return detect(registry, pixels)
    except RuntimeError as error: raise HTTPException(503, str(error)) from error

@app.post("/api/purify")
async def purification(request: ClassifyRequest):
    require("purifier"); _, pixels = image_or_400(request.image)
    result = await asyncio.to_thread(purify, registry, pixels)
    delta = result["image"] - pixels
    return {"image": encode_tensor(result["image"]), "change_mae": float(delta.abs().mean()), "change_mse": float(delta.square().mean()), "latency_ms": result["latency_ms"]}

async def save_result(result, original, request):
    record = {"id": str(uuid4()), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "attack": result["attack"], "parameters": result["parameters"], "success": result["success"], "linf": result["linf"], "l2": result["l2"], "before": result["before"], "after": result["after"]}
    record["original_image"] = encode_tensor(original)
    record["adversarial_image"] = encode_tensor(result["adversarial"])
    record["heatmap"] = encode_tensor(result["heatmap"])
    if "protected" in registry.models and "purifier" in registry.models:
        try:
            defense = await asyncio.to_thread(run_pipeline, registry, result["adversarial"], request.threshold, request.purify, original)
            record["defense"] = {"detection": defense.get("detection"), "detection_error": defense.get("detection_error"), "protected_clean": defense.get("protected_clean"), "protected": defense["protected"], "input_prediction": defense["input_prediction"]}
            if "purification" in defense:
                record["purified_image"] = encode_tensor(defense["purification"]["image"])
                record["defense"]["purification"] = {key: val for key, val in defense["purification"].items() if key != "image"}
        except RuntimeError as error: record["defense_error"] = str(error)
    history.insert(0, record); del history[100:]
    return record

@app.post("/api/pipeline")
async def pipeline(request: ExperimentRequest):
    require("victim"); _, pixels = image_or_400(request.image)
    settings = {"epsilon": request.epsilon, "alpha": request.alpha, "steps": request.steps, "random_start": request.random_start, "c": request.c, "kappa": request.kappa, "learning_rate": request.learning_rate}
    async with lock:
        attack_result = await asyncio.to_thread(generate, registry, pixels, request.attack, settings)
        try: return await save_result(attack_result, pixels, request)
        except HTTPException: raise

@app.get("/api/history")
def get_history(): return history

@app.get("/api/samples")
def samples():
    candidates = [Path.home()/".cache/torch/datasets/cifar-10-batches-py", Path(__file__).resolve().parents[1]/"Data/cifar-10-batches-py", Path(__file__).resolve().parents[1]/"data/cifar-10-batches-py"]
    return {"available": any(path.exists() for path in candidates), "message": "Local CIFAR-10 test data found." if any(path.exists() for path in candidates) else "No local CIFAR-10 test set found. Upload an image to begin."}
