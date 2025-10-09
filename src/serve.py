import os
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from src.models import load_checkpoint, CheckpointMeta, SimpleNet
except ImportError:  # pragma: no cover
    from models import load_checkpoint, CheckpointMeta, SimpleNet

app = FastAPI(title="MLOps Demo Inference Service")


class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    predicted_class: int
    probabilities: list[float]


MODEL: Optional[SimpleNet] = None
META: Optional[CheckpointMeta] = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
def load_model():
    global MODEL, META
    model_dir = os.getenv("MODEL_DIR", "/model")
    model_path = Path(model_dir) / "model.pt"
    if not model_path.exists():
        # Fail fast so container restarts and surfaces the missing model
        raise RuntimeError(f"Model file not found at {model_path}")
    model, meta = load_checkpoint(str(model_path), DEVICE)
    MODEL = model
    META = meta


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if MODEL is None or META is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Validate feature length
    if len(req.features) != META.in_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {META.in_features} features, got {len(req.features)}",
        )
    x = torch.tensor(req.features, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred = int(torch.argmax(probs).item())
        return PredictResponse(
            predicted_class=pred,
            probabilities=[float(p) for p in probs.tolist()],
        )


# For local testing: `uvicorn src.serve:app --host 0.0.0.0 --port 8000`
