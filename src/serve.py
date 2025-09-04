import os
from pathlib import Path

import torch
from torch import nn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="MLOps Demo Inference Service")


class SimpleNet(nn.Module):
    def __init__(self, in_features: int = 20, hidden: int = 32, out_features: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        return self.net(x)


class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    predicted_class: int
    probabilities: list[float]


MODEL = None
META = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
def load_model():
    global MODEL, META
    model_dir = os.getenv("MODEL_DIR", "/model")
    model_path = Path(model_dir) / "model.pt"
    checkpoint = torch.load(model_path, map_location=DEVICE)
    in_features = checkpoint.get("in_features", 20)
    classes = checkpoint.get("classes", 2)
    model = SimpleNet(in_features=in_features, out_features=classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    MODEL = model
    META = {"in_features": in_features, "classes": classes}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    assert MODEL is not None, "Model not loaded"
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
