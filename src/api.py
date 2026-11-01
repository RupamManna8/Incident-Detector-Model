import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Allow `python src/api.py` to resolve this repository's `src` package before
# same-named packages that may exist elsewhere on the machine.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import CLASSES, DEFAULT_MODEL_PATH
from src.inference.predictor import load_model, predict_log


class LogRequest(BaseModel):
    service: str = Field(min_length=1)
    environment: str = Field(min_length=1)
    log_level: str = Field(min_length=1)
    message: str = Field(min_length=1)
    status_code: int | None = Field(default=None, ge=100, le=599)
    dependency: str | None = None


class PredictionResponse(BaseModel):
    risk_level: str
    risk_score: float
    confidence: float
    predicted_class_probability: float
    probabilities: dict[str, float]


def model_path() -> str:
    return os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))


@asynccontextmanager
async def lifespan(_: FastAPI):
    if Path(model_path()).exists():
        load_model(model_path())
    yield


app = FastAPI(title="Log Risk Detector API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    path = Path(model_path())
    return {"status": "ok" if path.exists() else "degraded", "model_loaded": path.exists(), "model_path": str(path), "classes": list(CLASSES)}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: LogRequest):
    if not Path(model_path()).exists():
        raise HTTPException(status_code=503, detail="Model artifact not found. Train the model first.")
    prediction = predict_log(**request.model_dump(), model_path=model_path())
    return prediction.__dict__


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")))
