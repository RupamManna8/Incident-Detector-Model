import os
from functools import lru_cache

from src.config import DEFAULT_MODEL_PATH
from src.model import LogRiskModel, Prediction


@lru_cache(maxsize=4)
def load_model(model_path: str = str(DEFAULT_MODEL_PATH)) -> LogRiskModel:
    return LogRiskModel.load(model_path)


def predict_log(service: str, environment: str, log_level: str, message: str, status_code: int | None, dependency: str | None, model_path=None) -> Prediction:
    model_path = model_path or os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    row = {"service": service, "environment": environment, "log_level": log_level, "message": message, "status_code": "" if status_code is None else str(status_code), "dependency": dependency or "", "exception_type": ""}
    return load_model(str(model_path)).predict([row])[0]
