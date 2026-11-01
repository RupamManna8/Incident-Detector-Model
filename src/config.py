from pathlib import Path

CLASSES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
ID_TO_CLASS = {index: name for name, index in CLASS_TO_ID.items()}
REQUIRED_COLUMNS = ("service", "environment", "log_level", "message", "status_code", "dependency", "risk_level")
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = BASE_DIR / "artifacts"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACTS / "model.joblib"
DEFAULT_RANDOM_STATE = 42
