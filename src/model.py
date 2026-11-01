from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import CLASSES, CLASS_TO_ID, ID_TO_CLASS


@dataclass
class Prediction:
    risk_level: str
    risk_score: float
    confidence: float
    predicted_class_probability: float
    probabilities: dict[str, float]


class LogRiskModel:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", random_state=42):
        self.embedding_model_name = embedding_model
        self.random_state = random_state
        self.encoder = DictVectorizer()
        self.embedding_model = SentenceTransformer(embedding_model)
        self.classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)

    @staticmethod
    def _text(row):
        parts = [row.get("message", ""), row.get("exception_type", "")]
        return " ".join(part for part in parts if part).strip() or "(empty message)"

    @staticmethod
    def _structured(row):
        status = row.get("status_code", "").strip()
        value = int(status) if status.isdigit() else -1
        return {"service": row.get("service", "").lower(), "environment": row.get("environment", "").lower(), "log_level": row.get("log_level", "").upper(), "dependency": row.get("dependency", "").lower() or "(none)", "status_class": str(value // 100) if value >= 100 else "missing", "is_4xx": str(value // 100 == 4), "is_5xx": str(value // 100 == 5), "is_2xx": str(value // 100 == 2)}

    def _features(self, rows, fit=False):
        embeddings = self.embedding_model.encode([self._text(row) for row in rows], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        structured = self.encoder.fit_transform([self._structured(row) for row in rows]) if fit else self.encoder.transform([self._structured(row) for row in rows])
        return hstack([csr_matrix(embeddings), structured], format="csr")

    def fit(self, rows):
        self.classifier.fit(self._features(rows, fit=True), [CLASS_TO_ID[row["risk_level"]] for row in rows])
        return self

    def predict(self, rows):
        probabilities = self.classifier.predict_proba(self._features(rows))
        full_classes = [int(value) for value in self.classifier.classes_]
        predictions = []
        for probability in probabilities:
            full = np.zeros(len(CLASSES), dtype=float)
            for index, class_id in enumerate(full_classes):
                full[class_id] = probability[index]
            predicted_id = int(np.argmax(full))
            predictions.append(Prediction(ID_TO_CLASS[predicted_id], float(np.dot(full, [0, 1, 2, 3]) / 3), float(full.max()), float(full[predicted_id]), {name: float(full[index]) for index, name in enumerate(CLASSES)}))
        return predictions

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"embedding_model_name": self.embedding_model_name, "random_state": self.random_state, "encoder": self.encoder, "classifier": self.classifier}, path)

    @classmethod
    def load(cls, path):
        saved = joblib.load(path)
        model = cls(saved["embedding_model_name"], saved["random_state"])
        model.encoder = saved["encoder"]
        model.classifier = saved["classifier"]
        return model
