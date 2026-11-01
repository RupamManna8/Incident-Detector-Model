import json
from pathlib import Path

import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from src.config import CLASSES, CLASS_TO_ID
from src.data.loader import load_records
from src.data.splitter import split_records
from src.data.validator import validate_records
from src.model import LogRiskModel


def metrics(actual, predicted):
    actual_ids = [CLASS_TO_ID[x] for x in actual]
    predicted_ids = [CLASS_TO_ID[x] for x in predicted]
    return {"accuracy": float(accuracy_score(actual_ids, predicted_ids)), "macro_precision": float(precision_score(actual_ids, predicted_ids, labels=range(4), average="macro", zero_division=0)), "macro_recall": float(recall_score(actual_ids, predicted_ids, labels=range(4), average="macro", zero_division=0)), "macro_f1": float(f1_score(actual_ids, predicted_ids, labels=range(4), average="macro", zero_division=0)), "weighted_f1": float(f1_score(actual_ids, predicted_ids, average="weighted", zero_division=0)), "ordinal_mae": float(np.mean(np.abs(np.array(actual_ids) - np.array(predicted_ids)))), "within_one_level_accuracy": float(np.mean(np.abs(np.array(actual_ids) - np.array(predicted_ids)) <= 1)), "confusion_matrix": confusion_matrix(actual_ids, predicted_ids, labels=range(4)).tolist(), "per_class": classification_report(actual_ids, predicted_ids, labels=range(4), target_names=CLASSES, output_dict=True, zero_division=0)}


def train(data_path, artifacts="artifacts", random_state=42):
    records, load_stats = load_records(data_path)
    report = validate_records(records, load_stats)
    if not report.valid:
        raise ValueError("Dataset validation failed: " + "; ".join(report.errors))
    records = [row for row in records if "_invalid" not in row]
    train_rows, validation_rows, test_rows = split_records(records, random_state)
    model = LogRiskModel(random_state=random_state).fit(train_rows)
    predictions = model.predict(test_rows)
    result = metrics([row["risk_level"] for row in test_rows], [item.risk_level for item in predictions])
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    model.save(artifacts / "model.joblib")
    baseline = max((row["risk_level"] for row in train_rows), key=[row["risk_level"] for row in train_rows].count)
    baseline_result = metrics([row["risk_level"] for row in test_rows], [baseline] * len(test_rows))
    result["majority_baseline"] = baseline_result
    (artifacts / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plt.figure(figsize=(6, 5)); plt.imshow(result["confusion_matrix"], cmap="Blues"); plt.colorbar(); plt.xticks(range(4), CLASSES); plt.yticks(range(4), CLASSES); plt.xlabel("Predicted"); plt.ylabel("Actual")
    for row_index in range(4):
        for column_index in range(4):
            plt.text(column_index, row_index, result["confusion_matrix"][row_index][column_index], ha="center", va="center")
    plt.tight_layout(); plt.savefig(artifacts / "confusion_matrix.png", dpi=150); plt.close()
    with (artifacts / "errors.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["actual", "predicted", "confidence", "message", "service", "environment", "log_level", "status_code", "dependency"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        errors = [(row, prediction) for row, prediction in zip(test_rows, predictions) if row["risk_level"] != prediction.risk_level]
        for row, prediction in sorted(errors, key=lambda pair: pair[1].confidence, reverse=True):
            writer.writerow({"actual": row["risk_level"], "predicted": prediction.risk_level, "confidence": prediction.confidence, **{field: row.get(field, "") for field in fields[3:]}})
    metadata = {"model_name": "log-risk-classifier", "model_version": "0.1.0", "embedding_model": "all-MiniLM-L6-v2", "classifier": "LogisticRegression", "classes": list(CLASSES), "training_samples": len(train_rows), "validation_samples": len(validation_rows), "test_samples": len(test_rows), **{key: result[key] for key in ("macro_f1", "accuracy", "ordinal_mae")}}
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return result, report, load_stats
