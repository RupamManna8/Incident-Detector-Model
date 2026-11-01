# Log Risk Detector

Offline risk classification for individual application log records. The model uses the frozen `sentence-transformers/all-MiniLM-L6-v2` encoder for message and exception semantics, concatenated with one-hot structured fields and status-code indicators, then trains multinomial Logistic Regression. Labels are ordinal and fixed as `LOW < MEDIUM < HIGH < CRITICAL`.

## Setup and commands

```bash
pip install -r requirements.txt
python -m src.cli validate --data dataset.csv
python -m src.cli train --data dataset.csv
python -m src.cli evaluate
python -m src.cli predict --service payment-service --environment production --log-level ERROR --message "Payment provider connection timed out" --status-code 503 --dependency stripe
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## HTTP API

Start the service after training with `uvicorn src.api:app --host 0.0.0.0 --port 8000`. `GET /health` reports model availability. `POST /predict` accepts JSON with `service`, `environment`, `log_level`, `message`, optional `status_code`, and optional `dependency`, and returns the same typed prediction contract as the Python API. Set `MODEL_PATH` to serve a different persisted model artifact.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"service\":\"payment-service\",\"environment\":\"production\",\"log_level\":\"ERROR\",\"message\":\"Payment provider connection timed out\",\"status_code\":503,\"dependency\":\"stripe\"}"
```

The first embedding run downloads `all-MiniLM-L6-v2` from Hugging Face. The resulting classifier and structured encoder are saved in `artifacts/model.joblib`; metrics, `artifacts/confusion_matrix.png`, and confidence-ranked `artifacts/errors.csv` are also produced. `data/unseen_logs.csv` and `data/adversarial_logs.csv` are held out fixtures for manual inference checks.

`risk_score` is expected ordinal risk: `(0*P(LOW) + 1*P(MEDIUM) + 2*P(HIGH) + 3*P(CRITICAL))/3`. `confidence` is the maximum class probability and is not claimed to be calibrated. The train-only structured encoder prevents preprocessing leakage; the pretrained embedding model is fixed and never fit on labels.

The supplied CSV has an optional `exception_type` column. 232 rows omit it; the loader repairs this known seven-field shape and reports the count. Unknown malformed rows fail validation. Duplicate rows, nulls, class counts, and invalid statuses are reported.

This predicts the risk of one log with its available fields. It does not understand history, frequency, blast radius, affected users, time series, or cross-service correlation, so it is not yet an incident-level risk engine. The synthetic dataset also contains shortcut risk: `log_level`, status, and repeated templates may dominate generalization. The next improvement should be a grouped/chronological split over independently authored production-like incidents, followed by probability calibration and a leakage ablation without `log_level`.
