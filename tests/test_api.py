from fastapi.testclient import TestClient

from src.api import app


def test_health_reports_trained_model():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_risk_contract():
    response = TestClient(app).post("/predict", json={"service": "payment-service", "environment": "production", "log_level": "ERROR", "message": "Payment provider connection timed out", "status_code": 503, "dependency": "stripe"})
    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-6
