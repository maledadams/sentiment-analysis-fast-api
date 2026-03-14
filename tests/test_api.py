from fastapi.testclient import TestClient

from app.main import app


def test_root_returns_service_message(monkeypatch):
    monkeypatch.setattr("app.main.load_model", lambda: None)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Sentiment Analysis API is running"}


def test_health_reports_model_status(monkeypatch):
    monkeypatch.setattr("app.main.load_model", lambda: None)
    monkeypatch.setattr(
        "app.main.get_model_status",
        lambda: {
            "status": "ok",
            "model_loaded": True,
            "model_name": "test-model",
        },
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "test-model",
    }


def test_predict_returns_sentiment_prediction(monkeypatch):
    monkeypatch.setattr("app.main.load_model", lambda: None)
    monkeypatch.setattr(
        "app.main.predict_sentiment",
        lambda text: {
            "text": text,
            "sentiment": "positive",
            "confidence": 0.99,
        },
    )

    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "This project is amazing"})

    assert response.status_code == 200
    assert response.json() == {
        "text": "This project is amazing",
        "sentiment": "positive",
        "confidence": 0.99,
    }


def test_predict_rejects_empty_text(monkeypatch):
    monkeypatch.setattr("app.main.load_model", lambda: None)

    with TestClient(app) as client:
        response = client.post("/predict", json={"text": ""})

    assert response.status_code == 422


def test_predict_returns_503_when_model_is_unavailable(monkeypatch):
    monkeypatch.setattr("app.main.load_model", lambda: None)

    def raise_not_loaded(_: str):
        raise RuntimeError("Sentiment model has not been loaded.")

    monkeypatch.setattr("app.main.predict_sentiment", raise_not_loaded)

    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "test"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Sentiment model has not been loaded."}
