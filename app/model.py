from __future__ import annotations

import os
from threading import Lock
from typing import Any

from transformers import pipeline

DEFAULT_MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL_NAME",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
)

_LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
    "POSITIVE": "positive",
}

_classifier = None
_lock = Lock()
_loaded_model_name = DEFAULT_MODEL_NAME


def load_model(model_name: str | None = None) -> None:
    global _classifier, _loaded_model_name

    if _classifier is not None:
        return

    chosen_model = model_name or DEFAULT_MODEL_NAME

    with _lock:
        if _classifier is None:
            _classifier = pipeline("text-classification", model=chosen_model)
            _loaded_model_name = chosen_model


def predict_sentiment(text: str) -> dict[str, Any]:
    if _classifier is None:
        raise RuntimeError("Sentiment model has not been loaded.")

    result = _classifier(text, truncation=True)[0]
    label = _normalize_label(str(result["label"]))

    return {
        "text": text,
        "sentiment": label,
        "confidence": float(result["score"]),
    }


def get_model_status() -> dict[str, Any]:
    return {
        "status": "ok" if _classifier is not None else "loading",
        "model_loaded": _classifier is not None,
        "model_name": _loaded_model_name,
    }


def _normalize_label(label: str) -> str:
    normalized = _LABEL_MAP.get(label.upper())
    if normalized is None:
        raise ValueError(f"Unexpected label returned by the model: {label}")
    return normalized
