from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .model import get_model_status, load_model, predict_sentiment
from .schemas import HealthResponse, PredictionResponse, TextInput


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Sentiment Analysis API",
    version="1.0.0",
    description="Analyze text sentiment with a startup-loaded transformer model.",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Sentiment Analysis API is running"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(**get_model_status())


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TextInput) -> PredictionResponse:
    try:
        return PredictionResponse(**predict_sentiment(payload.text))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
