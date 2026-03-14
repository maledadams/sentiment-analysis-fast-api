from typing import Literal

from pydantic import BaseModel, Field


class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Text to analyze for sentiment.",
        examples=["I love building APIs."],
    )


class PredictionResponse(BaseModel):
    text: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(..., ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: Literal["ok", "loading"]
    model_loaded: bool
    model_name: str
