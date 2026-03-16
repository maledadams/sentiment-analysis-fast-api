# sentiment-analysis-fast-api

A REST API for real-time sentiment classification built with FastAPI.

The API accepts text input and returns predicted sentiment labels and confidence scores returns `positive`, `negative`, or `neutral` with a confidence score..
The project demonstrates how machine learning models can be deployed as scalable APIs.

## Architecture

Client Request
      ↓
FastAPI Endpoint
      ↓
Sentiment Model
      ↓
Prediction Output
      ↓
JSON Response

## Stack

- FastAPI for the REST API
- Hugging Face `transformers` for inference
- PyTorch as the model backend
- Pydantic for request and response validation

## Model

The API loads `cardiffnlp/twitter-roberta-base-sentiment-latest` at startup. That model supports three sentiment classes, which makes it a better fit than the default binary sentiment pipeline.

You can override the model with the `SENTIMENT_MODEL_NAME` environment variable.

## Setup

1. Move into the project folder:

```bash
cd sentiment-analysis-fast-api
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the API server:

```bash
python -m uvicorn app.main:app --reload
```

4. Open the API in your browser:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

The first startup can take longer because the model may need to be downloaded from Hugging Face.

## How to use the API

### Option 1: Use the Swagger UI

Open:

```text
http://127.0.0.1:8000/docs
```

Then:

1. Expand `POST /predict`
2. Click `Try it out`
3. Paste a request body like this:

```json
{
  "text": "I love this movie so much"
}
```

4. Click `Execute`

### Option 2: Use PowerShell

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -Body '{"text":"This project is amazing"}'
```

### Option 3: Use curl

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"This project is amazing\"}"
```

## Tests

Run the API tests with:

```bash
python -m pytest -q
```

Expected result:

```text
5 passed
```

## Endpoints

### `GET /`

Returns a basic service message.

Example:

```json
{
  "message": "Sentiment Analysis API is running"
}
```

### `GET /health`

Returns the service health and whether the model is loaded.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}
```

### `POST /predict`

Request body:

```json
{
  "text": "This movie was surprisingly good"
}
```

Example response:

```json
{
  "text": "This movie was surprisingly good",
  "sentiment": "positive",
  "confidence": 0.997
}
```

Another example:

Request:

```json
{
  "text": "I hate this class"
}
```

Possible response:

```json
{
  "text": "I hate this class",
  "sentiment": "negative",
  "confidence": 0.98
}
```

## Notes

- The model is loaded once during application startup and reused for all requests.
- The first startup may take longer because the model has to be downloaded from Hugging Face.
