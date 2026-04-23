from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from typing import Literal

app = FastAPI(title="Fraud Detection Inference API", version="0.1.0")


# Input schema for /predict
class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject unknown extra fields

    transaction_amount: float
    transaction_type: Literal["payment", "transfer", "cash_out", "debit"]
    account_age_days: int
    is_foreign_transaction: bool


# Output schema for /predict
class PredictResponse(BaseModel):
    prediction: Literal["fraud", "not_fraud"]
    confidence: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {
        "app_version": "0.1.0",
        "model_version": "baseline-v1"
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    # Placeholder logic for Loop 1 (real model comes in Loop 2)
    if payload.transaction_amount > 1000 and payload.is_foreign_transaction:
        pred = "fraud"
        conf = 0.91
    else:
        pred = "not_fraud"
        conf = 0.13

    return PredictResponse(
        prediction=pred,
        confidence=conf,
        model_version="baseline-v1"
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Friendly validation error format
    return JSONResponse(
        status_code=422,
        content={
            "message": "Invalid request payload",
            "errors": exc.errors()
        },
    )