import time
import logging
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fraud_detection_monitor.data import generate_transactions
from fraud_detection_monitor.model import build_fraud_artifacts, score_transaction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

API_VERSION = "0.1.0"
app = FastAPI(title="Fraud Detection Inference API", version=API_VERSION)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: float = Field(gt=0)
    hour: int = Field(ge=0, le=23)
    merchant_risk: int = Field(ge=1, le=5)
    velocity_1h: int = Field(ge=0)
    device_trust: float = Field(ge=0.0, le=1.0)
    cross_border: int = Field(ge=0, le=1)
    new_payee: int = Field(ge=0, le=1)
    failed_logins: int = Field(ge=0)
    weekend: int = Field(ge=0, le=1)


class PredictResponse(BaseModel):
    prediction: Literal["fraud", "not_fraud"]
    confidence: float
    model_version: str


@lru_cache(maxsize=1)
def get_artifacts() -> dict[str, Any]:
    logger.info("Loading model artifacts at startup...")
    dataset = generate_transactions()
    artifacts = build_fraud_artifacts(dataset)
    model_name = str(artifacts["leaderboard"].iloc[0]["model"])
    logger.info(f"Model ready: {model_name}")
    return artifacts


@app.on_event("startup")
def load_model_on_startup() -> None:
    get_artifacts()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = []
    for error in exc.errors():
        location = [str(item) for item in error.get("loc", []) if item != "body"]
        details.append({
            "field": ".".join(location) if location else "body",
            "message": error.get("msg", "Invalid value"),
        })
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "message": "Invalid request payload. Fix the fields listed in details.",
            "details": details,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "Something went wrong. Check server logs.",
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    artifacts = get_artifacts()
    model_name = str(artifacts["leaderboard"].iloc[0]["model"])
    return {"app_version": API_VERSION, "model_version": model_name}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    start_time = time.perf_counter()

    artifacts = get_artifacts()
    model = artifacts["best_model"]
    model_name = str(artifacts["leaderboard"].iloc[0]["model"])

    transaction_input = payload.model_dump()
    fraud_probability = score_transaction(model, transaction_input)

    is_fraud = fraud_probability >= 0.5
    prediction = "fraud" if is_fraud else "not_fraud"

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(f"prediction={prediction} | confidence={round(fraud_probability, 4)} | latency={latency_ms}ms")

    return PredictResponse(
        prediction=prediction,
        confidence=round(fraud_probability, 6),
        model_version=model_name,
    )