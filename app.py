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

API_VERSION = "0.1.0"


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["ok"]


class VersionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    service: str
    version: str


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    amount: float = Field(gt=0)
    hour: int = Field(ge=0, le=23)
    merchant_risk: int = Field(ge=1, le=5)
    velocity_1h: int = Field(ge=0)
    device_trust: float = Field(ge=0, le=1)
    cross_border: int = Field(ge=0, le=1)
    new_payee: int = Field(ge=0, le=1)
    failed_logins: int = Field(ge=0)
    weekend: int = Field(ge=0, le=1)


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    fraud_probability: float = Field(ge=0, le=1)
    is_fraud: bool
    risk_label: Literal["high_risk", "low_risk"]
    model_name: str
    version: str


app = FastAPI(title="Fraud Detection Monitor API", version=API_VERSION)


@lru_cache(maxsize=1)
def get_artifacts() -> dict[str, Any]:
    dataset = generate_transactions()
    return build_fraud_artifacts(dataset)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    details = []
    for error in exc.errors():
        location = [str(item) for item in error.get("loc", []) if item != "body"]
        details.append(
            {
                "field": ".".join(location) if location else "body",
                "message": error.get("msg", "Invalid value"),
            }
        )
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "message": "Invalid request payload. Fix the fields listed in 'details' and try again.",
            "details": details,
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "Something went wrong while processing the request.",
            "path": str(request.url.path),
        },
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/health/ready", response_model=HealthResponse)
def health_ready() -> HealthResponse:
    # Ready endpoint ensures the model artifacts can be loaded.
    get_artifacts()
    return HealthResponse(status="ok")


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(service="fraud-detection-monitor", version=API_VERSION)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    artifacts = get_artifacts()
    model = artifacts["best_model"]
    model_name = str(artifacts["leaderboard"].iloc[0]["model"])

    transaction_input = payload.model_dump()
    fraud_probability = score_transaction(model, transaction_input)
    is_fraud = fraud_probability >= 0.5

    return PredictResponse(
        fraud_probability=round(fraud_probability, 6),
        is_fraud=is_fraud,
        risk_label="high_risk" if is_fraud else "low_risk",
        model_name=model_name,
        version=API_VERSION,
    )