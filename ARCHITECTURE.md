# Fraud Detection Inference API - Architecture Note

## What this API does
Exposes a trained fraud detection model as a REST API with three endpoints:
- GET /health — liveness check for monitoring
- GET /version — app and model version for traceability
- POST /predict — accepts transaction features and returns fraud prediction with confidence score

---

## Why this API design

### FastAPI + Pydantic schema validation
Every request to /predict is validated against a strict schema before reaching model code.
Wrong types, missing fields, or unknown extra fields are rejected immediately with a clear error message.
This means the model never receives garbage input silently.
In interviews: "Validation errors are caught at the API boundary, not inside model logic."

### Three endpoints (not one)
- Health is separate so monitoring tools can check uptime without touching model code.
- Version is separate so you can trace which model artifact is running in production at any time.
- Predict is the only endpoint that touches the model.

---

## How model loading is handled
The model is trained and cached in memory once at server startup using Python's lru_cache.
Every subsequent request reuses the same in-memory object.
Loading is never triggered per request.

Why this matters:
- No disk I/O on each prediction request.
- Latency stays stable across requests.
- If loading fails at startup, server fails fast before accepting any traffic.

---

## How latency and failures are tracked

### Latency
Each /predict call measures wall-clock time using time.perf_counter() before and after inference.
Result is logged in milliseconds as a structured log line:
prediction=fraud | confidence=0.8734 | latency=14.2ms

### Failures
- Client errors (bad payload) return 422 with field-level details.
- Server errors return 500 with a safe generic message. Real error is logged internally.
- This prevents leaking stack traces to clients while keeping full detail for debugging.

---

## Main trade-offs

| Decision | Benefit | Cost |
|---|---|---|
| In-memory model | Low latency, simple | Memory grows with model size |
| Strict schema | Safe, debuggable | Less flexible for evolving inputs |
| Train at startup | No saved artifact needed | Startup is slower |
| Sync inference | Simple code | Blocks under high concurrency |

---

## Failure modes

1. Model training fails at startup — server refuses to start, caught immediately.
2. Feature mismatch — input has correct types but wrong feature distribution, model predicts silently with poor accuracy. Mitigation: input range validation on each field.
3. Memory pressure — large model + high traffic can exhaust container memory. Mitigation: move to dedicated model server like TorchServe or BentoML.
4. No request logging persistence — current logs go to stdout only, lost on restart. Mitigation: add log sink like CloudWatch or Datadog.

---

## What I would do for production scale

1. Save trained model to disk as .joblib and load artifact file at startup instead of retraining.
2. Add request ID to each log line for tracing across distributed logs.
3. Add Prometheus metrics endpoint for latency histograms and error rates.
4. Move to async endpoint with background worker queue for high-concurrency prediction.
5. Add model monitoring: track prediction distribution over time to detect data drift.
6. Containerize with Docker, deploy on Kubernetes with horizontal pod autoscaling.

---

## 60-second verbal explanation

I built a FastAPI inference service with three endpoints: health, version, and predict.
Health checks uptime, version gives model traceability, predict runs real fraud inference.
I use strict Pydantic schemas so bad payloads are rejected at the API boundary before touching model code.
The model is loaded once at startup using lru_cache and kept in memory, so latency stays stable.
Each predict request preprocesses input into model features, runs inference, and returns prediction plus confidence score.
I log structured timing per request so latency and failures are observable.
Main trade-off is in-memory model keeps things simple but limits horizontal scaling for very large models.