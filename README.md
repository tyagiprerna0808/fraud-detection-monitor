# Fraud Detection Monitor

Fraud Detection Monitor is a Python fintech project for transaction risk operations. It simulates a realistic card and wallet transaction stream, benchmarks fraud classifiers, and now exposes a simple prediction API.

## What it demonstrates

- Payments fraud analytics framing
- Model comparison with fraud-focused precision and recall tradeoffs
- Real-time transaction risk scoring
- Alert-rate and fraud distribution monitoring

## Stack

- Python
- pandas
- numpy
- scikit-learn
- FastAPI
- Uvicorn

## Run locally

```powershell
pip install -r requirements.txt
uvicorn app:app --reload
```

The API runs at `http://127.0.0.1:8000`.

## API endpoints

- `GET /health` returns service health.
- `GET /health/ready` verifies model artifacts are available.
- `GET /version` returns API service/version info.
- `POST /predict` scores one transaction.

## Docker run

```powershell
docker build -t fraud-detection-monitor:latest .
docker run --rm -e PORT=8000 -p 8000:8000 fraud-detection-monitor:latest
```

### 1) Health check

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"
```

Expected response:

```json
{
	"status": "ok"
}
```

### 2) Version check

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/version"
```

### 3) Predict with valid payload

```powershell
$body = @{
	amount = 1450.50
	hour = 22
	merchant_risk = 4
	velocity_1h = 3
	device_trust = 0.33
	cross_border = 1
	new_payee = 1
	failed_logins = 2
	weekend = 0
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body
```

### 4) Predict with invalid payload (validation error)

```powershell
$badBody = @{
	amount = -10
	hour = 40
	merchant_risk = 7
	velocity_1h = -2
	device_trust = 1.5
	cross_border = 2
	new_payee = 2
	failed_logins = -1
	weekend = 3
} | ConvertTo-Json

try {
	Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $badBody
} catch {
	$_.Exception.Response.GetResponseStream() | % {
		$reader = New-Object System.IO.StreamReader($_)
		$reader.ReadToEnd()
	}
}
```

Expected behavior: HTTP 422 with a clear JSON message and field-level validation details.

Run a config-driven training benchmark:

```powershell
python train.py --config configs/train_config.json --save-artifacts
```

This command writes `artifacts/train_metrics.json`.

## Day 1 production baseline

Generate reproducible baseline metrics for tracking the production migration plan.

```powershell
python scripts/baseline_benchmark.py
```

This command writes:

- `artifacts/baseline_metrics.json`
- `artifacts/baseline_metrics.md`

It also uses the Day 1 charter at `docs/production_day1_charter.md` as the execution reference.

## Business framing

This project is positioned like an internal fraud operations console for a payments or wallet product. It is designed to show practical fintech judgment rather than generic machine learning alone.