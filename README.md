# Fraud Detection Monitor

Fraud Detection Monitor is a Python and Streamlit fintech project for transaction risk operations. It simulates a realistic card and wallet transaction stream, benchmarks fraud classifiers, and lets a reviewer score a transaction in an analyst-style dashboard.

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
- Streamlit
- Plotly

## Run locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

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