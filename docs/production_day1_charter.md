# Production ML Service Day 1 Charter

## Objective
Convert Fraud Detection Monitor from a demo-first app into a production-ready ML service track with measurable quality, latency, and reliability outcomes.

## Scope for Week 1
- Define baseline model and evaluation metrics.
- Package reproducible benchmark output for tracking.
- Prepare deployment-oriented backlog for API, containerization, CI, and monitoring.

## Non-negotiable weekly output
- 1 shipped artifact
- 1 measurable metric update
- 1 interview-ready engineering story

## Baseline definitions
- Dataset source: synthetic transaction generator in `src/fraud_detection_monitor/data.py`.
- Candidate models: Logistic Regression and Random Forest in `src/fraud_detection_monitor/model.py`.
- Selection rule: best model by AUC, then recall.

## Day 1 deliverables
- Project charter (this document).
- Reproducible baseline benchmark script.
- Baseline metrics artifact stored under `artifacts/`.

## Metrics tracked from Day 1
- Model quality: AUC, precision, recall, F1, accuracy.
- Data profile: transaction count, fraud rate.
- Service metric placeholder for Day 3: P95 inference latency.

## Risks and controls
- Risk: synthetic data can overstate real-world stability.
- Control: treat values as engineering benchmark only, not business production calibration.

- Risk: metric drift from non-reproducible seeds.
- Control: fixed random seed and deterministic baseline script output.

## Next milestones
- Day 2: data and training job hardening.
- Day 3: inference API with request validation and error handling.
- Day 4: Docker and CI checks.
- Day 5: monitoring and health probes.
- Day 6: deployment and rollback note.
- Day 7: architecture summary and interview narrative.
