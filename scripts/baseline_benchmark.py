from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_detection_monitor.data import generate_transactions
from fraud_detection_monitor.model import build_fraud_artifacts


def main() -> None:
    dataset = generate_transactions(rows=9000, seed=42)
    artifacts = build_fraud_artifacts(dataset, random_state=42)

    leaderboard = artifacts["leaderboard"].copy()
    leaderboard_records = leaderboard.round(4).to_dict(orient="records")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "rows": int(len(dataset)),
            "fraud_rate": round(float(dataset["is_fraud"].mean()), 6),
        },
        "best_model": str(leaderboard.iloc[0]["model"]),
        "leaderboard": leaderboard_records,
    }

    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_json = artifacts_dir / "baseline_metrics.json"
    output_md = artifacts_dir / "baseline_metrics.md"

    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Baseline Benchmark")
    lines.append("")
    lines.append(f"Generated at (UTC): {summary['generated_at_utc']}")
    lines.append("")
    lines.append(f"Dataset rows: {summary['dataset']['rows']}")
    lines.append(f"Fraud rate: {summary['dataset']['fraud_rate']:.4f}")
    lines.append(f"Best model: {summary['best_model']}")
    lines.append("")
    lines.append("## Leaderboard")
    lines.append("")
    lines.append("| Model | AUC | Precision | Recall | F1 | Accuracy |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for row in summary["leaderboard"]:
        lines.append(
            f"| {row['model']} | {row['auc']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['accuracy']:.4f} |"
        )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved {output_json}")
    print(f"Saved {output_md}")


if __name__ == "__main__":
    main()
