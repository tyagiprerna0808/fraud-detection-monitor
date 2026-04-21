from pathlib import Path
import sys
import argparse
import json

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fraud_detection_monitor.data import generate_transactions
from fraud_detection_monitor.model import build_fraud_artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and benchmark fraud models")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train_config.json",
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Write leaderboard metrics to artifacts/train_metrics.json",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> dict[str, int]:
    default_config = {"rows": 9000, "seed": 42, "random_state": 42}
    if not config_path.exists():
        return default_config
    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "rows": int(loaded.get("rows", default_config["rows"])),
        "seed": int(loaded.get("seed", default_config["seed"])),
        "random_state": int(loaded.get("random_state", default_config["random_state"])),
    }


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    dataset = generate_transactions(rows=config["rows"], seed=config["seed"])
    artifacts = build_fraud_artifacts(dataset, random_state=config["random_state"])
    print("Fraud Detection Monitor leaderboard")
    print(artifacts["leaderboard"].to_string(index=False))

    if args.save_artifacts:
        output_dir = ROOT / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": config,
            "dataset_rows": int(len(dataset)),
            "fraud_rate": float(dataset["is_fraud"].mean()),
            "leaderboard": artifacts["leaderboard"].round(6).to_dict(orient="records"),
        }
        output_file = output_dir / "train_metrics.json"
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()