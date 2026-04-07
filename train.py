from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fraud_detection_monitor.data import generate_transactions
from fraud_detection_monitor.model import build_fraud_artifacts


def main() -> None:
    dataset = generate_transactions()
    artifacts = build_fraud_artifacts(dataset)
    print("Fraud Detection Monitor leaderboard")
    print(artifacts["leaderboard"].to_string(index=False))


if __name__ == "__main__":
    main()