from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURE_COLUMNS


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    metrics: dict[str, float]


def _candidate_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        max_depth=8,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                )
            ]
        ),
    }


def _metric_row(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> TrainedModel:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "auc": roc_auc_score(y_test, probabilities),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "accuracy": accuracy_score(y_test, predictions),
    }
    return TrainedModel(name=name, pipeline=model, metrics=metrics)


def _extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    else:
        importance = abs(estimator.coef_[0])
    return pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importance}).sort_values(
        "importance", ascending=False
    )


def build_fraud_artifacts(dataset: pd.DataFrame, random_state: int = 42) -> dict[str, object]:
    x = dataset[FEATURE_COLUMNS]
    y = dataset["is_fraud"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=random_state,
    )

    trained_models: list[TrainedModel] = []
    for name, pipeline in _candidate_models(random_state).items():
        pipeline.fit(x_train, y_train)
        trained_models.append(_metric_row(name, pipeline, x_test, y_test))

    leaderboard = pd.DataFrame(
        [{"model": item.name, **item.metrics} for item in trained_models]
    ).sort_values(["auc", "recall"], ascending=False, ignore_index=True)

    best_name = leaderboard.iloc[0]["model"]
    best_model = next(item.pipeline for item in trained_models if item.name == best_name)
    probabilities = best_model.predict_proba(x_test)[:, 1]

    test_predictions = pd.DataFrame({"fraud_probability": probabilities, "is_fraud": y_test.reset_index(drop=True)})
    test_predictions["label"] = test_predictions["is_fraud"].map({0: "Legit", 1: "Fraud"})

    return {
        "best_model": best_model,
        "leaderboard": leaderboard,
        "feature_importance": _extract_feature_importance(best_model),
        "test_predictions": test_predictions,
    }


def score_transaction(model: Pipeline, transaction_input: dict[str, float]) -> float:
    frame = pd.DataFrame([{feature: transaction_input[feature] for feature in FEATURE_COLUMNS}])
    return float(model.predict_proba(frame)[0, 1])