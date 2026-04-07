from pathlib import Path
import sys

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fraud_detection_monitor.data import FEATURE_COLUMNS, generate_transactions
from fraud_detection_monitor.model import build_fraud_artifacts, score_transaction


st.set_page_config(page_title="Fraud Detection Monitor", page_icon="🛡️", layout="wide")


@st.cache_data
def get_dataset():
    return generate_transactions()


@st.cache_resource
def get_artifacts():
    dataset = get_dataset()
    return build_fraud_artifacts(dataset)


dataset = get_dataset()
artifacts = get_artifacts()
leaderboard = artifacts["leaderboard"]
best_row = leaderboard.iloc[0]

st.title("Fraud Detection Monitor")
st.caption("Transaction fraud analytics for fintech payments operations.")

row = st.columns(4)
row[0].metric("Transactions", f"{len(dataset):,}")
row[1].metric("Fraud rate", f"{dataset['is_fraud'].mean():.1%}")
row[2].metric("Best model", best_row["model"])
row[3].metric("Best recall", f"{best_row['recall']:.2f}")

left, right = st.columns((1.15, 1))

with left:
    st.subheader("Model leaderboard")
    st.dataframe(
        leaderboard.style.format(
            {
                "auc": "{:.3f}",
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1": "{:.3f}",
                "accuracy": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Fraud probability distribution")
    probability_chart = px.histogram(
        artifacts["test_predictions"],
        x="fraud_probability",
        color="label",
        nbins=30,
        barmode="overlay",
        color_discrete_map={"Legit": "#0f7c82", "Fraud": "#c75c3f"},
    )
    probability_chart.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(probability_chart, use_container_width=True)

with right:
    st.subheader("Top risk drivers")
    importance_chart = px.bar(
        artifacts["feature_importance"].head(10).sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Sunset",
    )
    importance_chart.update_layout(margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
    st.plotly_chart(importance_chart, use_container_width=True)

    st.subheader("Score a transaction")
    medians = dataset[FEATURE_COLUMNS].median().to_dict()
    transaction_input = {}
    with st.form("fraud-transaction-form"):
        for feature in FEATURE_COLUMNS:
            low = float(dataset[feature].min())
            high = float(dataset[feature].max())
            default = float(medians[feature])
            transaction_input[feature] = st.slider(feature.replace("_", " ").title(), low, high, default)
        submitted = st.form_submit_button("Estimate fraud risk")

    if submitted:
        probability = score_transaction(artifacts["best_model"], transaction_input)
        status = "High fraud risk" if probability >= 0.5 else "Likely legitimate"
        color = "#c75c3f" if probability >= 0.5 else "#0f7c82"
        st.markdown(
            f"<div style='padding: 1rem; border-radius: 14px; background: {color}; color: white;'>"
            f"<strong>{status}</strong><br/>Estimated fraud probability: {probability:.1%}"
            f"</div>",
            unsafe_allow_html=True,
        )