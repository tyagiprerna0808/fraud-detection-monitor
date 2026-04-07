import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "amount",
    "hour",
    "merchant_risk",
    "velocity_1h",
    "device_trust",
    "cross_border",
    "new_payee",
    "failed_logins",
    "weekend",
]


def generate_transactions(rows: int = 9000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = rng.integers(0, 24, size=rows)
    amount = rng.lognormal(mean=4.1, sigma=0.9, size=rows)
    merchant_risk = rng.integers(1, 6, size=rows)
    velocity_1h = rng.poisson(1.4, size=rows)
    device_trust = rng.uniform(0.05, 1.0, size=rows)
    cross_border = rng.binomial(1, 0.14, size=rows)
    new_payee = rng.binomial(1, 0.17, size=rows)
    failed_logins = np.clip(rng.poisson(0.4, size=rows), 0, 4)
    weekend = rng.binomial(1, 0.28, size=rows)

    night = ((hours <= 5) | (hours >= 23)).astype(int)
    score = (
        -5.9
        + 0.018 * amount
        + 0.52 * merchant_risk
        + 0.35 * velocity_1h
        - 2.25 * device_trust
        + 1.1 * cross_border
        + 0.95 * new_payee
        + 0.7 * failed_logins
        + 0.25 * weekend
        + 0.3 * night
    )
    probability = 1 / (1 + np.exp(-score))
    is_fraud = rng.binomial(1, probability)

    frame = pd.DataFrame(
        {
            "amount": amount.round(2),
            "hour": hours,
            "merchant_risk": merchant_risk,
            "velocity_1h": velocity_1h,
            "device_trust": device_trust.round(3),
            "cross_border": cross_border,
            "new_payee": new_payee,
            "failed_logins": failed_logins,
            "weekend": weekend,
            "is_fraud": is_fraud,
        }
    )
    return frame