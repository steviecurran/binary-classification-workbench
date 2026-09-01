"""
Generates a fully synthetic dataset for money-laundering-style anomaly
detection in an online gambling context.

This is NOT derived from any real company's data, assignment, or feature
schema — the feature set below was designed from scratch to be
representative of the general problem domain (account behaviour, deposit/
withdrawal patterns, device/IP signals, structuring indicators) without
mirroring any specific employer's or interview assignment's exact fields.

Safe for public portfolio use.

Usage:
    python generate_dataset.py
Produces:
    synthetic_gambling_aml.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 25000
POSITIVE_RATE = 0.015  # ~1.5% flagged accounts, realistic for AML alerting


def generate():
    n = N

    # --- Baseline account attributes (mostly independent of label) ---
    account_tenure_days = RNG.exponential(scale=400, size=n).round().astype(int)
    account_tenure_days = np.clip(account_tenure_days, 1, 3650)

    customer_age = RNG.normal(loc=38, scale=12, size=n).round().astype(int)
    customer_age = np.clip(customer_age, 18, 85)

    kyc_verified = RNG.binomial(1, 0.9, size=n)
    is_vip_tier = RNG.binomial(1, 0.05, size=n)

    num_deposit_methods_used = RNG.poisson(1.3, size=n) + 1
    num_withdrawal_methods_used = RNG.poisson(1.1, size=n) + 1

    deposit_count_30d = RNG.poisson(8, size=n)
    withdrawal_count_30d = RNG.poisson(4, size=n)

    total_deposit_amount_30d = RNG.lognormal(mean=5.5, sigma=1.0, size=n)
    total_withdrawal_amount_30d = total_deposit_amount_30d * RNG.beta(2, 3, size=n)

    avg_bet_size = RNG.lognormal(mean=3.0, sigma=0.8, size=n)

    num_devices_used_30d = RNG.poisson(1.4, size=n) + 1
    num_distinct_ip_countries_30d = RNG.poisson(0.3, size=n) + 1

    pct_deposits_near_reporting_threshold = RNG.beta(1.5, 8, size=n)
    median_hours_deposit_to_withdrawal = RNG.gamma(shape=3, scale=15, size=n)

    cross_border_payment_flag = RNG.binomial(1, 0.08, size=n)
    night_time_transaction_pct = RNG.beta(2, 6, size=n)

    # --- Latent suspicion score: weighted combination of "risk" features ---
    # (Coefficients chosen to create realistic, imperfect separability —
    # not a clean linear split, so the classification task is non-trivial.)
    z = (
        1.8 * (num_devices_used_30d > 3)
        + 1.6 * (num_distinct_ip_countries_30d > 2)
        + 2.0 * (pct_deposits_near_reporting_threshold > 0.5)
        + 1.7 * (median_hours_deposit_to_withdrawal < 6)
        + 1.3 * cross_border_payment_flag
        + 1.1 * (num_withdrawal_methods_used > 3)
        + 0.9 * (night_time_transaction_pct > 0.5)
        + 0.8 * (kyc_verified == 0)
        + 0.6 * (account_tenure_days < 30)
        - 0.5 * is_vip_tier  # VIPs slightly less likely to be flagged (more scrutiny already)
        + RNG.normal(0, 1.4, size=n)  # noise, keeps the problem realistically hard
    )

    # Pick the threshold on z that yields approximately the target positive rate
    cutoff = np.quantile(z, 1 - POSITIVE_RATE)
    is_suspicious_activity = (z >= cutoff).astype(int)

    df = pd.DataFrame({
        "account_tenure_days": account_tenure_days,
        "customer_age": customer_age,
        "kyc_verified": kyc_verified,
        "is_vip_tier": is_vip_tier,
        "num_deposit_methods_used": num_deposit_methods_used,
        "num_withdrawal_methods_used": num_withdrawal_methods_used,
        "deposit_count_30d": deposit_count_30d,
        "withdrawal_count_30d": withdrawal_count_30d,
        "total_deposit_amount_30d": total_deposit_amount_30d.round(2),
        "total_withdrawal_amount_30d": total_withdrawal_amount_30d.round(2),
        "avg_bet_size": avg_bet_size.round(2),
        "num_devices_used_30d": num_devices_used_30d,
        "num_distinct_ip_countries_30d": num_distinct_ip_countries_30d,
        "pct_deposits_near_reporting_threshold": pct_deposits_near_reporting_threshold.round(3),
        "median_hours_deposit_to_withdrawal": median_hours_deposit_to_withdrawal.round(1),
        "cross_border_payment_flag": cross_border_payment_flag,
        "night_time_transaction_pct": night_time_transaction_pct.round(3),
        "is_suspicious_activity": is_suspicious_activity,
    })

    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("synthetic_gambling_aml.csv", index=False)
    print(f"Generated {len(df):,} rows")
    print(f"Positive rate: {df['is_suspicious_activity'].mean():.3%}")
    print("Written to synthetic_gambling_aml.csv")
    print(df.head())
