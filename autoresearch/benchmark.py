#!/usr/bin/env python3
"""
Fixed benchmark for the autoresearch loop.

Trains a Series A PKBoost model on a fixed temporal split of real data
and reports PR AUC. This is the single metric the loop optimizes.

Usage:
    python benchmark.py                     # baseline (focal_gamma=0)
    FOCAL_GAMMA=1.0 python benchmark.py     # test focal loss
    N_ESTIMATORS=100 python benchmark.py    # faster iteration

Returns a single JSON line to stdout:
    {"pr_auc": 0.0412, "roc_auc": 0.821, "train_time": 45.2, ...}
"""

import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

# Resolve paths
PKBOOST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(PKBOOST_DIR, "..", "svp-company-scoring-pipeline")
DATA_PATH = os.path.join(PIPELINE_DIR, "data", "features_full_4to7months.parquet")

sys.path.insert(0, PKBOOST_DIR)
sys.path.insert(0, os.path.join(PKBOOST_DIR, "pkboost_sklearn"))
sys.path.insert(0, PIPELINE_DIR)

from pkboost_sklearn.sklearn_interface import PKBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

# Fixed train/test split — never changes between experiments
# Train: Q1 2019 – Q4 2022 (16 quarters of history)
# Test:  Q1 2023 – Q4 2023 (held-out future)
TRAIN_START = "2019-01-01"
TRAIN_END = "2022-10-01"
TEST_START = "2023-01-01"
TEST_END = "2023-10-01"

# Hyperparams from production (fixed — only Rust core changes are tested)
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "200"))
FOCAL_GAMMA = float(os.environ.get("FOCAL_GAMMA", "0.0"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "8"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.05"))
REG_LAMBDA = float(os.environ.get("REG_LAMBDA", "1.0"))
SUBSAMPLE = float(os.environ.get("SUBSAMPLE", "0.8"))
COLSAMPLE = float(os.environ.get("COLSAMPLE", "0.8"))
GAMMA = float(os.environ.get("GAMMA", "0.1"))


def load_and_filter(path):
    """Load parquet and apply Series A cohort filter + feature engineering."""
    import polars as pl

    df = pl.read_parquet(path)

    # Series A entry: raised seed OR total_funding >= $1M; upper bound $7.5M
    entry = (pl.col("did_raise_seed").fill_null(0) == 1) | (
        pl.col("total_funding_amount").fill_null(0) >= 1_000_000
    )
    upper = pl.col("total_funding_amount").fill_null(0) <= 7_500_000
    df = df.filter(entry & upper)

    # Drop companies that already raised Series A or are out of business
    if "did_raise_a" in df.columns:
        df = df.filter(pl.col("did_raise_a").fill_null(0) != 1)
    if "oob_before_a" in df.columns:
        df = df.filter(
            (pl.col("oob_before_a") != True) | pl.col("oob_before_a").is_null()
        )

    # Log-transform skewed funding features
    LOG_FEATURES = [
        "seed_amount",
        "raised_before_a",
        "last_round_amount",
        "total_funding_amount",
        "funding_sum_last_12m",
        "funding_sum_last_18m",
        "funding_sum_last_24m",
        "avg_round_size_last_12m",
        "avg_round_size_last_18m",
        "avg_round_size_last_24m",
    ]
    log_exprs = []
    for feat in LOG_FEATURES:
        if feat in df.columns:
            log_exprs.append(
                pl.col(feat).clip(lower_bound=0).log1p().alias(f"log_{feat}")
            )
    if log_exprs:
        df = df.with_columns(log_exprs)

    # Interaction terms
    INTERACTIONS = {
        "int_peer_ratio_funding12": ("peer_round_ratio", "funding_sum_last_12m"),
        "int_peer_share_raised": ("peer_investor_share_pre_a", "raised_before_a"),
        "int_founder_seed_amount": ("founder_signal_sum", "seed_amount"),
        "int_last_round_peer": ("last_round_has_peer", "last_round_amount"),
    }
    int_exprs = []
    for new_col, (left, right) in INTERACTIONS.items():
        if left in df.columns and right in df.columns:
            int_exprs.append(
                (pl.col(left).fill_null(0) * pl.col(right).fill_null(0)).alias(new_col)
            )
    if int_exprs:
        df = df.with_columns(int_exprs)

    return df


def get_feature_columns(df):
    """Return numeric feature columns available in the dataframe."""
    # Import from pipeline if available, otherwise use all numeric columns
    try:
        from svp_company_scoring.models.series_a_model import SERIES_A_FEATURES
        available = [f for f in SERIES_A_FEATURES if f in df.columns]
        # Drop categorical that need target encoding — use numeric only for benchmark speed
        cat_cols = {"industry", "pb_deal_type", "pb_ownership_status", "pb_primary_industry_sector",
                    "pb_primary_industry_group", "pb_primary_industry_code", "pb_last_financing_status"}
        return [f for f in available if f not in cat_cols]
    except ImportError:
        import polars as pl
        return [
            c for c in df.columns
            if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.UInt64, pl.UInt32)
            and c not in {"as_of_date", "f_series_a", "salesforce_id"}
        ]


def run_benchmark():
    import polars as pl

    if not os.path.exists(DATA_PATH):
        print(
            json.dumps({"error": f"Data not found at {DATA_PATH}. "
                        "Run the pipeline to generate features_full_4to7months.parquet"}),
            file=sys.stderr,
        )
        sys.exit(1)

    df = load_and_filter(DATA_PATH)
    features = get_feature_columns(df)

    from datetime import date as _date
    train_start = _date.fromisoformat(TRAIN_START)
    train_end = _date.fromisoformat(TRAIN_END)
    test_start = _date.fromisoformat(TEST_START)
    test_end = _date.fromisoformat(TEST_END)

    df_train = df.filter(
        (pl.col("as_of_date") >= train_start) & (pl.col("as_of_date") <= train_end)
    )
    df_test = df.filter(
        (pl.col("as_of_date") >= test_start) & (pl.col("as_of_date") <= test_end)
    )

    TARGET = "f_series_a"

    X_train = df_train.select(features).to_numpy().astype(np.float64)
    y_train = df_train[TARGET].to_numpy().astype(np.float64)
    X_test = df_test.select(features).to_numpy().astype(np.float64)
    y_test = df_test[TARGET].to_numpy().astype(np.float64)

    # Median imputation (no target encoding — keeps benchmark fast & pure)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = PKBoostClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        min_child_weight=1.0,
        min_samples_split=20,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        gamma=GAMMA,
        reg_lambda=REG_LAMBDA,
        scale_pos_weight=scale_pos_weight,
        focal_gamma=FOCAL_GAMMA,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)

    result = {
        "pr_auc": round(pr_auc, 6),
        "roc_auc": round(roc_auc, 6),
        "train_time_s": round(train_time, 1),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "pos_rate_train": round(float(n_pos / len(y_train)), 6),
        "pos_rate_test": round(float(y_test.mean()), 6),
        "n_features": len(features),
        "focal_gamma": FOCAL_GAMMA,
        "n_estimators": N_ESTIMATORS,
    }

    print(json.dumps(result))
    return pr_auc


if __name__ == "__main__":
    run_benchmark()
