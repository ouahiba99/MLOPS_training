"""
Recreates exactly what Notebook 5 does to a raw order row: engineer
checkout-time features, drop leakage/ID columns, apply the fitted smooth
target encoder to the high-cardinality columns, then apply the fitted
ColumnTransformer (median-impute + scale numeric, constant-impute + one-hot
categorical).

This is the same sequence Notebook 5's own "Section 8 — Verification" cell
runs to prove a reload matches training — so as long as target_encoder.joblib
and preprocessor.joblib in models/artifacts/ are the ones Notebook 5 saved,
this reproduces Notebook 5/6 output exactly. Nothing here is ever fit.
"""

import numpy as np
import pandas as pd

from src.data_access import load_feature_config, load_joblib_artifact


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ported verbatim from Notebook 5, section 3."""
    df = df.copy()

    for col in ["order_purchase_timestamp", "order_estimated_delivery_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if (
        "order_estimated_delivery_date" in df.columns
        and "order_purchase_timestamp" in df.columns
    ):
        df["estimated_delivery_days"] = (
            df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
        ).dt.total_seconds() / (24 * 3600)

    if "order_purchase_timestamp" in df.columns:
        ts = df["order_purchase_timestamp"]
        df["purchase_month"] = ts.dt.month.astype("float32")
        df["purchase_weekday"] = ts.dt.weekday.astype("float32")
        df["purchase_hour"] = ts.dt.hour.astype("float32")
        df["is_weekend"] = ts.dt.weekday.isin([5, 6]).astype("float32")

    df["freight_ratio"] = df["total_freight_value"] / (df["total_item_value"] + 1.0)
    df["price_per_item"] = df["total_item_value"] / (
        df["item_count"].replace(0, np.nan)
    )
    df["payment_ratio"] = df["total_payment_value"] / (df["total_item_value"] + 1.0)
    df["is_multi_seller"] = (df["unique_sellers"] > 1).astype("float32")
    df["is_multi_product"] = (df["unique_products"] > 1).astype("float32")

    return df


def apply_target_encoding(
    df: pd.DataFrame, target_encoder, high_card_cols: list[str]
) -> pd.DataFrame:
    """Ported from Notebook 5, section 5 — uses the fitted maps_/global_mean_,
    never re-fits. Unseen categories fall back to the global mean, same as training."""
    df = df.copy()
    for col in high_card_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .map(target_encoder.maps_[col])
                .fillna(target_encoder.global_mean_)
                .astype("float64")
            )
    return df


def preprocess(raw_df: pd.DataFrame) -> np.ndarray:
    """Raw order row(s) -> the exact numeric matrix the model expects."""
    feature_config = load_feature_config()
    target_encoder = load_joblib_artifact("target_encoder")
    preprocessor = load_joblib_artifact("preprocessor")

    df = engineer_features(raw_df)
    df = df.drop(columns=feature_config["leakage_cols_dropped"], errors="ignore")
    df = apply_target_encoding(
        df, target_encoder, feature_config["high_cardinality_cols_target_encoded"]
    )
    return preprocessor.transform(df)
