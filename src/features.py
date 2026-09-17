"""
Builds the final numeric feature matrix for the model.

preprocess() already returns columns in the exact order the fitted
ColumnTransformer produces (matching feature_names.json) — there's no
extra column selection to do. What this module adds is the same shape
check Notebook 5's own verification cell does on reload, so a mismatched
artifact set fails loudly here instead of inside the model.
"""

import numpy as np
import pandas as pd

from src.data_access import load_feature_names
from src.preprocessing import preprocess


def build_features(raw_df: pd.DataFrame) -> np.ndarray:
    transformed = preprocess(raw_df)
    expected_n = len(load_feature_names())
    if transformed.shape[1] != expected_n:
        raise ValueError(
            f"Expected {expected_n} features per feature_names.json, "
            f"got {transformed.shape[1]} — artifacts may be mismatched."
        )
    return transformed
