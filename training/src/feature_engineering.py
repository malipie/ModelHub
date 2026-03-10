"""Feature engineering pipeline for Churn prediction."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def build_preprocessing_pipeline(
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for preprocessing.

    Numeric columns → StandardScaler.
    Categorical columns → OneHotEncoder (ignores unknown categories at inference).

    Args:
        categorical_columns: List of column names with categorical values.
        numeric_columns: List of column names with numeric values.

    Returns:
        Fitted-ready ColumnTransformer instance.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",  # silently ignores unseen categories
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",  # drop columns not in either list (e.g. customerID)
        verbose_feature_names_out=True,
    )

    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target column and drop unwanted columns.

    Args:
        df: Raw cleaned DataFrame.
        config: Parsed YAML config dict (uses data.target_column and data.drop_columns).

    Returns:
        Tuple of (X DataFrame, y Series).
    """
    target_col: str = config["data"]["target_column"]
    drop_cols: list[str] = config["data"].get("drop_columns", [])

    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop + [target_col])
    y = df[target_col].astype(int)

    logger.info(
        "Prepared features: %d columns, %d samples. Class balance — 0: %d, 1: %d.",
        X.shape[1],
        len(y),
        (y == 0).sum(),
        (y == 1).sum(),
    )
    return X, y


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract human-readable feature names from a fitted ColumnTransformer.

    Args:
        preprocessor: A *fitted* ColumnTransformer.

    Returns:
        List of output feature names.
    """
    try:
        names: list[str] = list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback for older sklearn versions
        names = []
        for _, transformer, cols in preprocessor.transformers_:
            if hasattr(transformer, "get_feature_names_out"):
                names.extend(transformer.get_feature_names_out())
            else:
                names.extend(cols)
    return names


def check_no_nan(X_transformed: np.ndarray, feature_names: list[str]) -> None:
    """Assert that the transformed feature matrix contains no NaN values.

    Args:
        X_transformed: Numpy array output of a fitted preprocessor.
        feature_names: Corresponding feature names (used in error message).

    Raises:
        ValueError: If any NaN values are detected.
    """
    nan_mask = np.isnan(X_transformed)
    if nan_mask.any():
        nan_cols = [feature_names[i] for i in np.where(nan_mask.any(axis=0))[0]]
        raise ValueError(f"NaN detected after preprocessing in columns: {nan_cols}")
    logger.debug("No NaN values detected in transformed features.")
