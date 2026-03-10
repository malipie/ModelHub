"""Tests for training.src.feature_engineering."""

import numpy as np
import pandas as pd
import pytest

from training.src.data_loader import EcommerceChurnDataLoader
from training.src.feature_engineering import (
    build_preprocessing_pipeline,
    check_no_nan,
    get_feature_names,
    prepare_data,
)
from training.src.utils import get_project_root, load_config


@pytest.fixture(scope="module")
def config() -> dict:
    root = get_project_root()
    return load_config(root / "training" / "config" / "base_config.yaml")


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    loader = EcommerceChurnDataLoader()
    return loader.load_synthetic(n_samples=400, random_state=99)


@pytest.fixture(scope="module")
def X_y(raw_df: pd.DataFrame, config: dict):
    return prepare_data(raw_df, config)


@pytest.fixture(scope="module")
def fitted_preprocessor(X_y, config: dict):
    X, _ = X_y
    cat_cols = [c for c in config["features"]["categorical_columns"] if c in X.columns]
    num_cols = [c for c in config["features"]["numeric_columns"] if c in X.columns]
    preprocessor = build_preprocessing_pipeline(cat_cols, num_cols)
    preprocessor.fit(X)
    return preprocessor


class TestPrepareData:
    def test_returns_tuple(self, X_y) -> None:
        X, y = X_y
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_not_in_features(self, X_y) -> None:
        X, _ = X_y
        assert "Churn" not in X.columns

    def test_customerid_dropped(self, X_y) -> None:
        X, _ = X_y
        assert "customerID" not in X.columns

    def test_y_binary(self, X_y) -> None:
        _, y = X_y
        assert set(y.unique()).issubset({0, 1})

    def test_no_nan_in_y(self, X_y) -> None:
        _, y = X_y
        assert not y.isna().any()

    def test_lengths_match(self, X_y, raw_df) -> None:
        X, y = X_y
        assert len(X) == len(y) == len(raw_df)


class TestPreprocessingPipeline:
    def test_output_is_numpy(self, fitted_preprocessor, X_y) -> None:
        X, _ = X_y
        result = fitted_preprocessor.transform(X)
        assert isinstance(result, np.ndarray)

    def test_output_2d(self, fitted_preprocessor, X_y) -> None:
        X, _ = X_y
        result = fitted_preprocessor.transform(X)
        assert result.ndim == 2

    def test_output_rows_match_input(self, fitted_preprocessor, X_y) -> None:
        X, _ = X_y
        result = fitted_preprocessor.transform(X)
        assert result.shape[0] == len(X)

    def test_no_nan_in_output(self, fitted_preprocessor, X_y) -> None:
        X, _ = X_y
        result = fitted_preprocessor.transform(X)
        feature_names = get_feature_names(fitted_preprocessor)
        check_no_nan(result, feature_names)  # should not raise

    def test_output_columns_positive(self, fitted_preprocessor, X_y) -> None:
        X, _ = X_y
        result = fitted_preprocessor.transform(X)
        assert result.shape[1] > 0

    def test_feature_names_count_matches_output(self, fitted_preprocessor, X_y) -> None:
        X, _ = X_y
        result = fitted_preprocessor.transform(X)
        names = get_feature_names(fitted_preprocessor)
        assert len(names) == result.shape[1]

    def test_numeric_columns_scaled(self, fitted_preprocessor, X_y) -> None:
        """After StandardScaler, numeric features should have ~zero mean."""
        X, _ = X_y
        result = fitted_preprocessor.transform(X)
        # Numeric columns are first in ColumnTransformer (see build_preprocessing_pipeline)
        # E-Commerce dataset has 16 numeric features
        num_count = 16
        num_features = result[:, :num_count]
        assert abs(num_features.mean()) < 0.5

    def test_unknown_category_handled(self, fitted_preprocessor, X_y, config) -> None:
        """Unseen categories at transform time should not raise (handle_unknown='ignore')."""
        X, _ = X_y
        X_copy = X.copy()
        # Inject a completely unseen category value
        if "country" in X_copy.columns:
            X_copy.loc[X_copy.index[0], "country"] = "Unknown_XYZ"
        fitted_preprocessor.transform(X_copy)  # must not raise


class TestCheckNoNan:
    def test_passes_for_clean_array(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        check_no_nan(arr, ["a", "b"])  # should not raise

    def test_raises_for_nan_array(self) -> None:
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])
        with pytest.raises(ValueError, match="NaN detected"):
            check_no_nan(arr, ["a", "b"])
