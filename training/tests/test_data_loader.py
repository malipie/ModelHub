"""Tests for training.src.data_loader (E-Commerce Churn dataset)."""

import pandas as pd
import pytest

from training.src.data_loader import EcommerceChurnDataLoader


@pytest.fixture
def loader() -> EcommerceChurnDataLoader:
    return EcommerceChurnDataLoader()


@pytest.fixture
def synthetic_df(loader: EcommerceChurnDataLoader) -> pd.DataFrame:
    """Small synthetic dataset for fast unit tests."""
    return loader.load_synthetic(n_samples=500, random_state=42)


class TestLoadSynthetic:
    def test_returns_dataframe(self, synthetic_df: pd.DataFrame) -> None:
        assert isinstance(synthetic_df, pd.DataFrame)

    def test_row_count(self, loader: EcommerceChurnDataLoader) -> None:
        df = loader.load_synthetic(n_samples=300, random_state=0)
        assert len(df) == 300

    def test_required_columns_present(self, synthetic_df: pd.DataFrame) -> None:
        required = {
            "customer_id",
            "country",
            "tenure_months",
            "subscription_type",
            "monthly_spend_eur",
            "total_spent_eur",
            "satisfaction_score_1_5",
            "churn",
        }
        missing = required - set(synthetic_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_churn_binary(self, synthetic_df: pd.DataFrame) -> None:
        assert set(synthetic_df["churn"].unique()).issubset({0, 1})

    def test_no_negative_tenure(self, synthetic_df: pd.DataFrame) -> None:
        assert (synthetic_df["tenure_months"] >= 1).all()

    def test_no_negative_spend(self, synthetic_df: pd.DataFrame) -> None:
        assert (synthetic_df["monthly_spend_eur"] >= 0).all()
        assert (synthetic_df["total_spent_eur"] >= 0).all()

    def test_churn_rate_reasonable(self, synthetic_df: pd.DataFrame) -> None:
        churn_rate = synthetic_df["churn"].mean()
        assert 0.05 <= churn_rate <= 0.50, f"Unexpected churn rate: {churn_rate:.2f}"

    def test_subscription_types_valid(self, synthetic_df: pd.DataFrame) -> None:
        valid = {"Basic", "Standard", "Premium"}
        assert set(synthetic_df["subscription_type"].unique()).issubset(valid)

    def test_account_age_categories_valid(self, synthetic_df: pd.DataFrame) -> None:
        valid = {"0-3 months", "3-12 months", "1-2 years", "2+ years"}
        assert set(synthetic_df["account_age_category"].unique()).issubset(valid)

    def test_preferred_category_valid(self, synthetic_df: pd.DataFrame) -> None:
        valid = {"Electronics", "Fashion", "Home & Garden", "Sports", "Books", "Beauty"}
        assert set(synthetic_df["preferred_category"].unique()).issubset(valid)

    def test_loyalty_member_binary(self, synthetic_df: pd.DataFrame) -> None:
        assert set(synthetic_df["loyalty_program_member"].unique()).issubset({0, 1})

    def test_satisfaction_score_range(self, synthetic_df: pd.DataFrame) -> None:
        assert (synthetic_df["satisfaction_score_1_5"] >= 1).all()
        assert (synthetic_df["satisfaction_score_1_5"] <= 5).all()

    def test_reproducibility(self, loader: EcommerceChurnDataLoader) -> None:
        df1 = loader.load_synthetic(n_samples=100, random_state=7)
        df2 = loader.load_synthetic(n_samples=100, random_state=7)
        pd.testing.assert_frame_equal(df1, df2)


class TestClean:
    def test_integer_columns_are_int(self, synthetic_df: pd.DataFrame) -> None:
        int_cols = [
            "tenure_months",
            "num_product_categories",
            "support_tickets_last_month",
            "reviews_left_count",
            "loyalty_program_member",
            "churn",
        ]
        for col in int_cols:
            assert synthetic_df[col].dtype in [
                int,
                "int64",
                "int32",
            ], f"Column {col} should be int, got {synthetic_df[col].dtype}"

    def test_no_nan_in_required_columns(self, synthetic_df: pd.DataFrame) -> None:
        required_no_nan = [
            "tenure_months",
            "monthly_spend_eur",
            "satisfaction_score_1_5",
            "subscription_type",
            "churn",
        ]
        for col in required_no_nan:
            assert synthetic_df[col].notna().all(), f"NaN found in column: {col}"
