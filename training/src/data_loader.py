"""Data loading and schema validation for the E-Commerce Customer Churn dataset."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from training.src.utils import get_project_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pandera schema — validates the E-Commerce Churn CSV
# ---------------------------------------------------------------------------
_ECOMMERCE_SCHEMA = DataFrameSchema(
    columns={
        "customer_id": Column(str, nullable=False),
        "country": Column(str),
        "tenure_months": Column(int, Check.ge(0)),
        "account_age_category": Column(
            str,
            Check.isin(["0-3 months", "3-12 months", "1-2 years", "2+ years"]),
        ),
        "subscription_type": Column(str, Check.isin(["Basic", "Standard", "Premium"])),
        "monthly_spend_eur": Column(float, Check.ge(0)),
        "total_spent_eur": Column(float, Check.ge(0)),
        "avg_order_value_eur": Column(float, Check.ge(0)),
        "purchase_frequency_per_month": Column(float, Check.ge(0)),
        "num_product_categories": Column(int, Check.ge(1)),
        "preferred_category": Column(str),
        "support_tickets_last_month": Column(int, Check.ge(0)),
        "website_sessions_per_month": Column(int, Check.ge(0)),
        "cart_abandonment_rate_percent": Column(float, Check.in_range(0, 100)),
        "email_engagement_rate_percent": Column(float, Check.in_range(0, 100)),
        "reviews_left_count": Column(int, Check.ge(0)),
        "returns_count_12m": Column(int, Check.ge(0)),
        "last_purchase_days_ago": Column(int, Check.ge(1)),
        "satisfaction_score_1_5": Column(float, Check.in_range(1, 5)),
        "loyalty_program_member": Column(int, Check.isin([0, 1])),
        "payment_methods_used": Column(int, Check.ge(1)),
        "churn": Column(int, Check.isin([0, 1])),
    },
    strict=False,  # registration_date and other extra columns are allowed
)

_SUBSCRIPTION_TYPES = ["Basic", "Standard", "Premium"]
_ACCOUNT_AGE_CATEGORIES = ["0-3 months", "3-12 months", "1-2 years", "2+ years"]
_COUNTRIES = ["Poland", "Germany", "France", "UK", "Spain", "Netherlands"]
_PREFERRED_CATEGORIES = [
    "Electronics",
    "Fashion",
    "Home & Garden",
    "Sports",
    "Books",
    "Beauty",
]


class EcommerceChurnDataLoader:
    """Handles loading and validating the E-Commerce Customer Churn dataset.

    The dataset is stored locally at `Data/raw/ecommerce_churn.csv`
    (generated specifically for this project — not downloaded from the internet).

    Args:
        raw_path: Path to the CSV file (relative to project root or absolute).
    """

    def __init__(self, raw_path: str | Path = "Data/raw/ecommerce_churn.csv") -> None:
        root = get_project_root()
        self.raw_path: Path = Path(raw_path) if Path(raw_path).is_absolute() else root / raw_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, validate: bool = True) -> pd.DataFrame:
        """Load and optionally validate the dataset.

        Args:
            validate: Whether to run Pandera schema validation.

        Returns:
            Cleaned DataFrame ready for feature engineering.
        """
        if not self.raw_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.raw_path}. "
                "Make sure Data/raw/ecommerce_churn.csv is present."
            )

        logger.info("Loading dataset from %s …", self.raw_path)
        df = pd.read_csv(self.raw_path)

        df = self._clean(df)

        if validate:
            self._validate(df)

        logger.info(
            "Loaded %d rows, %d columns. Churn rate: %.1f%%.",
            len(df),
            df.shape[1],
            df["churn"].mean() * 100,
        )
        return df

    def load_synthetic(self, n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
        """Generate a synthetic E-Commerce churn dataset for testing.

        Mirrors the real dataset's schema and realistic value distributions
        so that unit tests can run without access to the real data file.

        Args:
            n_samples: Number of rows to generate.
            random_state: Seed for reproducibility.

        Returns:
            Synthetic DataFrame matching the E-Commerce Churn schema.
        """
        rng = np.random.default_rng(random_state)
        n = n_samples

        # Core numeric features
        tenure = rng.integers(1, 61, size=n)
        monthly_spend = rng.uniform(5, 300, size=n).round(2)
        total_spent = (tenure * monthly_spend + rng.normal(0, 100, size=n)).clip(3.59).round(2)
        avg_order_value = rng.uniform(10, 200, size=n).round(2)
        purchase_freq = rng.uniform(0.1, 10, size=n).round(2)
        num_categories = rng.integers(1, 16, size=n)
        support_tickets = rng.integers(0, 6, size=n)
        sessions = rng.integers(0, 21, size=n)
        cart_abandon = rng.uniform(0, 100, size=n).round(2)
        email_engage = rng.uniform(0, 100, size=n).round(2)
        reviews = rng.integers(0, 11, size=n)
        returns = rng.integers(0, 10, size=n)
        last_purchase = rng.integers(1, 366, size=n)
        satisfaction = rng.uniform(1, 5, size=n).round(2)
        loyalty = rng.integers(0, 2, size=n)
        payment_methods = rng.integers(1, 5, size=n)

        # Churn probability (mirrors real dataset logic)
        churn_prob = 1 / (
            1
            + np.exp(
                -(
                    -satisfaction * 0.8
                    - monthly_spend / 150
                    - tenure / 30
                    + support_tickets * 0.3
                    + returns * 0.15
                    + last_purchase / 200
                    + 0.5
                )
            )
        )
        churn = (rng.uniform(size=n) < churn_prob).astype(int)

        cat = lambda choices, size: rng.choice(choices, size=size)  # noqa: E731

        df = pd.DataFrame(
            {
                "customer_id": [f"CUST_{i:06d}" for i in range(n)],
                "country": cat(_COUNTRIES, n),
                "tenure_months": tenure,
                "account_age_category": cat(_ACCOUNT_AGE_CATEGORIES, n),
                "registration_date": pd.Timestamp("2024-01-01"),
                "subscription_type": cat(_SUBSCRIPTION_TYPES, n),
                "monthly_spend_eur": monthly_spend,
                "total_spent_eur": total_spent,
                "avg_order_value_eur": avg_order_value,
                "purchase_frequency_per_month": purchase_freq,
                "num_product_categories": num_categories,
                "preferred_category": cat(_PREFERRED_CATEGORIES, n),
                "support_tickets_last_month": support_tickets,
                "website_sessions_per_month": sessions,
                "cart_abandonment_rate_percent": cart_abandon,
                "email_engagement_rate_percent": email_engage,
                "reviews_left_count": reviews,
                "returns_count_12m": returns,
                "last_purchase_days_ago": last_purchase,
                "satisfaction_score_1_5": satisfaction,
                "loyalty_program_member": loyalty,
                "payment_methods_used": payment_methods,
                "churn": churn,
            }
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data type normalisation and basic cleaning."""
        int_cols = [
            "tenure_months",
            "num_product_categories",
            "support_tickets_last_month",
            "website_sessions_per_month",
            "reviews_left_count",
            "returns_count_12m",
            "last_purchase_days_ago",
            "loyalty_program_member",
            "payment_methods_used",
            "churn",
        ]
        for col in int_cols:
            if col in df.columns and df[col].dtype != int:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        """Validate DataFrame against Pandera schema."""
        logger.info("Validating schema …")
        try:
            _ECOMMERCE_SCHEMA.validate(df, lazy=True)
            logger.info("Schema validation passed.")
        except pa.errors.SchemaErrors as exc:
            logger.error("Schema validation failed:\n%s", exc.failure_cases)
            raise
