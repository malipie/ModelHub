"""Pydantic request/response schemas for the serving API."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class PredictionRequest(BaseModel):
    """Input features for churn prediction.

    Mirrors the feature set used during training (base_config.yaml).
    ``request_id`` is used for deterministic A/B routing — if omitted, a
    random UUID is generated automatically.
    """

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # --- Numeric features ---
    tenure_months: int = Field(..., ge=0, description="Months since first purchase")
    monthly_spend_eur: float = Field(..., ge=0, description="Average monthly spend (EUR)")
    total_spent_eur: float = Field(..., ge=0, description="Total spend to date (EUR)")
    avg_order_value_eur: float = Field(..., ge=0, description="Average order value (EUR)")
    purchase_frequency_per_month: float = Field(..., ge=0, description="Purchases per month")
    num_product_categories: int = Field(..., ge=1, description="Distinct categories bought")
    support_tickets_last_month: int = Field(..., ge=0, description="Support tickets last 30 days")
    website_sessions_per_month: int = Field(..., ge=0, description="Website visits per month")
    cart_abandonment_rate_percent: float = Field(
        ..., ge=0, le=100, description="Cart abandonment rate (%)"
    )
    email_engagement_rate_percent: float = Field(
        ..., ge=0, le=100, description="Email open/click rate (%)"
    )
    reviews_left_count: int = Field(..., ge=0, description="Product reviews written")
    returns_count_12m: int = Field(..., ge=0, description="Returns in last 12 months")
    last_purchase_days_ago: int = Field(..., ge=1, description="Days since last purchase")
    satisfaction_score_1_5: float = Field(..., ge=1, le=5, description="CSAT score (1–5)")
    loyalty_program_member: int = Field(..., ge=0, le=1, description="In loyalty programme (0/1)")
    payment_methods_used: int = Field(..., ge=1, description="Distinct payment methods used")

    # --- Categorical features ---
    country: Literal["Poland", "Germany", "France", "UK", "Spain", "Netherlands"]
    account_age_category: Literal["0-3 months", "3-12 months", "1-2 years", "2+ years"]
    subscription_type: Literal["Basic", "Standard", "Premium"]
    preferred_category: Literal[
        "Electronics", "Fashion", "Home & Garden", "Sports", "Books", "Beauty"
    ]

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure_months": 24,
                "monthly_spend_eur": 89.99,
                "total_spent_eur": 2159.76,
                "avg_order_value_eur": 44.99,
                "purchase_frequency_per_month": 2.0,
                "num_product_categories": 4,
                "support_tickets_last_month": 2,
                "website_sessions_per_month": 12,
                "cart_abandonment_rate_percent": 45.0,
                "email_engagement_rate_percent": 20.0,
                "reviews_left_count": 3,
                "returns_count_12m": 2,
                "last_purchase_days_ago": 14,
                "satisfaction_score_1_5": 2.5,
                "loyalty_program_member": 0,
                "payment_methods_used": 2,
                "country": "Poland",
                "account_age_category": "1-2 years",
                "subscription_type": "Standard",
                "preferred_category": "Electronics",
            }
        }
    }


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    """Churn prediction output."""

    prediction: int = Field(..., description="Churn prediction: 1 = will churn, 0 = will stay")
    probability: float = Field(..., description="Churn probability (0–1)")
    model_version: str = Field(..., description="MLflow model version that served the request")
    model_name: str = Field(..., description="Model role: 'champion' or 'challenger'")
    request_id: str = Field(..., description="Echo of the request ID for traceability")


# ---------------------------------------------------------------------------
# Admin schemas
# ---------------------------------------------------------------------------


class ABConfig(BaseModel):
    """Payload for PUT /ab/config — adjust champion/challenger traffic split."""

    champion_pct: int = Field(..., ge=0, le=100, description="% of traffic routed to champion")


class ModelInfo(BaseModel):
    """Info about a single loaded model."""

    model_name: str
    version: str
    stage: str
    loaded: bool


class ModelsInfoResponse(BaseModel):
    """Response for GET /model/info."""

    champion: ModelInfo
    challenger: ModelInfo | None = None
