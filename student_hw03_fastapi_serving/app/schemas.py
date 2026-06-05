from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ListingFeatures(BaseModel):
    """One ML-ready listing feature row."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "room_type": "Private room",
                "property_type": "Private room in houseboat",
                "neighbourhood_name": "Centrum-West",
                "accommodates": 2,
                "bedrooms": 1.0,
                "beds": 1.0,
                "bathrooms": 1.5,
                "listing_price": 132.0,
                "minimum_nights": 3,
                "maximum_nights": 356,
                "instant_bookable": False,
                "host_is_superhost": True,
                "host_listing_count": 1,
                "total_reviews_before_cutoff": 311.0,
                "unique_reviewers_before_cutoff": 311.0,
                "avg_comment_len_before_cutoff": 302.1672,
                "max_comment_len_before_cutoff": 1917.0,
                "days_since_last_review": 94.0,
                "available_days_last_90d": 16,
                "available_rate_last_90d": 0.1758,
                "avg_minimum_nights_calendar_last_90d": 3.0,
                "avg_maximum_nights_calendar_last_90d": 30.0,
                "available_days_last_30d": 11,
                "available_rate_last_30d": 0.3548,
                "avg_minimum_nights_calendar_last_30d": 3.0,
                "avg_maximum_nights_calendar_last_30d": 30.0,
            }
        },
    )

    room_type: str
    property_type: str
    neighbourhood_name: str

    accommodates: int = Field(..., ge=0)
    bedrooms: Optional[float] = Field(None, ge=0)
    beds: Optional[float] = Field(None, ge=0)
    bathrooms: Optional[float] = Field(None, ge=0)
    listing_price: Optional[float] = Field(None, ge=0)
    minimum_nights: int = Field(..., ge=0)
    maximum_nights: int = Field(..., ge=0)

    instant_bookable: bool
    host_is_superhost: bool
    host_listing_count: int = Field(..., ge=0)

    total_reviews_before_cutoff: Optional[float] = Field(None, ge=0)
    unique_reviewers_before_cutoff: Optional[float] = Field(None, ge=0)
    avg_comment_len_before_cutoff: Optional[float] = Field(None, ge=0)
    max_comment_len_before_cutoff: Optional[float] = Field(None, ge=0)
    days_since_last_review: Optional[float] = Field(None, ge=0)

    available_days_last_90d: int = Field(..., ge=0)
    available_rate_last_90d: float = Field(..., ge=0, le=1)
    avg_minimum_nights_calendar_last_90d: Optional[float] = Field(None, ge=0)
    avg_maximum_nights_calendar_last_90d: Optional[float] = Field(None, ge=0)

    available_days_last_30d: int = Field(..., ge=0)
    available_rate_last_30d: float = Field(..., ge=0, le=1)
    avg_minimum_nights_calendar_last_30d: Optional[float] = Field(None, ge=0)
    avg_maximum_nights_calendar_last_30d: Optional[float] = Field(None, ge=0)


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: List[ListingFeatures] = Field(..., min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: Optional[float]
    threshold: float


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    tracking_uri: str
    experiment_name: str
    model_uri: Optional[str] = None
    run_id: Optional[str] = None
    run_name: Optional[str] = None
    dataset_version: str
    target: str
    threshold: float
    metrics: dict = Field(default_factory=dict)
    params: dict = Field(default_factory=dict)
    tags: dict = Field(default_factory=dict)
    error: Optional[str] = None