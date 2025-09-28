"""Feature source registry."""

from .base import FeatureSource, FeatureSourceRegistry, register_source
from .technical import TechnicalIndicatorSource
from .web3 import (
    FundingRateSource,
    MarketEventsSource,
    MarketMetricsSource,
    OpenInterestSource,
    SentimentIndexSource,
)

__all__ = [
    "FeatureSource",
    "FeatureSourceRegistry",
    "register_source",
    "TechnicalIndicatorSource",
    "FundingRateSource",
    "OpenInterestSource",
    "SentimentIndexSource",
    "MarketMetricsSource",
    "MarketEventsSource",
]
