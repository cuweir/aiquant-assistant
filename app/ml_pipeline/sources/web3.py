from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..schemas import FeatureSourceSpec
from .base import FeatureSource, register_source


class _ExternalDataMixin:
    _base_external_dir = Path("data/external")

    def _load_external_csv(self, path: str) -> pd.Series:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"External data file missing: {csv_path}")
        df = pd.read_csv(csv_path, index_col=0)
        index = pd.to_datetime(df.index, utc=True, errors="coerce")
        if index.isna().any():
            index = pd.to_datetime(df.index, utc=True, errors="coerce", format="ISO8601")
        if index.isna().any():
            raise ValueError(f"Failed to parse dates in external CSV: {csv_path}")
        series = df.iloc[:, 0]
        if index.tz is not None:
            index_naive = index.tz_convert("UTC").tz_localize(None)
        else:
            index_naive = index
        series.index = index_naive
        return series.sort_index()

    def _align_to_index(self, series: pd.Series, target_index: pd.Index) -> pd.Series:
        src_index = series.index
        tgt_index = target_index

        if isinstance(src_index, pd.DatetimeIndex) and isinstance(tgt_index, pd.DatetimeIndex):
            if src_index.tz is not None:
                src_index = src_index.tz_convert("UTC").tz_localize(None)
                series.index = src_index
            if tgt_index.tz is not None:
                tgt_index = tgt_index.tz_convert("UTC").tz_localize(None)

        aligned = series.reindex(tgt_index)
        aligned = aligned.ffill().bfill()
        return aligned

    def _find_default_csv(
        self,
        symbol: Optional[str],
        dataset_subdir: str,
        provider_priority: Optional[Iterable[str]] = None,
    ) -> Optional[str]:
        if not symbol:
            return None
        sanitized = symbol.replace("/", "")
        providers: List[str]
        if provider_priority is None:
            providers = ["binance", "coinglass"]
        else:
            providers = list(provider_priority)
        for provider in providers:
            candidate = self._base_external_dir / provider / dataset_subdir / f"{sanitized}.csv"
            if candidate.exists():
                return str(candidate)
        return None

    @staticmethod
    def _normalize_provider_priority(provider_priority: Optional[object]) -> Optional[List[str]]:
        if provider_priority is None:
            return None
        if isinstance(provider_priority, str):
            return [provider_priority]
        try:
            return list(provider_priority)  # type: ignore[arg-type]
        except TypeError:
            return None


@register_source("funding_rate")
class FundingRateSource(_ExternalDataMixin, FeatureSource):
    """Provides perp funding rates; falls back to constant zero when data missing."""

    def compute(self, price_df: pd.DataFrame) -> pd.DataFrame:
        params: Dict[str, str] = self.spec.params or {}
        series: pd.Series

        csv_path = params.get("csv_path")
        if not csv_path:
            symbol = params.get("symbol")
            providers = self._normalize_provider_priority(params.get("provider_priority"))
            csv_path = self._find_default_csv(symbol, "funding", providers)
        if csv_path:
            series = self._load_external_csv(csv_path)
        else:
            constant = params.get("default", 0.0)
            series = pd.Series(constant, index=price_df.index)
        aligned = self._align_to_index(series, price_df.index)
        return pd.DataFrame({"funding_rate": aligned})


@register_source("open_interest")
class OpenInterestSource(_ExternalDataMixin, FeatureSource):
    def compute(self, price_df: pd.DataFrame) -> pd.DataFrame:
        params: Dict[str, str] = self.spec.params or {}
        csv_path = params.get("csv_path")
        if not csv_path:
            symbol = params.get("symbol")
            providers = self._normalize_provider_priority(params.get("provider_priority"))
            csv_path = self._find_default_csv(symbol, "open_interest", providers)
        if csv_path:
            series = self._load_external_csv(csv_path)
        else:
            multiplier = params.get("vol_multiplier", 1.0)
            series = pd.Series(price_df["volume"] * multiplier, index=price_df.index)
        aligned = self._align_to_index(series, price_df.index)
        return pd.DataFrame({"open_interest": aligned})


@register_source("sentiment_index")
class SentimentIndexSource(_ExternalDataMixin, FeatureSource):
    def compute(self, price_df: pd.DataFrame) -> pd.DataFrame:
        params: Dict[str, str] = self.spec.params or {}
        csv_path = params.get("csv_path")
        if not csv_path:
            symbol = params.get("symbol")
            providers = self._normalize_provider_priority(params.get("provider_priority"))
            csv_path = self._find_default_csv(symbol, "long_short_ratio", providers)
        if csv_path:
            series = self._load_external_csv(csv_path)
        else:
            neutral = params.get("neutral_value", 50.0)
            series = pd.Series(neutral, index=price_df.index)
        aligned = self._align_to_index(series, price_df.index)
        return pd.DataFrame({"sentiment_index": aligned})
