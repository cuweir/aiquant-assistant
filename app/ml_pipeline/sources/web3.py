from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

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

    def _load_external_frame(self, path: str) -> pd.DataFrame:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"External data file missing: {csv_path}")
        df = pd.read_csv(csv_path, index_col=0)
        index = pd.to_datetime(df.index, utc=True, errors="coerce")
        if index.isna().any():
            index = pd.to_datetime(df.index, utc=True, errors="coerce", format="ISO8601")
        if index.isna().any():
            raise ValueError(f"Failed to parse dates in external CSV: {csv_path}")
        if index.tz is not None:
            index_naive = index.tz_convert("UTC").tz_localize(None)
        else:
            index_naive = index
        df.index = index_naive
        return df.sort_index()

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
            base_dir = self._base_external_dir / provider / dataset_subdir
            candidate = base_dir / f"{sanitized}.csv"
            if candidate.exists():
                return str(candidate)
            raw_candidate = base_dir / f"{sanitized}_raw.csv"
            if raw_candidate.exists():
                return str(raw_candidate)
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


@register_source("market_metrics")
class MarketMetricsSource(_ExternalDataMixin, FeatureSource):
    """Derives richer open interest / long-short features from Binance metrics archives."""

    _METRIC_RENAME: Dict[str, str] = {
        "open_interest": "open_interest",
        "sum_open_interest": "open_interest",
        "sum_open_interest_value": "open_interest_value",
        "open_interest_value": "open_interest_value",
        "top_trader_long_short_ratio": "top_trader_long_short_ratio",
        "sum_toptrader_long_short_ratio": "top_trader_long_short_ratio",
        "global_long_short_ratio": "global_long_short_ratio",
        "sum_long_short_ratio": "global_long_short_ratio",
        "taker_long_short_vol_ratio": "taker_long_short_vol_ratio",
        "sum_taker_long_short_vol_ratio": "taker_long_short_vol_ratio",
    }

    def compute(self, price_df: pd.DataFrame) -> pd.DataFrame:
        params: Dict[str, object] = self.spec.params or {}
        csv_path = params.get("csv_path")
        if not csv_path:
            symbol = params.get("symbol")
            providers = self._normalize_provider_priority(params.get("provider_priority"))
            csv_path = self._find_default_csv(symbol, "metrics", providers)
        if not csv_path:
            raise FileNotFoundError("No metrics CSV available for market_metrics feature source")

        csv_file = Path(csv_path)
        if not csv_file.exists():
            alt = csv_file.with_name(csv_file.stem + "_raw.csv")
            if alt.exists():
                csv_file = alt
            else:
                raise FileNotFoundError(f"Metrics CSV not found at {csv_file}")

        frame = self._load_external_frame(str(csv_file))
        frame = frame.rename(columns={k: v for k, v in self._METRIC_RENAME.items() if k in frame.columns})
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
        for col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if "symbol" in frame.columns:
            frame.drop(columns=["symbol"], inplace=True)
        frame.sort_index(inplace=True)

        aligned = frame.reindex(price_df.index).ffill().bfill()

        features = pd.DataFrame(index=price_df.index)
        oi = aligned.get("open_interest")
        oi_val = aligned.get("open_interest_value")
        top_ratio = aligned.get("top_trader_long_short_ratio")
        global_ratio = aligned.get("global_long_short_ratio")
        taker_ratio = aligned.get("taker_long_short_vol_ratio")

        if oi is not None:
            oi = oi.replace(0, np.nan)
            features["oi_pct_change_1h"] = oi.pct_change()
            features["oi_pct_change_6h"] = oi.pct_change(6)
            features["oi_log_diff_24h"] = np.log(oi).diff(24)
            rolling = oi.rolling(24, min_periods=6)
            features["oi_zscore_24h"] = (oi - rolling.mean()) / rolling.std()

        if oi_val is not None:
            oi_val = oi_val.replace(0, np.nan)
            features["oi_value_pct_change_1h"] = oi_val.pct_change()

        if top_ratio is not None and global_ratio is not None:
            spread = top_ratio - global_ratio
            features["ls_spread_top_vs_global"] = spread
            features["ls_spread_change_1h"] = spread.diff()
            features["top_trader_ls_change_1h"] = top_ratio.diff()
            features["global_ls_change_1h"] = global_ratio.diff()
            features["log_global_ls_ratio"] = np.log(global_ratio.replace(0, np.nan))

        if taker_ratio is not None:
            features["taker_ls_delta_1h"] = taker_ratio.diff()
            features["taker_ls_ma_6h"] = taker_ratio.rolling(6, min_periods=3).mean()

        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill()
        return features
