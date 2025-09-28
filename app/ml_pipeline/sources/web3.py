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
        features = pd.DataFrame({"funding_rate": aligned}, index=price_df.index)

        price_close = price_df.get("close")
        if price_close is not None:
            price_close = price_close.replace(0, np.nan).ffill().bfill()
        price_return = None
        if price_close is not None:
            price_return = price_close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

        features["funding_rate_change_1h"] = features["funding_rate"].diff()
        features["funding_rate_abs_change_1h"] = features["funding_rate_change_1h"].abs()
        features["funding_rate_ma_6h"] = features["funding_rate"].rolling(6, min_periods=3).mean()
        rolling_fr = features["funding_rate"].rolling(24, min_periods=6)
        features["funding_rate_zscore_24h"] = (features["funding_rate"] - rolling_fr.mean()) / rolling_fr.std()

        if price_return is not None:
            features["funding_price_return"] = features["funding_rate"] * price_return

        quantile_window = features["funding_rate"].rolling(48, min_periods=12)
        upper = quantile_window.quantile(0.9)
        lower = quantile_window.quantile(0.1)
        features["funding_rate_high_event"] = (features["funding_rate"] >= upper).astype(float)
        features["funding_rate_low_event"] = (features["funding_rate"] <= lower).astype(float)
        features["funding_rate_sign"] = np.sign(features["funding_rate"])

        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill()
        return features


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

        price_close = price_df.get("close")
        if price_close is not None:
            price_close = price_close.replace(0, np.nan).ffill().bfill()
        price_volume = price_df.get("volume")
        if price_volume is not None:
            price_volume = price_volume.replace(0, np.nan).ffill().bfill()

        features = pd.DataFrame(index=price_df.index)
        oi = aligned.get("open_interest")
        oi_val = aligned.get("open_interest_value")
        top_ratio = aligned.get("top_trader_long_short_ratio")
        global_ratio = aligned.get("global_long_short_ratio")
        taker_ratio = aligned.get("taker_long_short_vol_ratio")

        if oi is not None:
            oi = oi.replace(0, np.nan)
            features["oi_pct_change_1h"] = oi.pct_change(fill_method=None)
            features["oi_pct_change_6h"] = oi.pct_change(6, fill_method=None)
            features["oi_log_diff_24h"] = np.log(oi).diff(24)
            rolling = oi.rolling(24, min_periods=6)
            features["oi_zscore_24h"] = (oi - rolling.mean()) / rolling.std()
            features["oi_volatility_24h"] = rolling.std() / rolling.mean()
            if price_volume is not None:
                features["oi_to_volume_ratio"] = (oi / price_volume).replace([np.inf, -np.inf], np.nan)

        if oi_val is not None:
            oi_val = oi_val.replace(0, np.nan)
            features["oi_value_pct_change_1h"] = oi_val.pct_change(fill_method=None)
            if price_close is not None:
                features["oi_value_to_price"] = (oi_val / price_close).replace([np.inf, -np.inf], np.nan)

        if top_ratio is not None and global_ratio is not None:
            spread = top_ratio - global_ratio
            features["ls_spread_top_vs_global"] = spread
            features["ls_spread_change_1h"] = spread.diff()
            features["top_trader_ls_change_1h"] = top_ratio.diff()
            features["global_ls_change_1h"] = global_ratio.diff()
            features["log_global_ls_ratio"] = np.log(global_ratio.replace(0, np.nan))
            features["top_vs_global_rel"] = (top_ratio / global_ratio.replace(0, np.nan)) - 1
            global_roll = global_ratio.rolling(24, min_periods=6)
            features["global_ls_zscore_24h"] = (global_ratio - global_roll.mean()) / global_roll.std()

        if taker_ratio is not None:
            features["taker_ls_delta_1h"] = taker_ratio.diff()
            features["taker_ls_ma_6h"] = taker_ratio.rolling(6, min_periods=3).mean()
            taker_roll = taker_ratio.rolling(24, min_periods=6)
            features["taker_ls_zscore_24h"] = (taker_ratio - taker_roll.mean()) / taker_roll.std()

        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill()
        return features


@register_source("market_events")
class MarketEventsSource(_ExternalDataMixin, FeatureSource):
    """Generates event-style features combining funding, open interest, and sentiment ratios."""

    def compute(self, price_df: pd.DataFrame) -> pd.DataFrame:
        params: Dict[str, object] = self.spec.params or {}
        symbol = params.get("symbol")
        providers = self._normalize_provider_priority(params.get("provider_priority"))

        metrics_path = self._find_default_csv(symbol, "metrics", providers)
        if not metrics_path:
            raise FileNotFoundError("No metrics CSV available for market_events feature source")
        metrics_df = self._load_external_frame(metrics_path)

        funding_path = self._find_default_csv(symbol, "funding", providers)
        if not funding_path:
            raise FileNotFoundError("No funding CSV available for market_events feature source")
        funding_series = self._load_external_csv(funding_path)

        metrics_aligned = metrics_df.reindex(price_df.index).ffill().bfill()
        funding_aligned = funding_series.reindex(price_df.index).ffill().bfill()

        features = pd.DataFrame(index=price_df.index)

        oi = metrics_aligned.get("open_interest")
        top_ratio = metrics_aligned.get("top_trader_long_short_ratio")
        global_ratio = metrics_aligned.get("global_long_short_ratio")
        taker_ratio = metrics_aligned.get("taker_long_short_vol_ratio")

        oi_pct_change = None
        if oi is not None:
            oi = oi.replace(0, np.nan)
            oi_pct_change = oi.pct_change(fill_method=None)
            if oi_pct_change.notna().any():
                q_high = oi_pct_change.quantile(0.9)
                q_low = oi_pct_change.quantile(0.1)
                features["event_oi_surge"] = (oi_pct_change > q_high).astype(float)
                features["event_oi_drop"] = (oi_pct_change < q_low).astype(float)

        funding = funding_aligned.replace([-np.inf, np.inf], np.nan)
        funding_high = funding.quantile(0.9)
        funding_low = funding.quantile(0.1)
        if pd.isna(funding_high):
            funding_high = funding.max()
        if pd.isna(funding_low):
            funding_low = funding.min()
        features["event_funding_high"] = (funding >= funding_high).astype(float)
        features["event_funding_low"] = (funding <= funding_low).astype(float)
        features["event_funding_abs"] = funding.abs()
        features["event_funding_sign"] = np.sign(funding)
        features["event_funding_volatility_24h"] = funding.rolling(24, min_periods=6).std()

        if oi_pct_change is not None:
            interaction = (funding * oi_pct_change).replace([-np.inf, np.inf], np.nan)
            features["event_funding_oi_interaction"] = interaction
            features["event_long_crowding"] = ((funding >= funding_high) & (oi_pct_change > 0)).astype(float)
            features["event_short_crowding"] = ((funding <= funding_low) & (oi_pct_change < 0)).astype(float)

        if top_ratio is not None and global_ratio is not None:
            spread = top_ratio - global_ratio
            spread_abs = spread.abs()
            spread_threshold = spread_abs.quantile(0.9)
            if pd.isna(spread_threshold) or spread_threshold == 0:
                spread_threshold = spread_abs.max()
            features["event_spread_top_vs_global"] = spread
            features["event_top_vs_global_extreme"] = (spread_abs >= spread_threshold).astype(float)
            features["event_spread_sign"] = np.sign(spread)
            features["event_spread_change_1h"] = spread.diff()
            if oi_pct_change is not None:
                features["event_spread_oi_interaction"] = (spread * oi_pct_change).replace([-np.inf, np.inf], np.nan)

        if taker_ratio is not None:
            taker_change = taker_ratio.diff()
            features["event_taker_ratio_change_1h"] = taker_change
            surge_threshold = taker_change.quantile(0.9)
            drop_threshold = taker_change.quantile(0.1)
            if pd.isna(surge_threshold):
                surge_threshold = taker_change.max()
            if pd.isna(drop_threshold):
                drop_threshold = taker_change.min()
            features["event_taker_surge"] = (taker_change > surge_threshold).astype(float)
            features["event_taker_drop"] = (taker_change < drop_threshold).astype(float)

        price_close = price_df.get("close")
        price_volume = price_df.get("volume")
        price_return = None
        if price_close is not None:
            price_close = price_close.replace(0, np.nan).ffill().bfill()
            price_return = price_close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            features["event_price_return_1h"] = price_return
            features["event_price_return_6h"] = price_close.pct_change(6, fill_method=None).replace([np.inf, -np.inf], np.nan)
            price_roll = price_return.rolling(24, min_periods=6)
            features["event_price_volatility_24h"] = price_roll.std()
            features["event_price_zscore_24h"] = (price_return - price_roll.mean()) / price_roll.std()
            features["event_funding_price_positive"] = ((funding >= funding_high) & (price_return > 0)).astype(float)
            features["event_funding_price_negative"] = ((funding <= funding_low) & (price_return < 0)).astype(float)
            features["event_funding_price_interaction"] = (funding * price_return).replace([-np.inf, np.inf], np.nan)

        if price_volume is not None:
            price_volume = price_volume.replace(0, np.nan).ffill().bfill()
            volume_pct_change = price_volume.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            volume_roll = price_volume.rolling(24, min_periods=6)
            volume_zscore = (price_volume - volume_roll.mean()) / volume_roll.std()
            features["event_volume_pct_change_1h"] = volume_pct_change
            features["event_volume_zscore_24h"] = volume_zscore
            if oi_pct_change is not None:
                features["event_volume_oi_interaction"] = (volume_pct_change * oi_pct_change).replace([-np.inf, np.inf], np.nan)
            if price_return is not None:
                features["event_volume_price_interaction"] = (volume_pct_change * price_return).replace([-np.inf, np.inf], np.nan)

        if price_return is not None and oi_pct_change is not None:
            features["event_price_oi_interaction"] = (price_return * oi_pct_change).replace([-np.inf, np.inf], np.nan)
            same_sign = ((price_return > 0) & (oi_pct_change > 0)).astype(float)
            opp_sign = ((price_return < 0) & (oi_pct_change < 0)).astype(float)
            features["event_price_oi_same_direction"] = same_sign - opp_sign

        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill()
        return features
