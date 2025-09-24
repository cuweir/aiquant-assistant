#!/usr/bin/env python
"""Fetch Binance futures metrics (open interest & ratios) and funding rates from Binance Vision."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from app.core.config import settings
from scripts.fetch_binance_vision import (
    DATASETS as VISION_DATASETS,
    build_session as vision_build_session,
    build_suffixes as vision_build_suffixes,
    download_zip as vision_download_zip,
    extract_csv_bytes as vision_extract_csv_bytes,
    normalize_dataframe as vision_normalize_dataframe,
)

BASE_URLS = (
    "https://data.binance.vision/api/data",
    "https://data.binance.vision",
)

OUTPUT_ROOT = Path("data/external/binance")
METRICS_DIR = OUTPUT_ROOT / "metrics"
FUNDING_DIR = OUTPUT_ROOT / "funding"
COMBINED_DIR = Path("data/binance_metrics")


def sanitize_symbol(raw_symbol: str) -> str:
    """Convert user provided symbol into Binance Vision friendly form (e.g. BTC/USDT -> BTCUSDT)."""
    return raw_symbol.replace("/", "").upper()


def download_vision_dataset(
    dataset: str,
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    frequency: str,
    interval: str,
    proxy: Optional[str],
    timeout: float = 20.0,
) -> pd.DataFrame:
    """Download and concatenate Binance Vision archives for a dataset."""

    spec = VISION_DATASETS[dataset]
    template = spec.templates[frequency]
    requires_interval = spec.requires_interval

    session = vision_build_session(proxy)
    frames: list[pd.DataFrame] = []

    try:
        for suffix in vision_build_suffixes(start_date, end_date, frequency):
            parameters = {
                "symbol": symbol,
                "interval": interval,
                "suffix": suffix,
            }
            if not requires_interval:
                parameters.pop("interval")
            path_fragment = template.format(**parameters)
            zip_bytes = None
            for base in BASE_URLS:
                url = f"{base}{path_fragment}"
                zip_bytes = vision_download_zip(session, url, timeout)
                if zip_bytes is not None:
                    break
            if zip_bytes is None:
                print(f"Skipping {suffix}: archive not available")
                continue
            df_raw = vision_extract_csv_bytes(zip_bytes)
            df_norm = vision_normalize_dataframe(df_raw, spec)
            frames.append(df_norm)
            print(f"Downloaded {dataset} archive {suffix} ({len(df_norm)} rows)")
    finally:
        session.close()

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_dataframe(df: pd.DataFrame, directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    df.to_csv(path, index_label="timestamp")
    return path


def main(args: argparse.Namespace) -> None:
    proxy_option: Optional[str]
    if args.proxy is None:
        proxy_option = settings.PROXY_URL
    else:
        proxy_option = None if args.proxy.lower() == "none" else args.proxy

    symbol_id = sanitize_symbol(args.symbol)
    print(f"Resolved symbol {args.symbol} -> {symbol_id} for Binance Vision archives")

    metrics_df_raw = download_vision_dataset(
        "metrics",
        symbol_id,
        args.start_date,
        args.end_date,
        frequency=args.frequency,
        interval=args.interval,
        proxy=proxy_option,
    )

    funding_df_raw = download_vision_dataset(
        "funding_rate",
        symbol_id,
        args.start_date,
        args.end_date,
        frequency=args.frequency,
        interval=args.interval,
        proxy=proxy_option,
    )

    frames_saved: list[Path] = []

    if not metrics_df_raw.empty:
        metrics_df_raw = to_numeric(metrics_df_raw)
        metrics_raw_path = save_dataframe(metrics_df_raw, METRICS_DIR, f"{symbol_id}_raw.csv")
        frames_saved.append(metrics_raw_path)
        print(f"Saved raw metrics ({len(metrics_df_raw)} rows) to {metrics_raw_path}")

        resample_rule = args.interval.upper().replace("M", "T")
        metrics_df = metrics_df_raw.resample(resample_rule).last().dropna(how="all")
    else:
        metrics_df = pd.DataFrame()

    if not funding_df_raw.empty:
        funding_df_raw = to_numeric(funding_df_raw)
        funding_path = save_dataframe(funding_df_raw, FUNDING_DIR, f"{symbol_id}.csv")
        frames_saved.append(funding_path)
        print(f"Saved funding history ({len(funding_df_raw)} rows) to {funding_path}")

        resample_rule = args.interval.upper().replace("M", "T")
        funding_df = funding_df_raw.resample(resample_rule).ffill()
    else:
        funding_df = pd.DataFrame()

    if metrics_df.empty and funding_df.empty:
        print("No data downloaded. Nothing to combine.")
        return

    combined = metrics_df if not metrics_df.empty else pd.DataFrame(index=funding_df.index)
    if not funding_df.empty:
        combined = combined.join(funding_df, how="outer")
        combined.sort_index(inplace=True)

    combined_path = save_dataframe(combined, COMBINED_DIR, f"{symbol_id}_{args.interval}.csv")
    frames_saved.append(combined_path)
    print(f"Saved resampled combined metrics to {combined_path}")

    print("Files written:")
    for path in frames_saved:
        print(f" - {path}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Binance futures metrics + funding from Binance Vision archives")
    parser.add_argument("symbol", help="Trading pair, e.g. BTC/USDT or BTCUSDT")
    parser.add_argument("--start-date", required=True, help="UTC start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="UTC end date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1h", help="Resampled interval for the output, e.g. 1h, 4h")
    parser.add_argument("--frequency", choices=["daily", "monthly"], default="daily", help="Archive frequency to download")
    parser.add_argument("--proxy", default=None, help="Override proxy (use 'none' to disable)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
