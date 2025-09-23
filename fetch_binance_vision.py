#!/usr/bin/env python
"""Download Binance Vision historical futures metrics (open interest, long/short ratio)."""

from __future__ import annotations

import argparse
import datetime as dt
from io import BytesIO
from pathlib import Path
from typing import Dict, List
import zipfile

import pandas as pd
import requests

from app.core.config import settings

BASE_URL = "https://data.binance.vision/api/data"
FALLBACK_URL = "https://data.binance.vision"


class DatasetSpec:
    def __init__(self, templates: Dict[str, str], requires_interval: bool, rename: Dict[str, str], value_column: str):
        self.templates = templates
        self.requires_interval = requires_interval
        self.rename = rename
        self.value_column = value_column


DATASETS: Dict[str, DatasetSpec] = {
    "open_interest": DatasetSpec(
        templates={
            "daily": "/data/futures/um/daily/openInterest/{symbol}/{symbol}-openInterest-{suffix}.zip",
            "monthly": "/data/futures/um/monthly/openInterest/{symbol}/{symbol}-openInterest-{suffix}.zip",
        },
        requires_interval=False,
        rename={
            "sumOpenInterest": "sum_open_interest",
            "sumOpenInterestValue": "sum_open_interest_value",
            "openInterest": "open_interest",
        },
        value_column="open_interest",
    ),
    "long_short_ratio": DatasetSpec(
        templates={
            "daily": (
                "/data/futures/um/daily/metrics/globalLongShortRatio/{symbol}/"
                "{symbol}-globalLongShortAccountRatio-{interval}-{suffix}.zip"
            ),
            "monthly": (
                "/data/futures/um/monthly/metrics/globalLongShortRatio/{symbol}/"
                "{symbol}-globalLongShortAccountRatio-{interval}-{suffix}.zip"
            ),
        },
        requires_interval=True,
        rename={
            "longShortRatio": "long_short_ratio",
            "longAccount": "long_account_ratio",
            "shortAccount": "short_account_ratio",
        },
        value_column="long_short_ratio",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download historical metrics from data.binance.vision")
    parser.add_argument("dataset", choices=DATASETS.keys(), help="Dataset to download")
    parser.add_argument("symbol", help="Symbol, e.g. BTCUSDT")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1h", help="Interval (needed for long_short_ratio)")
    parser.add_argument("--frequency", choices=["daily", "monthly"], default="daily", help="Archive frequency")
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/binance"), help="Base output directory")
    parser.add_argument("--proxy", type=str, default=None, help="Override proxy (use 'none' to disable)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV instead of appending")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def build_suffixes(start: str, end: str, frequency: str) -> List[str]:
    start_dt = dt.datetime.strptime(start, "%Y-%m-%d").date()
    end_dt = dt.datetime.strptime(end, "%Y-%m-%d").date()
    if start_dt > end_dt:
        raise ValueError("start_date must be <= end_date")
    if frequency == "daily":
        days = (end_dt - start_dt).days
        return [(start_dt + dt.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days + 1)]

    # monthly
    suffixes: List[str] = []
    current = start_dt.replace(day=1)
    last = end_dt.replace(day=1)
    while current <= last:
        suffixes.append(current.strftime("%Y-%m"))
        year = current.year + (current.month // 12)
        month = current.month % 12 + 1
        current = dt.date(year, month, 1)
    return suffixes


def build_session(proxy: str | None) -> requests.Session:
    session = requests.Session()
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
    return session


def download_zip(session: requests.Session, url: str, timeout: float) -> bytes | None:
    headers = {"Referer": "https://data.binance.vision/"}
    response = session.get(url, timeout=timeout, headers=headers)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.content


def extract_csv_bytes(zip_bytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        members = zf.namelist()
        if not members:
            raise RuntimeError("Zip archive is empty")
        with zf.open(members[0]) as fh:
            return pd.read_csv(fh, encoding="utf-8-sig")


def normalize_dataframe(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    df = df.rename(columns=spec.rename)
    if "timestamp" not in df.columns:
        raise RuntimeError("Downloaded file missing 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    return df


def main() -> None:
    args = parse_args()
    spec = DATASETS[args.dataset]
    if spec.requires_interval and not args.interval:
        raise ValueError("This dataset requires --interval")

    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.symbol}.csv"

    proxy_override = args.proxy
    if proxy_override and proxy_override.lower() == "none":
        proxy_override = ""
    session_proxy = proxy_override if proxy_override is not None else getattr(settings, "PROXY_URL", None)
    session_proxy = session_proxy or None
    if session_proxy:
        print(f"Using proxy {session_proxy}")
    else:
        print("Using direct connection (no proxy)")
    session = build_session(session_proxy)

    existing = pd.DataFrame()
    if output_path.exists() and not args.overwrite:
        existing = pd.read_csv(output_path, parse_dates=["timestamp"], index_col="timestamp")

    frames: List[pd.DataFrame] = []
    if not existing.empty:
        frames.append(existing)

    template = spec.templates[args.frequency]

    for suffix in build_suffixes(args.start_date, args.end_date, args.frequency):
        url = BASE_URL + template.format(symbol=args.symbol, interval=args.interval, suffix=suffix)
        try:
            zip_bytes = download_zip(session, url, args.timeout)
            if zip_bytes is None:
                fallback = FALLBACK_URL + template.format(symbol=args.symbol, interval=args.interval, suffix=suffix)
                zip_bytes = download_zip(session, fallback, args.timeout)
            if zip_bytes is None:
                print(f"Skipping {suffix}: not available")
                continue
            df_raw = extract_csv_bytes(zip_bytes)
            df_norm = normalize_dataframe(df_raw, spec)
            frames.append(df_norm)
            print(f"Downloaded {args.dataset} for {suffix} ({len(df_norm)} rows)")
        except requests.HTTPError as exc:
            print(f"HTTP error for {suffix}: {exc}")
        except Exception as exc:
            print(f"Failed to process {suffix}: {exc}")

    if not frames:
        print("No data downloaded.")
        return

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.to_csv(output_path, index_label="timestamp")
    print(f"Saved {len(combined)} rows to {output_path}")


if __name__ == "__main__":
    main()
