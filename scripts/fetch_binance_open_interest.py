#!/usr/bin/env python
"""Download Binance futures open interest history and save to CSV."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

BINANCE_FUTURES_BASE = "https://fapi.binance.com"
OPEN_INTEREST_ENDPOINT = "/futures/data/openInterestHist"

PERIOD_TO_MS = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Binance open interest history")
    parser.add_argument("symbol", help="Perpetual contract symbol, e.g. BTC/USDT or BTCUSDT")
    parser.add_argument("--period", default="1h", choices=list(PERIOD_TO_MS.keys()), help="Sampling period")
    parser.add_argument("--since", type=str, default="2022-01-01", help="Start date (UTC, YYYY-MM-DD)")
    parser.add_argument("--until", type=str, default=None, help="End date (UTC, YYYY-MM-DD); defaults to now")
    parser.add_argument("--chunk", type=int, default=500, help="Rows per request (max 500)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external/binance/open_interest"),
        help="Directory to write the CSV file",
    )
    parser.add_argument("--proxy", type=str, default=None, help="Optional HTTP/HTTPS proxy")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout in seconds")
    parser.add_argument(
        "--contract-type",
        type=str,
        default="PERPETUAL",
        choices=["PERPETUAL", "CURRENT_QUARTER", "NEXT_QUARTER"],
        help="Contract type required by Binance endpoint",
    )
    return parser.parse_args()


def resolve_proxy(proxy_arg: Optional[str]) -> Optional[str]:
    if proxy_arg:
        return proxy_arg
    return (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
    )


def to_timestamp(value: str | None) -> int:
    if value is None:
        return int(datetime.now(UTC).timestamp() * 1000)
    dt = pd.Timestamp(value, tz="UTC")
    return int(dt.value / 1_000_000)


def fetch_open_interest(
    session: requests.Session,
    symbol: str,
    period: str,
    since_ms: int,
    until_ms: int,
    chunk: int,
    timeout: float,
    contract_type: str,
) -> List[Dict[str, str]]:
    sanitized = symbol.replace("/", "")
    cursor = until_ms
    step = PERIOD_TO_MS[period]
    collected: List[Dict[str, str]] = []

    while cursor > since_ms:
        params = {
            "symbol": sanitized,
            "period": period,
            "limit": min(chunk, 500),
            "endTime": cursor,
            "contractType": contract_type,
        }
        response = session.get(
            BINANCE_FUTURES_BASE + OPEN_INTEREST_ENDPOINT,
            params=params,
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Request failed ({response.status_code}): {response.text}")
        payload = response.json()
        if not payload:
            break
        collected.extend(payload)

        first_ts = int(payload[0]["timestamp"])
        if first_ts <= since_ms or len(payload) < params["limit"]:
            break
        cursor = first_ts - step

    filtered = []
    seen = set()
    for row in collected:
        ts = int(row.get("timestamp", 0))
        if ts < since_ms or ts > until_ms:
            continue
        if ts in seen:
            continue
        seen.add(ts)
        filtered.append(row)

    filtered.sort(key=lambda r: int(r["timestamp"]))
    return filtered


def normalize_dataframe(rows: List[Dict[str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No open interest records fetched; nothing to save.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    numeric_cols = ["sumOpenInterest", "sumOpenInterestValue", "openInterest"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    df.rename(
        columns={
            "sumOpenInterest": "sum_open_interest",
            "sumOpenInterestValue": "sum_open_interest_value",
            "openInterest": "open_interest",
        },
        inplace=True,
    )
    keep_cols = [col for col in ["open_interest", "sum_open_interest", "sum_open_interest_value"] if col in df.columns]
    return df[keep_cols]


def save_to_csv(df: pd.DataFrame, symbol: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{symbol.replace('/', '')}.csv"
    df.to_csv(file_path, index_label="timestamp")
    return file_path


def main() -> None:
    args = parse_args()

    proxy = resolve_proxy(args.proxy)
    session = requests.Session()
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})

    since_ms = to_timestamp(args.since)
    until_ms = to_timestamp(args.until)

    try:
        rows = fetch_open_interest(
            session,
            args.symbol,
            args.period,
            since_ms,
            until_ms,
            args.chunk,
            args.timeout,
            args.contract_type,
        )
    finally:
        session.close()

    df = normalize_dataframe(rows)
    path = save_to_csv(df, args.symbol, args.output_dir)
    print(f"Saved {len(df)} rows to {path}")


if __name__ == "__main__":
    main()
