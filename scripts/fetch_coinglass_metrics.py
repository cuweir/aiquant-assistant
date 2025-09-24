#!/usr/bin/env python
"""Download historical open interest and long/short ratio data from Coinglass."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.services.coinglass_client import CoinglassClient


DATASETS = {"open_interest", "long_short_ratio"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical futures metrics from Coinglass")
    parser.add_argument("dataset", choices=sorted(DATASETS), help="Metric to download")
    parser.add_argument("symbol", help="Trading pair, e.g. BTCUSDT or BTC/USDT")
    parser.add_argument("--start-date", required=True, help="UTC start date (YYYY-MM-DD or ISO8601)")
    parser.add_argument("--end-date", required=True, help="UTC end date (inclusive, YYYY-MM-DD or ISO8601)")
    parser.add_argument("--interval", default="1h", help="Sampling interval (e.g. 1h, 4h)")
    parser.add_argument("--exchange", default="Binance", help="Exchange name for open interest endpoint (use 'none' to omit)")
    parser.add_argument("--currency", default="USDT", help="Quote currency for open interest endpoint (use 'none' to omit)")
    parser.add_argument("--coinglass-symbol", dest="cg_symbol", default=None, help="Override base asset symbol for Coinglass API")
    parser.add_argument("--api-key", default=None, help="Coinglass API key; defaults to COINGLASS_API_KEY env var")
    parser.add_argument(
        "--endpoint",
        default="auto",
        help="Override open interest API endpoint suffix (e.g. 'futures/openInterest/v2')",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="Override HTTP(S) proxy. Use 'none' to disable environment proxies.",
    )
    parser.add_argument(
        "--base-url",
        action="append",
        default=None,
        help="Override Coinglass base URL (can be provided multiple times).",
    )
    parser.add_argument(
        "--extra-param",
        action="append",
        default=None,
        help="Additional query parameters in key=value form (can repeat).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/coinglass"), help="Base output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite instead of merging with existing CSV")
    parser.add_argument("--chunk-days", type=int, default=30, help="Maximum days per API request (<= 30 recommended)")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def parse_utc(value: str, is_end: bool = False) -> dt.datetime:
    ts = pd.to_datetime(value, utc=True)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    result = ts.to_pydatetime()
    if is_end and ts.time() == dt.time(0, 0):
        result = result + dt.timedelta(days=1) - dt.timedelta(milliseconds=1)
    return result


def derive_coinglass_symbol(raw_symbol: str) -> str:
    symbol = raw_symbol.upper().replace("/", "")
    for suffix in ("USDT", "USD", "BUSD", "PERP"):
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def merge_with_existing(df: pd.DataFrame, output_path: Path, overwrite: bool) -> pd.DataFrame:
    if output_path.exists() and not overwrite:
        existing = pd.read_csv(output_path, parse_dates=["timestamp"], index_col="timestamp")
        combined = pd.concat([existing, df])
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()
    return df


def main() -> None:
    load_dotenv()
    args = parse_args()
    start = parse_utc(args.start_date, is_end=False)
    end = parse_utc(args.end_date, is_end=True)
    if start >= end:
        raise ValueError("start-date must be earlier than end-date")

    api_key = args.api_key or os.getenv("COINGLASS_API_KEY")
    if not api_key:
        raise RuntimeError("Set --api-key or COINGLASS_API_KEY environment variable")

    cg_symbol = args.cg_symbol or derive_coinglass_symbol(args.symbol)
    exchange_arg = None if args.exchange and args.exchange.lower() == "none" else args.exchange
    currency_arg = None if args.currency and args.currency.lower() == "none" else args.currency
    proxy_arg = args.proxy
    if proxy_arg and proxy_arg.lower() == "none":
        proxy_arg = ""
    elif proxy_arg is None:
        proxy_arg = os.getenv("COINGLASS_PROXY") or os.getenv("PROXY_URL")
    trust_env = proxy_arg not in ("", None)
    base_urls = args.base_url
    client = CoinglassClient(
        api_key=api_key,
        timeout=args.timeout,
        proxy=proxy_arg,
        trust_env=trust_env,
        base_urls=base_urls,
    )

    if args.dataset == "long_short_ratio":
        df = client.fetch_long_short_ratio(
            cg_symbol,
            args.interval,
            start,
            end,
            chunk_days=args.chunk_days,
        )
    else:
        extra_params = {}
        if args.extra_param:
            for item in args.extra_param:
                if "=" not in item:
                    raise ValueError(f"Invalid --extra-param value: {item}")
                key, value = item.split("=", 1)
                extra_params[key] = value
        df = client.fetch_open_interest(
            cg_symbol,
            exchange_arg or "",
            currency_arg or "",
            args.interval,
            start,
            end,
            chunk_days=args.chunk_days,
            endpoint=args.endpoint,
            extra_params=extra_params,
        )

    if df.empty:
        print("No data downloaded from Coinglass.")
        return

    df.index.name = "timestamp"
    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    sanitized_symbol = args.symbol.replace("/", "")
    output_path = output_dir / f"{sanitized_symbol}.csv"

    merged = merge_with_existing(df, output_path, args.overwrite)
    existing_order = list(merged.columns)
    priority_order = {
        "long_short_ratio": 0,
        "long_account_ratio": 1,
        "short_account_ratio": 2,
        "open_interest": 0,
        "open_interest_value": 1,
    }
    ordered_columns = sorted(
        merged.columns,
        key=lambda col: (priority_order.get(col, len(priority_order)), existing_order.index(col)),
    )
    merged = merged[ordered_columns]

    merged.to_csv(output_path, index_label="timestamp")
    print(f"Saved {len(merged)} rows to {output_path}")


if __name__ == "__main__":
    main()
