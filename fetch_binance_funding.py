#!/usr/bin/env python
"""Download Binance perpetual funding rates and store them as a CSV for feature generation."""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import ccxt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Binance funding rate history and save to CSV")
    parser.add_argument("symbol", help="Perpetual contract symbol, e.g. BTC/USDT")
    parser.add_argument(
        "--since",
        type=str,
        default="2022-01-01",
        help="Start date (UTC, YYYY-MM-DD) for fetching history",
    )
    parser.add_argument(
        "--until",
        type=str,
        default=None,
        help="Optional end date (UTC, YYYY-MM-DD); defaults to now",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1000,
        help="Number of records per request (Binance allows up to 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external/binance/funding"),
        help="Directory to write the CSV file",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Optional HTTP/HTTPS proxy (e.g. http://127.0.0.1:7890)",
    )
    return parser.parse_args()


def resolve_proxy(proxy_arg: Optional[str]) -> Optional[dict]:
    candidate = proxy_arg or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if not candidate:
        return None
    return {"http": candidate, "https": candidate}


def to_milliseconds(date_str: str) -> int:
    timestamp = pd.Timestamp(date_str, tz="UTC")
    return int(timestamp.timestamp() * 1000)


def fetch_funding_rates(exchange: ccxt.Exchange, symbol: str, since_ms: int, until_ms: int | None, chunk: int) -> List[dict]:
    records: List[dict] = []
    cursor = since_ms
    market_symbol = symbol
    try:
        market = exchange.market(symbol)
        market_symbol = market["symbol"]
    except ccxt.BadSymbol:
        sanitized = symbol.replace("/", "")
        market = exchange.markets_by_id.get(sanitized)
        if market:
            market_symbol = market["symbol"]
        else:
            market_symbol = sanitized

    while True:
        batch = exchange.fetchFundingRateHistory(market_symbol, since=cursor, limit=chunk)
        if not batch:
            break

        exceeded_until = False
        for entry in batch:
            timestamp = entry.get("timestamp")
            if timestamp is None:
                continue
            if until_ms and timestamp > until_ms:
                exceeded_until = True
                break
            records.append(entry)
        if exceeded_until or (until_ms and batch[-1]["timestamp"] >= until_ms):
            break
        if len(batch) < chunk:
            break
        cursor = batch[-1]["timestamp"] + 1
    return records


def save_to_csv(symbol: str, records: List[dict], output_dir: Path) -> Path:
    if not records:
        raise RuntimeError("No funding rate records fetched; nothing to save.")
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    columns = {
        "fundingRate": "funding_rate",
        "fundingInterval": "funding_interval",
    }
    df.rename(columns=columns, inplace=True)
    df = df[[col for col in ["funding_rate", "funding_interval"] if col in df.columns]]
    if "funding_rate" in df.columns:
        df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    symbol_slug = symbol.replace("/", "")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{symbol_slug}.csv"
    df.to_csv(file_path, index_label="timestamp")
    return file_path


def main() -> None:
    args = parse_args()
    until_ms = to_milliseconds(args.until) if args.until else None
    since_ms = to_milliseconds(args.since)

    proxies = resolve_proxy(args.proxy)
    exchange_config = {"enableRateLimit": True}
    if proxies:
        exchange_config["proxies"] = proxies
        exchange_config["aiohttp_proxy"] = proxies["https"]

    exchange_class = ccxt.binanceusdm
    exchange = exchange_class(exchange_config)
    exchange.load_markets()
    try:
        records = fetch_funding_rates(exchange, args.symbol, since_ms, until_ms, args.chunk)
        file_path = save_to_csv(args.symbol, records, args.output_dir)
        print(f"Saved {len(records)} funding rate records to {file_path}")
    finally:
        # ccxt sync exchanges expose a `session` (requests.Session) we can close safely
        session = getattr(exchange, "session", None)
        if session:
            session.close()


if __name__ == "__main__":
    main()
