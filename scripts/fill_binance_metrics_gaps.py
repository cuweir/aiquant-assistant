#!/usr/bin/env python
"""Fill missing timestamps in the Binance metrics CSV by time interpolation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill missing timestamps in metrics CSV")
    parser.add_argument("symbol", help="Symbol, e.g. BTCUSDT or BTC/USDT")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/external/binance/metrics"),
        help="Directory containing the metrics CSV (defaults to data/external/binance/metrics)",
    )
    parser.add_argument(
        "--freq",
        default="5min",
        help="Target frequency for resampling (default: 5min)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Keep a copy of the original file with .bak suffix",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()
    return df


def fill_gaps(df: pd.DataFrame, freq: str, symbol_id: str) -> pd.DataFrame:
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    reindexed = df.reindex(full_index)
    numeric_cols = reindexed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    interpolated = reindexed.copy()
    if numeric_cols:
        interpolated[numeric_cols] = interpolated[numeric_cols].interpolate(method='time', limit_direction='both')
    remaining_cols = [col for col in interpolated.columns if col not in numeric_cols]
    for col in remaining_cols:
        interpolated[col] = interpolated[col].ffill().bfill()
    if 'symbol' in interpolated.columns:
        interpolated['symbol'] = symbol_id
    interpolated.index.name = 'timestamp'
    return interpolated


def main() -> None:
    args = parse_args()
    symbol_id = args.symbol.replace('/', '').upper()
    csv_path = args.input_dir / f"{symbol_id}.csv"
    if not csv_path.exists():
        raw_path = args.input_dir / f"{symbol_id}_raw.csv"
        if raw_path.exists():
            csv_path = raw_path
        else:
            raise FileNotFoundError(f"Neither {csv_path} nor {raw_path} exists")

    df = load_metrics(csv_path)
    before_count = len(df)
    filled = fill_gaps(df, args.freq, symbol_id)
    after_count = len(filled)

    if args.backup:
        backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
        csv_path.rename(backup_path)
        print(f"Backed up original file to {backup_path}")

    filled.to_csv(csv_path, index_label='timestamp')
    print(f"Filled metrics for {symbol_id}: {before_count} -> {after_count} rows")
    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
