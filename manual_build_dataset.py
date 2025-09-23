#!/usr/bin/env python
"""Helper script to generate a dataset bundle for model training."""

import argparse
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.ml_pipeline.dataset_builder import DatasetBuilder, save_bundle
from app.ml_pipeline.schemas import (
    FeatureConfig,
    FeatureSourceKind,
    FeatureSourceSpec,
    FillMethod,
    LabelConfig,
    LabelMethod,
    ScalingMethod,
)
from app.services.backtest.db_data_fetcher import fetch_df_from_postgres


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML dataset bundle")
    parser.add_argument("symbol", help="Trading pair, e.g. BTC/USDT")
    parser.add_argument("timeframe", help="Timeframe to fetch, e.g. 1h")
    parser.add_argument("--lookback-days", type=int, default=180, help="Number of days to pull from the database")
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date (UTC, YYYY-MM-DD or ISO8601). Defaults to now.",
    )
    parser.add_argument("--horizon", type=int, default=12, help="Label horizon in bars")
    parser.add_argument("--output", type=Path, default=Path("data/processed"), help="Output directory for the dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end_date:
        end = pd.to_datetime(args.end_date, utc=True)
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
    else:
        end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=args.lookback_days)
    price_df = fetch_df_from_postgres(args.symbol, args.timeframe, start, end)

    feature_config = FeatureConfig(
        sources=[
            FeatureSourceSpec(
                name="technical_indicators",
                kind=FeatureSourceKind.TECHNICAL,
                timeframe=args.timeframe,
                params={
                    "indicators": [
                        {"name": "rsi", "length": 14},
                        {"name": "macd", "fast": 12, "slow": 26, "signal": 9},
                        {"name": "adx", "length": 14},
                        {"name": "atr", "length": 14},
                        {"name": "bollinger_bandwidth", "length": 20, "std": 2},
                    ]
                },
            ),
            FeatureSourceSpec(
                name="funding_rate",
                kind=FeatureSourceKind.MARKET_MICRO,
                timeframe=args.timeframe,
                params={"symbol": args.symbol},
            ),
            FeatureSourceSpec(
                name="open_interest",
                kind=FeatureSourceKind.MARKET_MICRO,
                timeframe=args.timeframe,
                params={"symbol": args.symbol},
            ),
            FeatureSourceSpec(name="sentiment_index", kind=FeatureSourceKind.SENTIMENT, timeframe=args.timeframe),
        ],
        lags=[1, 4, 12],
        scaling=ScalingMethod.ZSCORE,
        fill_method=FillMethod.FILL_FORWARD,
    )

    label_config = LabelConfig(
        method=LabelMethod.FIXED_HORIZON,
        horizon=args.horizon,
        take_profit=0.01,
        stop_loss=0.01,
        neutral_zone=0.002,
        directional=True,
    )

    builder = DatasetBuilder(feature_config, label_config)
    bundle = builder.build(args.symbol, args.timeframe, price_df)
    save_dir = save_bundle(bundle, args.output)
    print(f"Dataset saved to {save_dir} with {bundle.meta['rows']} rows")
    print(f"Database URL (masked): {settings.DATABASE_URL[:10]}...")


if __name__ == "__main__":
    main()
