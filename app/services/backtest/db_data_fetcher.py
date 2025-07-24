# services/backtest/db_data_fetcher.py

import pandas as pd
from sqlalchemy import create_engine, select
import datetime

from ...db.models import HistoricalOhlcv, Symbol
from ...core.config import settings


def fetch_df_from_postgres(
        symbol_name: str,
        timeframe: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime
) -> pd.DataFrame:
    """
    Connects to the PostgreSQL database and fetches OHLCV data for a given
    symbol and timeframe into a pandas DataFrame.
    """
    print(f"Fetching data for {symbol_name} ({timeframe}) from PostgreSQL...")

    engine = create_engine(settings.DATABASE_URL)

    with engine.connect() as connection:
        # First, get the symbol_id for the given symbol_name
        symbol_query = select(Symbol.id).where(Symbol.name == symbol_name)
        symbol_result = connection.execute(symbol_query).scalar_one_or_none()
        if not symbol_result:
            raise ValueError(f"Symbol '{symbol_name}' not found in the database.")
        symbol_id = symbol_result

        # Now, build the main query for OHLCV data
        query = (
            select(
                HistoricalOhlcv.open_time,
                HistoricalOhlcv.open,
                HistoricalOhlcv.high,
                HistoricalOhlcv.low,
                HistoricalOhlcv.close,
                HistoricalOhlcv.volume
            )
            .where(
                HistoricalOhlcv.symbol_id == symbol_id,
                HistoricalOhlcv.timeframe == timeframe,
                HistoricalOhlcv.open_time >= start_date,
                HistoricalOhlcv.open_time <= end_date
            )
            .order_by(HistoricalOhlcv.open_time.asc())
        )

        # Execute the query and load into a DataFrame
        df = pd.read_sql_query(query, connection, index_col='open_time')

        if df.empty:
            raise ValueError(f"No data found for {symbol_name} ({timeframe}) in the specified date range.")

        # Backtrader expects the index to be named 'datetime'
        df.index.name = 'datetime'

        # Convert decimal columns from DB to float for indicator calculations
        cols_to_convert = ['open', 'high', 'low', 'close', 'volume']
        for col in cols_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"  > Fetched {len(df)} rows.")
        return df