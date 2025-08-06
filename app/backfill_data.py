import asyncio
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import select, func as sql_func, text
import ccxt.async_support as ccxt
from typing import List, Set
import datetime
import time

# To run this script standalone, we need to adjust the Python path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from app.core.config import settings
from app.db.session import SessionLocal
from app.db.models import Symbol, HistoricalOhlcv
from app.services.data_fetcher import data_fetcher_instance


class BackfillService:
    """
    A service dedicated to backfilling historical OHLCV data from an exchange
    into the local database.
    """

    def __init__(self, symbols_to_backfill: List[str], timeframes_to_backfill: List[str]):
        if not symbols_to_backfill or not timeframes_to_backfill:
            raise ValueError("Symbols and timeframes for backfill cannot be empty.")

        self.symbols = symbols_to_backfill
        self.timeframes = timeframes_to_backfill
        self.exchange = data_fetcher_instance.exchange
        print(f"BackfillService initialized for symbols: {self.symbols}, timeframes: {self.timeframes}")

    async def run_backfill(self, start_date_str: str):
        """
        Main method to run the backfill process.

        Args:
            start_date_str: The starting date for backfilling in 'YYYY-MM-DD' format.
        """
        try:
            start_timestamp_ms = int(datetime.datetime.strptime(start_date_str, "%Y-%m-%d").replace(
                tzinfo=datetime.timezone.utc).timestamp() * 1000)
        except ValueError:
            print("Error: Invalid date format. Please use 'YYYY-MM-DD'.")
            return

        db: Session = SessionLocal()
        try:
            # Ensure all symbol records exist in the DB
            symbol_records = self._get_or_create_symbols(db)

            for symbol_record in symbol_records:
                for timeframe in self.timeframes:
                    await self._backfill_symbol_timeframe(db, symbol_record, timeframe, start_timestamp_ms)

            print("\n--- BACKFILL PROCESS COMPLETED SUCCESSFULLY ---")

        except Exception as e:
            print(f"An unexpected error occurred during the backfill process: {e}")
            import traceback
            traceback.print_exc()
        finally:
            db.close()
            await self.exchange.close()
            print("Database connection and exchange connection closed.")

    def _get_or_create_symbols(self, db: Session) -> List[Symbol]:
        """Ensures all symbols for backfill exist in the database."""
        existing_symbols = {s.name: s for s in db.query(Symbol).filter(Symbol.name.in_(self.symbols)).all()}
        new_symbols_to_add = []
        for symbol_name in self.symbols:
            if symbol_name not in existing_symbols:
                print(f"Symbol '{symbol_name}' not found in DB, creating new entry.")
                new_symbols_to_add.append(Symbol(name=symbol_name))

        if new_symbols_to_add:
            db.add_all(new_symbols_to_add)
            db.commit()

        return list(db.query(Symbol).filter(Symbol.name.in_(self.symbols)).all())

    async def _backfill_symbol_timeframe(self, db: Session, symbol: Symbol, timeframe: str, start_timestamp_ms: int):
        """
        Handles the backfilling for a single symbol/timeframe pair.
        It fetches data in chunks from the start_date up to the earliest data point in the DB.
        """
        print(f"\n--- Starting backfill for {symbol.name} ({timeframe}) ---")

        # Find the earliest data we already have in our database for this pair
        earliest_db_time_obj = db.query(sql_func.min(HistoricalOhlcv.open_time)).filter(
            HistoricalOhlcv.symbol_id == symbol.id,
            HistoricalOhlcv.timeframe == timeframe
        ).scalar()

        end_timestamp_ms = None
        if earliest_db_time_obj:
            end_timestamp_ms = int(earliest_db_time_obj.timestamp() * 1000)
            print(f"Earliest data in DB is from {earliest_db_time_obj}. Will fetch data before this date.")
        else:
            print("No existing data found in DB. Will fetch up to the current time.")
            # If no data exists, end_timestamp_ms remains None, so fetch_ohlcv fetches up to now.

        current_timestamp_ms = start_timestamp_ms
        limit = 1000  # Max chunk size per request

        while True:
            # Check if we have reached the end of the required backfill range
            if end_timestamp_ms and current_timestamp_ms >= end_timestamp_ms:
                print("Reached the earliest data point in the database. Backfill for this pair is complete.")
                break

            try:
                print(
                    f"Fetching chunk for {symbol.name}/{timeframe} starting from {pd.to_datetime(current_timestamp_ms, unit='ms', utc=True)}...")

                # Fetch a chunk of OHLCV data
                ohlcv = await self.exchange.fetch_ohlcv(symbol.name, timeframe, since=current_timestamp_ms, limit=limit)

                if not ohlcv or len(ohlcv) <= 1:
                    print("No more historical data returned from the exchange. Backfill for this pair is complete.")
                    break

                new_records = []
                for row in ohlcv:
                    open_time = pd.to_datetime(row[0], unit='ms', utc=True)
                    # Create a new record object
                    new_records.append(
                        HistoricalOhlcv(
                            symbol_id=symbol.id,
                            timeframe=timeframe,
                            open_time=open_time.to_pydatetime(),
                            open=row[1],
                            high=row[2],
                            low=row[3],
                            close=row[4],
                            volume=row[5]
                        )
                    )

                # Use a bulk insert with a mechanism to ignore duplicates
                if new_records:
                    # For PostgreSQL, we can use ON CONFLICT DO NOTHING
                    # This requires a more direct way to insert than standard session.add_all
                    from sqlalchemy.dialects.postgresql import insert

                    insert_stmt = insert(HistoricalOhlcv).values([
                        {
                            'symbol_id': r.symbol_id, 'timeframe': r.timeframe, 'open_time': r.open_time,
                            'open': r.open, 'high': r.high, 'low': r.low, 'close': r.close, 'volume': r.volume
                        } for r in new_records
                    ])
                    # This tells PostgreSQL: if a row with the same unique constraint keys already exists, do nothing.
                    on_conflict_stmt = insert_stmt.on_conflict_do_nothing(
                        index_elements=['symbol_id', 'timeframe', 'open_time']
                    )
                    db.execute(on_conflict_stmt)
                    db.commit()
                    print(f"  > Saved/updated {len(new_records)} candles.")

                # Update the timestamp for the next chunk request
                # The next request should start after the last candle of the current chunk
                last_candle_timestamp = ohlcv[-1][0]
                current_timestamp_ms = last_candle_timestamp + self.exchange.parse_timeframe(timeframe) * 1000

                # Respect exchange rate limits
                await asyncio.sleep(self.exchange.rateLimit / 1000)

            except ccxt.NetworkError as e:
                print(f"Network error, retrying in 10 seconds... Error: {e}")
                await asyncio.sleep(10)
            except Exception as e:
                print(f"An error occurred in the fetch loop: {e}")
                break

        print(f"--- Finished backfill for {symbol.name} ({timeframe}) ---")


async def main():
    # --- Configuration ---
    # Define which assets and timeframes you want to backfill
    SYMBOLS_TO_FILL = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT"]
    TIMEFRAMES_TO_FILL = ["15m", "1h", "4h"]

    # Define the start date for the backfill
    # IMPORTANT: Exchanges have limits on how far back you can go.
    # For Binance, 2017 is generally a safe start for BTC.
    START_DATE = "2022-01-01"

    print("=" * 50)
    print("Starting Historical Data Backfill Process")
    print(f"Symbols: {SYMBOLS_TO_FILL}")
    print(f"Timeframes: {TIMEFRAMES_TO_FILL}")
    print(f"Start Date: {START_DATE}")
    print("=" * 50)

    backfiller = BackfillService(
        symbols_to_backfill=SYMBOLS_TO_FILL,
        timeframes_to_backfill=TIMEFRAMES_TO_FILL
    )
    await backfiller.run_backfill(start_date_str=START_DATE)


if __name__ == "__main__":
    # This allows running the script directly
    asyncio.run(main())