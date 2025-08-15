import asyncio
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import select, func as sql_func
import ccxt.async_support as ccxt
from typing import List, Set

from ..core.config import settings
from ..db.session import SessionLocal
from ..db.models import Symbol, HistoricalOhlcv
from .data_fetcher import data_fetcher_instance


class DataUpdaterService:
    """
    A service dedicated to updating and maintaining the local historical OHLCV database.
    It runs periodically to fetch the latest candle data and backfill if necessary.
    """

    def __init__(self):
        self.symbols_to_monitor: List[str] = settings.SYMBOLS_TO_MONITOR

        # Collect all unique timeframes from settings to avoid hardcoding
        self.timeframes_to_monitor: Set[str] = {
            settings.SIGNAL_TIMEFRAME,
            settings.TREND_TIMEFRAME_SHORT,
            "15m",  # Ensure 15m is always included for short-term data needs
        }
        print("DataUpdaterService initialized.")
        print(f"Monitoring symbols: {self.symbols_to_monitor}")
        print(f"Ensuring data is available for timeframes: {sorted(list(self.timeframes_to_monitor))}")

    async def run_update(self):
        """
        Main method to run the update cycle.
        It fetches data for all monitored symbols and timeframes.
        """
        print(f"\n--- DATA UPDATER TASK RUNNING at {pd.Timestamp.now(tz='UTC')} ---")
        db: Session = SessionLocal()
        try:
            symbol_records = self._get_or_create_symbols(db)

            update_tasks = []
            for symbol_record in symbol_records:
                for timeframe in self.timeframes_to_monitor:
                    update_tasks.append(
                        self._update_symbol_timeframe(db, symbol_record, timeframe)
                    )

            await asyncio.gather(*update_tasks)

            db.commit()
            print(f"--- DATA UPDATER TASK FINISHED SUCCESSFULLY at {pd.Timestamp.now(tz='UTC')} ---")

        except Exception as e:
            print(f"An error occurred in DataUpdaterService: {e}")
            import traceback
            traceback.print_exc()
            db.rollback()
        finally:
            db.close()

    def _get_or_create_symbols(self, db: Session) -> List[Symbol]:
        """
        Ensures all symbols from the config exist in the database.
        """
        existing_symbols = {s.name: s for s in db.query(Symbol).all()}
        new_symbols_to_add = []
        for symbol_name in self.symbols_to_monitor:
            if symbol_name not in existing_symbols:
                print(f"Symbol '{symbol_name}' not found in DB, creating new entry.")
                new_symbols_to_add.append(Symbol(name=symbol_name))

        if new_symbols_to_add:
            db.add_all(new_symbols_to_add)
            db.commit()

        return list(db.query(Symbol).filter(Symbol.name.in_(self.symbols_to_monitor)).all())

    async def _update_symbol_timeframe(self, db: Session, symbol: Symbol, timeframe: str):
        """
        Handles the update for a single symbol/timeframe pair.
        """
        try:
            latest_timestamp_in_db = self._get_latest_timestamp(db, symbol.id, timeframe)

            since = None
            limit = 1000
            if latest_timestamp_in_db:
                since = int(latest_timestamp_in_db.timestamp() * 1000)
                limit = 100

            print(
                f"Fetching {symbol.name}/{timeframe}. Since: {pd.to_datetime(since, unit='ms', utc=True) if since else 'Beginning'}")
            ohlcv = await data_fetcher_instance.exchange.fetch_ohlcv(
                symbol=symbol.name, timeframe=timeframe, since=since, limit=limit
            )

            if not ohlcv:
                print(f"No new data returned for {symbol.name}/{timeframe}.")
                return

            new_records = []
            for row in ohlcv:
                open_time = pd.to_datetime(row[0], unit='ms', utc=True)

                # [THE FIX] Skip the record if its open_time is less than or equal to the latest one we already have.
                # This prevents attempting to insert a duplicate of the last known candle.
                if latest_timestamp_in_db and open_time <= latest_timestamp_in_db:
                    continue

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

            if new_records:
                db.add_all(new_records)
                print(f"  > Saved {len(new_records)} new candles for {symbol.name}/{timeframe}.")
            else:
                print(f"  > No new unique candles to save for {symbol.name}/{timeframe}.")

        except ccxt.NetworkError as e:
            print(f"CCXT Network Error for {symbol.name}/{timeframe}: {e}")
        except ccxt.ExchangeError as e:
            print(f"CCXT Exchange Error for {symbol.name}/{timeframe}: {e}")
        except Exception as e:
            print(f"General Error updating {symbol.name}/{timeframe}: {e}")

    def _get_latest_timestamp(self, db: Session, symbol_id: int, timeframe: str) -> pd.Timestamp | None:
        """
        Finds the most recent candle's open_time for a given symbol and timeframe.
        """
        latest_time = db.query(sql_func.max(HistoricalOhlcv.open_time)).filter(
            HistoricalOhlcv.symbol_id == symbol_id,
            HistoricalOhlcv.timeframe == timeframe
        ).scalar()

        return pd.to_datetime(latest_time, utc=True) if latest_time else None