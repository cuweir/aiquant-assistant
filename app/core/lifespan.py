from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pandas as pd

from ..core.config import settings
from ..llm_providers import get_llm_strategy
from ..services.data_fetcher import data_fetcher_instance
from ..services.data_updater import DataUpdaterService
from ..services.order_executor import OrderExecutor
from ..services.parameter_manager import ParameterManager
from ..services.analysis_service import AnalysisService
from ..containers import container

scheduler = AsyncIOScheduler(timezone="UTC")


async def scheduled_analysis_task():
    """The function the scheduler will run periodically for market analysis."""
    print(f"\n--- ANALYSIS TASK RUNNING at {pd.Timestamp.now(tz='UTC')} ---")
    analysis_service = container.analysis_service
    try:
        # Run the first analysis immediately on start if needed, or just wait for schedule
        tasks = [
            analysis_service.generate_comprehensive_analysis(symbol_name=symbol)
            for symbol in settings.SYMBOLS_TO_MONITOR
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for symbol, result in zip(settings.SYMBOLS_TO_MONITOR, results):
            if isinstance(result, Exception):
                print(f"Error analyzing symbol {symbol}: {result}")

    except Exception as e:
        print(f"An unexpected error occurred in the scheduled_analysis_task: {e}")
    print(f"--- ANALYSIS TASK FINISHED at {pd.Timestamp.now(tz='UTC')} ---")


async def scheduled_data_update_task():
    """The function the scheduler will run to update local OHLCV data."""
    await container.data_updater.run_update()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup...")

    # --- Schedule Data Update Job ---
    # Runs every minute to keep the local database fresh.
    scheduler.add_job(
        scheduled_data_update_task,
        trigger=CronTrigger(minute="*/15", second="5"), # Runs at the start of every minute
        id="data_update_job",
        name="15-Minute OHLCV Data Update Job",
        replace_existing=True,
    )
    print("Scheduler: Minute OHLCV Data Update Job scheduled.")

    # --- Schedule Analysis Job ---
    if settings.SIGNAL_TIMEFRAME == "15m":
        # Run at 08:00:15, 08:15:15, etc.
        trigger = CronTrigger(minute="*/15", second="15")
        job_name = "15-Minute Analysis Job"
    elif settings.SIGNAL_TIMEFRAME == "1h":
        # Run at 08:00:15, 09:00:15, etc.
        # It will use the fresh data fetched by the 15-min updater at 08:00:05.
        trigger = CronTrigger(hour="*", minute="0", second="15")
        job_name = "Hourly Analysis Job"
    elif settings.SIGNAL_TIMEFRAME == "4h":
        # Run at 00:00:15, 04:00:15, 08:00:15, etc.
        trigger = CronTrigger(hour="*/4", minute="0", second="15")
        job_name = "4-Hour Analysis Job"
    else:
        raise ValueError(f"Unsupported SIGNAL_TIMEFRAME for scheduler: {settings.SIGNAL_TIMEFRAME}")

    scheduler.add_job(
        scheduled_analysis_task,
        trigger=trigger,
        id="market_analysis_job",
        name=job_name,
        replace_existing=True,
    )
    print(f"Scheduler: {job_name} scheduled.")

    scheduler.start()
    print("Scheduler started.")

    # Run the data update once on startup to ensure data is available.
    print("Running initial data update on startup...")
    await scheduled_data_update_task()
    print("Initial data update complete.")


    yield

    print("Application shutdown...")
    if scheduler.running:
        scheduler.shutdown()
        print("Scheduler shut down.")

    await data_fetcher_instance.close_exchange()
    await analysis_service.close_llm_resources()
    await order_executor.close_connections()
    print("Resources cleaned up.")