from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pandas as pd

from ..core.config import settings

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
    """
        The application's lifespan manager.
        """
    print("Application startup...")
    # Initialize services
    await container.order_executor.initialize()
    print("Services initialized successfully.")

    # --- Scheduler Setup ---
    # Data update job remains every 15 minutes
    scheduler.add_job(
        scheduled_data_update_task,
        trigger=CronTrigger(minute="*/15", second="5"),
        id="data_update_job",
        name="15-Minute OHLCV Data Update Job"
    )
    print("Scheduler: 15-Minute Data Update Job scheduled.")

    # Production analysis job remains every hour
    scheduler.add_job(
        scheduled_analysis_task,
        trigger=CronTrigger(hour="*", minute="0", second="15"),
        id="market_analysis_job",
        name="Hourly Analysis Job"
    )
    print(f"Scheduler: Hourly Analysis Job scheduled.")

    scheduler.start()
    print("Scheduler started.")

    # --- [CRITICAL FIX] IMMEDIATE VALIDATION LOGIC ---
    # We will run the tasks once on startup to allow for immediate verification.

    print("\n" + "=" * 50)
    print("RUNNING IMMEDIATE VALIDATION TASKS ON STARTUP")
    print("=" * 50 + "\n")

    # Run initial data update first
    print("--- Running initial data update... ---")
    await scheduled_data_update_task()
    print("--- Initial data update complete. ---\n")

    # Add a small delay to ensure everything is settled
    await asyncio.sleep(2)

    # Then, run the analysis task to see the results immediately
    print("--- Running initial analysis task for immediate verification... ---")
    await scheduled_analysis_task()
    print("--- Initial analysis task complete. The system will now run on its schedule. ---")
    print("=" * 50 + "\n")

    yield

    print("Application shutdown...")
    if scheduler.running:
        scheduler.shutdown()
        print("Scheduler shut down.")
    await container.order_executor.close_connections()
    print("Resources cleaned up.")