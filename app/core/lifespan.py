from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pandas as pd
from sqlalchemy.orm import Session
from ..db.session import SessionLocal

from ..core.config import settings

from ..containers import container

scheduler = AsyncIOScheduler(timezone="UTC")


async def scheduled_analysis_task():
    """The function the scheduler will run periodically for market analysis."""
    print(f"\n--- ANALYSIS TASK RUNNING at {pd.Timestamp.now(tz='UTC')} ---")
    analysis_service = container.analysis_service
    try:
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

async def scheduled_position_sync_task():
    """The function the scheduler will run to sync position states."""
    db: Session = SessionLocal()
    try:
        await container.trading_service.check_and_sync_positions(db)
    except Exception as e:
        print(f"An unexpected error occurred in the scheduled_position_sync_task: {e}")
    finally:
        db.close()

# [NEW] Create a scheduled job for cleaning up old tasks
def scheduled_task_cleanup_job():
    """The function the scheduler will run to clean up old task results from memory."""
    print(f"\n--- TASK CLEANUP JOB RUNNING at {pd.Timestamp.now(tz='UTC')} ---")
    try:
        container.job_manager.cleanup_old_tasks()
    except Exception as e:
        print(f"An unexpected error occurred in the scheduled_task_cleanup_job: {e}")
    print(f"--- TASK CLEANUP JOB FINISHED at {pd.Timestamp.now(tz='UTC')} ---")


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
    scheduler.add_job(
        scheduled_data_update_task,
        trigger=CronTrigger(minute="*/15", second="5"),
        id="data_update_job",
        name="15-Minute OHLCV Data Update Job"
    )
    print("Scheduler: 15-Minute Data Update Job scheduled.")

    scheduler.add_job(
        scheduled_analysis_task,
        trigger=CronTrigger(hour="*", minute="0", second="15"),
        id="market_analysis_job",
        name="Hourly Analysis Job"
    )
    print(f"Scheduler: Hourly Analysis Job scheduled.")

    scheduler.add_job(
        scheduled_task_cleanup_job,
        trigger=CronTrigger(hour="*", minute="30", second="0"), # Runs at half-past every hour
        id="task_cleanup_job",
        name="Hourly Task Cleanup Job"
    )
    print("Scheduler: Hourly Task Cleanup Job scheduled.")

    scheduler.add_job(
        scheduled_position_sync_task,
        trigger=CronTrigger(minute="*", second="30"), # Runs every minute at the 30s mark
        id="position_sync_job",
        name="1-Minute Position State Sync Job"
    )
    print("Scheduler: 1-Minute Position Sync Job scheduled.")

    scheduler.start()
    print("Scheduler started.")

    # --- [CRITICAL FIX] IMMEDIATE VALIDATION LOGIC ---
    print("\n" + "=" * 50)
    print("RUNNING IMMEDIATE VALIDATION TASKS ON STARTUP")
    print("=" * 50 + "\n")

    print("--- Running initial data update... ---")
    await scheduled_data_update_task()
    print("--- Initial data update complete. ---\n")

    await asyncio.sleep(2)

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