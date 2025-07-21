from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pandas as pd

from ..core.config import settings
from ..services.analysis_service import analysis_service
from ..services.data_fetcher import data_fetcher_instance

scheduler = AsyncIOScheduler()


async def scheduled_analysis_task():
    """The function the scheduler will run periodically."""
    print(f"\n--- SCHEDULLED TASK RUNNING at {pd.Timestamp.now(tz='UTC')} ---")
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
    print(f"--- SCHEDULLED TASK FINISHED at {pd.Timestamp.now(tz='UTC')} ---")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup...")

    # Schedule the job to run based on the SIGNAL_TIMEFRAME
    if settings.SIGNAL_TIMEFRAME == "15m":
        trigger = CronTrigger(minute="*/15", second="5")
        job_name = "15-Minute Analysis Job"
    elif settings.SIGNAL_TIMEFRAME == "1h":
        trigger = CronTrigger(minute="0", second="5")
        job_name = "Hourly Analysis Job"
    else:  # Fallback or error
        raise ValueError(f"Unsupported SIGNAL_TIMEFRAME for scheduler: {settings.SIGNAL_TIMEFRAME}")

    scheduler.add_job(
        scheduled_analysis_task,
        trigger=trigger,
        id="market_analysis_job",
        name=job_name,
        replace_existing=True,
    )

    scheduler.start()
    print(f"Scheduler started. {job_name} scheduled.")

    yield

    print("Application shutdown...")
    if scheduler.running:
        scheduler.shutdown()
        print("Scheduler shut down.")

    await data_fetcher_instance.close_exchange()
    await analysis_service.close_llm_resources()
    print("Resources cleaned up.")