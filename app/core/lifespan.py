from contextlib import asynccontextmanager
import asyncio
import pandas as pd
from fastapi import FastAPI

from ..core.config import settings
from ..services.analysis_service import analysis_service
from ..services.data_fetcher import data_fetcher_instance

async def monitor_markets_periodically():
    """Background task to periodically check markets."""
    while True:
        print(f"\n[{pd.Timestamp.now(tz='UTC')}] Running periodic market check...")
        try:
            for symbol in settings.SYMBOLS_TO_MONITOR:
                print(f"Comprehensive check for symbol: {symbol}")
                await analysis_service.generate_comprehensive_analysis(
                    symbol_name=symbol,
                    timeframe=settings.DEFAULT_TIMEFRAME
                )
        except Exception as e:
            print(f"Error in monitoring_markets_periodically: {e}")
            import traceback
            traceback.print_exc()
        await asyncio.sleep(60 * 15)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup...")
    app.state.monitoring_task = asyncio.create_task(monitor_markets_periodically())
    print("Market monitoring task started.")
    yield
    print("Application shutdown...")
    app.state.monitoring_task.cancel()
    try:
        await app.state.monitoring_task
    except asyncio.CancelledError:
        print("Market monitoring task cancelled.")
    await data_fetcher_instance.close_exchange()
    await analysis_service.close_llm_resources()
    print("Resources cleaned up.")