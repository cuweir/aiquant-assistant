import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
import asyncio

from .core.config import settings
from .services.data_fetcher import data_fetcher_instance, DataFetcher
from .services.trading_logic import trading_logic_service_instance, TradingLogicService, ai_analysis_cache
from .core.llm_client import llm_client_instance, LLMClient
from .models.schemas import SignalInput, AIAnalysisOutput, AllAnalysesOutput


# Lifespan manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize resources
    print("Application startup...")
    # You could initialize data_fetcher_instance and llm_client_instance here if not global
    # Start background tasks if any
    app.state.data_fetcher = data_fetcher_instance  # Make accessible via app.state
    app.state.llm_client = llm_client_instance
    app.state.trading_logic = trading_logic_service_instance
    app.state.monitoring_task = asyncio.create_task(monitor_markets_periodically())
    print("Market monitoring task started.")
    yield
    # Shutdown: Cleanup resources
    print("Application shutdown...")
    app.state.monitoring_task.cancel()
    try:
        await app.state.monitoring_task
    except asyncio.CancelledError:
        print("Market monitoring task cancelled.")
    await app.state.data_fetcher.close_exchange()
    await app.state.llm_client.close_http_client()
    print("Resources cleaned up.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan
)


async def monitor_markets_periodically():
    """
    Background task to periodically check for signals.
    This is a simplified version. In a real app, you'd use a proper scheduler
    or a more robust async loop management.
    """
    while True:
        print(f"\n[{pd.Timestamp.now()}] Running periodic market check...")
        try:
            for symbol_ccxt_format in settings.SYMBOLS_TO_MONITOR:  # Use CCXT standard format
                # Example: for spot "BTC/USDT", for futures "BTC/USDT:USDT"
                # Ensure your SYMBOLS_TO_MONITOR list in config.py uses the correct CCXT format
                # for the market type (spot/future) you intend to query.
                print(f"Checking symbol: {symbol_ccxt_format}")
                analysis_result = await trading_logic_service_instance.check_rsi_signal(
                    symbol=symbol_ccxt_format,
                    timeframe=settings.RSI_TIMEFRAME
                )
                if analysis_result:
                    print(f"AI Analysis generated for {symbol_ccxt_format}:")
                    # print(analysis_result.get('ai_analysis')) # Don't print full prompt in production
        except Exception as e:
            print(f"Error in monitoring_markets_periodically: {e}")
        await asyncio.sleep(60 * 5)  # Check every 5 minutes (adjust as needed)


@app.get("/", summary="Root endpoint, returns project info")
async def read_root():
    return {"project": settings.PROJECT_NAME, "version": settings.PROJECT_VERSION}


@app.post("/trigger-analysis", response_model=AIAnalysisOutput, summary="Manually trigger AI analysis for a symbol")
async def trigger_ai_analysis(signal_input: SignalInput):
    """
    Manually triggers a local signal check and subsequent AI analysis if a signal is found.
    This is more for testing; the primary analysis is done by the background task.
    """
    analysis_result = await trading_logic_service_instance.check_rsi_signal(
        symbol=signal_input.symbol,  # Expect CCXT format, e.g., "BTC/USDT" or "BTC/USDT:USDT"
        timeframe=signal_input.timeframe
    )
    if not analysis_result:
        raise HTTPException(status_code=404,
                            detail=f"No local signal found or AI analysis could not be generated for {signal_input.symbol} on {signal_input.timeframe}")

    # Convert dict to Pydantic model for response
    return AIAnalysisOutput(
        timestamp=analysis_result["timestamp"],
        symbol=signal_input.symbol,  # Use the input symbol for consistency in response
        timeframe=signal_input.timeframe,
        local_signal=analysis_result["local_signal"],
        rsi=analysis_result["rsi"],
        price=analysis_result["price"],
        ai_analysis=analysis_result["ai_analysis"]
        # prompt=analysis_result.get("prompt") # Optionally include
    )


@app.get("/get-latest-analysis", response_model=AIAnalysisOutput, summary="Get latest cached AI analysis")
async def get_latest_analysis(symbol: str, timeframe: str, signal_type: str):
    # Note: This requires knowing the signal_type. A better approach might be to
    # just get the latest for a symbol/timeframe regardless of signal_type.
    cached = await trading_logic_service_instance.get_cached_analysis(symbol, timeframe, signal_type)
    if not cached:
        raise HTTPException(status_code=404, detail="No cached analysis found for the given criteria.")
    return AIAnalysisOutput(**cached)


@app.get("/get-all-analyses", response_model=AllAnalysesOutput, summary="Get all cached AI analyses")
async def get_all_analyses_endpoint():
    all_data = await trading_logic_service_instance.get_all_cached_analyses()
    # Pydantic validation will occur if the structure matches
    # We might need to re-structure 'all_data' if it's not directly compatible
    # For now, assume ai_analysis_cache stores data in AIAnalysisOutput compatible dicts
    valid_analyses = {}
    for key, value_dict in all_data.items():
        try:
            # Add symbol and timeframe from key if not in value_dict, or ensure they are present
            parts = key.split('_')
            if len(parts) >= 2:  # symbol_timeframe_...
                value_dict.setdefault('symbol', parts[0])
                value_dict.setdefault('timeframe', parts[1])
            valid_analyses[key] = AIAnalysisOutput(**value_dict)
        except Exception as e:
            print(f"Skipping cache entry {key} due to validation error: {e}")

    return AllAnalysesOutput(analyses=valid_analyses)

# To run: uvicorn app.main:app --reload