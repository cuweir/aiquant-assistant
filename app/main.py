import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import asyncio

from .core.config import settings
from .services.data_fetcher import data_fetcher_instance
from .services.trading_logic import trading_logic_service_instance  # This instance is now created with the strategy
from .models.schemas import SignalInput, AIAnalysisOutput, AllAnalysesOutput


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup...")
    # data_fetcher_instance is already initialized globally
    # trading_logic_service_instance is also initialized globally, and its __init__ now sets up the LLM strategy
    app.state.data_fetcher = data_fetcher_instance
    app.state.trading_logic = trading_logic_service_instance  # This now has the LLM strategy
    app.state.monitoring_task = asyncio.create_task(monitor_markets_periodically())
    print(f"Market monitoring task started for symbols: {settings.SYMBOLS_TO_MONITOR}")
    yield
    # Shutdown: Cleanup resources
    print("Application shutdown...")
    if app.state.monitoring_task:  # Check if task exists
        app.state.monitoring_task.cancel()
        try:
            await app.state.monitoring_task
        except asyncio.CancelledError:
            print("Market monitoring task cancelled.")
        except Exception as e:
            print(f"Error during monitoring task shutdown: {e}")

    if hasattr(app.state.data_fetcher, 'close_exchange'):  # Good practice to check
        await app.state.data_fetcher.close_exchange()

    # Close LLM resources via the trading_logic_service instance
    if hasattr(app.state.trading_logic, 'close_llm_resources') and callable(
            app.state.trading_logic.close_llm_resources):
        await app.state.trading_logic.close_llm_resources()
    else:
        print(
            "Trading logic service does not have a callable 'close_llm_resources' method or app.state.trading_logic not set.")
    print("Resources cleaned up.")


# ... (rest of main.py - endpoints - should remain the same as your previous version) ...
# Ensure your FastAPI app uses this lifespan manager:
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan
)


async def monitor_markets_periodically():
    while True:
        print(f"\n[{pd.Timestamp.now(tz='UTC')}] Running periodic market check (Comprehensive Analysis)...")
        try:
            for symbol_ccxt_format in settings.SYMBOLS_TO_MONITOR:
                print(f"Comprehensive check for symbol: {symbol_ccxt_format}")
                analysis_result_dict = await trading_logic_service_instance.generate_comprehensive_analysis(
                    symbol=symbol_ccxt_format,
                    timeframe=settings.DEFAULT_TIMEFRAME
                )
                if analysis_result_dict:
                    print(f"Comprehensive AI Analysis generated and cached for {symbol_ccxt_format}.")
        except Exception as e:
            print(f"Error in monitoring_markets_periodically: {e}")
            import traceback
            traceback.print_exc()
        await asyncio.sleep(60 * 15)


@app.get("/", summary="Root endpoint, returns project info")
async def read_root():
    return {"project": settings.PROJECT_NAME, "version": settings.PROJECT_VERSION}


@app.post("/trigger-analysis", response_model=AIAnalysisOutput, summary="Manually trigger comprehensive AI analysis")
async def trigger_comprehensive_ai_analysis(signal_input: SignalInput):
    analysis_result_dict = await trading_logic_service_instance.generate_comprehensive_analysis(
        symbol=signal_input.symbol,
        timeframe=signal_input.timeframe
    )
    if not analysis_result_dict:
        raise HTTPException(status_code=404,
                            detail=f"Comprehensive analysis could not be generated for {signal_input.symbol} on {signal_input.timeframe}")
    # The Pydantic model AIAnalysisOutput will automatically validate the dict.
    # If stop_loss or take_profit are missing (e.g., for neutral signals), they will be set to None
    # as defined in the schema (Optional[float] = None).
    # This conversion will work as long as the keys match.
    return AIAnalysisOutput(**analysis_result_dict)


@app.get("/get-all-analyses", response_model=AllAnalysesOutput, summary="Get all cached comprehensive AI analyses")
async def get_all_analyses_endpoint():
    all_data_dicts = await trading_logic_service_instance.get_all_cached_analyses()
    validated_analyses: dict[str, AIAnalysisOutput] = {}
    for key, value_dict in all_data_dicts.items():
        try:
            response_data = {k: v for k, v in value_dict.items() if k != 'details'}
            # Pydantic will handle missing optional fields like stop_loss/take_profit
            # by setting them to their default (None).
            validated_analyses[key] = AIAnalysisOutput(**response_data)
        except Exception as e:
            print(f"Skipping cache entry {key} due to data error: {e}. Data: {value_dict}")
    return AllAnalysesOutput(analyses=validated_analyses)