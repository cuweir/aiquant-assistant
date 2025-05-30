import os
from dotenv import load_dotenv

load_dotenv() # Loads variables from .env file into environment variables

class Settings:
    PROJECT_NAME: str = "AI Quant Assistant"
    PROJECT_VERSION: str = "0.1.0"

    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY_READONLY")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET_READONLY")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-pro")

    # Example strategy params (could be more complex)
    RSI_TIMEFRAME: str = "1h"
    RSI_PERIOD: int = 14
    RSI_OVERBOUGHT: int = 70
    RSI_OVERSOLD: int = 30
    SYMBOLS_TO_MONITOR: list[str] = ["BTC/USDT", "ETH/USDT"] # CCXT format for spot
    # For USDT-margined perpetual futures, use e.g., "BTC/USDT:USDT"
    # SYMBOLS_TO_MONITOR_FUTURES: list[str] = ["BTC/USDT:USDT"]


settings = Settings()

if not all([settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET, settings.GEMINI_API_KEY]):
    raise ValueError("One or more API keys are missing. Please set them in your .env file.")