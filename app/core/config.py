import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "AI Quant Assistant"
    PROJECT_VERSION: str = "0.3.1" # Version up for strategy pattern

    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY_READONLY")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET_READONLY")

    # --- LLM Provider Configuration ---
    # Set this in your .env file. For now, it will effectively always be "gemini".
    ACTIVE_LLM_PROVIDER: str = os.getenv("ACTIVE_LLM_PROVIDER", "gemini").lower()

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")

    # --- OpenAI Configuration (Commented out as only Gemini is implemented for now) ---
    # OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    # OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

    # --- Strategy Parameters (from your previous version) ---
    DEFAULT_TIMEFRAME: str = "1h"
    SYMBOLS_TO_MONITOR: list[str] = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    ]
    RSI_PERIOD: int = 14
    RSI_OVERBOUGHT: int = 70
    RSI_OVERSOLD: int = 30
    MACD_FAST_PERIOD: int = 12
    MACD_SLOW_PERIOD: int = 26
    MACD_SIGNAL_PERIOD: int = 9
    MA_SHORT_PERIOD: int = 10
    MA_LONG_PERIOD: int = 30
    BBANDS_PERIOD: int = 20
    BBANDS_STD_DEV: int = 2
    ATR_STOP_LOSS_MULTIPLIER: float = 2.0
    ATR_TAKE_PROFIT_MULTIPLIER: float = 4.0
    WEIGHT_RSI_SIGNAL: int = 1
    WEIGHT_MACD_CROSS: int = 2
    WEIGHT_MA_CROSS: int = 2
    WEIGHT_BBANDS_BREAKOUT: int = 1
    WEIGHT_VOLUME_CONFIRMATION: int = 1
    BUY_SCORE_THRESHOLD: int = 2
    SELL_SCORE_THRESHOLD: int = -2
    ATR_PERIOD: int = 14


settings = Settings()

# Validate necessary API keys based on active provider
if settings.ACTIVE_LLM_PROVIDER == "gemini" and not settings.GEMINI_API_KEY:
    raise ValueError("ACTIVE_LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is missing in .env file.")
# if settings.ACTIVE_LLM_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
#     raise ValueError("ACTIVE_LLM_PROVIDER is 'openai' but OPENAI_API_KEY is missing in .env file.")

if not all([settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET]):
    raise ValueError("Binance API keys are missing. Please set them in your .env file.")

print(f"AI Quant Assistant configured with LLM Provider: {settings.ACTIVE_LLM_PROVIDER.upper()}")
print(f"Using LLM Model: {settings.GEMINI_MODEL_NAME if settings.ACTIVE_LLM_PROVIDER == 'gemini' else 'N/A'}")
# print(f"Using LLM Model: {settings.OPENAI_MODEL_NAME if settings.ACTIVE_LLM_PROVIDER == 'openai' else 'N/A'}")