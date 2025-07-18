import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "AI Quant Assistant"
    PROJECT_VERSION: str = "0.4.0" # Version up for DB persistence
    SERVER_PUBLIC_IP: str = os.getenv("SERVER_PUBLIC_IP")

    # --- Database ---
    DATABASE_URL: str = os.getenv("DATABASE_URL")

    # --- Binance ---
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY_READONLY")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET_READONLY")

    # --- LLM Provider Configuration ---
    ACTIVE_LLM_PROVIDER: str = os.getenv("ACTIVE_LLM_PROVIDER", "gemini").lower()
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")

    # --- OpenAI Configuration (Commented out as only Gemini is implemented for now) ---
    # OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    # OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

    # --- Strategy & Exit Parameters ---
    DEFAULT_TIMEFRAME: str = "1h"
    SYMBOLS_TO_MONITOR: list[str] = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    ]
    # ... (RSI, MACD, MA, BBANDS parameters as before) ...
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
    WEIGHT_VOLUME_CONFIRMATION: int = 1
    ATR_PERIOD: int = 14
    ATR_STOP_LOSS_MULTIPLIER: float = 2.0
    RISK_REWARD_RATIO: float = 2.0

    # --- Scoring ---
    WEIGHT_RSI_SIGNAL: int = 1
    WEIGHT_MACD_CROSS: int = 1
    WEIGHT_MA_CROSS: int = 2
    # Weights for the new "Stateful" scoring model
    WEIGHT_MA_STATE: int = 1  # Score for being in a golden/death cross state
    WEIGHT_MA_EVENT: int = 1  # Bonus score for the moment of crossover
    WEIGHT_MACD_STATE: int = 1  # Score for MACD line being above/below signal line
    WEIGHT_MACD_EVENT: int = 1  # Bonus score for the moment of crossover
    WEIGHT_RSI_EXTREME: int = 2  # Score for being in overbought/oversold
    WEIGHT_RSI_TREND: float = 0.5  # Score for being in bullish/bearish zone (above/below 50)
    WEIGHT_BBANDS_BREAKOUT: int = 1  # Score for breaking out of the bands
    # New Thresholds for the stateful model
    BUY_SCORE_THRESHOLD: int = 4
    SELL_SCORE_THRESHOLD: int = -4

settings = Settings()

# Validate necessary API keys based on active provider
if settings.ACTIVE_LLM_PROVIDER == "gemini" and not settings.GEMINI_API_KEY:
    raise ValueError("ACTIVE_LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is missing in .env file.")
# if settings.ACTIVE_LLM_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
#     raise ValueError("ACTIVE_LLM_PROVIDER is 'openai' but OPENAI_API_KEY is missing in .env file.")

if not all([settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET]):
    raise ValueError("Binance API keys are missing. Please set them in your .env file.")

if not settings.DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the .env file.")

print(f"AI Quant Assistant configured with LLM Provider: {settings.ACTIVE_LLM_PROVIDER.upper()}")
print(f"Using LLM Model: {settings.GEMINI_MODEL_NAME if settings.ACTIVE_LLM_PROVIDER == 'gemini' else 'N/A'}")
# print(f"Using LLM Model: {settings.OPENAI_MODEL_NAME if settings.ACTIVE_LLM_PROVIDER == 'openai' else 'N/A'}")