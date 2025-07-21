from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class TradingStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @abstractmethod
    async def generate_signals(self, df_signal: pd.DataFrame, df_trend: pd.DataFrame) -> Dict[str, Any]:
        """
        Takes a signal timeframe DataFrame and a trend timeframe DataFrame,
        and returns a dictionary of analysis results.
        """
        pass