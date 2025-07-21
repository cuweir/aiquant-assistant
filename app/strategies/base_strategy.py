from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class TradingStrategy(ABC):
    @abstractmethod
    async def generate_signals(
        self,
        df_signal: pd.DataFrame,
        df_trend_short: pd.DataFrame,
        df_trend_long: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Takes DataFrames for signal, short-term trend, and long-term trend,
        and returns a dictionary of analysis results.
        """
        pass