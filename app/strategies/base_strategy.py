from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class TradingStrategy(ABC):
    @abstractmethod
    async def generate_signals(self, df_signal: pd.DataFrame, df_regime: pd.DataFrame) -> Dict[str, Any] | None:
        """
        The core method for a strategy to generate trading signals.

        Args:
            df_signal: DataFrame for the primary signal timeframe.
            df_regime: DataFrame for the higher timeframe regime filter.

        Returns:
            A dictionary containing the analysis result (e.g., signal, price, sl),
            or None if no signal is generated.
        """
        pass