from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class TradingStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @abstractmethod
    async def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Takes a DataFrame with OHLCV data and returns a dictionary of analysis results.

        The returned dictionary should contain at least:
        - 'signals_details': A list of individual indicator signals.
        - 'total_score': The final composite score.
        - 'current_price': The price at which signals were generated.
        - 'suggested_sl': Suggested stop loss price.
        - 'suggested_tp': Suggested take profit price.
        - 'overall_signal': The final signal label (e.g., POTENTIAL_BUY).
        """
        pass