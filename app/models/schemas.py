import pandas as pd
from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict, Any
import datetime

class SignalDetail(BaseModel):
    indicator: str
    signal: str
    value: Any # Can be float or string
    score_change: float

class AnalysisDetails(BaseModel):
    composite_score: float
    individual_signals_details: List[SignalDetail]

class SignalInput(BaseModel):
    symbol: str
    timeframe: str

class LocalSignalOutput(BaseModel):
    symbol: str
    timeframe: str
    signal_type: Optional[str] = None
    rsi_value: Optional[float] = None
    current_price: Optional[float] = None
    message: str

class AIAnalysisOutput(BaseModel):
    timestamp: datetime.datetime
    symbol: str
    timeframe: str
    local_signal: str
    rsi: Optional[float] = None
    price: float
    ai_analysis: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    details: Optional[AnalysisDetails] = None
    # prompt: Optional[str] = None # Optionally include for debugging

    # Pydantic v2 aotmatically handles timezone conversion if the string is correctly formatted.
    # The custom validator can be simplified or removed if input is always timezone-aware.
    @field_validator('rsi', mode='before')
    @classmethod
    def rsi_must_be_float_or_none(cls, v):
        if v is not None and pd.isna(v):
            return None
        return v