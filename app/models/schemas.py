from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import datetime

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
    rsi: float
    price: float
    ai_analysis: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    # prompt: Optional[str] = None # Optionally include for debugging

class AllAnalysesOutput(BaseModel):
    analyses: Dict[str, AIAnalysisOutput]