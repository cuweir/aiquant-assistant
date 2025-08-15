# app/models/schemas.py

from pydantic import BaseModel
from typing import Optional, Dict, Any
import datetime

class AnalysisReport(BaseModel):

    timestamp: datetime.datetime
    symbol: str
    timeframe: str
    price: float
    signal: str  # This is the overall_signal

    # Optional fields that might not always be present
    ai_analysis: Optional[str] = None

    # The 'snapshot' now holds ALL the rich details from the strategy
    snapshot: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

class ManualTriggerInput(BaseModel):
    symbol: str