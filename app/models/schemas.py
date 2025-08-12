# app/models/schemas.py

from pydantic import BaseModel
from typing import Optional, Dict, Any
import datetime

# These models define the structure of our NEW API responses.

class RiskManagement(BaseModel):
    suggested_sl: Optional[float] = None
    take_profit_condition: Optional[str] = None

class Confidence(BaseModel):
    score: Optional[float] = None
    volatility_regime: Optional[str] = None

class KeyFactors(BaseModel):
    is_bull_regime: Optional[bool] = None
    adx_value: Optional[float] = None
    adx_threshold: Optional[float] = None
    ma_slope: Optional[float] = None

class AnalysisReport(BaseModel):
    timestamp: datetime.datetime
    symbol: str
    timeframe: str
    price: float
    signal: str

    ai_analysis: Optional[str] = None
    risk_management: Optional[RiskManagement] = None
    confidence: Optional[Confidence] = None
    key_factors: Optional[KeyFactors] = None
    snapshot: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True # Used to be orm_mode = True

class ManualTriggerInput(BaseModel):
    symbol: str