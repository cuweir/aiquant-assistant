import uuid
from sqlalchemy import (Column, Integer, String, Text, Boolean, DateTime,
                        Numeric, ForeignKey, Index, UniqueConstraint, Float)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from .base_class import Base


class Strategy(Base):
    __tablename__ = "strategies"
    id = Column(Integer, primary_key=True, index=True)
    strategy_name = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text)
    config = Column(JSONB)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    results = relationship("AnalysisResult", back_populates="strategy")


class Symbol(Base):
    __tablename__ = "symbols"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    exchange = Column(String, default="binance", nullable=False)
    asset_type = Column(String, default="spot", nullable=False)
    is_monitored = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    results = relationship("AnalysisResult", back_populates="symbol")
    historical_data = relationship("HistoricalOhlcv", back_populates="symbol", cascade="all, delete-orphan")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False, index=True)
    timeframe = Column(String, nullable=False)
    current_price = Column(Numeric(18, 8), nullable=False)
    composite_score = Column(Float)
    overall_signal = Column(String)
    suggested_sl = Column(Numeric(18, 8))
    suggested_tp1 = Column(Numeric(18, 8))
    suggested_tp = Column(Numeric(18, 8))
    llm_queried = Column(Boolean, nullable=False)
    llm_analysis = Column(Text)
    indicator_details = Column(JSONB)

    symbol = relationship("Symbol", back_populates="results")
    strategy = relationship("Strategy", back_populates="results")

    __table_args__ = (
        Index('ix_analysis_results_timestamp_desc', timestamp.desc()),
    )


class HistoricalOhlcv(Base):
    """
    Stores historical OHLCV data for all symbols and timeframes.
    This table acts as the local data center for analysis and backtesting.
    """
    __tablename__ = "historical_ohlcv"

    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False)
    open_time = Column(DateTime(timezone=True), nullable=False)
    timeframe = Column(String, nullable=False)

    open = Column(Numeric(18, 8), nullable=False)
    high = Column(Numeric(18, 8), nullable=False)
    low = Column(Numeric(18, 8), nullable=False)
    close = Column(Numeric(18, 8), nullable=False)
    volume = Column(Numeric(24, 8), nullable=False)

    symbol = relationship("Symbol", back_populates="historical_data")

    __table_args__ = (
        # Ensure that for a given symbol and timeframe, each candle's open time is unique.
        UniqueConstraint('symbol_id', 'timeframe', 'open_time', name='uq_symbol_timeframe_opentime'),
        # Create a composite index for fast retrieval of time-series data for a specific symbol/timeframe.
        Index('ix_symbol_timeframe_opentime_desc', 'symbol_id', 'timeframe', open_time.desc()),
    )