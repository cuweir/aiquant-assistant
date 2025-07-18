import uuid
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Numeric, ForeignKey, Index, Float
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
    results = relationship("AnalysisResult", back_populates="symbol")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False, index=True)
    timeframe = Column(String, nullable=False)
    current_price = Column(Numeric(18, 8), nullable=False)  # Example precision
    composite_score = Column(Float)
    overall_signal = Column(String)
    suggested_sl = Column(Numeric(18, 8))
    suggested_tp = Column(Numeric(18, 8))
    llm_queried = Column(Boolean, nullable=False)
    llm_analysis = Column(Text)
    indicator_details = Column(JSONB)

    symbol = relationship("Symbol", back_populates="results")
    strategy = relationship("Strategy", back_populates="results")

    __table_args__ = (
        Index('ix_analysis_results_timestamp_desc', timestamp.desc()),
    )