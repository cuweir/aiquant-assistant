# app/services/analysis_service.py
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Type
from sqlalchemy.orm import Session, joinedload
import json

from ..core.config import settings
from ..llm_providers.base import LLMStrategy
from ..db.session import SessionLocal
from ..db.models import AnalysisResult, Symbol, Strategy
from ..models.schemas import AnalysisReport  # Import the new simplified model
from ..utils.formatters import format_price_dynamically
from .backtest.db_data_fetcher import fetch_df_from_postgres
from .parameter_manager import ParameterManager
from ..strategies.multi_indicator_strategy import AlphaRegimeStrategy
from ..strategies.base_strategy import TradingStrategy
from .trading_service import TradingService

STRATEGY_MAP: Dict[str, Type[TradingStrategy]] = {
    "AlphaRegimeStrategy": AlphaRegimeStrategy,
}


def make_dict_json_serializable(d: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in d.items():
        if isinstance(value, dict):
            make_dict_json_serializable(value)
        elif isinstance(value, (datetime.datetime, datetime.date, pd.Timestamp)):
            d[key] = value.isoformat()
        elif isinstance(value, (np.bool_, np.int64, np.float64)):
            d[key] = value.item()
    return d


class AnalysisService:
    def __init__(self, param_manager: ParameterManager, trading_service: TradingService, llm_strategy: LLMStrategy):
        self.param_manager = param_manager
        self.trading_service = trading_service
        self.llm_strategy = llm_strategy

    def _get_or_create_symbol(self, db: Session, symbol_name: str) -> Symbol:
        symbol = db.query(Symbol).filter(Symbol.name == symbol_name).first()
        if not symbol: symbol = Symbol(name=symbol_name); db.add(symbol); db.commit(); db.refresh(symbol)
        return symbol

    def _get_or_create_strategy(self, db: Session, strategy_name: str, config: dict) -> Strategy:
        strategy = db.query(Strategy).filter(Strategy.strategy_name == strategy_name).first()
        if not strategy: strategy = Strategy(strategy_name=strategy_name, config=config); db.add(
            strategy); db.commit(); db.refresh(strategy)
        return strategy

    async def generate_comprehensive_analysis(self, symbol_name: str) -> Dict[str, Any] | None:
        print(f"\n--- Analyzing {symbol_name} ---")
        db: Session = SessionLocal()
        try:
            symbol_params = self.param_manager.get_params_for_symbol(symbol_name)
            strategy_name = symbol_params.get("strategy_name", "AlphaRegimeStrategy")
            StrategyClass = STRATEGY_MAP.get(strategy_name)
            if not StrategyClass: return None

            print(f"  > Using strategy: {strategy_name}")
            trading_strategy = StrategyClass(params=symbol_params)

            end_date = pd.Timestamp.now(tz='UTC')
            start_date = end_date - pd.Timedelta(days=200)
            df_signal = fetch_df_from_postgres(symbol_name, settings.SIGNAL_TIMEFRAME, start_date, end_date)
            df_regime = fetch_df_from_postgres(symbol_name, settings.TREND_TIMEFRAME_SHORT, start_date, end_date)

            if df_signal is None or df_regime is None or df_signal.empty or df_regime.empty: return None

            strategy_result = await trading_strategy.generate_signals(df_signal, df_regime)
            if not strategy_result: return None

            print(
                f"  > Strategy Result for {symbol_name}:\n{json.dumps(make_dict_json_serializable(strategy_result.copy()), indent=2)}")

            strategy_record = self._get_or_create_strategy(db, strategy_name, symbol_params)
            symbol_record = self._get_or_create_symbol(db, symbol_name)

            serializable_result = make_dict_json_serializable(strategy_result)

            db_analysis_result = AnalysisResult(
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                symbol_id=symbol_record.id,
                strategy_id=strategy_record.id,
                timeframe=settings.SIGNAL_TIMEFRAME,
                current_price=serializable_result['current_price'],
                overall_signal=serializable_result['overall_signal'],
                llm_queried=False,
                llm_analysis="AI analysis disabled.",
                indicator_details=serializable_result
            )
            db.add(db_analysis_result);
            db.commit()
            print(f"  > Analysis report for {symbol_name} saved to database.")

            await self.trading_service.process_signal(db, symbol_record, strategy_result)

            # [THE FIX] Construct the API response using the new simplified Pydantic model.
            # All the rich details from the strategy are now correctly placed in the 'snapshot' field.
            api_response = AnalysisReport(
                timestamp=db_analysis_result.timestamp,
                symbol=symbol_name,
                timeframe=settings.SIGNAL_TIMEFRAME,
                price=serializable_result['current_price'],
                signal=serializable_result['overall_signal'],
                snapshot=serializable_result  # Pass the entire result blob to the snapshot
            )

            return api_response.model_dump()

        except Exception as e:
            print(f"  > An unexpected error during analysis for {symbol_name}: {e}")
            import traceback;
            traceback.print_exc()
            return None
        finally:
            db.close()

    def get_all_analyses_from_db(self, db: Session, skip: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
        db_results = db.query(AnalysisResult).options(joinedload(AnalysisResult.symbol)).order_by(
            AnalysisResult.timestamp.desc()).offset(skip).limit(limit).all()

        response_list = []
        for r in db_results:
            try:
                # [THE FIX] The entire strategy result is the snapshot.
                snapshot_details = r.indicator_details or {}
                if not isinstance(snapshot_details, dict): continue

                report_dict = {
                    "timestamp": r.timestamp.isoformat(),
                    "symbol": r.symbol.name,
                    "timeframe": r.timeframe,
                    "price": float(r.current_price),
                    "signal": r.overall_signal,
                    "snapshot": snapshot_details  # Pass the entire blob as the snapshot
                }
                response_list.append(report_dict)
            except Exception as e:
                print(f"Skipping malformed DB record {r.id} due to error: {e}")
                continue
        return response_list