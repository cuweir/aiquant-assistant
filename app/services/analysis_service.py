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
from ..models.schemas import AnalysisReport, Confidence, KeyFactors, RiskManagement
from ..utils.formatters import format_price_dynamically
from .backtest.db_data_fetcher import fetch_df_from_postgres
from .parameter_manager import ParameterManager
from ..strategies.multi_indicator_strategy import AlphaRegimeStrategy, LongOnlyTrendStrategy
from ..strategies.base_strategy import TradingStrategy
from .trading_service import TradingService

STRATEGY_MAP: Dict[str, Type[TradingStrategy]] = {
    "AlphaRegimeStrategy": AlphaRegimeStrategy,
    "LongOnlyTrendStrategy": LongOnlyTrendStrategy,
}

def make_dict_json_serializable(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively converts non-serializable items in a dictionary."""
    for key, value in d.items():
        if isinstance(value, dict): make_dict_json_serializable(value)
        elif isinstance(value, (datetime.datetime, datetime.date, pd.Timestamp)): d[key] = value.isoformat()
        elif isinstance(value, np.bool_): d[key] = bool(value)
    return d

class AnalysisService:
    def __init__(self, param_manager: ParameterManager, trading_service: TradingService, llm_strategy: LLMStrategy):
        self.param_manager = param_manager
        self.trading_service = trading_service
        self.llm_strategy = llm_strategy

    def _get_or_create_symbol(self, db: Session, symbol_name: str) -> Symbol:
        symbol = db.query(Symbol).filter(Symbol.name == symbol_name).first()
        if not symbol:
            symbol = Symbol(name=symbol_name)
            db.add(symbol)
            db.commit()
            db.refresh(symbol)
        return symbol

    def _get_or_create_strategy(self, db: Session, strategy_name: str, config: dict) -> Strategy:
        strategy = db.query(Strategy).filter(Strategy.strategy_name == strategy_name).first()
        if not strategy:
            strategy = Strategy(strategy_name=strategy_name, description="V7 Volatility Adaptive", config=config)
            db.add(strategy)
            db.commit()
            db.refresh(strategy)
        return strategy

    async def generate_comprehensive_analysis(self, symbol_name: str) -> Dict[str, Any] | None:
        print(f"\n--- Analyzing {symbol_name} ---")
        db: Session = SessionLocal()
        try:
            # 1. Get parameters and determine which strategy to use
            symbol_params = self.param_manager.get_params_for_symbol(symbol_name)
            strategy_name = symbol_params.get("strategy_name", "AlphaRegimeStrategy")  # Default to Alpha

            StrategyClass = STRATEGY_MAP.get(strategy_name)
            if not StrategyClass:
                print(f"  > [ERROR] Strategy '{strategy_name}' not found for symbol '{symbol_name}'.")
                return None

            print(f"  > Using strategy: {strategy_name}")
            trading_strategy = StrategyClass(params=symbol_params)

            # 2. Fetch data
            end_date = pd.Timestamp.now(tz='UTC')
            start_date = end_date - pd.Timedelta(days=200)  # Increased data for longer MAs
            df_signal = fetch_df_from_postgres(symbol_name, settings.SIGNAL_TIMEFRAME, start_date, end_date)
            df_regime = fetch_df_from_postgres(symbol_name, settings.TREND_TIMEFRAME_SHORT, start_date, end_date)
            if df_signal is None or df_regime is None or df_signal.empty or df_regime.empty:
                print(f"  > [ERROR] Insufficient data for {symbol_name}.")
                return None

            # 3. Generate the raw signal from the chosen strategy
            strategy_result = await trading_strategy.generate_signals(df_signal, df_regime)
            if not strategy_result: return None

            # 4. Log the detailed snapshot
            snapshot = strategy_result.get("snapshot", {})
            print(f"  > Strategy Result for {symbol_name}: {strategy_result}")
            print(json.dumps(snapshot, indent=4, default=str))

            components = snapshot.get("components", {})

            confidence_obj = Confidence(
                score=snapshot.get("total_score"),
                volatility_regime='High' if snapshot.get("is_high_vol") else 'Low'
            )

            key_factors_obj = KeyFactors(
                is_bull_regime=snapshot.get("is_bull_regime"),
                adx_value=components.get("strength", {}).get("value"),
                adx_threshold=components.get("strength", {}).get("threshold"),
                ma_slope=components.get("slope", {}).get("value")
            )

            risk_management_obj = RiskManagement(
                suggested_sl=strategy_result.get("risk_management", {}).get("suggested_sl"),
                take_profit_condition=strategy_result.get("risk_management", {}).get("take_profit_condition")
            )

            try:
                report = AnalysisReport(
                    timestamp=pd.Timestamp.now(tz='UTC').to_pydatetime(),
                    symbol=symbol_name,
                    timeframe=settings.SIGNAL_TIMEFRAME,
                    price=strategy_result.get("current_price"),
                    signal=strategy_result.get("overall_signal", "ERROR"),
                    ai_analysis="AI analysis not triggered.",
                    risk_management=risk_management_obj,  # Use the created object
                    confidence=confidence_obj,  # Use the created object
                    key_factors=key_factors_obj,  # Use the created object
                    snapshot=snapshot
                )
            except Exception as e:
                print(f"  > FAILED to build Pydantic model from strategy result: {e}")
                return None

            report_data = report.model_dump()

            # 6. LLM Analysis (logic remains the same)
            should_call_llm = report_data["signal"] == "POTENTIAL_BUY"
            if should_call_llm:
                print(f"  > Signal is valid. Querying LLM...")
                prompt = self._build_llm_prompt(report_data)
                ai_suggestion = await self.llm_strategy.generate_analysis(prompt)
                report_data["ai_analysis"] = ai_suggestion

            serializable_details = make_dict_json_serializable(report_data.copy())

            # 4. Save to DB
            strategy_record = self._get_or_create_strategy_record(db, strategy_name, symbol_params)
            db_analysis_result = AnalysisResult(
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                symbol_id=self._get_or_create_symbol(db, symbol_name).id,
                strategy_id=strategy_record.id,
                timeframe=settings.SIGNAL_TIMEFRAME,
                current_price=strategy_result['current_price'],
                overall_signal=strategy_result['overall_signal'],
                llm_queried=False,  # For simplicity, disable LLM for now
                llm_analysis="AI analysis disabled in this version.",
                indicator_details=make_dict_json_serializable(strategy_result)
            )
            db.add(db_analysis_result)
            db.commit()
            print(f"  > Analysis report for {symbol_name} using {strategy_name} saved to database.")

            # 8. Pass the signal report to the TradingService for action (logic remains the same)
            # await self.trading_service.process_signal(db, symbol_record, strategy_result)

            return strategy_result

        except Exception as e:
            print(f"  > An unexpected error during analysis for {symbol_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            db.close()

    def _build_llm_prompt(self, report: Dict[str, Any]) -> str:
        # This function now correctly uses the unified report object
        prompt = f"""
        Analyze this high-quality trading signal for {report['symbol']} on the {report['timeframe']} chart.

        - Signal: {report['signal']}
        - Current Price: {format_price_dynamically(report['price'])}

        - Risk Management:
          - Stop Loss: {format_price_dynamically(report['risk_management']['suggested_sl'])}
          - Take Profit Condition: {report['risk_management']['take_profit_condition']}

        - Key Confidence Factors:
          - Signal Score: {report['confidence']['score']}
          - Volatility Regime: {report['confidence']['volatility_regime']}

        AI Analyst Task:
        1.  Provide a concise, professional final verdict: **[Strong Buy / Cautious Buy / Hold]**.
        2.  In 1-2 sentences, justify your verdict.
        3.  Mention the single most important risk to watch for.
        """
        return prompt.strip()

    def get_all_analyses_from_db(self, db: Session, skip: int = 0, limit: int = 20) -> List[AnalysisResult]:
        return db.query(AnalysisResult).options(
      joinedload(AnalysisResult.symbol),
            joinedload(AnalysisResult.strategy)
        ).order_by(AnalysisResult.timestamp.desc()).offset(skip).limit(limit).all()