from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

from .base_strategy import TradingStrategy
from ..ml_pipeline.feature_engineering import FeatureEngineer
from ..ml_pipeline.predictor import PredictionService


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared.ffill(inplace=True)
    prepared.dropna(inplace=True)
    return prepared


class MLDynamicDecisionStrategy(TradingStrategy):
    """Strategy wrapper that turns model forecasts into trading directives."""

    def __init__(self, params: Dict[str, Any]):
        if "model_name" not in params:
            raise ValueError("MLDynamicDecisionStrategy requires 'model_name' in params")

        base_path = params.get("model_base_path")
        run_id = params.get("model_run_id")
        self.prediction_service = PredictionService(
            model_name=params["model_name"],
            base_path=Path(base_path) if base_path else None,
            run_id=run_id,
        )
        self.feature_engineer = FeatureEngineer(self.prediction_service.model.feature_config)

        thresholds = params.get("thresholds", {})
        self.thresholds = {
            "enter_long": thresholds.get("enter_long", 0.6),
            "enter_short": thresholds.get("enter_short", 0.6),
            "mean_revert": thresholds.get("mean_revert", 0.65),
            "bias": thresholds.get("bias", 0.55),
        }

        risk = params.get("risk", {})
        self.risk = {
            "atr_period": risk.get("atr_period", 14),
            "atr_multiplier": risk.get("atr_multiplier", 1.8),
            "tp_multiplier": risk.get("tp_multiplier", 3.0),
            "mean_revert_margin": risk.get("mean_revert_margin", 0.003),
            "bb_length": risk.get("bb_length", 20),
            "bb_std": risk.get("bb_std", 2.0),
        }

    async def generate_signals(self, df_signal: pd.DataFrame, df_regime: pd.DataFrame) -> Dict[str, Any] | None:
        df_signal = _prepare_dataframe(df_signal)
        if df_signal.empty:
            return {"overall_signal": "INSUFFICIENT_DATA", "current_price": None}

        feature_matrix = self.feature_engineer.build(df_signal)
        feature_df = feature_matrix.dataframe.dropna()
        if feature_df.empty:
            return {"overall_signal": "INSUFFICIENT_FEATURES", "current_price": float(df_signal["close"].iloc[-1])}

        latest_index = feature_df.index[-1]
        latest_features = feature_df.tail(1)
        proba_df = self.prediction_service.predict_probabilities(latest_features)
        forecast_series = proba_df.iloc[-1]
        forecast = {k: float(v) for k, v in forecast_series.to_dict().items()}
        ranked_forecast = sorted(forecast.items(), key=lambda item: item[1], reverse=True)

        action, confidence, rationale = self._decide_action(forecast)
        risk_management = self._build_risk_management(action, df_signal)

        current_price = float(df_signal["close"].iloc[-1])
        feature_snapshot = {
            column: float(latest_features.iloc[-1][column])
            for column in latest_features.columns
            if np.isfinite(latest_features.iloc[-1][column])
        }

        return {
            "overall_signal": f"ML_{action}",
            "action": action,
            "action_confidence": float(confidence),
            "action_rationale": rationale,
            "probabilistic_forecast": forecast,
            "ranked_forecast": ranked_forecast,
            "current_price": current_price,
            "risk_management": risk_management,
            "feature_timestamp": latest_index.isoformat() if hasattr(latest_index, "isoformat") else str(latest_index),
            "feature_snapshot": feature_snapshot,
        }

    def _decide_action(self, forecast: Dict[str, float]) -> Tuple[str, float, str]:
        strong_bull = forecast.get("strong_bullish", 0.0)
        strong_bear = forecast.get("strong_bearish", 0.0)
        grind_up = forecast.get("grinding_up", 0.0)
        grind_down = forecast.get("grinding_down", 0.0)
        ranging = forecast.get("ranging", 0.0)

        if strong_bull >= self.thresholds["enter_long"]:
            return "ENTER_LONG", strong_bull, "Strong bullish regime probability above long threshold."
        if strong_bear >= self.thresholds["enter_short"]:
            return "ENTER_SHORT", strong_bear, "Strong bearish regime probability above short threshold."
        if ranging >= self.thresholds["mean_revert"]:
            return "RANGE_MEAN_REVERT", ranging, "Ranging probability indicates mean-reversion opportunity."
        if grind_up >= self.thresholds["bias"]:
            return "LEAN_LONG", grind_up, "Grinding up environment; consider light long bias."
        if grind_down >= self.thresholds["bias"]:
            return "LEAN_SHORT", grind_down, "Grinding down environment; consider protective short bias."
        max_label = max(forecast, key=forecast.get)
        return "STAY_FLAT", forecast[max_label], f"No probabilities above thresholds; dominant state is {max_label}."

    def _build_risk_management(self, action: str, df_signal: pd.DataFrame) -> Dict[str, Any]:
        price = float(df_signal["close"].iloc[-1])
        atr_series = ta.atr(high=df_signal["high"], low=df_signal["low"], close=df_signal["close"], length=self.risk["atr_period"])
        atr_value = float(atr_series.iloc[-1]) if atr_series is not None and not atr_series.empty else float("nan")

        risk_info: Dict[str, Any] = {
            "atr": atr_value,
            "atr_period": self.risk["atr_period"],
            "atr_multiplier": self.risk["atr_multiplier"],
            "tp_multiplier": self.risk["tp_multiplier"],
        }

        if np.isnan(atr_value):
            atr_value = price * 0.01  # Fallback to 1% move if ATR unavailable
        risk_info["atr"] = atr_value

        stop_distance = atr_value * self.risk["atr_multiplier"]
        take_profit_distance = atr_value * self.risk["tp_multiplier"]

        if action in {"ENTER_LONG", "LEAN_LONG"}:
            risk_info["suggested_sl"] = price - stop_distance
            risk_info["suggested_tp"] = price + take_profit_distance
        elif action in {"ENTER_SHORT", "LEAN_SHORT"}:
            risk_info["suggested_sl"] = price + stop_distance
            risk_info["suggested_tp"] = price - take_profit_distance

        if action == "RANGE_MEAN_REVERT":
            bb = ta.bbands(close=df_signal["close"], length=self.risk["bb_length"], std=self.risk["bb_std"])
            if bb is not None and not bb.empty:
                lower = float(bb.iloc[-1, 0])
                upper = float(bb.iloc[-1, 2])
            else:
                spread = price * self.risk["mean_revert_margin"]
                lower = price - spread
                upper = price + spread
            risk_info["range_entry"] = {"buy_zone": lower, "sell_zone": upper}

        risk_info["reference_price"] = price
        return risk_info
