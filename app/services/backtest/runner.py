# runner.py

import backtrader as bt
import datetime
import json
from typing import Dict, Any

from app.services.backtest.db_data_fetcher import fetch_df_from_postgres
from backtrader.feeds import PandasData
# We need to import all possible strategies here for the runner function
from app.services.backtest.strategies import MtfaStrategyAlpha, MtfaStrategyBt, BuyAndHold

# [REFACTOR] A map to resolve strategy names to their actual classes
STRATEGY_MAP = {
    "MtfaStrategyAlpha": MtfaStrategyAlpha,
    "MtfaStrategyBt": MtfaStrategyBt,
    "BuyAndHold": BuyAndHold
}


def get_regime_timeframe(signal_tf: str) -> str:
    if signal_tf == "1h": return "4h"
    if signal_tf == "4h": return "1d"
    return "1h"


# [REFACTOR] This function is now a self-contained, parallelizable unit of work.
def run_single_backtest_instance(strategy_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    This function is designed to be called in a separate process.
    It takes all necessary info, runs one backtest, and returns the result.
    """
    try:
        strategy_class = STRATEGY_MAP.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"Strategy '{strategy_name}' not found in STRATEGY_MAP.")

        cerebro = bt.Cerebro(stdstats=False)  # Disable standard stats for cleaner output

        strategy_params = config.get("strategy_params", {})
        valid_params = strategy_class.params._getkeys()
        params_to_pass = {k: v for k, v in strategy_params.items() if k in valid_params}
        cerebro.addstrategy(strategy_class, **params_to_pass)

        start_date = datetime.datetime.fromisoformat(config["start_date"])
        end_date = datetime.datetime.fromisoformat(config["end_date"])
        signal_tf = config.get("signal_timeframe", "1h")
        regime_tf = get_regime_timeframe(signal_tf)

        # IMPORTANT: Data fetching must happen inside the parallel function
        df_signal = fetch_df_from_postgres(config["symbol"], signal_tf, start_date, end_date)
        cerebro.adddata(PandasData(dataname=df_signal))

        if strategy_class in [MtfaStrategyAlpha, MtfaStrategyBt]:
            df_regime = fetch_df_from_postgres(config["symbol"], regime_tf, start_date, end_date)
            cerebro.adddata(PandasData(dataname=df_regime))

        cerebro.broker.setcash(config["cash"])
        cerebro.broker.setcommission(commission=config.get("commission", 0.001))
        sizer_config = config.get("sizer")
        if sizer_config and sizer_config["type"] == "Percent":
            cerebro.addsizer(bt.sizers.PercentSizer, percents=sizer_config["params"]["percents"])

        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        strat = cerebro.run()[0]

        trade_analysis = strat.analyzers.trade_analyzer.get_analysis()
        return {
            "config": config,
            "final_value": strat.broker.getvalue(),
            "net_profit_pct": (strat.broker.getvalue() / config.get("cash", 10000.0) - 1) * 100,
            "sharpe_ratio": strat.analyzers.sharpe_ratio.get_analysis().get('sharperatio'),
            "max_drawdown_pct": strat.analyzers.drawdown.get_analysis().max.drawdown,
            "total_trades": trade_analysis.total.total if trade_analysis else 0,
            "win_rate_pct": (trade_analysis.won.total / trade_analysis.total.total * 100) if (
                        trade_analysis and trade_analysis.total.total > 0) else 0,
            "profit_factor": (trade_analysis.won.pnl.total / abs(trade_analysis.lost.pnl.total)) if (
                        trade_analysis and trade_analysis.lost.pnl.total != 0) else float('inf'),
        }
    except Exception as e:
        # In a parallel process, it's crucial to catch errors and return them
        import traceback
        error_str = traceback.format_exc()
        print(f"A backtest instance failed: {e}")
        return {"config": config, "error": error_str}


# This part is now only for manual, direct script execution for debugging
if __name__ == '__main__':
    test_config = {
        "symbol": "BTC/USDT", "cash": 10000.0, "commission": 0.001,
        "sizer": {"type": "Percent", "params": {"percents": 90}},
        "strategy_params": {},
        "signal_timeframe": "1h",
        "start_date": "2022-01-01", "end_date": "2025-08-12"
    }
    results = run_single_backtest_instance("MtfaStrategyAlpha", test_config)
    print(json.dumps(results, indent=4))