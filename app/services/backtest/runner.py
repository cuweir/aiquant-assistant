# runner.py

import backtrader as bt
import datetime
import json
from typing import Dict, Any

from app.services.backtest.db_data_fetcher import fetch_df_from_postgres
from backtrader.feeds import PandasData
from app.services.backtest.strategies import MtfaStrategyBt, BuyAndHold


def get_regime_timeframe(signal_tf: str) -> str:
    if signal_tf == "1h": return "4h"
    if signal_tf == "4h": return "1d"
    return "1h"


def run_single_strategy(strategy_class: bt.Strategy, config: Dict[str, Any]):
    cerebro = bt.Cerebro()

    strategy_params = config.get("strategy_params", {})
    valid_params = strategy_class.params._getkeys()
    params_to_pass = {k: v for k, v in strategy_params.items() if k in valid_params}
    cerebro.addstrategy(strategy_class, **params_to_pass)

    start_date = datetime.datetime.fromisoformat(config["start_date"])
    end_date = datetime.datetime.fromisoformat(config["end_date"])
    signal_tf = config.get("signal_timeframe", "1h")
    regime_tf = config.get("regime_timeframe", "4h")

    print(f"Fetching data for backtest... Signal: {signal_tf}, Regime: {regime_tf}")
    df_signal = fetch_df_from_postgres(config["symbol"], signal_tf, start_date, end_date)
    cerebro.adddata(PandasData(dataname=df_signal))

    # Add regime data ONLY if the strategy needs it
    if strategy_class is MtfaStrategyBt:
        df_regime = fetch_df_from_postgres(config["symbol"], regime_tf, start_date, end_date)
        cerebro.adddata(PandasData(dataname=df_regime))

    cerebro.broker.setcash(config["cash"])
    cerebro.broker.setcommission(commission=config["commission"])

    sizer_config = config.get("sizer")
    if sizer_config:
        if sizer_config["type"] == "Percent":
            cerebro.addsizer(bt.sizers.PercentSizer, percents=sizer_config["params"]["percents"])
        else:
            cerebro.addsizer(bt.sizers.FixedSize, stake=sizer_config.get("params", {}).get("stake", 0.01))

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

    results = cerebro.run()
    return results[0]


def run_parameterized_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    This is the core function called by the optimizer. It runs a single backtest
    for our main strategy and returns a structured result dictionary.
    """
    try:
        # We use run_single_strategy to execute the backtest
        strat = run_single_strategy(MtfaStrategyBt, config)

        # --- Result Extraction ---
        final_value = strat.broker.getvalue()
        returns_analysis = strat.analyzers.returns.get_analysis()
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        sharpe_analysis = strat.analyzers.sharpe_ratio.get_analysis()
        trade_analysis = strat.analyzers.trade_analyzer.get_analysis()

        output = {
            "config": config,
            "initial_value": config.get("cash", 10000.0),
            "final_value": final_value,
            "net_profit_pct": (final_value / config.get("cash", 10000.0) - 1) * 100,
            "sharpe_ratio": sharpe_analysis.get('sharperatio'),
            "max_drawdown_pct": drawdown_analysis.max.drawdown,
            "total_trades": trade_analysis.total.total if trade_analysis else 0,
            "win_rate_pct": (trade_analysis.won.total / trade_analysis.total.total * 100) if (
                        trade_analysis and trade_analysis.total.total > 0) else 0,
            "profit_factor": (trade_analysis.won.pnl.total / abs(trade_analysis.lost.pnl.total)) if (
                        trade_analysis and trade_analysis.lost.pnl.total != 0) else float('inf'),
            "total_compound_return_pct": returns_analysis.get('rtot', 0.0) * 100,
            "annualized_return_pct": returns_analysis.get('rnorm100', 0.0),
            "trades": []  # Simplified for now
        }
        return output
    except (KeyError, AttributeError, IndexError) as e:
        # This can happen if no trades are made or data is insufficient
        print(f"  > Backtest run for {config.get('strategy_params')} resulted in an error: {e}")
        # Return a dictionary indicating failure for this specific run
        return {
            "config": config,
            "error": str(e)
        }

def run_final_validation():
    # --- SCENARIO 1: Our proven champion on the 1-hour chart (2022-2025) ---

    # [CRITICAL FIX] Provide the COMPLETE and CORRECT parameter set
    champion_params_1h = {
        "low_vol_ma_long": 80,
        "high_vol_ma_long": 40,
        "low_vol_adx_threshold": 30,
        "high_vol_adx_threshold": 30,
        "low_vol_atr_sl_multiplier": 2.0,
        "high_vol_atr_sl_multiplier": 2.5,
        "low_vol_ma_short": 25,
        "high_vol_ma_short": 15
        # The rest of the params will use the defaults from the strategy class
    }
    config_1h = {
        "symbol": "BTC/USDT", "cash": 10000.0, "commission": 0.001,
        "sizer": {"type": "Percent", "params": {"percents": 90}},
        "strategy_params": champion_params_1h,
        "signal_timeframe": "1h", "regime_timeframe": "4h",
        "start_date": "2022-01-01", "end_date": "2025-08-04"
    }
    run_backtest_scenario("CHAMPION MODEL ON 1H CHART (2023-2024)", config_1h)

    # --- SCENARIO 2: Test the champion model on the 4-hour chart (2020-2024) ---

    champion_params_4h = {
        "low_vol_ma_short": 25 * 4, "low_vol_ma_long": 80 * 4,
        "high_vol_ma_short": 15 * 4, "high_vol_ma_long": 40 * 4,
        "low_vol_adx_threshold": 30, "high_vol_adx_threshold": 30,
        "regime_ma_period": 50,  # 200 on 1d is roughly 50 on 4h
        "low_vol_atr_sl_multiplier": 2.0, "high_vol_atr_sl_multiplier": 2.5,
    }
    config_4h = {
        "symbol": "BTC/USDT", "cash": 10000.0, "commission": 0.001,
        "sizer": {"type": "Percent", "params": {"percents": 90}},
        "strategy_params": champion_params_4h,
        "signal_timeframe": "4h", "regime_timeframe": "1d",
        "start_date": "2020-01-01", "end_date": "2025-08-04",
    }
    run_backtest_scenario("CHAMPION MODEL ON 4H CHART (2020-2024)", config_4h)


def run_backtest_scenario(scenario_name: str, config: Dict[str, Any]):
    try:
        print(f"\n--- Starting Backtest Scenario: [{scenario_name}] ---")

        our_strategy_result = run_single_strategy(MtfaStrategyBt, config)

        benchmark_config = config.copy()
        benchmark_config.pop("sizer", None)
        benchmark_result = run_single_strategy(BuyAndHold, benchmark_config)

        our_final_val = our_strategy_result.broker.getvalue()
        bm_final_val = benchmark_result.broker.getvalue()

        our_dd = our_strategy_result.analyzers.drawdown.get_analysis().max.drawdown
        bm_dd = benchmark_result.analyzers.drawdown.get_analysis().max.drawdown

        comparison = {
            "strategy_net_profit_pct": (our_final_val / config["cash"] - 1) * 100,
            "buy_and_hold_net_profit_pct": (bm_final_val / config["cash"] - 1) * 100,
            "alpha_pct": ((our_final_val / config["cash"] - 1) - (bm_final_val / config["cash"] - 1)) * 100,
            "strategy_max_drawdown_pct": our_dd,
            "buy_and_hold_max_drawdown_pct": bm_dd
        }

        print(f"\n--- RESULT FOR SCENARIO: [{scenario_name}] ---")
        print("--- BENCHMARK COMPARISON ---")
        print(json.dumps(comparison, indent=4))
        print("------------------------------------------")

    except Exception as e:
        print(f"\n--- SCENARIO [{scenario_name}] FAILED ---")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_final_validation()