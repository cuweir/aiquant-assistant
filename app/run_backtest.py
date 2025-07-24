import backtrader as bt
import datetime
import json
from typing import Dict, Any

from app.services.backtest.db_data_fetcher import fetch_df_from_postgres
from backtrader.feeds import PandasData

from app.services.backtest.strategies import MtfaStrategyBt
from app.core.config import settings

def run_parameterized_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs a backtest with a given configuration dictionary.
    This is the final, definitive version.
    """
    cerebro = bt.Cerebro()

    # --- Strategy, Data, Broker, Sizer setup (no changes) ---
    strategy_params = config.get("strategy_params", {})
    cerebro.addstrategy(MtfaStrategyBt, **strategy_params)
    start_date = datetime.datetime.fromisoformat(config["start_date"])
    end_date = datetime.datetime.fromisoformat(config["end_date"])
    print("Fetching data for backtest...")
    df_signal = fetch_df_from_postgres(config["symbol"], settings.SIGNAL_TIMEFRAME, start_date, end_date)
    df_trend_short = fetch_df_from_postgres(config["symbol"], settings.TREND_TIMEFRAME_SHORT, start_date, end_date)
    df_trend_long = fetch_df_from_postgres(config["symbol"], settings.TREND_TIMEFRAME_LONG, start_date, end_date)
    data_signal = PandasData(dataname=df_signal)
    data_trend_short = PandasData(dataname=df_trend_short)
    data_trend_long = PandasData(dataname=df_trend_long)
    cerebro.adddata(data_signal)
    cerebro.adddata(data_trend_short)
    cerebro.adddata(data_trend_long)
    cerebro.broker.setcash(config.get("cash", 10000.0))
    cerebro.broker.setcommission(commission=config.get("commission", 0.001))
    sizer_config = config.get("sizer", {})
    if sizer_config.get("type") == "Percent":
        cerebro.addsizer(bt.sizers.PercentSizer, percents=sizer_config.get("params", {}).get("percents", 10))
    else:
        stake_size = 0.01
        cerebro.addsizer(bt.sizers.FixedSize, stake=sizer_config.get("params", {}).get("stake", stake_size))

    # --- Analyzers ---
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # --- Run ---
    print(f"Executing backtest for config: {config['strategy_params']}")
    results = cerebro.run()
    strat = results[0]

    # --- Result Extraction ---
    try:
        returns_analysis = strat.analyzers.getbyname('returns').get_analysis()
        drawdown_analysis = strat.analyzers.getbyname('drawdown').get_analysis()
        sharpe_analysis = strat.analyzers.getbyname('sharpe_ratio').get_analysis()
        trade_analysis = strat.analyzers.getbyname('trade_analyzer').get_analysis()
    except KeyError as e:
        raise RuntimeError(f"Could not find analyzer by name: {e}. This might happen if no trades were made.")

    final_value = cerebro.broker.getvalue()

    # [REPLACED] The 'pyfolio' key is replaced with a simpler, more robust 'trades' key.
    trades_list = []
    if trade_analysis and trade_analysis.get('trades'):
        for trade in trade_analysis.trades:
            trades_list.append({
                "pnl": trade.get('pnl'),
                "pnl_net": trade.get('pnlcomm'),
                "open_datetime": trade.get('dtopen').strftime('%Y-%m-%d %H:%M:%S'),
                "close_datetime": trade.get('dtclose').strftime('%Y-%m-%d %H:%M:%S'),
                "duration_bars": trade.get('barlen'),
                "status": trade.get('status'),
                "type": "buy" if trade.get('isbuy') else "sell"
            })

    output = {
        "config": config,
        "initial_value": config.get("cash", 10000.0),
        "final_value": final_value,
        "net_profit_pct": (final_value / config.get("cash", 10000.0) - 1) * 100,
        "sharpe_ratio": sharpe_analysis.get('sharperatio'),
        "max_drawdown_pct": drawdown_analysis.max.drawdown,
        "total_trades": trade_analysis.total.total if trade_analysis else 0,
        "win_rate_pct": (trade_analysis.won.total / trade_analysis.total.total * 100) if (trade_analysis and trade_analysis.total.total > 0) else 0,
        "profit_factor": (trade_analysis.won.pnl.total / abs(trade_analysis.lost.pnl.total)) if (trade_analysis and trade_analysis.lost.pnl.total != 0) else float('inf'),
        "total_compound_return_pct": returns_analysis.get('rtot', 0.0) * 100,
        "annualized_return_pct": returns_analysis.get('rnorm100', 0.0),
        "trades": trades_list # Add the clean list of trades
    }
    return output

# This 'main' block for testing remains the same and will work with the new structure
if __name__ == '__main__':
    # default_config = {
    #     "symbol": "BTC/USDT",
    #     "start_date": "2025-01-01",
    #     "end_date": "2025-07-23",
    #     "cash": 10000.0,
    #     "commission": 0.001,
    #     "sizer": {"type": "Percent", "percents": 20},
    #     "strategy_params": {
    #         "rsi_period": 14,
    #         "buy_score_threshold": 5
    #     }
    # }
    # results = run_parameterized_backtest(default_config)
    # results_summary = {k: v for k, v in results.items() if k != 'pyfolio'}
    # print(json.dumps(results_summary, indent=4))
    test_config = {
        "symbol": "BTC/USDT",
        "start_date": "2024-01-01",
        "end_date": "2025-07-23",
        "cash": 10000.0,
        "commission": 0.001,
        "sizer": {"type": "Percent", "percents": 90},
        "strategy_params": {
            "ma_short_period": 20,
            "ma_long_period": 50,
            "adx_period": 14,
            "adx_threshold": 25,
            "atr_period": 14,
            "atr_sl_multiplier": 2.0, # Keep the stop loss
            # "risk_reward_ratio_tp" is removed
        }
    }

    try:
        print("--- Starting Manual Baseline (SMA Crossover) Backtest ---")
        results = run_parameterized_backtest(test_config)
        results_summary = {k: v for k, v in results.items() if k != 'trades'}
        print("\n--- MANUAL BASELINE RESULT ---")
        print(json.dumps(results_summary, indent=4))
        print("----------------------------")
    except Exception as e:
        print(f"\n--- MANUAL BACKTEST FAILED ---")
        import traceback

        traceback.print_exc()
        print("----------------------------")