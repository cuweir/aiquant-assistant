import backtrader as bt
import datetime
import json
from typing import Dict, Any

# [FIX] Import the new data fetching utility and the standard PandasData feed
from app.services.backtest.db_data_fetcher import fetch_df_from_postgres
from backtrader.feeds import PandasData

from app.services.backtest.strategies import MtfaStrategyBt
from app.core.config import settings


def run_parameterized_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs a backtest with a given configuration dictionary.
    This is the definitively corrected version.
    """
    cerebro = bt.Cerebro()

    # --- Strategy ---
    strategy_params = config.get("strategy_params", {})
    cerebro.addstrategy(MtfaStrategyBt, **strategy_params)

    # --- Data Feeds ---
    start_date = datetime.datetime.fromisoformat(config["start_date"])
    end_date = datetime.datetime.fromisoformat(config["end_date"])

    print("Fetching data for backtest...")
    # [FIX] Step 1: Fetch all dataframes BEFORE initializing Cerebro data feeds
    df_signal = fetch_df_from_postgres(config["symbol"], settings.SIGNAL_TIMEFRAME, start_date, end_date)
    df_trend_short = fetch_df_from_postgres(config["symbol"], settings.TREND_TIMEFRAME_SHORT, start_date, end_date)
    df_trend_long = fetch_df_from_postgres(config["symbol"], settings.TREND_TIMEFRAME_LONG, start_date, end_date)

    # [FIX] Step 2: Create standard PandasData instances from the pre-fetched data
    data_signal = PandasData(dataname=df_signal)
    data_trend_short = PandasData(dataname=df_trend_short)
    data_trend_long = PandasData(dataname=df_trend_long)

    # [FIX] Step 3: Add the DATA FEED INSTANCES to Cerebro
    cerebro.adddata(data_signal)
    cerebro.adddata(data_trend_short)
    cerebro.adddata(data_trend_long)

    # --- Broker Settings ---
    cerebro.broker.setcash(config.get("cash", 10000.0))
    cerebro.broker.setcommission(commission=config.get("commission", 0.001))

    # --- Sizer ---
    sizer_config = config.get("sizer", {"type": "Fixed", "stake": 1})
    if sizer_config["type"] == "Fixed":
        cerebro.addsizer(bt.sizers.FixedSize, stake=sizer_config.get("stake", 1))
    elif sizer_config["type"] == "Percent":
        cerebro.addsizer(bt.sizers.PercentSizer, percents=sizer_config.get("percents", 10))

    # --- Analyzers ---
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    # --- Run ---
    print(f"Executing backtest for config: {config['strategy_params']}")
    results = cerebro.run()
    strat = results[0]

    # --- Result Extraction (no changes here) ---
    returns = strat.analyzers.returns.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    sharpe = strat.analyzers.sharpe_ratio.get_analysis()
    trade_analysis = strat.analyzers.trade_analyzer.get_analysis()
    pyfolio_data = strat.analyzers.pyfolio.get_analysis()
    final_value = cerebro.broker.getvalue()
    output = {
        "config": config,
        "initial_value": config.get("cash", 10000.0),
        "final_value": final_value,
        "net_profit_pct": (final_value / config.get("cash", 10000.0) - 1) * 100,
        "sharpe_ratio": sharpe.get('sharperatio'),
        "max_drawdown_pct": drawdown.max.drawdown,
        "total_trades": trade_analysis.total.total if trade_analysis else 0,
        "win_rate_pct": (trade_analysis.won.total / trade_analysis.total.total * 100) if (
                    trade_analysis and trade_analysis.total.total > 0) else 0,
        "profit_factor": (trade_analysis.won.pnl.total / abs(trade_analysis.lost.pnl.total)) if (
                    trade_analysis and trade_analysis.lost.pnl.total != 0) else float('inf'),
        "pyfolio": {
            "returns": {k.strftime('%Y-%m-%d'): v for k, v in pyfolio_data['returns'].items()},
            "positions": [p.get_analysis() for p in pyfolio_data['positions']],
            "transactions": [[t[0].strftime('%Y-%m-%d %H:%M:%S'), t[1], t[2]] for t in pyfolio_data['transactions']]
        }
    }
    return output


# This 'main' block for testing remains the same and will work with the new structure
if __name__ == '__main__':
    default_config = {
        "symbol": "BTC/USDT",
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "cash": 10000.0,
        "commission": 0.001,
        "sizer": {"type": "Percent", "percents": 20},
        "strategy_params": {
            "rsi_period": 14,
            "buy_score_threshold": 5
        }
    }
    results = run_parameterized_backtest(default_config)
    results_summary = {k: v for k, v in results.items() if k != 'pyfolio'}
    print(json.dumps(results_summary, indent=4))