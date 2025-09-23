## Historical Futures Metrics

This repository now supports pulling long-form open interest and long/short account ratio history from Coinglass so that model training is no longer limited to the ~30 days exposed by the public Binance endpoints.

1. Obtain a Coinglass API key (free tier works) and expose it to the tooling:
   ```bash
   export COINGLASS_API_KEY="your-secret"
   ```
2. Collect the metrics you need. The script merges with any existing CSV so you can re-run it to extend the range.
   ```bash
   # Open interest (per-exchange, hourly bars)
   ./fetch_coinglass_metrics.py open_interest BTCUSDT \
       --start-date 2022-01-01 --end-date 2024-12-31 \
       --interval 1h --exchange Binance --currency USDT \
        --endpoint auto  # 可选：如官方提示 deprecated，可改成 futures/openInterest/v2

   # 若接口仍返回 deprecated/500，可尝试调整 currency / exchange 参数，例如：
   ./fetch_coinglass_metrics.py open_interest BTCUSDT \
       --start-date 2022-01-01 --end-date 2024-12-31 \
        --interval 1h --exchange BINANCE --currency USD

   # 如果在终端里自动走系统代理导致连接失败，可禁用代理：
   ./fetch_coinglass_metrics.py open_interest BTCUSDT \
       --start-date 2022-01-01 --end-date 2024-12-31 \
       --interval 1h --exchange Binance --currency USDT \
        --proxy none

   # 新版 Coinglass API 若迁移到其它 base url，可显式指定：
   ./fetch_coinglass_metrics.py open_interest BTCUSDT \
       --start-date 2022-01-01 --end-date 2024-12-31 \
       --interval 1h --exchange Binance --currency USDT \
       --base-url https://open-api.coinglass.com/api/pro/v2 \
       --extra-param contract=PERPETUAL

   # 若要省略 exchange/currency 参数，可传入 --exchange none --currency none

   # Global long/short account ratio
   ./fetch_coinglass_metrics.py long_short_ratio BTCUSDT \
       --start-date 2022-01-01 --end-date 2024-12-31 \
       --interval 1h
   ```
3. The files are written to `data/external/coinglass/<dataset>/<symbol>.csv`. The feature sources will automatically prefer Binance files when they exist and fall back to Coinglass, so simply re-running your dataset build (e.g. `manual_build_dataset.py BTC/USDT 1h --lookback-days 720`) will blend in the extended history.

If you want to force a specific provider, pass `"provider_priority": ["coinglass", "binance"]` (or a single string) in the relevant feature source config. You can still point to a custom CSV via the existing `csv_path` parameter.
