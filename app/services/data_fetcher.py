from typing import Union

import ccxt.async_support as ccxt # Use async version of ccxt
import pandas as pd
from ..core.config import settings

class DataFetcher:
    def __init__(self):
        self.exchange_id = 'binance' # Or your preferred exchange
        self.exchange_class = getattr(ccxt, self.exchange_id)
        exchange_config = {
            'apiKey': settings.BINANCE_API_KEY,
            'secret': settings.BINANCE_API_SECRET,
            'enableRateLimit': True,
            "headers": {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            }
        }
        if settings.PROXY_URL:
            exchange_config['aiohttp_proxy'] = settings.PROXY_URL

        self.exchange = self.exchange_class(exchange_config)
        print(f"Initialized {self.exchange_id} Data Fetcher (Read-Only) with proxy.")

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Union[pd.DataFrame, None]:
        try:
            # For USDT-margined perpetuals, CCXT often expects symbol format like "BTC/USDT:USDT"
            # For spot, "BTC/USDT" is fine.
            # You might need to adjust symbol formatting based on exchange.load_markets() output.
            print(f"Fetching OHLCV for {symbol} ({timeframe})...")
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                cols_to_convert = ['open', 'high', 'low', 'close', 'volume']
                for col in cols_to_convert:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df.dropna(subset=cols_to_convert, inplace=True)
                return df
            return None
        except ccxt.NetworkError as e:
            print(f"CCXT Network Error fetching {symbol}: {e}")
        except ccxt.ExchangeError as e:
            print(f"CCXT Exchange Error fetching {symbol}: {e}")
        except Exception as e:
            print(f"General Error fetching {symbol}: {e}")
        return None

    async def close_exchange(self):
        await self.exchange.close()

# Global instance
data_fetcher_instance = DataFetcher()