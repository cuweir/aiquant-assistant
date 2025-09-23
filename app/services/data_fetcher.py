from __future__ import annotations

import asyncio
from typing import List

import requests

from ..core.config import settings


class DataFetcher:
    """Lightweight Binance Futures client using REST endpoints via requests."""

    BASE_URL = "https://fapi.binance.com"

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
        )
        if settings.PROXY_URL:
            self.session.proxies.update(
                {
                    "http": settings.PROXY_URL,
                    "https": settings.PROXY_URL,
                }
            )

        proxy_message = (
            f"with proxy ({settings.PROXY_URL})" if settings.PROXY_URL else "without proxy"
        )
        print(f"Initialized Binance Futures Data Fetcher {proxy_message}.")

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int = 1000,
    ) -> List[List[float]]:
        """Fetch klines asynchronously by delegating to a thread."""

        return await asyncio.to_thread(
            self._fetch_ohlcv_sync,
            symbol,
            timeframe,
            since,
            limit,
        )

    def _fetch_ohlcv_sync(
        self,
        symbol: str,
        timeframe: str,
        since: int | None,
        limit: int,
    ) -> List[List[float]]:
        params = {
            "symbol": symbol.replace("/", ""),
            "interval": timeframe,
            "limit": min(limit, 1500),
        }
        if since is not None:
            params["startTime"] = since

        url = f"{self.BASE_URL}/fapi/v1/klines"
        response = self.session.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected response payload: {data}")
        return data

    async def close_exchange(self) -> None:
        await asyncio.to_thread(self.session.close)


data_fetcher_instance = DataFetcher()
