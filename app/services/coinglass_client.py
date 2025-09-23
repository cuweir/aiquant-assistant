from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests


class CoinglassError(RuntimeError):
    """Raised when the Coinglass API returns an error."""


@dataclass
class TimeChunk:
    start: dt.datetime
    end: dt.datetime

    def as_query(self) -> Dict[str, int]:
        return {
            "start_time": int(self.start.timestamp() * 1000),
            "end_time": int(self.end.timestamp() * 1000),
        }


def _chunk_timerange(start: dt.datetime, end: dt.datetime, days: int) -> Iterable[TimeChunk]:
    current = start
    delta = dt.timedelta(days=days)
    while current < end:
        chunk_end = min(current + delta, end)
        # Coinglass API expects inclusive end; add 1 millisecond to avoid overlap on next chunk.
        yield TimeChunk(start=current, end=chunk_end)
        current = chunk_end + dt.timedelta(milliseconds=1)


class CoinglassClient:
    """Lightweight HTTP client for the Coinglass futures metrics API."""

    DEFAULT_BASE_URLS = (
        "https://open-api.coinglass.com/api/pro/v1",
        "https://open-api.coinglass.com/api/pro/v2",
    )
    OPEN_INTEREST_ENDPOINTS = (
        "/futures/openInterest",
        "/futures/openInterest/v2",
        "/futures/openInterest/chart",
        "/futures/openInterest/total",
    )

    def __init__(
        self,
        api_key: str,
        session: Optional[requests.Session] = None,
        timeout: float = 20.0,
        proxy: Optional[str] = None,
        trust_env: bool = True,
        base_urls: Optional[Iterable[str]] = None,
    ) -> None:
        if not api_key:
            raise ValueError("Coinglass API key is required")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.timeout = timeout
        self.session.trust_env = trust_env
        if proxy == "":
            self.session.trust_env = False
        elif proxy:
            self.session.trust_env = False
            self.session.proxies.update({"http": proxy, "https": proxy})
        self.base_urls = list(base_urls) if base_urls else list(self.DEFAULT_BASE_URLS)

    def _request(self, path: str, params: Dict[str, object]) -> Dict[str, object]:
        last_error: Optional[Exception] = None
        for base_url in self.base_urls:
            url = f"{base_url}{path}"
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers={"coinglassSecret": self.api_key},
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                continue
            if response.status_code == 401:
                raise CoinglassError("Unauthorized. Check COINGLASS API key or account permissions.")
            if not response.ok:
                last_error = CoinglassError(f"HTTP {response.status_code}: {response.text}")
                continue
            payload = response.json()
            if payload.get("code") != 200:
                message = payload.get("msg", "Unknown Coinglass API error")
                print(f"Coinglass API response error: {payload} (url={url}, params={params})")
                last_error = CoinglassError(message)
                if message.lower().startswith("deprecated"):
                    continue
            else:
                return payload
        if last_error:
            raise last_error
        raise CoinglassError("Failed to fetch data from all Coinglass base URLs")

    def _collect_paginated(
        self,
        path: str,
        base_params: Dict[str, object],
        start: dt.datetime,
        end: dt.datetime,
        chunk_days: int,
        data_key: str = "data",
    ) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for chunk in _chunk_timerange(start, end, chunk_days):
            params = dict(base_params)
            params.update(chunk.as_query())
            payload = self._request(path, params)
            items = payload.get(data_key) or []
            if not isinstance(items, list):
                raise CoinglassError(f"Unexpected payload shape for {path}: {payload}")
            results.extend(items)
        return results

    def fetch_long_short_ratio(
        self,
        symbol: str,
        interval: str,
        start: dt.datetime,
        end: dt.datetime,
        *,
        chunk_days: int = 30,
    ) -> pd.DataFrame:
        base_params = {
            "symbol": symbol,
            "interval": interval,
        }
        records = self._collect_paginated(
            "/futures/longShortPositionRatio",
            base_params,
            start,
            end,
            chunk_days=chunk_days,
        )
        return _normalize_long_short(records)

    def fetch_open_interest(
        self,
        symbol: str,
        exchange: str,
        currency: str,
        interval: str,
        start: dt.datetime,
        end: dt.datetime,
        *,
        chunk_days: int = 30,
        endpoint: str = "auto",
        extra_params: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        symbol = symbol.upper()
        endpoints = self._resolve_open_interest_endpoints(endpoint)
        last_error: Optional[Exception] = None

        for symbol_option in self._iter_symbol_options(symbol):
            for exchange_option in self._iter_exchange_options(exchange):
                for currency_option in self._iter_currency_options(currency):
                    base_params = {
                        "symbol": symbol_option,
                        "interval": interval,
                    }
                    if extra_params:
                        base_params.update(extra_params)
                    if exchange_option:
                        base_params["exchange"] = exchange_option
                        base_params.setdefault("exchCode", exchange_option)
                    if currency_option:
                        base_params["currency"] = currency_option
                        base_params.setdefault("quote", currency_option)

                    for path in endpoints:
                        try:
                            records = self._collect_paginated(
                                path,
                                base_params,
                                start,
                                end,
                                chunk_days=chunk_days,
                            )
                            if records:
                                return _normalize_open_interest(records)
                            last_error = None
                        except CoinglassError as exc:
                            last_error = exc
                            message = str(exc).lower()
                            if "deprecated" in message:
                                print(f"Endpoint {path} is deprecated, trying next fallback...")
                                continue
                            if any(
                                code in message
                                for code in ("http 500", "http 503", "internal server error", "service unavailable")
                            ):
                                print(f"Endpoint {path} returned server error, trying next fallback...")
                                continue
                            raise
        if last_error:
            raise last_error
        return _normalize_open_interest([])

    def _resolve_open_interest_endpoints(self, endpoint: str) -> Iterable[str]:
        if endpoint == "auto":
            return self.OPEN_INTEREST_ENDPOINTS
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return (endpoint,)

    @staticmethod
    @staticmethod
    def _iter_symbol_options(symbol: str) -> Iterable[str]:
        base = symbol.upper()
        core = base
        for suffix in ("USDT", "USD", "PERP"):
            if core.endswith(suffix):
                core = core[: -len(suffix)]
                break
        seen: set[str] = set()
        options: List[str] = []

        def add(option: str) -> None:
            option = option.upper()
            if option not in seen:
                seen.add(option)
                options.append(option)

        add(base)
        add(core)
        add(f"{core}USDT")
        add(f"{core}USD")
        return options

    @staticmethod
    def _iter_currency_options(currency: Optional[str]) -> Iterable[Optional[str]]:
        seen: set[Optional[str]] = set()
        options: List[Optional[str]] = []

        def add(option: Optional[str]) -> None:
            if option not in seen:
                seen.add(option)
                options.append(option)

        if currency:
            add(currency)
            upper = currency.upper()
            add(upper)
        add("USDT")
        add("USD")
        add(None)
        return options

    @staticmethod
    def _iter_exchange_options(exchange: Optional[str]) -> Iterable[Optional[str]]:
        seen: set[Optional[str]] = set()
        options: List[Optional[str]] = []

        def add(option: Optional[str]) -> None:
            if option not in seen:
                seen.add(option)
                options.append(option)

        if exchange:
            add(exchange)
            add(exchange.upper())
            add(exchange.lower())
            add(exchange.capitalize())
        add(None)
        return options


def _extract_timestamp(entry: Dict[str, object]) -> dt.datetime:
    for key in ("timestamp", "time", "t", "startTime", "start_time", "openTime"):
        if key in entry:
            value = entry[key]
            if value is None:
                continue
            # Coinglass timestamps are milliseconds.
            return dt.datetime.fromtimestamp(float(value) / 1000.0, tz=dt.timezone.utc)
    raise CoinglassError(f"Unable to locate timestamp in entry: {entry}")


def _normalize_long_short(records: List[Dict[str, object]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["timestamp", "long_short_ratio", "long_account_ratio", "short_account_ratio"]).set_index(
            "timestamp"
        )
    rows = []
    for entry in records:
        ts = _extract_timestamp(entry)
        rows.append(
            {
                "timestamp": ts,
                "long_short_ratio": float(entry.get("longShortRatio")),
                "long_account_ratio": float(entry.get("longAccount")),
                "short_account_ratio": float(entry.get("shortAccount")),
            }
        )
    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
    )
    return df.set_index("timestamp")


def _normalize_open_interest(records: List[Dict[str, object]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["timestamp", "open_interest", "open_interest_value"]).set_index("timestamp")
    rows = []
    for entry in records:
        ts = _extract_timestamp(entry)
        oi = entry.get("openInterest") or entry.get("sumOpenInterest")
        oi_value = entry.get("openInterestValue") or entry.get("sumOpenInterestValue")
        rows.append(
            {
                "timestamp": ts,
                "open_interest": float(oi) if oi is not None else None,
                "open_interest_value": float(oi_value) if oi_value is not None else None,
            }
        )
    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
    )
    return df.set_index("timestamp")
