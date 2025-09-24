from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import time

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
        yield TimeChunk(start=current, end=chunk_end)
        current = chunk_end + dt.timedelta(milliseconds=1)


class CoinglassClient:
    """Lightweight HTTP client for the Coinglass futures metrics API."""

    # [MODIFIED] Updated base URLs. The new public endpoint uses a different base.
    DEFAULT_BASE_URLS = [
        "https://open-api.coinglass.com",  # For new /public/v2 endpoints
        "https://open-api.coinglass.com/api/pro/v2",  # Kept for other potential endpoints
        "https://open-api.coinglass.com/api/pro/v1",  # Kept for other potential endpoints
    ]

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

    def _request(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for base_url in self.base_urls:
            url = f"{base_url.rstrip('/')}{path}"
            request_params = {k: v for k, v in params.items() if v is not None}

            print(f"DEBUG: Attempting request to URL: {url} with params: {request_params}")

            try:
                response = self.session.get(
                    url,
                    params=request_params,
                    headers={"coinglassSecret": self.api_key},
                    timeout=self.timeout,
                )
                time.sleep(0.5)  # Add a small delay to avoid rate limiting
            except requests.RequestException as exc:
                print(f"DEBUG: RequestException for {url}: {exc}")
                last_error = exc
                continue

            if response.status_code == 401:
                raise CoinglassError("Unauthorized. Check COINGLASS API key or account permissions.")
            if not response.ok:
                print(f"DEBUG: HTTP Error {response.status_code} for {url}: {response.text}")
                last_error = CoinglassError(f"HTTP {response.status_code}: {response.text}")
                continue

            payload = response.json()
            if isinstance(payload, list):
                return {"data": payload}

            # The new public API has a different success code structure
            if payload.get("code") in ("0", 0, 200):
                return payload

            # Handle old API error structure and other potential errors
            message = payload.get("msg", "Unknown Coinglass API error")
            print(f"Coinglass API response error: {payload} (url={url}, params={request_params})")
            last_error = CoinglassError(message)
            if "deprecated" in message.lower():
                continue

        if last_error:
            raise last_error
        raise CoinglassError("Failed to fetch data from all Coinglass base URLs")

    def _collect_paginated(
            self,
            path: str,
            base_params: Dict[str, Any],
            start: dt.datetime,
            end: dt.datetime,
            chunk_days: int,
            data_key: str = "data",
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for chunk in _chunk_timerange(start, end, chunk_days):
            params = dict(base_params)
            params.update(chunk.as_query())
            payload = self._request(path, params)
            items = payload.get(data_key) or []
            if not isinstance(items, list):
                # New API might nest data differently, e.g. data -> list
                if isinstance(items, dict) and 'list' in items:
                    items = items['list']
                else:
                    raise CoinglassError(f"Unexpected payload shape for {path}: {payload}")
            results.extend(items)
        return results

    def _iter_symbol_options(self, symbol: Optional[str]) -> List[str]:
        if not symbol:
            return []
        cleaned = symbol.replace("/", "").upper()
        candidates = [symbol, symbol.upper(), symbol.lower(), cleaned]
        seen: List[str] = []
        for item in candidates:
            if item not in seen:
                seen.append(item)
        return seen

    def _iter_exchange_options(self, exchange: Optional[str]) -> List[Optional[str]]:
        seen: List[Optional[str]] = []

        def add(value: Optional[str]) -> None:
            if value not in seen:
                seen.append(value)

        if exchange:
            add(exchange)
            add(exchange.upper())
            add(exchange.lower())
            add(exchange.capitalize())
        add(None)
        return seen

    def _iter_currency_options(self, currency: Optional[str]) -> List[Optional[str]]:
        seen: List[Optional[str]] = []

        def add(value: Optional[str]) -> None:
            if value not in seen:
                seen.append(value)

        if currency:
            add(currency)
            add(currency.upper())
        add("USDT")
        add("USD")
        add(None)
        return seen

    def fetch_long_short_ratio(
            self,
            symbol: str,
            interval: str,
            start: dt.datetime,
            end: dt.datetime,
            *,
            chunk_days: int = 30,
    ) -> pd.DataFrame:
        # This endpoint might still use the old structure, keeping it as is for now.
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

    def _resolve_open_interest_endpoints(self, endpoint: str) -> List[str]:
        if endpoint != "auto":
            if not endpoint.startswith("/"):
                endpoint = f"/{endpoint}"
            return [endpoint]
        return [
            "/futures/openInterest",
            "/futures/openInterest/v2",
            "/futures/openInterest/chart",
            "/futures/openInterest/total",
            "/public/v2/indicator/open_interest_ohlc",
        ]

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
        endpoints = self._resolve_open_interest_endpoints(endpoint)
        symbol_options = self._iter_symbol_options(symbol) or [symbol]
        exchange_options = self._iter_exchange_options(exchange)
        currency_options = self._iter_currency_options(currency)
        last_error: Optional[Exception] = None

        for symbol_option in symbol_options:
            pair = symbol_option.replace("/", "").upper()
            for exchange_option in exchange_options:
                for currency_option in currency_options:
                    for path in endpoints:
                        params: Dict[str, Any]
                        if path.startswith("/public/v2/indicator"):
                            params = {
                                "ex": exchange_option,
                                "pair": pair,
                                "interval": interval.replace("h", "H"),
                            }
                            data_key = "data"
                        else:
                            params = {
                                "symbol": symbol_option,
                                "interval": interval,
                                "exchange": exchange_option,
                                "currency": currency_option,
                                "exchCode": exchange_option,
                            }
                            data_key = "data"

                        if extra_params:
                            params.update(extra_params)

                        try:
                            records = self._collect_paginated(
                                path,
                                params,
                                start,
                                end,
                                chunk_days=chunk_days,
                                data_key=data_key,
                            )
                            df = _normalize_open_interest(records)
                            if not df.empty:
                                return df
                        except CoinglassError as exc:
                            last_error = exc
                            message = str(exc).lower()
                            if any(keyword in message for keyword in ("deprecated", "server error", "not found")):
                                continue
                            raise

        if last_error:
            raise last_error
        return pd.DataFrame()


# --- Helper Functions for Data Normalization ---
# Note: These are now outside the class as standalone functions for clarity

def _extract_timestamp(entry: Dict[str, Any]) -> dt.datetime:
    # This function is robust and can be kept as is.
    for key in ("timestamp", "time", "t", "startTime", "start_time", "openTime"):
        if key in entry:
            value = entry[key]
            if value is None:
                continue
            return dt.datetime.fromtimestamp(float(value) / 1000.0, tz=dt.timezone.utc)
    raise CoinglassError(f"Unable to locate timestamp in entry: {entry}")


def _normalize_long_short(records: List[Dict[str, Any]]) -> pd.DataFrame:
    # This function is for a different endpoint and can be kept as is.
    if not records:
        return pd.DataFrame(
            columns=["timestamp", "long_short_ratio", "long_account_ratio", "short_account_ratio"]).set_index(
            "timestamp")
    rows = []
    for entry in records:
        ts = _extract_timestamp(entry)
        rows.append({
            "timestamp": ts,
            "long_short_ratio": float(entry.get("longShortRatio")),
            "long_account_ratio": float(entry.get("longAccount")),
            "short_account_ratio": float(entry.get("shortAccount")),
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    return df.set_index("timestamp")


def _normalize_open_interest(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["timestamp", "open_interest", "open_interest_value"]).set_index("timestamp")

    rows = []
    for entry in records:
        ts = (entry.get("t")
              or entry.get("timestamp")
              or entry.get("startTime")
              or entry.get("openTime"))
        if ts is None:
            continue
        ts_dt = pd.to_datetime(int(ts), unit="ms", utc=True)

        value_candidates = [
            entry.get("c"),
            entry.get("openInterestValue"),
            entry.get("sumOpenInterestValue"),
        ]
        amount_candidates = [
            entry.get("vol"),
            entry.get("openInterest"),
            entry.get("openInterestAmount"),
            entry.get("sumOpenInterest"),
        ]

        def _to_float(candidates: List[Any]) -> Optional[float]:
            for candidate in candidates:
                if candidate in (None, ""):
                    continue
                try:
                    return float(candidate)
                except (TypeError, ValueError):
                    continue
            return None

        rows.append({
            "timestamp": ts_dt,
            "open_interest_value": _to_float(value_candidates),
            "open_interest": _to_float(amount_candidates),
        })

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    ordered_cols = [col for col in ["open_interest", "open_interest_value"] if col in df.columns]
    if ordered_cols:
        df = df[ordered_cols]
    return df


# Backwards compatibility export
_normalize_open_interest_v2 = _normalize_open_interest
