import aiohttp
import ccxt.async_support as ccxt
import asyncio
import time
from typing import Dict, Any, Literal

from ..core.config import settings


class OrderExecutor:
    """
    A service class responsible for executing trades on the Binance Futures exchange.
    This is the final, robust, and stateless version.
    """

    def __init__(self, is_testnet: bool = True):
        self.is_testnet = is_testnet
        if is_testnet:
            api_key = settings.BINANCE_FUTURES_TESTNET_API_KEY
            secret = settings.BINANCE_FUTURES_TESTNET_API_SECRET
        else:
            api_key = settings.BINANCE_FUTURES_LIVE_API_KEY
            secret = settings.BINANCE_FUTURES_LIVE_API_SECRET

        if not api_key or not secret:
            raise ValueError("Binance Futures API keys are not set.")

        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
            "headers": {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            }
        })

        # Set proxy session for CCXT's aiohttp
        if settings.PROXY_URL:
            self.exchange.session = aiohttp.ClientSession(
                proxy=settings.PROXY_URL
            )

        mode_message = "TESTNET" if is_testnet else "LIVE"
        proxy_message = f"with proxy ({settings.PROXY_URL})" if settings.PROXY_URL else "without proxy"
        print(f"OrderExecutor initialized in {mode_message} mode {proxy_message}.")

        if is_testnet:
            self.exchange.set_sandbox_mode(True)
            print("OrderExecutor initialized in TESTNET mode.")
        else:
            print("OrderExecutor initialized in LIVE mode.")

        self.markets_loaded = False

    async def initialize(self):
        if self.markets_loaded:
            return
        print("Performing initial setup for OrderExecutor...")
        try:
            # We wrap ALL network-dependent startup calls in a single block.
            print("  > Attempting to connect to Binance for setup...")

            # Step 1: Time Synchronization
            print("    - Step 1: Checking time synchronization...")
            server_time = await self.exchange.fetch_time()
            local_time = int(time.time() * 1000)
            time_diff = server_time - local_time
            print(f"    - Time sync successful. Difference: {time_diff} ms")
            if abs(time_diff) > 1000:
                self.exchange.options['adjustForTimeDifference'] = True

            # Step 2: Loading Markets
            print("    - Step 2: Loading exchange markets...")
            await self.exchange.load_markets()
            self.markets_loaded = True
            print("  > Exchange setup successful. OrderExecutor is fully operational.")

        except Exception as e:
            print("\n" + "#"*80)
            print("### CRITICAL STARTUP WARNING: COULD NOT CONNECT TO BINANCE API ###")
            print(f"### Error: {e}")
            print("### The application is in a DEGRADED state. ALL live trading will fail. ###")
            print("#"*80 + "\n")

    async def close_connections(self):
        if self.exchange.session:
            await self.exchange.session.close()
        print("OrderExecutor connection closed.")

    async def get_balance(self, currency: str = 'USDT') -> float:
        await self.initialize()
        try:
            balance = await self.exchange.fetch_balance()
            return balance['total'][currency]
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return 0.0

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        await self.initialize()
        try:
            print(f"Setting leverage for {symbol} to {leverage}x...")
            await self.exchange.set_leverage(leverage, symbol)
            print(f"  > Leverage for {symbol} set to {leverage}x successfully.")
            return True
        except Exception as e:
            print(f"Error setting leverage for {symbol}: {e}")
            return False

    async def create_market_order(self, symbol: str, side: Literal['buy', 'sell'], amount: float,
                                  position_side: Literal['LONG', 'SHORT']) -> Dict[str, Any] | None:
        await self.initialize()
        if not self.markets_loaded: return None
        try:
            params = {'positionSide': position_side}
            order = await self.exchange.create_market_order(symbol, side, amount, params=params)
            print(f"  > Market {side} order for {position_side} position created successfully. Order ID: {order['id']}")
            return order
        except Exception as e:
            print(f"Error creating market order for {symbol}: {e}")
            return None

    async def create_stop_loss_order(self, symbol: str, side: Literal['buy', 'sell'], amount: float, stop_price: float,
                                     position_side: Literal['LONG', 'SHORT']) -> Dict[str, Any] | None:
        await self.initialize()
        if not self.markets_loaded: return None
        try:
            params = {'stopPrice': stop_price, 'positionSide': position_side, 'reduceOnly': True}
            order = await self.exchange.create_order(symbol, 'STOP_MARKET', side, amount, params=params)
            print(f"  > Stop loss order created successfully. Order ID: {order['id']}")
            return order
        except Exception as e:
            print(f"Error creating stop loss order for {symbol}: {e}")
            return None

    async def create_take_profit_order(self, symbol: str, side: Literal['buy', 'sell'], amount: float, price: float,
                                       position_side: Literal['LONG', 'SHORT']) -> Dict[str, Any] | None:
        await self.initialize()
        if not self.markets_loaded: return None
        try:
            params = {'positionSide': position_side, 'reduceOnly': True}
            order = await self.exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', side, amount, price, params=params)
            print(f"  > Take profit order created successfully. Order ID: {order['id']}")
            return order
        except Exception as e:
            print(f"Error creating take profit order for {symbol}: {e}")
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        await self.initialize()
        if not self.markets_loaded: return False
        try:
            await self.exchange.cancel_order(order_id, symbol)
            print(f"  > Order {order_id} cancelled successfully.")
            return True
        except ccxt.OrderNotFound:
            print(f"  > Order {order_id} not found. It might have been already filled/cancelled.")
            return True
        except Exception as e:
            print(f"  > Error cancelling order {order_id}: {e}")
            return False

    async def close_position_market(self, symbol: str, position_side: Literal['LONG', 'SHORT'], quantity: float) -> \
    Dict[str, Any] | None:
        await self.initialize()
        if not self.markets_loaded: return None
        side: Literal['buy', 'sell'] = 'sell' if position_side == 'LONG' else 'buy'
        try:
            params = {'positionSide': position_side, 'reduceOnly': True}
            order = await self.exchange.create_market_order(symbol, side, quantity, params=params)
            print(f"  > Market close order for {position_side} position sent successfully.")
            return order
        except Exception as e:
            print(f"  > Error closing {position_side} position for {symbol}: {e}")
            return None


    async def create_market_order_by_notional(self, symbol: str, side: Literal['buy', 'sell'], notional_usdt: float) -> \
    Dict[str, Any] | None:
        """
        Places a market order based on the desired notional value in USDT.
        """
        await self.initialize()
        try:
            print(f"Creating market {side} order for {symbol} with notional value of ~{notional_usdt:.2f} USDT...")

            ticker = await self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            if current_price <= 0:
                raise ValueError("Invalid current price for calculation.")

            amount_in_coin = notional_usdt / current_price
            formatted_amount = self.exchange.amount_to_precision(symbol, amount_in_coin)

            print(
                f"  > Calculated amount: {amount_in_coin} -> Formatted to: {formatted_amount} at price {current_price}")

            # Add position side parameter for futures trading
            params = {}
            if side == 'buy':
                params['positionSide'] = 'LONG'
            elif side == 'sell':
                params['positionSide'] = 'SHORT'

            order = await self.exchange.create_market_order(symbol, side, float(formatted_amount), params=params)
            print(f"  > Market order created successfully. Order ID: {order['id']}")
            return order
        except Exception as e:
            print(f"Error creating market order by notional value for {symbol}: {e}")
            return None

    async def get_open_positions(self, symbol: str) -> Dict[str, Any] | None:
        await self.initialize()
        try:
            positions = await self.exchange.fetch_positions([symbol])
            open_positions = [p for p in positions if p.get('contracts') is not None and float(p['contracts']) != 0]
            if open_positions:
                return open_positions[0]
            return None
        except Exception as e:
            print(f"Error fetching open positions for {symbol}: {e}")
            return None

    async def close_market_position(self, symbol: str, position_side: Literal['LONG', 'SHORT'], quantity: float) -> \
    Dict[str, Any] | None:
        await self.initialize()
        close_side: Literal['buy', 'sell'] = 'sell' if position_side.upper() == 'LONG' else 'buy'
        try:
            print(f"Closing {position_side} position for {quantity:.8f} {symbol} with a market order...")

            # Try different parameter combinations for closing position
            params_options = [
                # Option 1: With position side (for hedge mode)
                {'positionSide': position_side.upper()},
                # Option 2: With reduceOnly (for one-way mode)
                {'reduceOnly': True},
                # Option 3: No special parameters
                {}
            ]

            order = None
            for i, params in enumerate(params_options):
                try:
                    print(f"  > Trying close position option {i+1}...")
                    order = await self.exchange.create_market_order(symbol, close_side, quantity, params=params)
                    print(f"  > Position closed successfully with option {i+1}")
                    break
                except Exception as e:
                    print(f"  > Option {i+1} failed: {e}")
                    if i == len(params_options) - 1:  # Last option
                        raise e

            if not order:
                raise Exception("All close position parameter combinations failed")
            print(f"  > Position closing order created successfully. Order ID: {order['id']}")
            return order
        except Exception as e:
            print(f"  > Error closing position for {symbol}: {e}")
            return None

    async def get_open_position_by_symbol(self, symbol: str) -> Dict[str, Any] | None:
        """
        Fetches the current open position for a single symbol from the exchange.
        Returns the position details if one exists, otherwise None.
        """
        await self.initialize()
        if not self.markets_loaded: return None
        try:
            # fetch_positions can take a list of symbols
            all_positions = await self.exchange.fetch_positions([symbol])

            # Filter for positions that actually have a size
            for position in all_positions:
                # 'contracts' or 'size' can be used depending on exchange, ccxt standardizes to 'contracts'
                size = position.get('contracts')
                if size is not None and float(size) != 0:
                    return position  # Return the first non-zero position found

            return None  # No open position for this symbol
        except Exception as e:
            print(f"Error fetching open position for {symbol}: {e}")
            return None