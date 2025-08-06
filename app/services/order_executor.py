# app/services/order_executor.py

import ccxt.async_support as ccxt
from typing import Dict, Any

from ..core.config import settings


class OrderExecutor:
    """
    A service class responsible for executing trades on the Binance Futures exchange.
    It handles API connections, order placement, and position management.
    """

    def __init__(self, is_testnet: bool = True):
        """
        Initializes the OrderExecutor.

        Args:
            is_testnet: If True, connects to the Binance Futures Testnet.
                        Otherwise, connects to the live market.
        """
        self.is_testnet = is_testnet

        api_key = settings.BINANCE_FUTURES_TESTNET_API_KEY if is_testnet else settings.BINANCE_API_KEY
        secret = settings.BINANCE_FUTURES_TESTNET_API_SECRET if is_testnet else settings.BINANCE_API_SECRET

        if not api_key or not secret:
            raise ValueError(f"Binance Futures {'Testnet' if is_testnet else 'Live'} API keys are not set.")

        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'options': {
                'defaultType': 'future',  # IMPORTANT: Specify futures trading
            },
        })

        # Set sandbox mode if using testnet
        if is_testnet:
            self.exchange.set_sandbox_mode(True)
            print("OrderExecutor initialized in TESTNET mode.")
        else:
            print("OrderExecutor initialized in LIVE mode. REAL FUNDS WILL BE USED.")

    async def close_connections(self):
        """Closes the connection to the exchange."""
        await self.exchange.close()
        print("OrderExecutor connection closed.")

    async def get_balance(self, currency: str = 'USDT') -> float:
        """
        Fetches the free balance for a specific currency in the futures account.

        Args:
            currency: The currency to check (e.g., 'USDT').

        Returns:
            The available balance as a float.
        """
        try:
            balance = await self.exchange.fetch_balance()
            return balance['free'][currency]
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return 0.0

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Sets the leverage for a specific symbol.

        Args:
            symbol: The trading symbol (e.g., 'BTC/USDT').
            leverage: The desired leverage (e.g., 10 for 10x).

        Returns:
            True if successful, False otherwise.
        """
        try:
            print(f"Setting leverage for {symbol} to {leverage}x...")
            await self.exchange.set_leverage(leverage, symbol)
            print(f"  > Leverage for {symbol} set to {leverage}x successfully.")
            return True
        except Exception as e:
            print(f"Error setting leverage for {symbol}: {e}")
            return False

    async def create_market_order(self, symbol: str, side: str, amount: float) -> Dict[str, Any] | None:
        """
        Places a market order.

        Args:
            symbol: The trading symbol (e.g., 'BTC/USDT').
            side: 'buy' for long, 'sell' for short.
            amount: The quantity of the asset to trade (e.g., 0.01 for BTC).

        Returns:
            The order object from the exchange if successful, None otherwise.
        """
        try:
            print(f"Creating market {side} order for {amount} {symbol}...")
            order = await self.exchange.create_market_order(symbol, side, amount)
            print(f"  > Market order created successfully. Order ID: {order['id']}")
            return order
        except Exception as e:
            print(f"Error creating market order for {symbol}: {e}")
            return None

    async def create_stop_loss_order(self, symbol: str, side: str, amount: float, stop_price: float) -> Dict[
                                                                                                            str, Any] | None:
        """
        Places a stop-loss order. This is a separate order to protect a position.

        Args:
            symbol: The trading symbol.
            side: 'sell' to protect a long position, 'buy' to protect a short position.
            amount: The quantity to sell/buy if the stop price is hit.
            stop_price: The price at which the stop order triggers.

        Returns:
            The order object if successful, None otherwise.
        """
        try:
            print(f"Creating stop loss {side} order for {amount} {symbol} at price {stop_price}...")
            # For stop loss, we use a STOP_MARKET order.
            # 'reduceOnly' ensures this order only closes an existing position.
            params = {'stopPrice': stop_price, 'reduceOnly': True}
            order = await self.exchange.create_order(symbol, 'STOP_MARKET', side, amount, params=params)
            print(f"  > Stop loss order created successfully. Order ID: {order['id']}")
            return order
        except Exception as e:
            print(f"Error creating stop loss order for {symbol}: {e}")
            return None

    async def get_open_positions(self, symbol: str) -> Dict[str, Any] | None:
        """
        Checks if there is an open position for a given symbol.

        Args:
            symbol: The trading symbol.

        Returns:
            The position object if a position exists, None otherwise.
        """
        try:
            positions = await self.exchange.fetch_positions([symbol])
            # Filter out positions with zero amount
            open_positions = [p for p in positions if p.get('contracts') is not None and float(p['contracts']) != 0]
            if open_positions:
                return open_positions[0]  # Return the first open position
            return None
        except Exception as e:
            print(f"Error fetching open positions for {symbol}: {e}")
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancels a specific open order.

        Args:
            order_id: The ID of the order to cancel.
            symbol: The trading symbol of the order.

        Returns:
            True if cancellation was successful or order doesn't exist, False otherwise.
        """
        try:
            print(f"Attempting to cancel order {order_id} for {symbol}...")
            await self.exchange.cancel_order(order_id, symbol)
            print(f"  > Order {order_id} cancelled successfully.")
            return True
        except ccxt.OrderNotFound:
            # This is not an error, it just means the order was already filled or cancelled.
            print(f"  > Order {order_id} not found. It might have been already filled/cancelled.")
            return True
        except Exception as e:
            print(f"  > Error cancelling order {order_id}: {e}")
            return False

    async def close_market_position(self, symbol: str, position_side: str, quantity: float) -> Dict[str, Any] | None:
        """
        Closes an existing position with a market order.

        Args:
            symbol: The trading symbol (e.g., 'BTC/USDT').
            position_side: 'LONG' or 'SHORT'.
            quantity: The amount of the position to close.

        Returns:
            The closing order object if successful, None otherwise.
        """
        # To close a LONG position, we need to SELL. To close a SHORT, we need to BUY.
        close_side = 'sell' if position_side.upper() == 'LONG' else 'buy'

        try:
            print(f"Closing {position_side} position for {quantity} {symbol} with a market order...")
            # 'reduceOnly' is crucial here. It ensures this order only reduces or closes
            # an existing position, and never opens a new one in the opposite direction.
            params = {'reduceOnly': True}
            order = await self.exchange.create_market_order(symbol, close_side, quantity, params=params)
            print(f"  > Position closing order created successfully. Order ID: {order['id']}")
            return order
        except Exception as e:
            print(f"  > Error closing position for {symbol}: {e}")
            return None
