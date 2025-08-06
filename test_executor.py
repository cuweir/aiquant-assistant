# test_executor.py

import asyncio
import sys, os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.services.order_executor import OrderExecutor


async def main():
    print("--- Testing Order Executor ---")
    # IMPORTANT: Ensure your .env file has the TESTNET keys set

    # We create a new instance for this test
    executor = OrderExecutor(is_testnet=True)

    # 1. Test fetching balance
    balance = await executor.get_balance('USDT')
    print(f"\n[TEST] Current Testnet Balance: {balance} USDT")

    # 2. Test setting leverage
    symbol_to_test = 'BTC/USDT'
    await executor.set_leverage(symbol_to_test, 10)

    # 3. Test opening a small long position
    # First, check if a position already exists
    existing_position = await executor.get_open_positions(symbol_to_test)
    if existing_position:
        print(f"\n[WARN] A position for {symbol_to_test} already exists. Skipping order creation.")
        print(existing_position)
    else:
        print(f"\n[TEST] No open position found. Placing a new order...")
        # Get the current price to calculate a reasonable SL
        ticker = await executor.exchange.fetch_ticker(symbol_to_test)
        current_price = ticker['last']

        # Place a small market buy order (e.g., 0.001 BTC)
        order_amount = 0.001
        buy_order = await executor.create_market_order(symbol_to_test, 'buy', order_amount)

        # If the buy order was successful, place a stop loss
        if buy_order:
            stop_loss_price = current_price * 0.98  # Set a 2% stop loss
            # The SL order is a 'sell' order to close the 'buy' position
            await executor.create_stop_loss_order(symbol_to_test, 'sell', order_amount, stop_loss_price)

    # Clean up connections
    await executor.close_connections()
    print("\n--- Test Finished ---")


if __name__ == "__main__":
    asyncio.run(main())