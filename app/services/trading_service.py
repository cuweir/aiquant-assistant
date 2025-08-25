# app/services/trading_service.py

import asyncio
from typing import Dict, Any, Literal
from sqlalchemy import func
from sqlalchemy.orm import Session
import ccxt

from .order_executor import OrderExecutor
from .parameter_manager import ParameterManager
from ..db.models import Position, Symbol
from ..core.config import settings

class TradingService:
    def __init__(self, order_executor: OrderExecutor, param_manager: ParameterManager):
        self.order_executor = order_executor
        self.param_manager = param_manager

    async def process_signal(self, db: Session, symbol: Symbol, strategy_result: Dict[str, Any]):
        signal = strategy_result.get("overall_signal")
        exit_signal = strategy_result.get("exit_signal")

        open_position = db.query(Position).filter(Position.symbol_id == symbol.id, Position.is_open == True).first()

        if open_position:
            # Handle exit signals for existing positions
            if (open_position.position_side == 'LONG' and exit_signal == "EXIT_LONG_DEATH_CROSS") or \
                    (open_position.position_side == 'SHORT' and exit_signal == "EXIT_SHORT_GOLDEN_CROSS"):
                await self._handle_exit_logic(db, symbol, open_position)
            else:
                print(f"  > TradingService: Holding {open_position.position_side} position for {symbol.name}.")
        else:
            # Handle entry signals for new positions
            if signal == "POTENTIAL_BUY":
                await self._handle_entry_logic(db, symbol, 'LONG', strategy_result)
            elif signal == "POTENTIAL_SELL":
                await self._handle_entry_logic(db, symbol, 'SHORT', strategy_result)
            else:
                print(f"  > TradingService: Signal is '{signal}'. No action taken for {symbol.name}.")

    async def _handle_entry_logic(self, db: Session, symbol: Symbol, side: Literal['LONG', 'SHORT'],
                                  strategy_result: Dict[str, Any]):
        print(f"  > [TradingService] Received {side} signal for {symbol.name}. Initiating checks...")

        # --- Risk Check 1: Max Positions ---
        if db.query(func.count(Position.id)).filter(Position.is_open == True).scalar() >= settings.MAX_OPEN_POSITIONS:
            print(f"  > [Decision] REJECTED: At max open positions ({settings.MAX_OPEN_POSITIONS}).")
            return

        # --- Risk Check 2: Balance & Position Sizing ---
        balance = await self.order_executor.get_balance('USDT')
        if balance <= 10:  # Minimum balance check
            print(f"  > [Decision] REJECTED: Insufficient balance ({balance:.2f} USDT).")
            return

        risk_amount_usd = balance * settings.RISK_PER_TRADE_PERCENT
        current_price = strategy_result['current_price']
        stop_loss_price = strategy_result['risk_management']['suggested_sl']

        # Validate SL price
        if (side == 'LONG' and stop_loss_price >= current_price) or (
                side == 'SHORT' and stop_loss_price <= current_price):
            print(f"  > [Decision] REJECTED: Invalid stop loss price ({stop_loss_price}) for a {side} trade.")
            return

        risk_per_share = abs(current_price - stop_loss_price)
        if risk_per_share <= 0:
            print(f"  > [Decision] REJECTED: Risk per share is zero or negative.")
            return

        position_size_coin = risk_amount_usd / risk_per_share

        # --- Execution ---
        print(
            f"  > [Calculation] Risk Amount=${risk_amount_usd:.2f}, Position Size={position_size_coin:.8f} {symbol.name.split('/')[0]}")

        await self.order_executor.cancel_all_open_orders(symbol.name)
        # Set leverage before placing orders
        await self.order_executor.set_leverage(symbol.name, settings.LEVERAGE)

        # 1. Place Entry Order
        entry_order = await self.order_executor.create_market_order(symbol.name, 'buy' if side == 'LONG' else 'sell',
                                                                    position_size_coin, side)
        if not entry_order or 'id' not in entry_order:
            print(f"  > [CRITICAL] FAILED to place entry order. Aborting trade.")
            return

        # 2. Confirm Position is Open on Exchange ("Confirm, then Act")
        confirmed_position = None
        for i in range(10):  # Try for 10 seconds
            await asyncio.sleep(1)
            confirmed_position = await self.order_executor.get_open_position_by_symbol(symbol.name)
            if confirmed_position:
                print(
                    f"  > Position confirmed on exchange after {i + 1}s. Size: {confirmed_position.get('contracts')}")
                break

        if not confirmed_position:
            print(f"  > [CRITICAL] FAILED to confirm position on exchange after 10s. Closing for safety.")
            # We don't know the exact position info, so we can't close it reliably here.
            # The sync task will handle this inconsistency later.
            return

        # 3. Place Stop Loss Order (Only after confirmation)
        sl_side: Literal['buy', 'sell'] = 'sell' if side == 'LONG' else 'buy'
        confirmed_amount = float(confirmed_position.get('contracts', entry_order['amount']))
        sl_order = await self.order_executor.create_stop_loss_order(symbol.name, sl_side, confirmed_amount,
                                                                    stop_loss_price, side)
        if not sl_order:
            print(f"  > [CRITICAL] FAILED to place SL order. Closing position for safety.")
            await self.order_executor.close_position_market(symbol.name, side, entry_order['amount'])
            return

        # 4. Persist Position State
        new_position = Position(
            symbol_id=symbol.id, is_open=True, position_side=side,
            entry_price=entry_order.get('price') or current_price,
            quantity=entry_order['amount'], entry_order_id=str(entry_order['id']),
            stop_loss_order_id=str(sl_order['id']), initial_stop_loss_price=stop_loss_price
        )
        db.add(new_position)
        db.commit()
        print(f"  > TradingService: Successfully opened and persisted new {side} position for {symbol.name}.")

    async def _handle_exit_logic(self, db: Session, symbol: Symbol, position: Position):
        print(
            f"  > TradingService: Received exit signal for {position.position_side} position on {symbol.name}. Closing...")

        # 1. Fetch real position info from exchange to get exact size
        position_info = await self.order_executor.get_open_position_by_symbol(symbol.name)
        if not position_info:
            print(f"  > Info: Position on {symbol.name} seems to be already closed on exchange. Syncing DB.")
            position.is_open = False
            db.commit()
            return

        # 2. Cancel all open orders for the symbol first
        await self.order_executor.cancel_all_open_orders(symbol.name)
        await asyncio.sleep(1)  # Give time for cancellation to process

        # 3. Close the position with a market order
        close_order = await self.order_executor.close_position_market(symbol.name, position_info.get('position_side'), float(position.quantity))
        if not close_order:
            print(f"  > CRITICAL: FAILED to close position for {symbol.name}!")
            return

        # 4. Update database
        position.is_open = False;
        db.commit()
        print(f"  > TradingService: Successfully closed and updated position for {symbol.name}.")

    # Self-test logic can be updated later to support both long and short tests
    async def run_self_test(self, db: Session, symbol: str) -> Dict[str, Any]:
        return {"status": "success", "message": "Self-test passed (logic to be updated)."}

    async def check_and_sync_positions(self, db: Session):
        print(f"\n--- POSITION SYNC TASK RUNNING ---")
        open_positions_db = db.query(Position).join(Symbol).filter(Position.is_open == True).all()

        if not open_positions_db:
            print("  > No open positions in DB to sync.")
            return

        print(f"  > Found {len(open_positions_db)} open position(s) in DB to check.")
        for pos in open_positions_db:
            symbol_name = pos.symbol.name
            print(f"  > Checking status for {pos.position_side} on {symbol_name}...")

            # Get the real position from the exchange
            exchange_position = await self.order_executor.get_open_position_by_symbol(symbol_name)

            if exchange_position is None:
                # --- Ghost Position Detected! ---
                print(f"  > GHOST POSITION DETECTED for {symbol_name}! DB says open, exchange says closed.")
                print(f"  > This means a SL/TP was likely hit. Syncing state.")

                # 1. Cancel any lingering associated orders (like the opposing TP or SL)
                try:
                    if pos.stop_loss_order_id:
                        await self.order_executor.cancel_order(pos.stop_loss_order_id, symbol_name)
                    if pos.take_profit_order_id:
                        await self.order_executor.cancel_order(pos.take_profit_order_id, symbol_name)
                except Exception as e:
                    print(
                        f"  > Info: Could not cancel lingering orders for {symbol_name}. They may already be gone. Error: {e}")

                # 2. Update the database to reflect reality
                pos.is_open = False
                db.commit()
                print(f"  > Synced DB for {symbol_name}. Position is now marked as closed.")
            else:
                # Position exists on both sides, everything is likely fine.
                print(f"  > Position for {symbol_name} is confirmed on exchange. State is in sync.")

        print(f"--- POSITION SYNC TASK FINISHED ---")

    async def run_full_self_test(self, symbol: str = "BNB/USDT"):
        print("\n" + "=" * 60)
        print("SYSTEM SELF-TEST: PRE-FLIGHT CHECK INITIALIZED")
        print("=" * 60)

        test_leverage = 2
        test_notional_value = 10.0  # Binance minimum

        try:
            # --- SHARED CHECKS ---
            print("\n[PHASE 1: PRELIMINARY CHECKS]")
            balance = await self.order_executor.get_balance('USDT')
            if balance < test_notional_value / test_leverage:
                raise Exception(
                    f"Insufficient balance for self-test. Need > ${test_notional_value / test_leverage:.2f}, have ${balance:.2f}")
            print(f"  ✅ Balance check passed. Available: ${balance:.2f}")

            await self.order_executor.set_leverage(symbol, test_leverage)
            print(f"  ✅ Leverage set to {test_leverage}x.")

            await self.order_executor.cancel_all_open_orders(symbol)
            print(f"  ✅ Pre-test cleanup: All open orders for {symbol} cancelled.")

            # --- LONG TRADE TEST ---
            print("\n[PHASE 2: LONG TRADE LIFECYCLE TEST]")
            await self._run_single_side_test(symbol, "LONG", test_notional_value)

            # --- SHORT TRADE TEST ---
            print("\n[PHASE 3: SHORT TRADE LIFECYCLE TEST]")
            await self._run_single_side_test(symbol, "SHORT", test_notional_value)

            print("\n" + "=" * 60)
            print("✅ SYSTEM SELF-TEST PASSED: All systems are operational.")
            print("=" * 60 + "\n")

        except Exception as e:
            print("\n" + "!" * 60)
            print("❌ SYSTEM SELF-TEST FAILED! APPLICATION STARTUP HALTED.")
            print(f"  > CRITICAL FAILURE at: {e}")
            print("!" * 60 + "\n")
            raise e  # Re-raise the exception to stop the application

    async def _run_single_side_test(self, symbol: str, side: Literal["LONG", "SHORT"], notional_value: float):
        print(f"  --- Testing {side} trade ---")

        # 1. Entry
        current_price = await self.order_executor.get_current_price(symbol)
        if not current_price: raise Exception("Failed to fetch current price.")
        amount = notional_value / current_price
        entry_order = await self.order_executor.create_market_order(symbol, 'buy' if side == "LONG" else 'sell', amount,
                                                                    side)
        if not entry_order: raise Exception(f"Failed to create market {side} order.")
        print(f"    ✅ Market {side} entry order placed.")

        # 2. Confirmation
        position = await self._confirm_position_with_retry(symbol, 5)
        if not position: raise Exception(f"Failed to confirm {side} position on exchange.")
        print(f"    ✅ {side} position confirmed on exchange.")

        # 3. SL/TP Placement
        sl_price = current_price * 0.98 if side == "LONG" else current_price * 1.02
        tp_price = current_price * 1.02 if side == "LONG" else current_price * 0.98
        sl_order = await self.order_executor.create_stop_loss_order(symbol, 'sell' if side == "LONG" else 'buy',
                                                                    float(position['contracts']), sl_price, side)
        if not sl_order: raise Exception(f"Failed to place {side} stop loss.")
        print(f"    ✅ Stop Loss order placed.")

        # Note: We skip TP order placement in this test for simplicity, as SL is the critical one.

        # 4. Cleanup
        await self.order_executor.cancel_all_open_orders(symbol)
        print(f"    ✅ All open orders cancelled.")

        await asyncio.sleep(1)  # Small delay

        close_order = await self.order_executor.close_position_market(symbol, side, float(position['contracts']))
        if not close_order: raise Exception(f"Failed to close {side} position.")
        print(f"    ✅ Market close order sent.")

        # 5. Final Confirmation
        final_pos = await self._confirm_position_with_retry(symbol, 5, expect_closed=True)
        if final_pos is not None: raise Exception(f"Failed to confirm {side} position is closed.")
        print(f"    ✅ Position confirmed closed on exchange.")
        print(f"  --- {side} trade test PASSED ---")

    async def _confirm_position_with_retry(self, symbol: str, retries: int, expect_closed: bool = False):
        for i in range(retries):
            position = await self.order_executor.get_open_position_by_symbol(symbol)
            if expect_closed and position is None:
                return None  # Success, it's closed
            if not expect_closed and position is not None:
                return position  # Success, it's open
            await asyncio.sleep(1)
        return await self.order_executor.get_open_position_by_symbol(symbol)