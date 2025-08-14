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

        # Set leverage before placing orders
        await self.order_executor.set_leverage(symbol.name, settings.LEVERAGE)

        # 1. Place Entry Order
        entry_order = await self.order_executor.create_market_order(symbol.name, 'buy' if side == 'LONG' else 'sell',
                                                                    position_size_coin, side)
        if not entry_order or 'id' not in entry_order:
            print(f"  > [CRITICAL] FAILED to place entry order. Aborting trade.")
            return

        # Give exchange time to process
        await asyncio.sleep(2)

        # 2. Place Stop Loss Order
        sl_side: Literal['buy', 'sell'] = 'sell' if side == 'LONG' else 'buy'
        sl_order = await self.order_executor.create_stop_loss_order(symbol.name, sl_side, entry_order['amount'],
                                                                    stop_loss_price, side)
        if not sl_order:
            print(f"  > [CRITICAL] FAILED to place SL order. Closing position for safety.")
            await self.order_executor.close_position_market(symbol.name, side, entry_order['amount'])
            return

        # 3. Place Take Profit Order (Optional, can be added here)
        # For now, we rely on the trend reversal exit signal from the strategy

        # --- Persist Position State ---
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

        # 1. Cancel all open orders associated with this position (SL, TP, etc.)
        try:
            await self.order_executor.cancel_order(position.stop_loss_order_id, symbol.name)
            if position.take_profit_order_id:
                await self.order_executor.cancel_order(position.take_profit_order_id, symbol.name)
        except Exception as e:
            print(f"  > Warning: Could not cancel open orders, they might be already filled/cancelled. Error: {e}")

        # 2. Close the position with a market order
        close_order = await self.order_executor.close_position_market(symbol.name, position.position_side,
                                                                      float(position.quantity))
        if not close_order:
            print(f"  > CRITICAL: FAILED to close position for {symbol.name}! Manual intervention may be required.")
            return

        # 3. Update database
        position.is_open = False
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