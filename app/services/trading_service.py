# app/services/trading_service.py

from typing import Dict, Any
from sqlalchemy import func
from sqlalchemy.orm import Session


from .order_executor import OrderExecutor
from .parameter_manager import ParameterManager
from ..core.config import settings
from ..db.models import Position, Symbol


class TradingService:
    """
    The "Commander" of the trading system.
    It receives signals, manages position state, calculates risk,
    and makes the final decision to execute trades.
    """

    def __init__(self, order_executor: OrderExecutor, param_manager: ParameterManager):
        self.order_executor = order_executor
        self.param_manager = param_manager

    async def process_signal(self, db: Session, symbol: Symbol, strategy_result: Dict[str, Any]):
        """
        The main entry point for processing a new signal from the AnalysisService.
        """
        signal = strategy_result.get("overall_signal")

        # Check if we have an open position for this symbol
        open_position = db.query(Position).filter(
            Position.symbol_id == symbol.id, Position.is_open == True
        ).first()

        if open_position:
            if strategy_result.get("exit_signal"):
                await self._handle_exit_logic(db, symbol, open_position, strategy_result)
            else:
                print(f"  > TradingService: Holding position for {symbol.name}.")
        else:
            if signal == "POTENTIAL_BUY":
                await self._handle_entry_logic(db, symbol, strategy_result)

    async def _handle_entry_logic(self, db: Session, symbol: Symbol, strategy_result: Dict[str, Any]):
        print(f"  > TradingService: Received POTENTIAL_BUY for {symbol.name}.")

        current_open_positions = db.query(func.count(Position.id)).filter(Position.is_open == True).scalar()
        if current_open_positions >= settings.MAX_OPEN_POSITIONS:
            print(f"  > Risk Check FAILED: At max open positions ({settings.MAX_OPEN_POSITIONS}).")
            return

        balance = await self.order_executor.get_balance('USDT')
        if balance <= 0:
            print("  > Risk Check FAILED: Insufficient balance.")
            return

        risk_amount_per_trade = balance * settings.RISK_PER_TRADE_PERCENT
        stop_loss_price = strategy_result['risk_management']['suggested_sl']
        current_price = strategy_result['current_price']

        if not stop_loss_price or stop_loss_price >= current_price:
            print(f"  > Risk Check FAILED: Invalid stop loss price ({stop_loss_price}).")
            return

        risk_per_share = current_price - stop_loss_price
        if risk_per_share <= 0:
            print(f"  > Risk Check FAILED: Risk per share is zero or negative.")
            return
        order_amount = round(risk_amount_per_trade / risk_per_share, 3)

        print(
            f"  > Order Calc: Balance={balance:.2f}, Risk Amount={risk_amount_per_trade:.2f}, Order Size={order_amount}")

        if order_amount <= 0:
            print("  > Risk Check FAILED: Calculated order amount is zero.")
            return

        await self.order_executor.set_leverage(symbol.name, settings.LEVERAGE)
        entry_order = await self.order_executor.create_market_order(symbol.name, 'buy', order_amount)
        if not entry_order: return

        sl_order = await self.order_executor.create_stop_loss_order(symbol.name, 'sell', order_amount, stop_loss_price)
        if not sl_order:
            print(f"  > CRITICAL: FAILED to place SL. Closing position for safety.")
            await self.order_executor.close_market_position(symbol.name, 'LONG', order_amount)
            return

        new_position = Position(
            symbol_id=symbol.id, is_open=True, position_side='LONG',
            entry_price=entry_order.get('price') or current_price,
            quantity=entry_order['amount'], entry_order_id=str(entry_order['id']),
            stop_loss_order_id=str(sl_order['id']), initial_stop_loss_price=stop_loss_price
        )
        db.add(new_position)
        db.commit()
        print(f"  > TradingService: Successfully opened and persisted new LONG position for {symbol.name}.")

    async def _handle_exit_logic(self, db: Session, symbol: Symbol, position: Position,
                                 strategy_result: Dict[str, Any]):
        print(f"  > TradingService: Received EXIT_LONG signal for {symbol.name}. Closing position.")

        # 1. Close the position with a market order
        close_order = await self.order_executor.close_market_position(
            symbol.name, 'LONG', float(position.quantity)
        )
        if not close_order:
            print(f"  > CRITICAL: FAILED to close position for {symbol.name}. Manual intervention required!")
            return

        # 2. Cancel the now-redundant stop loss order
        await self.order_executor.cancel_order(position.stop_loss_order_id, symbol.name)

        # 3. Update the position status in the database
        position.is_open = False
        db.commit()
        print(f"  > TradingService: Successfully closed and updated position for {symbol.name}.")