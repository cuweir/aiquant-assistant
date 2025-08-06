# app/services/trading_service.py

from typing import Dict, Any
from sqlalchemy.orm import Session

from .order_executor import OrderExecutor
from .parameter_manager import ParameterManager
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
            # If a position is open, we only care about exit signals.
            await self._handle_exit_logic(db, symbol, open_position, strategy_result)
        else:
            # If no position is open, we only care about entry signals.
            if signal == "POTENTIAL_BUY":
                await self._handle_entry_logic(db, symbol, strategy_result)

    async def _handle_entry_logic(self, db: Session, symbol: Symbol, strategy_result: Dict[str, Any]):
        """Handles the logic for opening a new position."""
        print(f"  > TradingService: Received POTENTIAL_BUY for {symbol.name}. Executing entry logic.")

        # --- [RISK MANAGEMENT] ---
        # This is where we implement the "Total Risk Budget" model.
        # For now, we'll keep it simple with a fixed order amount.
        # TODO: Implement dynamic order size calculation based on risk.
        order_amount = 0.001  # Example: buy 0.001 BTC on testnet

        # --- [EXECUTION] ---
        # 1. Place the market buy order
        entry_order = await self.order_executor.create_market_order(symbol.name, 'buy', order_amount)
        if not entry_order:
            print(f"  > FAILED to place entry order for {symbol.name}. Aborting.")
            return

        # 2. Place the stop loss order
        sl_price = strategy_result['risk_management']['suggested_sl']
        sl_order = await self.order_executor.create_stop_loss_order(symbol.name, 'sell', order_amount, sl_price)
        if not sl_order:
            print(f"  > CRITICAL: FAILED to place stop loss. Attempting to close position for safety.")
            await self.order_executor.close_market_position(symbol.name, 'LONG', order_amount)
            return

        # 3. [PERSISTENCE] Save the new position to the database
        new_position = Position(
            symbol_id=symbol.id,
            is_open=True,
            position_side='LONG',
            entry_price=entry_order.get('price') or strategy_result['current_price'],
            quantity=entry_order['amount'],
            entry_order_id=entry_order['id'],
            stop_loss_order_id=sl_order['id'],
            initial_stop_loss_price=sl_price
        )
        db.add(new_position)
        db.commit()
        print(f"  > TradingService: Successfully opened and persisted new LONG position for {symbol.name}.")

    async def _handle_exit_logic(self, db: Session, symbol: Symbol, position: Position,
                                 strategy_result: Dict[str, Any]):
        """Handles the logic for monitoring and closing an open position."""
        # The exit signal is the trend reversal (death cross)
        # This logic is now inside the strategy, we just need to check the result.
        # For simplicity, we assume the strategy would return an 'EXIT_SIGNAL' if conditions are met.
        # This part needs to be built out in the MultiIndicatorStrategy.
        # For now, we'll simulate it.

        # A real implementation would get the exit signal from strategy_result
        # if strategy_result.get("overall_signal") == "EXIT_LONG":
        #    ... close position ...

        print(f"  > TradingService: Monitoring open position for {symbol.name}. No exit signal found in this cycle.")
        # [TODO] Add the full exit logic here, which would mirror the backtest's exit conditions.
        pass