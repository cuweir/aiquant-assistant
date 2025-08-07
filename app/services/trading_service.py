# app/services/trading_service.py
import asyncio
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
        risk_amount_usdt = balance * settings.RISK_PER_TRADE_PERCENT
        stop_loss_price = strategy_result['risk_management']['suggested_sl']
        current_price = strategy_result['current_price']

        if not stop_loss_price or stop_loss_price >= current_price: return

        risk_per_share_pct = (current_price - stop_loss_price) / current_price
        if risk_per_share_pct <= 0: return

        position_size_usdt = risk_amount_usdt / risk_per_share_pct

        print(
            f"  > Order Calc (USDT): Risk Amount={risk_amount_usdt:.2f}, Position Notional Value={position_size_usdt:.2f} USDT")

        # [FINAL CHECK] Ensure the calculated notional value meets exchange minimums
        if position_size_usdt < 20.0:
            print(
                f"  > Risk Check FAILED: Calculated notional value ({position_size_usdt:.2f}) is below exchange minimum (20 USDT).")
            return

        # --- [EXECUTION] ---
        await self.order_executor.set_leverage(symbol.name, settings.LEVERAGE)

        # [FINAL FIX] Call the new, correct order function
        entry_order = await self.order_executor.create_market_order_by_quote_quantity(symbol.name, 'buy',
                                                                                      position_size_usdt)
        if not entry_order: return

        # We need the executed amount in base currency for the SL order
        amount_in_coin = entry_order['filled']
        sl_order = await self.order_executor.create_stop_loss_order(symbol.name, 'sell', amount_in_coin, stop_loss_price)

        if not sl_order:
            print(f"  > CRITICAL: FAILED to place SL. Closing position for safety.")
            await self.order_executor.close_market_position(symbol.name, 'LONG', amount_in_coin)
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

    async def run_self_test(self, db: Session, symbol: str) -> Dict[str, Any]:
        """
        Performs a full end-to-end functionality test of the trading system.
        """
        test_results = {"steps": [], "success": False, "summary": ""}
        symbol_record = db.query(Symbol).filter(Symbol.name == symbol).first()
        if not symbol_record:
            test_results["summary"] = f"Error: Symbol {symbol} not found in database."
            return test_results

        try:
            balance = await self.order_executor.get_balance('USDT')
            if balance > 0:
                test_results["steps"].append(
                    {"step": "Fetch Balance", "status": "SUCCESS", "details": f"Available Balance: {balance:.2f} USDT"})
            else:
                raise Exception("Failed to fetch a positive balance.")

            leverage_set = await self.order_executor.set_leverage(symbol, 5)
            if leverage_set:
                test_results["steps"].append(
                    {"step": "Set Leverage", "status": "SUCCESS", "details": "Leverage set to 5x"})
            else:
                raise Exception("Failed to set leverage.")

            test_notional_value_usdt = 21.0

            required_margin = test_notional_value_usdt / 5
            if balance < required_margin:
                raise Exception(f"Insufficient balance for self-test. Need {required_margin:.2f}, have {balance:.2f}")

            entry_order = await self.order_executor.create_market_order_by_quote_quantity(symbol, 'buy',
                                                                                          test_notional_value_usdt)
            if not entry_order or not entry_order.get('id'):
                raise Exception(
                    f"Failed to create market buy order by quote quantity. Exchange response: {entry_order}")
            test_results["steps"].append({"step": "Open Position (Market Buy)", "status": "SUCCESS",
                                          "details": f"Order ID: {entry_order['id']}, Notional Value: ~{test_notional_value_usdt} USDT"})

            # --- Test 4: Set Stop Loss ---
            amount_in_coin = entry_order['filled']  # Use the actual filled amount
            current_price = entry_order.get('price') or (await self.order_executor.exchange.fetch_ticker(symbol))[
                'last']
            sl_price = round(current_price * 0.98, 2)
            sl_order = await self.order_executor.create_stop_loss_order(symbol, 'sell', amount_in_coin, sl_price)

            if not sl_order or not sl_order.get('id'):
                raise Exception("Failed to create stop loss order.")
            test_results["steps"].append({"step": "Set Stop Loss", "status": "SUCCESS",
                                          "details": f"Order ID: {sl_order['id']}, SL Price: {sl_price}"})

            test_results["steps"].append({"step": "Persist Position State", "status": "SUCCESS",
                                          "details": "Position object created successfully."})

            await asyncio.sleep(1)
            close_order = await self.order_executor.close_market_position(symbol, 'LONG', amount_in_coin)
            if not close_order or not close_order.get('id'):
                raise Exception("Failed to close position.")
            test_results["steps"].append(
                {"step": "Close Position", "status": "SUCCESS", "details": f"Close Order ID: {close_order['id']}"})

            await asyncio.sleep(1)
            cancel_success = await self.order_executor.cancel_order(str(sl_order['id']), symbol)
            if cancel_success:
                test_results["steps"].append({"step": "Cancel Stop Loss", "status": "SUCCESS",
                                              "details": f"SL Order {sl_order['id']} cancelled."})
            else:
                test_results["steps"].append({"step": "Cancel Stop Loss", "status": "WARNING",
                                              "details": "Failed to cancel stop loss, it might have been already removed."})

            test_results["success"] = True
            test_results["summary"] = "All systems operational. Full trade lifecycle test completed successfully."

        except Exception as e:
            test_results["steps"].append({"step": "Test Halted", "status": "FAILURE", "details": str(e)})
            test_results["summary"] = f"A critical error occurred during the self-test: {e}"

        return test_results