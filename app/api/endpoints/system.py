# app/api/endpoints/system.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...containers import container
from ...services.trading_service import TradingService
from ...db.session import get_db

router = APIRouter()


def get_trading_service() -> TradingService:
    return container.trading_service


@router.post("/run-self-test/{symbol}")
async def run_self_test_endpoint(
        symbol: str,
        db: Session = Depends(get_db),
        trading_service: TradingService = Depends(get_trading_service)
):
    """
    Triggers a full end-to-end system functionality test for a given symbol.
    This will execute a quick round-trip trade on the exchange (testnet by default).
    """
    # We need to format the symbol to match CCXT's format (e.g., BTC/USDT)
    formatted_symbol = symbol.upper().replace("-", "/")

    results = await trading_service.run_self_test(db, formatted_symbol)
    return results