from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Any, Dict

from pydantic import ValidationError
from sqlalchemy.orm import Session

from ...models.schemas import AnalysisReport, ManualTriggerInput
from ...services.analysis_service import AnalysisService
from ...db.session import get_db
from ...containers import container

router = APIRouter()


def get_analysis_service():
    return container.analysis_service


@router.post("/trigger-analysis", response_model=AnalysisReport)
async def trigger_manual_analysis(
        trigger_input: ManualTriggerInput,
        service: AnalysisService = Depends(get_analysis_service)
):
    """
    Manually triggers a comprehensive analysis for a single symbol
    and returns the complete, new AnalysisReport.
    """
    print(f"\n--- [MANUAL TRIGGER] Received request to analyze {trigger_input.symbol} ---")
    result = await service.generate_comprehensive_analysis(trigger_input.symbol)
    if result is None:
        raise HTTPException(status_code=500, detail="Analysis failed or produced no result. Check server logs.")
    return result


@router.get("/get-all-analyses", response_model=List[AnalysisReport])
async def get_all_analyses_endpoint(
        db: Session = Depends(get_db),
        service: AnalysisService = Depends(get_analysis_service)
):
    """
    Fetches the MOST RECENT analysis result for each monitored symbol.
    """
    results = service.get_all_analyses_from_db(db=db, skip=0, limit=20)
    return results


@router.get("/analysis-history/{symbol}", response_model=List[AnalysisReport])
async def get_analysis_history(
        symbol: str,
        hours: int = Query(12, ge=1, le=72),  # Default 12h, min 1h, max 72h
        db: Session = Depends(get_db),
        service: AnalysisService = Depends(get_analysis_service)
):
    """
    Fetches the analysis history for a specific symbol over a given
    number of past hours.
    """
    # FastAPI automatically handles the conversion of BTC-USDT to BTC/USDT if needed,
    # but it's good practice to standardize.
    formatted_symbol = symbol.upper().replace("-", "/")

    history = service.get_analysis_history_from_db(db=db, symbol_name=formatted_symbol, hours=hours)

    if not history:
        # It's not an error if there's no history, just return an empty list.
        return []

    return history