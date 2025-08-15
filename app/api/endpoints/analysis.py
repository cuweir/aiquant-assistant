from fastapi import APIRouter, HTTPException, Depends
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
    [FINAL & SIMPLIFIED]
    Fetches the most recent analysis results from the database.
    The service layer now handles all data transformation.
    """
    # The service method now returns a list of dictionaries that are already
    # perfectly formatted to match the AnalysisReport Pydantic model.
    # No further processing is needed here.
    results = service.get_all_analyses_from_db(db=db, skip=0, limit=20)

    # We can directly return the results. FastAPI will automatically validate
    # them against the response_model.
    return results