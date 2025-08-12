from fastapi import APIRouter, HTTPException, Depends
from typing import List

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

    # The service now returns the full report object, which we can directly return
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
    Fetches the most recent analysis results from the database.
    """
    db_results = service.get_all_analyses_from_db(db=db, skip=0, limit=20)
    response_list: List[AnalysisReport] = []
    for r in db_results:
        details_json = r.indicator_details or {}

        try:
            report_data = {
                # --- Core fields from the main table ---
                "timestamp": r.timestamp,
                "symbol": r.symbol.name,  # Get name from the relationship
                "timeframe": r.timeframe,
                "price": float(r.current_price),  # Convert Decimal to float
                "signal": r.overall_signal,
                "ai_analysis": r.llm_analysis,

                # --- Nested fields from the JSONB details ---
                "risk_management": details_json.get("risk_management"),
                "confidence": details_json.get("confidence"),
                "key_factors": details_json.get("key_factors"),
                "snapshot": details_json.get("snapshot")
            }

            # Use Pydantic to validate the final constructed data
            report = AnalysisReport.model_validate(report_data)
            response_list.append(report)

        except (ValidationError, TypeError) as e:
            print(f"Skipping malformed DB record {r.id} due to validation error: {e}")
            continue
    return response_list