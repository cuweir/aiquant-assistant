from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from sqlalchemy.orm import Session

from ...models.schemas import SignalInput, AIAnalysisOutput, AnalysisDetails, SignalDetail
from ...services.analysis_service import AnalysisService, analysis_service
from ...db.session import get_db

router = APIRouter()


def get_analysis_service():
    return analysis_service


@router.post("/trigger-analysis", response_model=AIAnalysisOutput)
async def trigger_comprehensive_ai_analysis(
        signal_input: SignalInput,
        service: AnalysisService = Depends(get_analysis_service)
):
    result_dict = await service.generate_comprehensive_analysis(
        symbol=signal_input.symbol,
        timeframe=signal_input.timeframe
    )
    if not result_dict:
        raise HTTPException(status_code=500, detail="Analysis could not be generated or saved.")
    # The returned dict is already shaped like the Pydantic model
    return result_dict


@router.get("/get-all-analyses", response_model=List[AIAnalysisOutput])
async def get_all_analyses_endpoint(
        db: Session = Depends(get_db),  # Inject the database session
        service: AnalysisService = Depends(get_analysis_service),
        skip: int = 0,
        limit: int = 20  # Add pagination
):
    db_results = service.get_all_analyses_from_db(db=db, skip=skip, limit=limit)

    # Convert DB ORM objects to Pydantic models for the response
    response_list = []
    for r in db_results:
        details_data = None
        if r.indicator_details is not None and r.composite_score is not None:
            details_data = AnalysisDetails(
                composite_score=r.composite_score,
                individual_signals_details=[SignalDetail(**detail) for detail in r.indicator_details]
            )
        # Safely get RSI value from the JSONB field
        rsi_val = float('nan')
        if r.indicator_details:
            rsi_val = next((d.get('value') for d in r.indicator_details if d.get('indicator') == 'RSI'), float('nan'))

        response_list.append(
            AIAnalysisOutput(
                timestamp=r.timestamp,
                symbol=r.symbol.name,  # Access related symbol name via relationship
                timeframe=r.timeframe,
                local_signal=r.overall_signal,
                rsi=rsi_val,
                price=float(r.current_price),
                ai_analysis=r.llm_analysis,
                stop_loss=float(r.suggested_sl) if r.suggested_sl else None,
                take_profit=float(r.suggested_tp) if r.suggested_tp else None,
                details=details_data
            )
        )
    return response_list