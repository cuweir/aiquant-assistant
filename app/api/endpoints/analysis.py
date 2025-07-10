from fastapi import APIRouter, HTTPException, Depends
from typing import Dict

from ...models.schemas import SignalInput, AIAnalysisOutput, AllAnalysesOutput
from ...services.analysis_service import AnalysisService, analysis_service

router = APIRouter()

# Dependency Injection for the service
def get_analysis_service():
    return analysis_service

@router.post("/trigger-analysis", response_model=AIAnalysisOutput)
async def trigger_comprehensive_ai_analysis(
    signal_input: SignalInput,
    service: AnalysisService = Depends(get_analysis_service)
):
    result = await service.generate_comprehensive_analysis(
        symbol=signal_input.symbol,
        timeframe=signal_input.timeframe
    )
    if not result:
        raise HTTPException(status_code=404, detail="Analysis could not be generated.")
    return AIAnalysisOutput(**result)

@router.get("/get-all-analyses", response_model=AllAnalysesOutput)
async def get_all_analyses_endpoint(
    service: AnalysisService = Depends(get_analysis_service)
):
    all_data = await service.get_all_cached_analyses()
    validated_analyses: Dict[str, AIAnalysisOutput] = {}
    for key, value_dict in all_data.items():
        try:
            validated_analyses[key] = AIAnalysisOutput(**value_dict)
        except Exception as e:
            print(f"Skipping cache entry {key} due to data error: {e}")
    return AllAnalysesOutput(analyses=validated_analyses)