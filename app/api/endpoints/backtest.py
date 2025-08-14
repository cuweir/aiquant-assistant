# api/endpoints/backtest.py

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.params import Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List

from app.job_manager import JobManager
from app.services.backtest.optimizer import GridSearchOptimizer
from ...containers import container

router = APIRouter()

class SizerConfig(BaseModel):
    type: str = "Percent"
    params: Dict[str, Any] = Field(default_factory=lambda: {"percents": 10})

class OptimizationRequest(BaseModel):
    strategy_name: str = "MtfaStrategyAlpha"
    symbol: str
    start_date: str
    end_date: str
    cash: float
    param_grid: Dict[str, List[Any]]
    sizer: SizerConfig = Field(default_factory=SizerConfig)
    signal_timeframe: str = "1h"

def get_job_manager() -> JobManager:
    return container.job_manager

def get_optimizer(request: OptimizationRequest) -> GridSearchOptimizer:
    base_config = {
        "symbol": request.symbol,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "cash": request.cash,
        "sizer": request.sizer.model_dump(),
        "signal_timeframe": request.signal_timeframe
    }
    return GridSearchOptimizer(base_config=base_config, param_grid=request.param_grid)

def background_optimizer_task(
    optimizer: GridSearchOptimizer,
    task_id: str,
    job_manager: JobManager,
    strategy_name: str
):
    try:
        best_result = optimizer.run_optimization(task_id, job_manager, strategy_name)
        job_manager.complete_task(task_id, best_result)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        job_manager.fail_task(task_id, str(error_details))

@router.post("/start-optimization")
def start_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    optimizer: GridSearchOptimizer = Depends(get_optimizer),
    job_manager: JobManager = Depends(get_job_manager)
):
    task_id = job_manager.create_task()
    background_tasks.add_task(background_optimizer_task, optimizer, task_id, job_manager, request.strategy_name)
    return {"message": "Optimization task started successfully.", "task_id": task_id}

@router.get("/task-status/{task_id}")
def poll_task_status(
    task_id: str,
    job_manager: JobManager = Depends(get_job_manager)
):
    status = job_manager.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status