# api/endpoints/backtest.py

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.params import Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List

from app.job_manager import create_task, get_task_status, complete_task, fail_task
from app.services.backtest.optimizer import GridSearchOptimizer
from ...containers import container

router = APIRouter()


class SizerConfig(BaseModel):
    type: str = "Percent"  # "Percent" or "Fixed"
    # Use Field to allow flexible parameters for different sizers
    params: Dict[str, Any] = Field(default_factory=lambda: {"percents": 10})


class OptimizationRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    cash: float
    param_grid: Dict[str, List[Any]]
    sizer: SizerConfig = Field(default_factory=SizerConfig)  # Add sizer to the request
    signal_timeframe: str = "15m"


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

def background_optimizer_task(optimizer: GridSearchOptimizer, task_id: str):
    """
    This function will be run in the background.
    It now receives a pre-configured optimizer instance.
    """
    try:
        best_result = optimizer.run_optimization(task_id=task_id)
        complete_task(task_id, best_result)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        fail_task(task_id, str(error_details))


@router.post("/start-optimization")
def start_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    # FastAPI will call get_optimizer to create and inject the optimizer instance
    optimizer: GridSearchOptimizer = Depends(get_optimizer)
):
    """
    Starts a strategy optimization process in the background.
    """
    task_id = create_task()
    # Pass the created optimizer instance to the background task
    background_tasks.add_task(background_optimizer_task, optimizer, task_id)
    return {"message": "Optimization task started successfully.", "task_id": task_id}

@router.get("/task-status/{task_id}")
def poll_task_status(task_id: str):
    status = get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status