# api/endpoints/backtest.py

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List

from app.job_manager import create_task, get_task_status, complete_task, fail_task
from app.services.backtest.optimizer import GridSearchOptimizer

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


def background_optimizer_task(config: OptimizationRequest, task_id: str):
    """
    This function will be run in the background.
    It performs the optimization and updates the task status.
    """
    try:
        base_config = {
            "symbol": config.symbol,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "cash": config.cash,
            "sizer": config.sizer.model_dump()
        }
        optimizer = GridSearchOptimizer(base_config=base_config, param_grid=config.param_grid)
        best_result = optimizer.run_optimization(task_id=task_id)

        complete_task(task_id, best_result)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Task [{task_id}] failed catastrophically: {e}")
        fail_task(task_id, str(error_details))


@router.post("/start-optimization")
def start_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    task_id = create_task()
    background_tasks.add_task(background_optimizer_task, request, task_id)
    return {"message": "Optimization task started successfully.", "task_id": task_id}


@router.get("/task-status/{task_id}")
def poll_task_status(task_id: str):
    status = get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status