# api/endpoints/backtest.py

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List

from app.job_manager import create_task, get_task_status, complete_task, fail_task
from app.services.backtest.optimizer import GridSearchOptimizer

router = APIRouter()


class OptimizationRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    cash: float
    param_grid: Dict[str, List[Any]]


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
            "cash": config.cash
        }
        optimizer = GridSearchOptimizer(base_config=base_config, param_grid=config.param_grid)
        best_result = optimizer.run_optimization(task_id=task_id)

        # When done, mark the task as complete with the result
        complete_task(task_id, best_result)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Task [{task_id}] failed catastrophically: {e}")
        # If something goes wrong, mark the task as failed
        fail_task(task_id, str(error_details))


@router.post("/start-optimization")
def start_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """
    Starts a strategy optimization process in the background using FastAPI's built-in mechanism.
    Returns a task ID for polling the status and result.
    """
    task_id = create_task()

    # Add the long-running process to the background
    background_tasks.add_task(background_optimizer_task, request, task_id)

    return {"message": "Optimization task started successfully.", "task_id": task_id}


@router.get("/task-status/{task_id}")
def poll_task_status(task_id: str):
    """
    Polls the status of a background task using our in-memory job manager.
    """
    status = get_task_status(task_id)
    print(f"Polling task {task_id}, current status being returned: {status}")
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")

    return status