# app/job_manager.py

from typing import Dict, Any
import uuid

# This is our simple, in-memory "database" for tasks.
# In a real multi-server setup, this would be Redis or a database table.
tasks_db: Dict[str, Dict[str, Any]] = {}

def create_task() -> str:
    """Creates a new task entry and returns its ID."""
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {
        "status": "PENDING",
        "result": None,
        "progress": {"current": 0, "total": 0, "step": "Initializing"}
    }
    return task_id

def get_task_status(task_id: str) -> Dict[str, Any] | None:
    """Gets the status of a task."""
    return tasks_db.get(task_id)

def update_task_progress(task_id: str, current: int, total: int, step: str):
    """Updates the progress of a task."""
    if task_id in tasks_db:
        tasks_db[task_id]["status"] = "PROGRESS"
        tasks_db[task_id]["progress"] = {"current": current, "total": total, "step": step}

def complete_task(task_id: str, result: Any):
    """Marks a task as complete and stores the result."""
    if task_id in tasks_db:
        tasks_db[task_id]["status"] = "SUCCESS"
        tasks_db[task_id]["result"] = result
        tasks_db[task_id]["progress"] = {} # Clear progress

def fail_task(task_id: str, error_message: str):
    """Marks a task as failed and stores the error."""
    if task_id in tasks_db:
        tasks_db[task_id]["status"] = "FAILURE"
        tasks_db[task_id]["result"] = error_message
        tasks_db[task_id]["progress"] = {} # Clear progress