# app/job_manager.py

import datetime
import uuid
from typing import Dict, Any

# The TTL (Time-to-Live) for completed tasks, in seconds.
# Tasks older than this will be purged from memory.
# 2 hours = 7200 seconds.
TASK_TTL_SECONDS = 7200


class JobManager:
    """
    A class-based job manager that stores task results in memory
    and includes a cleanup mechanism to prevent memory leaks.
    """

    def __init__(self):
        # This dictionary will now be an instance variable, not a global one.
        self.tasks_db: Dict[str, Dict[str, Any]] = {}
        print("JobManager initialized.")

    def create_task(self) -> str:
        """Creates a new task entry and returns its ID."""
        task_id = str(uuid.uuid4())
        self.tasks_db[task_id] = {
            "status": "PENDING",
            "result": None,
            "progress": {"current": 0, "total": 0, "step": "Initializing"},
            "created_at": datetime.datetime.now(datetime.timezone.utc)
        }
        return task_id

    def get_task_status(self, task_id: str) -> Dict[str, Any] | None:
        """Gets the status of a task."""
        return self.tasks_db.get(task_id)

    def update_task_progress(self, task_id: str, current: int, total: int, step: str):
        """Updates the progress of a task."""
        if task_id in self.tasks_db:
            self.tasks_db[task_id]["status"] = "PROGRESS"
            self.tasks_db[task_id]["progress"] = {"current": current, "total": total, "step": step}

    def _update_task_as_finished(self, task_id: str, status: str, result: Any):
        """Internal helper to mark a task as finished and clear progress."""
        if task_id in self.tasks_db:
            self.tasks_db[task_id]["status"] = status
            self.tasks_db[task_id]["result"] = result
            self.tasks_db[task_id]["progress"] = {}  # Clear progress
            # Add a timestamp for when it finished, for TTL calculation.
            self.tasks_db[task_id]["finished_at"] = datetime.datetime.now(datetime.timezone.utc)

    def complete_task(self, task_id: str, result: Any):
        """Marks a task as complete and stores the result."""
        self._update_task_as_finished(task_id, "SUCCESS", result)

    def fail_task(self, task_id: str, error_message: str):
        """Marks a task as failed and stores the error."""
        self._update_task_as_finished(task_id, "FAILURE", error_message)

    def cleanup_old_tasks(self):
        """
        [THE FIX] This method iterates through the tasks and removes old ones.
        It should be called periodically by a scheduler.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        tasks_to_delete = []

        print(f"Running task cleanup. Current task count: {len(self.tasks_db)}")

        for task_id, task_data in self.tasks_db.items():
            finished_at = task_data.get("finished_at")
            if finished_at:
                age = (now - finished_at).total_seconds()
                if age > TASK_TTL_SECONDS:
                    tasks_to_delete.append(task_id)

        if tasks_to_delete:
            for task_id in tasks_to_delete:
                del self.tasks_db[task_id]
            print(f"Cleaned up {len(tasks_to_delete)} old tasks.")
        else:
            print("No old tasks to clean up.")