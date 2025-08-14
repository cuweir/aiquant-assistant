import itertools
from typing import Dict, Any, List
import backtrader as bt
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# [MODIFIED] Import the new runner function
from .runner import run_single_backtest_instance
from app.job_manager import JobManager


class GridSearchOptimizer:
    def __init__(self, base_config: Dict[str, Any], param_grid: Dict[str, List[Any]]):
        self.base_config = base_config
        self.param_grid = param_grid
        self.results = []
        # Determine the number of CPU cores to use, leaving one for the system
        self.max_workers = max(1, os.cpu_count() - 1)

    def get_param_combinations(self):
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def run_optimization(self, task_id: str, job_manager: JobManager, strategy_name: str):
        param_combinations = self.get_param_combinations()
        total_runs = len(param_combinations)
        print(
            f"Task [{task_id}] Starting PARALLEL Grid Search for {strategy_name}: {total_runs} runs on {self.max_workers} workers.")

        completed_runs = 0

        # Use ProcessPoolExecutor to run backtests in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a future for each backtest run
            futures = []
            for params in param_combinations:
                current_config = self.base_config.copy()
                current_config["strategy_params"] = params
                # Submit the runner function to the process pool
                future = executor.submit(run_single_backtest_instance, strategy_name, current_config)
                futures.append(future)

            # Process results as they complete
            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)
                completed_runs += 1

                job_manager.update_task_progress(
                    task_id,
                    current=completed_runs,
                    total=total_runs,
                    step=f"Completed run {completed_runs}/{total_runs}"
                )
                print(f"--- Task [{task_id}] Completed [{completed_runs}/{total_runs}] ---")

        print(f"\n--- Task [{task_id}] Optimization Finished ---")
        return self.get_best_result()

    def get_best_result(self, metric: str = "sharpe_ratio"):
        if not self.results: return None
        # Filter out runs that resulted in an error
        valid_results = [r for r in self.results if 'error' not in r and r.get(metric) is not None]

        if not valid_results:
            all_errors = [f"Params: {r['config']['strategy_params']}, Error: {r.get('error', 'Unknown')[:200]}..."
                          for r in self.results if 'error' in r]
            return {"error": "All backtest runs failed.", "details": all_errors}

        # Find the best result based on the specified metric
        best = max(valid_results, key=lambda x: x.get(metric, -float('inf')))
        return best