import itertools
from typing import Dict, Any, List

from .runner import run_parameterized_backtest
from app.job_manager import update_task_progress  # <-- Import the update function


class GridSearchOptimizer:
    def __init__(self, base_config: Dict[str, Any], param_grid: Dict[str, List[Any]]):
        self.base_config = base_config
        self.param_grid = param_grid
        self.results = []

    def get_param_combinations(self):
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    # The key change is here: it now accepts a task_id
    def run_optimization(self, task_id: str):
        param_combinations = self.get_param_combinations()
        total_runs = len(param_combinations)
        print(f"Task [{task_id}] Starting Grid Search: {total_runs} total runs.")

        for i, params in enumerate(param_combinations):
            update_task_progress(
                task_id,
                current=i + 1,
                total=total_runs,
                step=f"Running with params: {params}"
            )
            print(f"\n--- Task [{task_id}] Running [{i + 1}/{total_runs}]: {params} ---")
            current_config = self.base_config.copy()
            current_config["strategy_params"] = params

            try:
                result = run_parameterized_backtest(current_config)
                self.results.append(result)
            except Exception as e:
                # [FIX] Capture the full traceback on failure
                import traceback
                error_details = traceback.format_exc()
                print(f"  > Task [{task_id}] Run failed for {params}: {e}")
                # [FIX] Append a failure record to results
                self.results.append({
                    "config": current_config,
                    "error": str(error_details)
                })

        print(f"\n--- Task [{task_id}] Optimization Finished ---")
        return self.get_best_result()

    def get_best_result(self, metric: str = "sharpe_ratio"):
        if not self.results: return None
        valid_results = [r for r in self.results if r.get(metric) is not None]
        if not valid_results:
            # [FIX] If no runs were successful, return a summary of errors
            all_errors = [f"Params: {r['config']['strategy_params']}, Error: {r['error'][:200]}..."
                          for r in self.results if 'error' in r]
            return {
                "error": "All backtest runs failed.",
                "details": all_errors
            }

        best = max(valid_results, key=lambda x: x.get(metric, -float('inf')))
        return best