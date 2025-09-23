# manual_test.py

import asyncio
import sys
import os

import pytest


pytestmark = pytest.mark.skip(reason="Manual validation script; not intended for automated pytest runs")

try:
    from app.services.backtest.runner import run_final_validation
except ImportError:  # pragma: no cover - defensive guard for pytest collection
    run_final_validation = None

# Add the project root to the Python path to allow imports from 'app'
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    print("Starting manual validation...")
    # Since run_final_validation is not async, we can call it directly.
    # If it were async, we would use asyncio.run()
    if run_final_validation is None:
        raise RuntimeError("run_final_validation is unavailable; ensure backtest runner exports it")
    run_final_validation()
    print("Manual validation finished.")
