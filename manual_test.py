# manual_test.py

import asyncio
import sys
import os

# Add the project root to the Python path to allow imports from 'app'
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.services.backtest.runner import run_final_validation

if __name__ == "__main__":
    print("Starting manual validation...")
    # Since run_final_validation is not async, we can call it directly.
    # If it were async, we would use asyncio.run()
    run_final_validation()
    print("Manual validation finished.")