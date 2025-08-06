# app/services/parameter_manager.py

import json
from pathlib import Path
from typing import Dict, Any


class ParameterManager:
    """
    A service class responsible for loading, managing, and providing
    strategy parameters from a JSON configuration file.
    """

    def __init__(self, file_path: str = "production_parameters.json"):
        self.params_file = Path(file_path)
        self.parameters = self._load_parameters()
        print("ParameterManager initialized.")
        print(f"Loaded parameters for symbols: {list(self.parameters.keys())}")

    def _load_parameters(self) -> Dict[str, Any]:
        """Loads the parameters from the JSON file."""
        if not self.params_file.exists():
            raise FileNotFoundError(f"Parameters file not found at: {self.params_file.resolve()}")

        with open(self.params_file, 'r') as f:
            return json.load(f)

    def get_params_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Gets the specific parameter set for a given symbol.

        Args:
            symbol: The symbol name (e.g., 'BTC/USDT').

        Returns:
            A dictionary of parameters for the symbol.

        Raises:
            KeyError: If the symbol is not found in the parameters file.
        """
        if symbol not in self.parameters:
            raise KeyError(f"No parameters found for symbol '{symbol}' in {self.params_file.name}")

        return self.parameters[symbol]


# Create a global instance to be used across the application
parameter_manager = ParameterManager()