# app/containers.py

from .core.config import settings
from .llm_providers import get_llm_strategy
from .services.parameter_manager import ParameterManager
from .services.order_executor import OrderExecutor
from .services.analysis_service import AnalysisService
from .services.data_updater import DataUpdaterService
from .services.trading_service import TradingService


class Container:
    """
    The central place for creating and wiring up all service instances.
    """

    def __init__(self):
        # 1. Create leaf-level services
        self.param_manager = ParameterManager()
        self.llm_strategy = get_llm_strategy(settings)
        self.order_executor = OrderExecutor(is_testnet=True)
        self.data_updater = DataUpdaterService()

        # 2. Create the TradingService, injecting its dependencies
        self.trading_service = TradingService(
            order_executor=self.order_executor,
            param_manager=self.param_manager
        )

        # 3. Create the AnalysisService, injecting the TradingService
        self.analysis_service = AnalysisService(
            param_manager=self.param_manager,
            trading_service=self.trading_service,  # <-- Injection
            llm_strategy=self.llm_strategy
        )


# Create the single, application-wide container instance
container = Container()
