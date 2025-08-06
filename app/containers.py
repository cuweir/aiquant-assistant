# app/containers.py

from .core.config import settings
from .llm_providers import get_llm_strategy
from .services.parameter_manager import ParameterManager
from .services.order_executor import OrderExecutor
from .services.analysis_service import AnalysisService
from .services.data_updater import DataUpdaterService


class Container:
    """
    A simple, centralized container for creating and holding service instances.
    This implements the Dependency Injection pattern.
    """

    def __init__(self):
        # 1. Create leaf-level services
        self.param_manager = ParameterManager()
        self.llm_strategy = get_llm_strategy(settings)
        # IMPORTANT: For real trading, is_testnet should be False
        self.order_executor = OrderExecutor(is_testnet=True)
        self.data_updater = DataUpdaterService()

        # 2. Create high-level services and "inject" their dependencies
        self.analysis_service = AnalysisService(
            param_manager=self.param_manager,
            order_executor=self.order_executor,
            llm_strategy=self.llm_strategy
        )


# Create a single, application-wide container instance
container = Container()