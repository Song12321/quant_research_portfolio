"""因子研究正式入口。"""

from projects._03_factor_selection.factory.enhanced_test_runner import (
    EnhancedTestRunner,
    run_test_by_config,
)

__all__ = ["EnhancedTestRunner", "run_test_by_config"]
