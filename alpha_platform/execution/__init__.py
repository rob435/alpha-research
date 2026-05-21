"""Execution layer - non-linear market impact and the strategy backtester."""
from alpha_platform.execution.backtester import StrategyBacktester, summarize
from alpha_platform.execution.impact import SquareRootImpactSimulator

__all__ = ["SquareRootImpactSimulator", "StrategyBacktester", "summarize"]
