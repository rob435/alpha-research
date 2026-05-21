"""
alpha_platform - Mid-Frequency Alpha Research Platform
======================================================

A production-grade research stack for intraday cross-sectional alpha built
purely on standard OHLCV klines. The platform deliberately rejects naive
daily assumptions and instead targets:

    * intraday cross-sectional RANK skill (not absolute return forecasting),
    * continuous asset-turnover VELOCITY (rank migration through time),
    * non-linear MARKET IMPACT (the square-root law, not infinite liquidity).

Five composable engines
-----------------------
    data           -> CrossSectionalResidualizer   (factor neutralisation)
    microstructure -> KlineMicrostructureEngine     (synthetic book metrics)
    execution      -> SquareRootImpactSimulator     (non-linear impact)
    alpha          -> LatentExtremityEvaluator      (latent extremity field)
    diagnostics    -> QuantDiagnosticSuite          (IC / convexity audit)

Module 6 — Microstructure Research (propagator / long-memory / metaorder)
--------------------------------------------------------------------------
    ImpactPropagator    response function R(l), flow ACF C(l), no-arb audit,
                        phantom-arbitrage demonstration
    FlowMemoryAnalyzer  trade-sign ACF, power-law γ fit, Hurst exponents
    MetaorderDetector   participation rates, clip regularity, sustained-
                        pressure windows (TWAP/VWAP footprint detection)
    ImpactAttributor    post-trade markout, permanent/transient split,
                        adverse-selection estimation

NOTE ON THE PACKAGE NAME
------------------------
The brief specified a top-level package literally named ``platform``. That
name shadows the Python standard-library ``platform`` module and produces
hard-to-debug import failures the moment any dependency does
``import platform``. The package is therefore shipped as ``alpha_platform``
while preserving the exact requested sub-module layout.
"""
from alpha_platform.data.residualizer import CrossSectionalResidualizer
from alpha_platform.microstructure.kline_engine import KlineMicrostructureEngine
from alpha_platform.microstructure.propagator   import ImpactPropagator
from alpha_platform.microstructure.flow_memory  import FlowMemoryAnalyzer
from alpha_platform.microstructure.metaorder    import MetaorderDetector
from alpha_platform.execution.impact      import SquareRootImpactSimulator
from alpha_platform.execution.backtester  import StrategyBacktester
from alpha_platform.execution.attribution import ImpactAttributor
from alpha_platform.alpha.latent     import LatentExtremityEvaluator
from alpha_platform.diagnostics.suite import QuantDiagnosticSuite
from alpha_platform.orderbook.vision_loader  import BinanceVisionLoader
from alpha_platform.orderbook.feature_engine import (
    OrderbookFeatureEngine,
    build_orderbook_panels,
)
from alpha_platform.orderbook.trade_flow import (
    TradeFlowEngine,
    build_trade_flow_panels,
)
from alpha_platform.microstructure.research import (
    run_propagator_study,
    run_flow_memory_study,
    run_attribution_study,
    run_metaorder_study,
)

__all__ = [
    # original five engines
    "CrossSectionalResidualizer",
    "KlineMicrostructureEngine",
    "SquareRootImpactSimulator",
    "StrategyBacktester",
    "LatentExtremityEvaluator",
    "QuantDiagnosticSuite",
    # order book data layer
    "BinanceVisionLoader",
    "OrderbookFeatureEngine",
    "build_orderbook_panels",
    "TradeFlowEngine",
    "build_trade_flow_panels",
    # Module 6 — microstructure research
    "ImpactPropagator",
    "FlowMemoryAnalyzer",
    "MetaorderDetector",
    "ImpactAttributor",
    "run_propagator_study",
    "run_flow_memory_study",
    "run_attribution_study",
    "run_metaorder_study",
]
__version__ = "0.3.0"
