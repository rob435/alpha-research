"""
Microstructure  —  kline engine + order-flow research tools
============================================================

Kline engine (Module 2):
    KlineMicrostructureEngine  synthetic book pressure from OHLCV geometry

Order-flow research (Module 6):
    ImpactPropagator   empirical response function R(l), flow ACF C(l),
                       no-arbitrage audit, phantom-arbitrage demo
    FlowMemoryAnalyzer trade-sign ACF, power-law fit, Hurst exponents
    MetaorderDetector  participation rates, clip regularity, sustained
                       pressure windows (TWAP/VWAP footprint detection)

Research entry points:
    run_propagator_study    one symbol  → propagator + no-arb audit
    run_flow_memory_study   universe    → long-memory report
    run_attribution_study   one symbol  → markout / permanent+transient
    run_metaorder_study     universe    → metaorder feature matrix
"""
from alpha_platform.microstructure.kline_engine import KlineMicrostructureEngine
from alpha_platform.microstructure.propagator   import ImpactPropagator
from alpha_platform.microstructure.flow_memory  import FlowMemoryAnalyzer
from alpha_platform.microstructure.metaorder    import MetaorderDetector
from alpha_platform.microstructure.research     import (
    run_propagator_study,
    run_flow_memory_study,
    run_attribution_study,
    run_metaorder_study,
)

__all__ = [
    "KlineMicrostructureEngine",
    "ImpactPropagator",
    "FlowMemoryAnalyzer",
    "MetaorderDetector",
    "run_propagator_study",
    "run_flow_memory_study",
    "run_attribution_study",
    "run_metaorder_study",
]
