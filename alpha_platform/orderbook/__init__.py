"""
Order Book Research Track  -  Binance Vision Core
=================================================

Real, free, historical microstructure alphas sourced from Binance Vision and
fed into the latent classifier (Module 4) and diagnostics (Module 5):

    trade_flow     - aggTrades Trade Flow Imbalance, the MULTI-YEAR backbone
                     (2019 -> present); see `run_backbone`.
    feature_engine - bookTicker Order Flow Imbalance + L1 companions, the
                     ~11-month high-resolution overlay; see `run_phase1`.

`vision_loader` handles download / cache / point-in-time universe discovery.
"""
from alpha_platform.orderbook.feature_engine import (
    OrderbookFeatureEngine,
    build_orderbook_panels,
)
from alpha_platform.orderbook.research import (
    run_backbone,
    run_phase1,
    run_strategy,
)
from alpha_platform.orderbook.trade_flow import (
    TradeFlowEngine,
    build_trade_flow_panels,
)
from alpha_platform.orderbook.vision_loader import BinanceVisionLoader

__all__ = [
    "BinanceVisionLoader",
    "OrderbookFeatureEngine",
    "build_orderbook_panels",
    "TradeFlowEngine",
    "build_trade_flow_panels",
    "run_backbone",
    "run_strategy",
    "run_phase1",
]
