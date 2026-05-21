"""
Order book research wiring  ->  the Latent Overextension Score
==============================================================

Three entry points, all feeding Module 4 (the Latent Overextension Score):

    run_backbone  - MULTI-YEAR aggTrades path. Trade Flow Imbalance + market-
                    neutral over-extension features over Binance's full
                    history. Trains the score, then runs Module 5 signal
                    diagnostics (IC + convexity audit).

    run_strategy  - the actual STRATEGY BACKTEST. Same aggTrades features and
                    score, then the dollar-neutral long/short overextension
                    book is simulated bar-by-bar with costs -> equity curve.

    run_phase1    - the ~11-month bookTicker path (OFI + L1 companions), a
                    high-resolution overlay; diagnostics only.

All paths train the latent model on the early slice and score forward only -
strict look-ahead safety.

SIGNAL DIAGNOSTICS vs STRATEGY BACKTEST
---------------------------------------
`run_backbone` / `run_phase1` answer "does the score have predictive rank
skill?" (IC, decile audit). `run_strategy` answers "does trading it make
money, after costs?" (equity curve, Sharpe, drawdown). They are complementary
- rank skill can mask negative P&L, which is the whole reason both exist.
"""
from __future__ import annotations

from alpha_platform.alpha.latent import LatentExtremityEvaluator
from alpha_platform.diagnostics.suite import QuantDiagnosticSuite
from alpha_platform.execution.backtester import StrategyBacktester, summarize
from alpha_platform.execution.impact import SquareRootImpactSimulator
from alpha_platform.orderbook.feature_engine import build_orderbook_panels
from alpha_platform.orderbook.trade_flow import build_trade_flow_panels

_OFI_FEATURES = ("ofi", "obi", "microprice_adj", "rel_spread", "update_intensity")
_BACKBONE_FEATURES = ("tfi", "trade_intensity", "runup", "runup_rank_velocity")


def _fit_score(features: dict, close, train_frac: float):
    """Fit the Latent Overextension Score on the early slice (index <=
    train_end) and score the whole timeline forward. Returns (score, train_end).
    """
    if len(close.index) < 10:
        raise ValueError("not enough bars to form a train/test split")
    train_end = close.index[int(len(close.index) * train_frac)]
    evaluator = LatentExtremityEvaluator(model="isolation_forest")
    evaluator.fit(features, train_end=train_end)
    return evaluator.score(features), train_end


def _diagnose(score, close, train_end, forward_bars: int) -> dict:
    """Module 5 diagnostics on the strictly out-of-sample slice.

    For a SHORT signal a working score yields a NEGATIVE Spearman IC against
    the raw forward return, so IC is also reported sign-flipped
    (`ic_short_oriented`, oriented so positive = genuine skill).
    """
    out_of_sample = score.index > train_end
    diagnostics = QuantDiagnosticSuite(score=score.loc[out_of_sample],
                                       close=close.loc[out_of_sample])
    raw_ic = diagnostics.ic_summary(n=forward_bars)
    decile_audit = diagnostics.decile_audit(n=forward_bars, verbose=True)
    short_ic = {f"short_{k}": (-v if k in ("ic_mean", "ic_ir") else v)
                for k, v in raw_ic.items()}
    return {"ic_raw": raw_ic, "ic_short_oriented": short_ic,
            "decile_audit": decile_audit}


def run_backbone(loader, symbols, start, end, freq: str = "1h",
                 train_frac: float = 0.6, forward_bars: int = 4,
                 runup_window: int = 8, velocity_window: int = 4,
                 normalize: str = "zscore") -> dict:
    """Multi-year aggTrades path: features -> Latent Overextension Score ->
    Module 5 signal diagnostics."""
    panels = build_trade_flow_panels(loader, symbols, start, end, freq,
                                     runup_window, velocity_window, normalize)
    features = {key: panels[key] for key in _BACKBONE_FEATURES}
    score, train_end = _fit_score(features, panels["close"], train_frac)
    result = _diagnose(score, panels["close"], train_end, forward_bars)
    result.update({"panels": panels, "overextension_score": score,
                   "train_end": train_end})
    return result


def run_strategy(loader, symbols, start, end, freq: str = "1h",
                 train_frac: float = 0.6, holding_bars: int = 4,
                 quantile: float = 0.10, stop_loss: float = 0.08,
                 rebalance_every: int = 1, runup_window: int = 8,
                 velocity_window: int = 4,
                 impact: SquareRootImpactSimulator | None = None) -> dict:
    """Full strategy backtest: aggTrades features -> Latent Overextension
    Score -> dollar-neutral long/short book simulated bar-by-bar with costs.

    The model is trained on the early slice; the book is traded only on the
    out-of-sample slice. Returns the backtester result (equity_curve, returns,
    trades, stats) plus the score and train_end.
    """
    panels = build_trade_flow_panels(loader, symbols, start, end, freq,
                                     runup_window, velocity_window)
    features = {key: panels[key] for key in _BACKBONE_FEATURES}
    score, train_end = _fit_score(features, panels["close"], train_frac)

    oos = score.index > train_end
    backtester = StrategyBacktester(
        impact=impact, holding_bars=holding_bars, quantile=quantile,
        stop_loss=stop_loss, rebalance_every=rebalance_every)
    result = backtester.run(score=score.loc[oos],
                            close=panels["close"].loc[oos],
                            dollar_volume=panels["dollar_volume"].loc[oos])
    summarize(result)
    result.update({"overextension_score": score, "train_end": train_end,
                   "panels": panels})
    return result


def run_phase1(loader, symbols, start, end, freq: str = "15min",
               train_frac: float = 0.6, forward_bars: int = 4,
               normalize: str = "zscore") -> dict:
    """High-resolution overlay: bookTicker OFI + L1 companions -> Latent
    Overextension Score -> Module 5 diagnostics. Limited to Binance's
    ~11-month bookTicker window (2023-05 .. 2024-04)."""
    panels = build_orderbook_panels(loader, symbols, start, end, freq, normalize)
    features = {key: panels[key] for key in _OFI_FEATURES}
    score, train_end = _fit_score(features, panels["mid_close"], train_frac)
    result = _diagnose(score, panels["mid_close"], train_end, forward_bars)
    result.update({"panels": panels, "overextension_score": score,
                   "train_end": train_end})
    return result


__all__ = ["run_backbone", "run_strategy", "run_phase1"]
