"""
End-to-end reference pipeline  (Module 1 -> Module 5)
=====================================================

A wiring example, not a trading strategy. It shows the intended data flow
and, critically, the single train/test split that keeps the WHOLE chain
look-ahead-safe: the latent classifier is fit only on the early slice, and
every feature upstream of it is already causal by construction.

Data contract
    klines : dict with keys {'open','high','low','close','volume'}, each a
    wide DataFrame (index = DatetimeIndex, columns = asset symbol).
"""
from __future__ import annotations

import pandas as pd

from alpha_platform.alpha.latent import LatentExtremityEvaluator
from alpha_platform.data.residualizer import CrossSectionalResidualizer
from alpha_platform.diagnostics.suite import QuantDiagnosticSuite
from alpha_platform.execution.impact import SquareRootImpactSimulator
from alpha_platform.microstructure.kline_engine import KlineMicrostructureEngine


class AlphaResearchPipeline:
    """Compose the five engines into one causal research run."""

    def __init__(self, klines: dict[str, pd.DataFrame], sector_map,
                 train_frac: float = 0.6) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(klines)
        if missing:
            raise ValueError(f"klines missing panels: {sorted(missing)}")
        self.klines = klines
        self.sector_map = sector_map
        self.train_frac = float(train_frac)

    def run(self, forward_n: int = 6, rank_window: int = 6) -> dict:
        k = self.klines

        # -- Module 1: factor-neutral residuals + rank dynamics ---------- #
        residualizer = CrossSectionalResidualizer(
            close=k["close"], volume=k["volume"], sector_map=self.sector_map,
        )
        residuals = residualizer.residuals()
        rank_velocity = residualizer.rank_velocity(window=rank_window)

        # -- Module 2: synthetic microstructure features ----------------- #
        micro = KlineMicrostructureEngine(
            k["open"], k["high"], k["low"], k["close"], k["volume"],
        )
        illiquidity = micro.illiquidity_proxy()
        wick = micro.wick_ratio()

        # -- Module 4: collapse features into the latent extremity field - #
        features = {
            "residual_return": residuals,
            "rank_velocity": rank_velocity,
            "illiquidity": illiquidity,
            "wick_ratio": wick,
        }
        split_pos = int(len(k["close"].index) * self.train_frac)
        train_end = k["close"].index[split_pos]
        evaluator = LatentExtremityEvaluator(model="isolation_forest")
        evaluator.fit(features, train_end=train_end)
        score = evaluator.score(features)

        # -- Module 5: diagnostics on the out-of-sample slice ------------ #
        oos = score.index > train_end
        diagnostics = QuantDiagnosticSuite(
            score=score.loc[oos], close=k["close"].loc[oos],
        )
        ic_summary = diagnostics.ic_summary(n=forward_n)
        decile_audit = diagnostics.decile_audit(n=forward_n, verbose=True)

        # -- Module 3: the impact model is consumed at execution time ---- #
        # (instantiated here so a backtest loop can charge every fill).
        impact = SquareRootImpactSimulator()

        return {
            "train_end": train_end,
            "residuals": residuals,
            "latent_extremity_score": score,
            "ic_summary": ic_summary,
            "decile_audit": decile_audit,
            "impact_model": impact,
        }


__all__ = ["AlphaResearchPipeline"]
