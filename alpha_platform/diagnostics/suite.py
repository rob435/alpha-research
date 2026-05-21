"""
Module 5 - Performance Diagnostics & Information Coefficient Suite
==================================================================

Rank skill and P&L are NOT the same thing. A signal can rank the
cross-section correctly on average - a healthy Information Coefficient - and
still lose money, because the short side gets destroyed by a handful of
convex continuation pumps: assets that were "correctly" flagged as extreme
and then kept running.

This suite measures both halves of the truth:

    * Information Coefficient  -> does the score RANK forward returns?
    * Convexity Tail Audit     -> does the top decile actually PAY OUT, or
                                  is its mean a lie told by a few fat tails?
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class QuantDiagnosticSuite:
    r"""Validation suite for the latent_extremity_score.

    Parameters
    ----------
    score : pd.DataFrame
        Wide panel of latent_extremity_score (timestamp x asset).
    close : pd.DataFrame
        Wide panel of close prices, used to build forward returns.
    """

    def __init__(self, score: pd.DataFrame, close: pd.DataFrame) -> None:
        self.score = score
        # Align close onto the score grid so every pooled observation has a
        # matching forward return.
        self.close = close.reindex(index=score.index, columns=score.columns)

    # ------------------------------------------------------------------ #
    # Forward returns (the only legitimate forward look)                 #
    # ------------------------------------------------------------------ #
    def forward_return(self, n: int = 6) -> pd.DataFrame:
        """Forward N-bar simple return:  close(t + n) / close(t) - 1.

        ``shift(-n)`` looks into the future ON PURPOSE - this is the
        prediction TARGET, the single place a forward look is valid. It must
        never be fed back as a model feature.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        return self.close.shift(-n) / self.close - 1.0

    # ------------------------------------------------------------------ #
    # Information Coefficient                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rowwise_rank_corr(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
        """Per-timestamp Spearman correlation across the asset cross-section.

        Spearman == Pearson computed on ranks. Doing it row-wise in pure
        pandas keeps the whole IC time series vectorised over millions of
        bars. Each bar is restricted to assets present in BOTH panels, so a
        partially observed cross-section cannot bias the correlation.
        """
        both = a.notna() & b.notna()
        rank_a = a.where(both).rank(axis=1)
        rank_b = b.where(both).rank(axis=1)
        # Centre each row, then form the Pearson correlation of the ranks.
        ca = rank_a.sub(rank_a.mean(axis=1), axis=0)
        cb = rank_b.sub(rank_b.mean(axis=1), axis=0)
        numerator = (ca * cb).sum(axis=1)
        denominator = np.sqrt((ca ** 2).sum(axis=1) * (cb ** 2).sum(axis=1))
        return (numerator / denominator.replace(0.0, np.nan)).rename("IC")

    def information_coefficient(self, n: int = 6, rolling: int = 0) -> pd.Series:
        """Spearman IC between score(t) and the forward N-bar return.

        rolling = 0  -> the raw per-bar cross-sectional IC series.
        rolling > 1  -> trailing mean IC over `rolling` bars (smoothed skill;
                        the mean is backward-looking, hence safe).
        """
        ic = self._rowwise_rank_corr(self.score, self.forward_return(n))
        if rolling and rolling > 1:
            return ic.rolling(rolling, min_periods=1).mean()
        return ic

    def ic_summary(self, n: int = 6) -> dict:
        """Headline IC statistics including the IC information ratio.

        ic_ir = mean(IC) / std(IC) - the stability-adjusted measure of rank
        skill; a high mean IC with an unstable sign is not tradeable.
        """
        ic = self.information_coefficient(n)
        mean = float(ic.mean())
        std = float(ic.std(ddof=0))
        return {
            "ic_mean": mean,
            "ic_std": std,
            "ic_ir": mean / std if std else np.nan,
            "ic_hit_rate": float((ic > 0).mean()),
            "n_obs": int(ic.notna().sum()),
        }

    # ------------------------------------------------------------------ #
    # Convexity Tail Audit                                               #
    # ------------------------------------------------------------------ #
    def decile_audit(self, n: int = 6, n_deciles: int = 10,
                     verbose: bool = True) -> pd.DataFrame:
        """Decile-spread analysis with an explicit convexity check.

        Every (score, forward-return) observation is pooled, bucketed into
        equal-population deciles BY SCORE, and each decile reports:

            median  - typical forward return  (the honest rank-skill payout)
            mean    - average forward return  (the structural P&L payout)
            p5, p95 - the tails               (who is actually carrying)

        The headline is the top decile's median-vs-mean gap. If the median
        is flat/negative but the mean is positive, the "edge" is being
        manufactured by a few runaway continuation pumps in the right tail -
        and on a short book those same pumps are pure, unbounded loss.
        """
        forward = self.forward_return(n)
        pooled = pd.DataFrame({
            "score": self.score.to_numpy().ravel(),
            "fwd": forward.to_numpy().ravel(),
        }).dropna()
        if pooled.empty:
            raise ValueError("no overlapping score / forward-return data")

        # Rank-based bucketing -> exactly equal-population deciles, robust to
        # the spiky, non-uniform distribution of the score itself.
        ranks = pooled["score"].rank(method="first")
        bucket = (ranks / (len(ranks) + 1) * n_deciles).astype(int)
        pooled["decile"] = np.minimum(bucket, n_deciles - 1)

        grouped = pooled.groupby("decile")["fwd"]
        table = pd.DataFrame({
            "count": grouped.size(),
            "median": grouped.median(),
            "mean": grouped.mean(),
            "p5": grouped.quantile(0.05),
            "p95": grouped.quantile(0.95),
        })
        table["mean_minus_median"] = table["mean"] - table["median"]

        if verbose:
            self._print_audit(table, n_deciles, n)
        return table

    @staticmethod
    def _print_audit(table: pd.DataFrame, n_deciles: int, n: int) -> None:
        """Render the audit and flag tail-driven (convex) top-decile payouts."""
        top = table.iloc[-1]
        print(f"\n=== Convexity Tail Audit (forward {n}-bar return) ===")
        print(table.to_string(float_format=lambda v: f"{v: .6f}"))
        print(f"\n--- Top decile  D{n_deciles - 1}  (most extreme signals) ---")
        print(f"  median fwd ret : {top['median']: .6f}   <- typical rank skill")
        print(f"  mean   fwd ret : {top['mean']: .6f}   <- actual structural payout")
        print(f"  p5  / p95 tail : {top['p5']: .6f} / {top['p95']: .6f}")

        gap = top["mean"] - top["median"]
        if top["mean"] > 0.0 >= top["median"]:
            print("  WARNING: positive mean on a non-positive median -> the payout")
            print("           is tail-manufactured (a few right-tail continuation")
            print("           pumps); short-side expectancy is almost certainly")
            print("           negative despite the apparent rank skill.")
        elif top["median"] > 0.0 >= top["mean"]:
            print("  WARNING: positive median but non-positive mean -> rank skill")
            print("           is MASKING a negative structural payout; convex")
            print("           losing tails are eating the edge. Do not size this.")
        elif abs(gap) > abs(top["median"]) * 0.5 + 1e-12:
            print("  NOTE: mean and median diverge materially -> convex tail")
            print("        present; stress the book against the p5/p95 tails.")
        else:
            print("  OK: mean and median agree -> payout is broad, not tail-driven.")
