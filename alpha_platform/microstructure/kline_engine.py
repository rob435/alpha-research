"""
Module 2 - Synthetic Kline-Microstructure Engine
=================================================

Multi-year L2/L3 order-book history is thin, expensive, and frequently
non-existent for the long tail of the universe. This engine reconstructs
*structural* book pressure straight from the GEOMETRY of an OHLCV candle:
the shape of a bar already encodes how aggressively price had to move to
clear the liquidity that was resting in the book.

Two continuous features are produced, both as wide panels
(index = timestamp, columns = asset) so they stack cleanly downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class KlineMicrostructureEngine:
    r"""Synthetic microstructure feature generator over OHLCV klines."""

    def __init__(
        self,
        open_: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> None:
        panels = {"open": open_, "high": high, "low": low,
                  "close": close, "volume": volume}
        reference = close.index
        for name, panel in panels.items():
            if not panel.index.equals(reference):
                raise ValueError(f"'{name}' index must match the close index")
            if not panel.columns.equals(close.columns):
                raise ValueError(f"'{name}' columns must match the close columns")
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    # ------------------------------------------------------------------ #
    # Volumetric Illiquidity Proxy  (Lambda)                             #
    # ------------------------------------------------------------------ #
    def illiquidity_proxy(self, use_dollar_volume: bool = True) -> pd.DataFrame:
        r"""Volumetric Illiquidity Proxy  Lambda.

            Lambda = |ln(Close / Open)| / Volume

        Intent
            An Amihud-style price-impact-per-unit-of-flow estimate. A large
            |log return| achieved on small volume means the book was thin:
            price slipped a long way per traded unit  =>  illiquid / fragile.
            A small move on huge volume means deep liquidity  =>  Lambda low.

        use_dollar_volume (default True)
            Divides by Close*Volume rather than raw base-asset Volume. This
            is STRONGLY recommended and is the default: raw base-asset volume
            is not comparable across assets priced at $0.0001 vs $60,000,
            whereas dollar volume is a common unit. The raw-volume form is
            retained only for fidelity to the literal brief.

        Safe divide
            A zero-volume (untraded) bar yields NaN, never +inf. An untraded
            bar carries no liquidity information and must not be allowed to
            poison the cross-sectional z-scores computed downstream.
        """
        log_move = np.abs(np.log(self.close / self.open))
        denominator = self.close * self.volume if use_dollar_volume else self.volume
        denominator = denominator.where(denominator > 0)  # 0 -> NaN
        return log_move / denominator

    # ------------------------------------------------------------------ #
    # Flow Deceleration Signature  (the Wick metric)                     #
    # ------------------------------------------------------------------ #
    def wick_ratio(self) -> pd.DataFrame:
        r"""Flow Deceleration Signature - the upper-wick metric.

            Wick_Ratio = (High - Close) / (High - Low)

        Intent
            Measures aggressive-buying EXHAUSTION. Price was bid all the way
            up to `High` but could not hold there and closed back down; the
            unfilled retreat (High - Close) as a fraction of the full bar
            range is the rejection strength.
                ~0  => closed on the highs  (buyers in control, continuation)
                ~1  => closed on the lows after a failed push (deceleration)

        Note on terminology
            The strict candlestick "upper wick" is High - max(Open, Close).
            The brief's (High - Close) form additionally folds in the body
            whenever Close < Open; it is implemented verbatim as specified,
            and ``upper_wick_fraction`` below offers the strict definition.

        Safe divide
            A flat doji bar (High == Low) carries no range information and
            yields NaN. The result is clipped to [0, 1] to absorb any
            floating-point spill outside the geometric bound.
        """
        bar_range = (self.high - self.low).where(self.high > self.low)
        return ((self.high - self.close) / bar_range).clip(0.0, 1.0)

    def upper_wick_fraction(self) -> pd.DataFrame:
        r"""Strict candlestick upper wick as a fraction of bar range.

            Upper_Wick = (High - max(Open, Close)) / (High - Low)

        Provided as the textbook-correct companion to ``wick_ratio``: it
        isolates the rejection tail ABOVE the body, excluding body travel.
        """
        body_top = np.maximum(self.open, self.close)
        bar_range = (self.high - self.low).where(self.high > self.low)
        return ((self.high - body_top) / bar_range).clip(0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Normalisation helpers (continuous risk arrays)                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
        """Per-timestamp (row-wise) z-score.

        Puts every asset on a common scale at each bar so a downstream
        classifier is not dominated by raw-unit magnitude differences. A
        zero-dispersion row degrades to NaN rather than dividing by zero.
        """
        mean = panel.mean(axis=1)
        std = panel.std(axis=1, ddof=0).replace(0.0, np.nan)
        return panel.sub(mean, axis=0).div(std, axis=0)

    @staticmethod
    def winsorize(panel: pd.DataFrame, lower: float = 0.01,
                  upper: float = 0.99) -> pd.DataFrame:
        """Clip each timestamp's cross-section to its [lower, upper] quantiles.

        Lambda in particular has a violently fat right tail; winsorising per
        bar tames it without distorting the cross-sectional ordering.
        """
        lo = panel.quantile(lower, axis=1)
        hi = panel.quantile(upper, axis=1)
        return panel.clip(lower=lo, upper=hi, axis=0)
