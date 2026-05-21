"""
Flow Long-Memory Analysis
=========================

Empirical evidence for the "central tension" of order-book-driven trading:

    signed trade flow has long memory  (Hurst H_flow >> 0.5)
    price mid-returns are near-diffusive (Hurst H_prices ≈ 0.5)

If impact were permanent, long-memory flow would imply long-memory prices —
trivially arbitrageable.  Markets are diffusive *because* the propagator G(τ)
decays transient enough to annul the flow autocorrelation.

This module measures the long-memory evidence directly from aggTrades data
with three tools:

  FlowMemoryAnalyzer.sign_acf         — trade-sign autocorrelation C(l)
  FlowMemoryAnalyzer.power_law_fit    — fit C(l) ∝ l^{-γ} and imply Hurst
  FlowMemoryAnalyzer.memory_summary   — full audit dict for one symbol

Key empirical benchmarks (Lillo & Farmer 2004; Bouchaud et al. 2018):
  γ ≈ 0.3–0.7  for most liquid instruments
  H_flow ≈ 1 - γ/2  ≈ 0.65–0.85
  H_prices ≈ 0.48–0.52
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_platform.microstructure.propagator import _hurst_dfa, _signed_flow


class FlowMemoryAnalyzer:
    """Measure long-memory properties of signed trade flow vs price returns.

    Parameters
    ----------
    freq : str
        Bar size for price-return Hurst estimation (default '1min').
    max_lag : int
        Maximum autocorrelation lag in number of *trades* (default 500).
    """

    def __init__(self, freq: str = "1min", max_lag: int = 500) -> None:
        self.freq    = freq
        self.max_lag = int(max_lag)

    # ── Sign ACF ──────────────────────────────────────────────────────────── #

    def sign_acf(self, trades: pd.DataFrame) -> pd.Series:
        """Autocorrelation of the binary trade sign (volume-weighted sign).

        C(l) = Cov(sign(ε_t), sign(ε_{t+l})) / Var(sign(ε))

        Using the signed quantity rather than a pure +1/-1 indicator captures
        the volume-weighted directionality.  Long-memory shows up as a
        power-law tail rather than exponential decay.

        Parameters
        ----------
        trades : aggTrades DataFrame from BinanceVisionLoader.

        Returns
        -------
        pd.Series indexed 0…max_lag, named 'sign_acf'.
        """
        sv    = _signed_flow(trades)
        signs = np.sign(sv.to_numpy()).astype(float)
        signs = signs[signs != 0]   # drop zero-volume events
        n     = len(signs)

        mu    = float(np.mean(signs))
        var   = float(np.var(signs))
        if var < 1e-15 or n < self.max_lag + 2:
            return pd.Series(np.zeros(self.max_lag + 1), name="sign_acf")

        sc  = signs - mu
        acf = np.zeros(self.max_lag + 1)
        acf[0] = 1.0
        for l in range(1, self.max_lag + 1):
            n_pairs = n - l
            if n_pairs <= 0:
                break
            acf[l] = float(np.dot(sc[: n - l], sc[l:]) / n_pairs / var)
        return pd.Series(acf, name="sign_acf")

    # ── Power-law tail fit ────────────────────────────────────────────────── #

    def power_law_fit(
        self, acf: pd.Series, min_lag: int = 10, max_lag: int | None = None
    ) -> dict:
        """Fit C(l) ∝ l^{-γ} over the positive tail of the ACF.

        Expected: γ ≈ 0.3–0.7.  The relationship to Hurst is H_flow ≈ 1 - γ/2,
        so γ = 0.4 implies H_flow ≈ 0.80 (strongly persistent).

        Parameters
        ----------
        min_lag : int    First lag to include in the log-log fit (skip the
                         noisy short-range part).
        max_lag : int    Last lag to include (default: len(acf) - 1).

        Returns
        -------
        dict: gamma, log_amplitude, r_squared, implied_hurst, interpretation.
        """
        end = (max_lag or len(acf) - 1)
        vals = acf.iloc[min_lag : end + 1].to_numpy()
        lags = np.arange(min_lag, min_lag + len(vals), dtype=float)

        # Fit only the positive part (negative ACF tail is noise / mean-reversion)
        mask = vals > 1e-6
        if mask.sum() < 8:
            return {
                "gamma": np.nan, "log_amplitude": np.nan,
                "r_squared": np.nan, "implied_hurst": np.nan,
                "interpretation": "insufficient positive-lag data for power-law fit",
            }

        log_l  = np.log(lags[mask])
        log_c  = np.log(vals[mask])
        coeffs = np.polyfit(log_l, log_c, 1)
        gamma  = -float(coeffs[0])
        amp    = float(coeffs[1])

        # R² of the log-log fit
        predicted = np.polyval(coeffs, log_l)
        ss_res    = float(np.sum((log_c - predicted) ** 2))
        ss_tot    = float(np.sum((log_c - log_c.mean()) ** 2))
        r_sq      = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        h_implied = 1.0 - gamma / 2.0
        mem_label = "long memory ✓" if h_implied > 0.55 else "weak/no long memory"
        return {
            "gamma":          round(gamma, 3),
            "log_amplitude":  round(amp, 3),
            "r_squared":      round(r_sq, 3),
            "implied_hurst":  round(h_implied, 3),
            "interpretation": (
                f"C(l) ∝ l^{{-{gamma:.2f}}} "
                f"(R²={r_sq:.2f}), implied H ≈ {h_implied:.2f} — {mem_label}"
            ),
        }

    # ── Full audit ────────────────────────────────────────────────────────── #

    def memory_summary(self, trades: pd.DataFrame) -> dict:
        """Complete long-memory audit for one symbol.

        Computes Hurst exponents for both flow and prices, the sign ACF, and
        the power-law fit.  Returns a flat dict suitable for appending to a
        cross-symbol summary DataFrame.

        Parameters
        ----------
        trades : aggTrades DataFrame.

        Returns
        -------
        dict with keys: hurst_flow, hurst_prices, h_gap, sign_acf_lag1/10/100,
        power_law (nested dict), central_tension (interpretation string).
        """
        sv       = _signed_flow(trades)
        flow_arr = sv.to_numpy()
        h_flow   = _hurst_dfa(flow_arr)

        price_s = pd.to_numeric(
            trades["price"], errors="coerce"
        ).resample(self.freq).last().ffill()
        ret_arr  = np.log(price_s).diff().fillna(0.0).to_numpy()
        h_prices = _hurst_dfa(ret_arr)

        acf = self.sign_acf(trades)
        pl  = self.power_law_fit(acf)

        def _acf_at(l: int) -> float:
            return round(float(acf.iloc[min(l, len(acf) - 1)]), 4)

        tension = (
            f"H_flow={h_flow:.2f} vs H_prices={h_prices:.2f} "
            f"(gap={h_flow - h_prices:.2f}) — "
            + (
                "consistent with transient-impact propagator annulling "
                "long-memory flow"
                if h_flow > h_prices + 0.10
                else "gap smaller than expected; check sample size or data quality"
            )
        )

        return {
            "hurst_flow":         round(h_flow, 3),
            "hurst_prices":       round(h_prices, 3),
            "h_gap":              round(h_flow - h_prices, 3),
            "sign_acf_lag1":      _acf_at(1),
            "sign_acf_lag10":     _acf_at(10),
            "sign_acf_lag100":    _acf_at(100),
            "power_law":          pl,
            "central_tension":    tension,
        }


__all__ = ["FlowMemoryAnalyzer"]
