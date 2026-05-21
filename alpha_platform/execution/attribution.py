"""
Post-Trade Impact Attribution
==============================

Markout analysis: the standard method for separating *own price impact*
from *underlying alpha* in a trade.

For any bar t carrying net signed flow ε_t:

    markout(t, l) = sign(ε_t) · [log P(t+l) − log P(t)]

Averaged over many t at each horizon l this gives the post-trade price
trajectory conditional on trade direction.

SHAPE INTERPRETATION
--------------------
  Monotone rise  → alpha-dominated  (the market kept going your way)
  Rise then revert → impact-dominated  (you moved it; it mean-reverted)
  Flat immediately  → no edge, no impact
  Mixed hump        → both: permanent component = long-lag plateau,
                      transient = peak minus plateau

PERMANENT vs TRANSIENT DECOMPOSITION
--------------------------------------
  permanent_component  = mean markout at long lags (≥ long_lag_cutoff bars)
                       = the informational content that the market permanently
                         priced in (alpha)
  transient_component  = peak markout − permanent_component
                       = mechanical price displacement that eventually reverted

ADVERSE SELECTION (PASSIVE FILLS)
----------------------------------
For a passive (maker) fill, you were the liquidity provider.  The incoming
aggressive order may have been informed — the subsequent price move tells
you how badly you were selected against.  Adverse selection cost at lag l =
−E[ sign(aggressor) · r(t, t+l) ] from the passive side's perspective.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_platform.microstructure.propagator import _signed_flow


class ImpactAttributor:
    """Post-trade markout analysis — impact vs alpha decomposition.

    Parameters
    ----------
    freq : str
        Bar size for aggregating trades (default '1min').
    horizons : list[int]
        Lag values in bars at which to evaluate the markout.
    long_lag_cutoff : int
        Lags >= this are averaged to estimate the permanent (alpha) component.
    """

    def __init__(
        self,
        freq: str = "1min",
        horizons: list[int] | None = None,
        long_lag_cutoff: int = 32,
    ) -> None:
        self.freq            = freq
        self.horizons        = sorted(horizons or [1, 2, 4, 8, 16, 32, 64, 128])
        self.long_lag_cutoff = int(long_lag_cutoff)

    # ── Markout profile ───────────────────────────────────────────────────── #

    def markout_profile(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Compute the average directional markout at each lag.

        Parameters
        ----------
        trades : aggTrades DataFrame from BinanceVisionLoader.

        Returns
        -------
        pd.DataFrame  (index = lag in bars), columns:
          mean_markout   E[ sign(ε) · r(t, t+l) ]  in log-return units
          std_markout    standard deviation of above
          n_obs          number of bar observations at this lag
          t_stat         mean / (std / √n)  — significance
          cumulative     cumulative sum of mean_markout (useful for plotting)
        """
        sv    = _signed_flow(trades)
        price = pd.to_numeric(trades["price"], errors="coerce")

        flow    = sv.resample(self.freq).sum()
        mid     = price.resample(self.freq).last().ffill()
        logmid  = np.log(mid.to_numpy())
        f       = flow.to_numpy()
        sgn     = np.sign(f)
        n       = len(logmid)

        records = []
        for l in self.horizons:
            if l >= n:
                continue
            r    = logmid[l:] - logmid[: n - l]   # l-bar log-return
            s    = sgn[: n - l]
            mask = s != 0
            if mask.sum() < 20:
                continue
            mo = (s * r)[mask]
            std = float(np.std(mo))
            records.append({
                "lag":          l,
                "mean_markout": float(np.mean(mo)),
                "std_markout":  std,
                "n_obs":        int(mask.sum()),
                "t_stat": (
                    float(np.mean(mo) / (std / np.sqrt(mask.sum())))
                    if std > 0 else 0.0
                ),
            })

        df = pd.DataFrame(records).set_index("lag")
        if not df.empty:
            df["cumulative"] = df["mean_markout"].cumsum()
        return df

    # ── Permanent / transient decomposition ──────────────────────────────── #

    def permanent_transient_split(self, trades: pd.DataFrame) -> dict:
        """Decompose average markout into permanent (alpha) and transient (impact).

        Returns
        -------
        dict:
          permanent_component  plateau level at long lags  (= informational alpha)
          transient_component  peak − plateau              (= mechanical reversion)
          peak_lag             lag at which markout peaks
          reversion_fraction   transient / total  ∈ [0, 1]
          interpretation       text verdict
        """
        df = self.markout_profile(trades)
        if df.empty:
            return {"error": "insufficient data"}

        mc       = df["mean_markout"]
        peak_idx = int(mc.idxmax())
        peak_val = float(mc.loc[peak_idx])

        long_mask = df.index >= self.long_lag_cutoff
        plateau   = float(mc.loc[long_mask].mean()) if long_mask.any() else peak_val

        transient   = max(0.0, peak_val - plateau)
        rev_frac    = transient / max(abs(peak_val), 1e-15)

        if rev_frac > 0.60:
            label = "impact-dominated — execution cost likely exceeds alpha"
        elif rev_frac < 0.20:
            label = "alpha-dominated — persistent informational move"
        else:
            label = "mixed — both alpha and transient impact present"

        return {
            "permanent_component":  round(plateau,    8),
            "transient_component":  round(transient,  8),
            "peak_lag":             peak_idx,
            "reversion_fraction":   round(rev_frac,   3),
            "interpretation":       label,
        }

    # ── Adverse selection ─────────────────────────────────────────────────── #

    def adverse_selection(self, trades: pd.DataFrame) -> dict:
        """Estimate adverse selection cost for passive (maker) fills.

        Approach: bars where net flow is positive (net buying aggression) are
        proxied as bars where a passive seller was filled.  The adverse
        selection at lag l is the expected log-return *against* the passive
        side: E[ r(t, t+l) | net buying at t ].  A positive value means
        the market continued to move against the passive seller — bad.

        Returns
        -------
        dict: adverse_selection_by_lag (dict lag→float), note.
        """
        sv    = _signed_flow(trades)
        price = pd.to_numeric(trades["price"], errors="coerce")

        flow   = sv.resample(self.freq).sum()
        mid    = price.resample(self.freq).last().ffill()
        logmid = np.log(mid.to_numpy())
        f      = flow.to_numpy()
        n      = len(logmid)

        # Proxy: net-buying bars → passive seller filled by aggressor
        buy_mask = f > 0

        result = {}
        for l in self.horizons:
            if l >= n:
                continue
            r    = logmid[l:] - logmid[: n - l]
            mask = buy_mask[: n - l]
            if mask.sum() < 10:
                continue
            # positive = price continued up after buy aggression = adverse for passive seller
            result[l] = round(float(np.mean(r[mask])), 8)

        return {
            "adverse_selection_by_lag": result,
            "note": (
                "Positive value at lag l = price moved against passive seller "
                "by that amount (adverse). Near-zero or negative = favourable."
            ),
        }


__all__ = ["ImpactAttributor"]
