"""
Metaorder Detection
===================

Execution algorithms (TWAP, VWAP, POV) slice large parent orders into
many child orders, leaving distinctive statistical footprints in the
aggTrades stream.  This module extracts those footprints as bar-level
features that can either feed directly into the LatentExtremityEvaluator
or support standalone rule-based detection.

FOOTPRINT SIGNATURES
--------------------
1. SUSTAINED DIRECTIONAL PRESSURE
   A metaorder keeps trading the same side for N consecutive bars.  Random
   walk sign-changes should appear much more frequently.  We flag windows
   where |TFI| > threshold for at least min_sustained bars in a row.

2. PARTICIPATION RATE
   buy_volume / total_volume  (and the sell analogue).  A metaorder
   anomalously dominates one side; the distribution of participation rates
   has a heavy right tail during metaorder periods.

3. CLIP SIZE REGULARITY (coefficient of variation)
   VWAP/TWAP clips tend to have similar sizes.  Low coefficient of
   variation in trade sizes within a bar ⇒ algorithmic.

4. INTER-TRADE INTERVAL ENTROPY
   TWAP clips arrive on a near-clock schedule.  Entropy of inter-trade
   interval distribution within a bar: low ⇒ clock-regular ⇒ TWAP.

Reference framing: Kyle (1985) participation rate; Tóth et al. (2011)
anomalous impact; Bershova & Rakhlin (2013) metaorder detection in
futures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_platform.microstructure.propagator import _signed_flow


def _entropy_bits(x: np.ndarray) -> float:
    """Shannon entropy in bits of an array of non-negative values."""
    x = x[x > 0]
    if len(x) < 2:
        return 0.0
    p = x / x.sum()
    return float(-np.sum(p * np.log2(p)))


class MetaorderDetector:
    """Extract metaorder-signature features from an aggTrades stream.

    Parameters
    ----------
    freq : str
        Bar aggregation frequency (default '5min').
    window_bars : int
        Rolling window (in bars) for the pressure-score feature.
    pressure_threshold : float
        |TFI| threshold above which a bar is "directionally pressured".
    min_sustained : int
        Minimum consecutive pressured bars to log a metaorder window.
    """

    def __init__(
        self,
        freq: str = "5min",
        window_bars: int = 12,
        pressure_threshold: float = 0.60,
        min_sustained: int = 4,
    ) -> None:
        self.freq               = freq
        self.window_bars        = int(window_bars)
        self.pressure_threshold = float(pressure_threshold)
        self.min_sustained      = int(min_sustained)

    # ── Participation rates ───────────────────────────────────────────────── #

    def participation_rate(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Buy / sell participation rates and TFI per bar.

        Columns: buy_participation, sell_participation, tfi, total_volume.
        """
        sv  = _signed_flow(trades)
        qty = pd.to_numeric(trades["quantity"], errors="coerce").fillna(0.0)

        buy_vol  = sv.clip(lower=0).resample(self.freq).sum()
        sell_vol = (-sv).clip(lower=0).resample(self.freq).sum()
        tot_vol  = qty.resample(self.freq).sum().replace(0.0, np.nan)

        return pd.DataFrame({
            "buy_participation":  buy_vol  / tot_vol,
            "sell_participation": sell_vol / tot_vol,
            "tfi":                (buy_vol - sell_vol) / tot_vol,
            "total_volume":       tot_vol,
        })

    # ── Clip size regularity ──────────────────────────────────────────────── #

    def size_cv(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Coefficient of variation of individual trade sizes per bar.

        Low CV ⇒ constant-size clips ⇒ VWAP / fixed-quantity algo.
        High CV ⇒ heterogeneous sizes ⇒ organic / retail flow.
        """
        qty = pd.to_numeric(trades["quantity"], errors="coerce").fillna(0.0)

        def _cv(x: pd.Series) -> float:
            v = x.to_numpy()
            mu = float(np.mean(v))
            if len(v) < 2 or mu < 1e-12:
                return np.nan
            return float(np.std(v) / mu)

        cv = qty.resample(self.freq).apply(_cv)
        cv.name = "size_cv"
        return cv.to_frame()

    # ── Inter-trade interval entropy ──────────────────────────────────────── #

    def interval_entropy(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Shannon entropy of inter-trade intervals within each bar.

        Low entropy ⇒ clock-regular spacing ⇒ TWAP signature.
        High entropy ⇒ Poisson-like / random arrival.
        """
        ts_ns     = trades.index.view(np.int64)
        gaps_ns   = np.diff(ts_ns).clip(0)
        gaps_ms   = pd.Series(gaps_ns / 1_000_000, index=trades.index[1:])

        def _ent(x: pd.Series) -> float:
            v = x.to_numpy()
            if len(v) < 3:
                return np.nan
            bins = np.histogram(v, bins=min(10, len(v)))[0].astype(float)
            return _entropy_bits(bins)

        ent = gaps_ms.resample(self.freq).apply(_ent)
        ent.name = "interval_entropy"
        return ent.to_frame()

    # ── Sustained pressure windows ────────────────────────────────────────── #

    def sustained_pressure_windows(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Detect windows of sustained one-sided aggressive flow.

        A window is logged when |TFI| >= pressure_threshold for at least
        min_sustained *consecutive* same-direction bars.

        Returns
        -------
        pd.DataFrame with columns: start, end, direction (+1=buy, -1=sell),
        n_bars, avg_tfi, max_tfi, total_volume.
        """
        pr  = self.participation_rate(trades)
        tfi = pr["tfi"].fillna(0.0).to_numpy()
        vol = pr["total_volume"].fillna(0.0).to_numpy()
        idx = pr.index.to_numpy()
        sgn = np.sign(tfi)

        pressured = (np.abs(tfi) >= self.pressure_threshold)
        windows   = []
        i         = 0

        while i < len(pressured):
            if not pressured[i]:
                i += 1
                continue
            d = sgn[i]
            j = i
            while j < len(pressured) and pressured[j] and sgn[j] == d:
                j += 1
            n_bars = j - i
            if n_bars >= self.min_sustained:
                windows.append({
                    "start":        pd.Timestamp(idx[i]),
                    "end":          pd.Timestamp(idx[j - 1]),
                    "direction":    int(d),
                    "n_bars":       n_bars,
                    "avg_tfi":      round(float(np.mean(np.abs(tfi[i:j]))), 3),
                    "max_tfi":      round(float(np.max(np.abs(tfi[i:j]))), 3),
                    "total_volume": round(float(vol[i:j].sum()), 2),
                })
            i = j

        return pd.DataFrame(windows)

    # ── Combined feature matrix ───────────────────────────────────────────── #

    def metaorder_features(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Full feature matrix aligned to bars.

        Columns: buy_participation, sell_participation, tfi, total_volume,
        size_cv, interval_entropy, pressure_score.

        pressure_score is the rolling fraction of bars in the last
        window_bars that exceeded the pressure_threshold in |TFI|.

        This DataFrame is directly compatible with LatentExtremityEvaluator
        as a feature input for cross-sectional scoring.
        """
        pr   = self.participation_rate(trades)
        cv   = self.size_cv(trades)
        ent  = self.interval_entropy(trades)
        out  = pr.join(cv, how="outer").join(ent, how="outer")

        abs_tfi = out["tfi"].abs().fillna(0.0)
        out["pressure_score"] = (
            abs_tfi
            .rolling(self.window_bars, min_periods=1)
            .apply(lambda x: float((x >= self.pressure_threshold).mean()))
        )
        return out


__all__ = ["MetaorderDetector"]
