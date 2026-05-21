"""
Module 1 - Data Alignment & Residualisation Engine
==================================================

Mid-frequency cross-sectional signals are contaminated by two systematic
components that have nothing to do with idiosyncratic asset skill:

    1. Market beta  - the whole universe moving together (risk-on/risk-off).
    2. Sector waves - co-movement inside a sector (all L1 tokens, all AI
                      tokens, ...) driven by narrative rotation.

If we rank RAW returns we are really ranking "who has the most beta", not
"who has the most alpha". This module strips both components out with a
rolling, look-ahead-safe time-series regression estimated per asset:

    r_i(t) = a_i(t) + b_i(t) * m(t) + c_i(t) * s_i(t) + e_i(t)

    r_i(t)  log return of asset i
    m(t)    equal-weighted MEDIAN market proxy return
    s_i(t)  leave-one-out, rolling volume-weighted return of i's sector
    e_i(t)  idiosyncratic residual  ->  the platform's clean alpha input

The coefficients (a, b, c) at time t are estimated ONLY from the trailing
window [t-L+1, t]. No observation after t ever enters a fit, so the residual
stream is causal and safe to feed into a backtest.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


class CrossSectionalResidualizer:
    r"""Rolling, look-ahead-safe cross-sectional residualiser.

    Parameters
    ----------
    close, volume : pd.DataFrame
        Wide panels - index = DatetimeIndex, columns = asset symbol.
    sector_map : Mapping[str, str] | pd.Series
        Asset symbol -> sector label.
    beta_window : int
        Length L of the trailing window used to estimate regression betas.
    vol_weight_window : int
        Window over which dollar-volume is averaged to form sector weights.
    ridge : float
        Tikhonov diagonal loading that stabilises near-singular windows
        (e.g. a stretch where a regressor barely moves).

    Why a LEAVE-ONE-OUT sector index?
        If asset i is included in its own sector index, the regressor s_i is
        mechanically correlated with the regressand r_i. That inflates the
        sector beta and bleeds genuine alpha into the "explained" bucket.
        Excluding self gives an honest sector factor.

    Look-ahead safety guarantees
        * every regressor at time t is known at t (returns/volumes are
          contemporaneous or lagged, never future);
        * all regression coefficients at t are functions of rolling SUMS
          over [t-L+1, t] - pandas .rolling() is strictly backward-looking;
        * sector volume WEIGHTS are lagged one bar, so the weight applied at
          t cannot peek at volume printed at t.
    """

    def __init__(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        sector_map: "Mapping[str, str] | pd.Series",
        beta_window: int = 120,
        vol_weight_window: int = 48,
        ridge: float = 1e-8,
    ) -> None:
        if not close.index.equals(volume.index):
            raise ValueError("close and volume must share the same index")
        if not close.columns.equals(volume.columns):
            raise ValueError("close and volume must share the same columns")
        if beta_window < 5:
            raise ValueError("beta_window too short for a stable 3-factor fit")

        self.close = close.sort_index()
        self.volume = volume.reindex(self.close.index)
        self.sector = pd.Series(dict(sector_map)).reindex(self.close.columns)
        if self.sector.isna().any():
            missing = list(self.sector[self.sector.isna()].index)
            raise ValueError(f"sector_map missing assets: {missing}")

        self.beta_window = int(beta_window)
        self.vol_weight_window = int(vol_weight_window)
        self.ridge = float(ridge)

        # Log returns are additive over time, which keeps the rolling-sum
        # normal-equation arithmetic exact (no compounding drift).
        self.returns = np.log(self.close).diff()
        self._dollar_volume = self.close * self.volume

        # Lazy caches - every public method memoises its result.
        self._market: pd.Series | None = None
        self._sector_index: pd.DataFrame | None = None
        self._residuals: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Factor construction                                                #
    # ------------------------------------------------------------------ #
    def market_proxy(self) -> pd.Series:
        """Equal-weighted MEDIAN of the cross-section of returns at each bar.

        The median (not the mean) is used deliberately: this proxy is
        subtracted from every single asset, so it must be robust. A handful
        of runaway pumps would drag a mean-based 'market' definition around
        and inject spurious negative beta into every other asset.
        """
        if self._market is None:
            self._market = self.returns.median(axis=1).rename("market")
        return self._market

    def sector_indices(self) -> pd.DataFrame:
        r"""Leave-one-out, rolling volume-weighted sector return per asset.

        For asset i in sector g:

            s_i(t) = ( SUM_{j in g} w_j(t) r_j(t)  -  w_i(t) r_i(t) )
                     / ( SUM_{j in g} w_j(t)       -  w_i(t) )

        w_j(t) is the trailing-mean dollar volume of asset j, lagged one bar.
        A singleton sector has an empty leave-one-out set -> NaN (such an
        asset simply cannot be sector-neutralised and will yield NaN
        residuals; surface it and assign a real sector).
        """
        if self._sector_index is not None:
            return self._sector_index

        # Rolling, one-bar-lagged dollar-volume weights (look-ahead-safe).
        weights = (
            self._dollar_volume
            .rolling(self.vol_weight_window, min_periods=1)
            .mean()
            .shift(1)
        )
        weighted_ret = weights * self.returns  # w_j * r_j contributions

        sector_idx = pd.DataFrame(
            index=self.returns.index, columns=self.returns.columns, dtype=float
        )
        for _sector, members in self.sector.groupby(self.sector).groups.items():
            cols = list(members)
            w_sum = weights[cols].sum(axis=1, min_count=1)         # SUM w_j
            wr_sum = weighted_ret[cols].sum(axis=1, min_count=1)   # SUM w_j r_j
            for col in cols:
                # Remove the asset's own contribution from numerator and
                # denominator -> honest "rest of sector" benchmark.
                num = wr_sum - weighted_ret[col]
                den = w_sum - weights[col]
                sector_idx[col] = num / den.where(den > 0)

        self._sector_index = sector_idx
        return sector_idx

    # ------------------------------------------------------------------ #
    # Residualisation                                                    #
    # ------------------------------------------------------------------ #
    def residuals(self) -> pd.DataFrame:
        """Factor-neutral idiosyncratic returns e_i(t) for the whole panel.

        Per asset, a 3-factor OLS [intercept, market, sector] is solved on
        every trailing window. The 3x3 systems for ALL timestamps are
        stacked and solved in a single batched ``np.linalg.solve`` call - the
        only Python-level loop is over assets (hundreds), never over time
        (millions of bars).
        """
        if self._residuals is not None:
            return self._residuals

        market = self.market_proxy()
        sectors = self.sector_indices()
        out = pd.DataFrame(
            index=self.returns.index, columns=self.returns.columns, dtype=float
        )
        for col in self.returns.columns:
            out[col] = self._rolling_ols_residual(
                self.returns[col], market, sectors[col], self.beta_window
            )
        self._residuals = out
        return out

    def _rolling_ols_residual(
        self, y: pd.Series, m: pd.Series, s: pd.Series, length: int
    ) -> pd.Series:
        r"""Solve  y ~ 1 + m + s  on every trailing window of size `length`.

        OLS via the normal equations:  (X'X) b = X'y  with X = [1, m, s].

        Every entry of the 3x3 matrix X'X and the 3-vector X'y is a rolling
        SUM of products of {1, m, s, y}. That is the key trick: the entire
        rolling estimator collapses to nine backward-looking pandas rolling
        sums, which is what makes the engine simultaneously vectorised AND
        strictly causal (no future index is ever touched inside a window).
        """
        frame = pd.DataFrame({"y": y, "m": m, "s": s}, dtype=float)
        # A window observation is usable only if all three series exist;
        # blank the whole row otherwise so partial rows cannot bias a fit.
        frame = frame.where(frame.notna().all(axis=1))

        def rolling_sum(series: pd.Series) -> np.ndarray:
            return series.rolling(length, min_periods=length).sum().to_numpy()

        count = frame["y"].rolling(length, min_periods=length).count().to_numpy()
        sum_m = rolling_sum(frame["m"])
        sum_s = rolling_sum(frame["s"])
        sum_y = rolling_sum(frame["y"])
        sum_mm = rolling_sum(frame["m"] * frame["m"])
        sum_ss = rolling_sum(frame["s"] * frame["s"])
        sum_ms = rolling_sum(frame["m"] * frame["s"])
        sum_my = rolling_sum(frame["m"] * frame["y"])
        sum_sy = rolling_sum(frame["s"] * frame["y"])

        n_rows = len(frame)
        xtx = np.empty((n_rows, 3, 3))
        xtx[:, 0, 0], xtx[:, 0, 1], xtx[:, 0, 2] = count, sum_m, sum_s
        xtx[:, 1, 0], xtx[:, 1, 1], xtx[:, 1, 2] = sum_m, sum_mm, sum_ms
        xtx[:, 2, 0], xtx[:, 2, 1], xtx[:, 2, 2] = sum_s, sum_ms, sum_ss
        xty = np.stack([sum_y, sum_my, sum_sy], axis=1)

        # Ridge loading keeps flat-regressor windows invertible.
        xtx[:, [0, 1, 2], [0, 1, 2]] += self.ridge

        betas = np.full((n_rows, 3), np.nan)
        valid = np.isfinite(xtx).all(axis=(1, 2)) & np.isfinite(xty).all(axis=1)
        if valid.any():
            # np.linalg.solve (numpy >= 2.0) treats a 2-D right-hand side as
            # a stack of MATRICES. Append a trailing axis so each rolling
            # window is solved as a single right-hand-side VECTOR instead.
            solved = np.linalg.solve(xtx[valid], xty[valid][:, :, np.newaxis])
            betas[valid] = solved[:, :, 0]

        m_val = frame["m"].to_numpy()
        s_val = frame["s"].to_numpy()
        y_val = frame["y"].to_numpy()
        fitted = betas[:, 0] + betas[:, 1] * m_val + betas[:, 2] * s_val
        return pd.Series(y_val - fitted, index=frame.index, name=y.name)

    # ------------------------------------------------------------------ #
    # Rank migration trajectories                                        #
    # ------------------------------------------------------------------ #
    def rank_trajectory(self) -> pd.DataFrame:
        """Cross-sectional percentile rank of the residual alpha each bar.

        0.0 = weakest idiosyncratic performer in the universe at t,
        1.0 = strongest. This is the "where does the asset currently sit"
        state variable that the rest of the platform reasons about.
        """
        return self.residuals().rank(axis=1, pct=True)

    def rank_velocity(self, window: int = 6) -> pd.DataFrame:
        """Rank-migration velocity: change in percentile rank over `window`.

        Captures continuous asset-turnover - an asset climbing from the 20th
        to the 80th percentile over a few bars carries large positive
        velocity. ``.diff(window)`` is backward-looking and therefore safe.

        `window` is intentionally free so the same engine serves sub-hourly
        and multi-hour horizons without code changes.
        """
        if window < 1:
            raise ValueError("window must be >= 1")
        return self.rank_trajectory().diff(window)
