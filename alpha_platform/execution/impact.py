"""
Module 3 - Non-Linear Market Impact Engine
===========================================

Naive backtests fill the entire requested size at the last printed tick, as
if liquidity were infinite. Real fills WALK THE BOOK: the more size you
demand, the worse the average price you receive. This simulator charges
every execution a non-linear cost via the empirically robust Square-Root
Law of market impact.

    Slippage_bps = Base_Fee + Gamma * Volatility_rolling
                              * sqrt(Position_Notional / ADV_24h) * 1e4

The square root is the entire point. Impact is CONCAVE in size: doubling
order size raises cost by only ~1.41x, not 2x. That concavity is what makes
order-splitting rational and what stops a backtest from minting free P&L by
assuming it can dump unlimited size at one price.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SquareRootImpactSimulator:
    r"""Square-Root-Law execution cost model.

        slippage_bps = base_fee_bps
                     + gamma * vol_rolling * sqrt(notional / ADV) * 1e4

    Term by term
        base_fee_bps  fixed exchange/clearing drag (taker fee + spread floor)
        gamma         dimensionless impact coefficient; empirical studies
                      (Almgren et al., BARRA, Toth et al.) place it near
                      0.3-1.0. Default 0.6.
        vol_rolling   trailing per-bar return volatility, a fraction (0.02 =
                      2%). More volatile names cost more to cross.
        notional/ADV  PARTICIPATION RATE - the order as a fraction of average
                      daily dollar volume. This is the true driver of impact.
        sqrt(.)       the concave core of the law.
        * 1e4         converts the fractional impact term into basis points.

    Default scalars (override via the constructor as needed)
        base_fee_bps = 5.0   gamma = 0.6
        vol_lookback = 24    adv_lookback = 24   (24 hourly bars = 1 day)
    """

    base_fee_bps: float = 5.0
    gamma: float = 0.6
    vol_lookback: int = 24      # bars used for trailing volatility
    adv_lookback: int = 24      # bars summed for trailing 24h dollar volume

    # ------------------------------------------------------------------ #
    # Trailing liquidity / risk inputs (all look-ahead-safe)             #
    # ------------------------------------------------------------------ #
    def rolling_volatility(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Trailing per-bar return volatility.

        Backward-looking standard deviation, then ``.shift(1)``: the cost
        charged for a trade AT bar t is built from information available
        strictly BEFORE bar t. Without the shift the simulator would price
        impact using the very bar it is trading into - a look-ahead leak.
        """
        return (
            returns.rolling(self.vol_lookback, min_periods=2)
            .std(ddof=0)
            .shift(1)
        )

    def rolling_adv(self, dollar_volume: pd.DataFrame) -> pd.DataFrame:
        """Trailing 24h Average Daily (dollar) Volume.

        Summed over ``adv_lookback`` bars and lagged one bar, so ADV never
        contains the live bar's own volume.
        """
        return (
            dollar_volume.rolling(self.adv_lookback, min_periods=1)
            .sum()
            .shift(1)
        )

    # ------------------------------------------------------------------ #
    # Cost model                                                         #
    # ------------------------------------------------------------------ #
    def slippage_bps(self, position_notional, adv, volatility):
        """Execution drag in basis points - fully vectorised.

        Accepts scalars or arbitrarily shaped numpy/pandas inputs (they are
        broadcast together). A non-positive or NaN ADV yields NaN: impact
        cannot be priced without a liquidity reference, and a NaN must
        propagate rather than silently collapse to the base fee.
        """
        notional = np.abs(np.asarray(position_notional, dtype=float))
        adv_arr = np.asarray(adv, dtype=float)
        vol = np.asarray(volatility, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            participation = np.where(adv_arr > 0, notional / adv_arr, np.nan)
            # sqrt of a clipped-non-negative participation rate.
            impact_frac = self.gamma * vol * np.sqrt(np.clip(participation, 0.0, None))
        return self.base_fee_bps + impact_frac * 1e4

    def execute(self, side, ref_price, position_notional, adv, volatility):
        """Realised fill price after impact.

        side : +1 buy / -1 sell. Entry, take-profit and stop-loss are ALL
               trades and ALL pay impact - there is no free exit.
        Cost always works against the trader: a buy fills ABOVE the
        reference price, a sell BELOW it.

            fill = ref_price * (1 + side * slippage_bps / 1e4)
        """
        side_arr = np.asarray(side, dtype=float)
        ref = np.asarray(ref_price, dtype=float)
        slip_frac = self.slippage_bps(position_notional, adv, volatility) / 1e4
        return ref * (1.0 + side_arr * slip_frac)

    def round_trip_cost_bps(self, position_notional, adv, volatility):
        """Total round-trip drag (entry + exit) in basis points.

        A convenience for net-edge screening: a signal whose gross forward
        edge does not clear this number is not tradeable at that size.
        """
        return 2.0 * self.slippage_bps(position_notional, adv, volatility)
