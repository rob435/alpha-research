"""
Strategy Backtester  -  cross-sectional dollar-neutral long/short
=================================================================

Event-driven backtester for the locked overextension strategy.

LOCKED STRATEGY SPEC
--------------------
  Entry  - each rebalance bar, rank the eligible universe by the Latent
           Overextension Score; SHORT the top quantile (most overextended),
           LONG the bottom quantile (least overextended).
  Sizing - equal-weight within each leg; the two legs dollar-balanced
           (dollar-neutral); per-name notional cap; liquidity-filtered.
  Exit   - a fixed `holding_bars` horizon, OR earlier on a hard stop-loss.
  Costs  - SquareRootImpactSimulator slippage baked into every fill price.

WHY AN EVENT LOOP, NOT A WEIGHT MATRIX
--------------------------------------
The stop-loss makes each position's lifetime PATH-DEPENDENT: you cannot know
the bar-(t+2) holding until you have seen whether the stop fired at t+1. A
vectorised weight matrix cannot express that; a bar-by-bar loop can.

LOOK-AHEAD SAFETY
-----------------
At bar t the score and the close are both already known. A rebalance decision
at t ranks on score_t and fills at close_t; an exit detected at t (stop
breached or horizon reached) fills at close_t. Every fill price uses only
information available at or before its own bar - no future data enters.
Realistic execution cost on top of the close is the slippage model's job.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_platform.execution.impact import SquareRootImpactSimulator


@dataclass
class _Position:
    """One open lot. Lots are independent - the same symbol may be held by
    several overlapping cohorts, each tracked separately."""
    symbol: str
    col: int                 # column index into the price/score matrices
    side: int                # +1 long, -1 short
    entry_idx: int           # bar index at which the lot was opened
    entry_fill: float        # fill price INCLUDING entry slippage
    qty: float
    last_price: float        # last valid price (carried over data gaps)


class StrategyBacktester:
    """Bar-by-bar backtester for the dollar-neutral overextension L/S book.

    Parameters
    ----------
    impact : SquareRootImpactSimulator
        Cost model; slippage is folded into every fill price.
    holding_bars : int
        Fixed exit horizon N.
    quantile : float
        Fraction of the eligible universe shorted (top) and longed (bottom).
    stop_loss : float
        Hard stop: a position is closed once its adverse move (close-to-close)
        reaches this fraction. Mandatory risk control for the short book -
        the continuation-pump tail is otherwise unbounded.
    rebalance_every : int
        Bars between rebalances. With rebalance_every < holding_bars, cohorts
        overlap; each cohort is sized down so total gross stays ~constant.
    gross_exposure : float
        Target gross book size (long notional + short notional).
    min_dollar_volume : float
        Per-bar dollar-volume floor for entry eligibility (can't short an
        illiquid name).
    max_weight : float
        Per-name notional cap, as a fraction of `gross_exposure`.
    """

    def __init__(self, impact: SquareRootImpactSimulator | None = None,
                 holding_bars: int = 4, quantile: float = 0.10,
                 stop_loss: float = 0.08, rebalance_every: int = 1,
                 gross_exposure: float = 1.0, min_dollar_volume: float = 0.0,
                 max_weight: float = 0.10) -> None:
        if not 0.0 < quantile <= 0.5:
            raise ValueError("quantile must be in (0, 0.5]")
        if holding_bars < 1 or rebalance_every < 1:
            raise ValueError("holding_bars and rebalance_every must be >= 1")
        if stop_loss <= 0.0:
            raise ValueError("stop_loss must be positive")
        self.impact = impact or SquareRootImpactSimulator()
        self.holding_bars = int(holding_bars)
        self.quantile = float(quantile)
        self.stop_loss = float(stop_loss)
        self.rebalance_every = int(rebalance_every)
        self.gross_exposure = float(gross_exposure)
        self.min_dollar_volume = float(min_dollar_volume)
        self.max_weight = float(max_weight)

    # ------------------------------------------------------------------ #
    # Fills                                                              #
    # ------------------------------------------------------------------ #
    def _fill(self, exec_side: int, ref_price: float, notional: float,
              adv: float, vol: float) -> float:
        """Fill price after square-root impact. `exec_side` is +1 for a buy
        (fills above ref) and -1 for a sell (fills below) - cost always works
        against the trader. If ADV/vol are unavailable (e.g. a delisting bar)
        impact is unpriceable, so only the fixed base fee is charged."""
        if np.isfinite(adv) and adv > 0.0 and np.isfinite(vol):
            slip = float(self.impact.slippage_bps(notional, adv, vol)) / 1e4
        else:
            slip = self.impact.base_fee_bps / 1e4
        return ref_price * (1.0 + exec_side * slip)

    # ------------------------------------------------------------------ #
    # Backtest loop                                                      #
    # ------------------------------------------------------------------ #
    def run(self, score: pd.DataFrame, close: pd.DataFrame,
            dollar_volume: pd.DataFrame) -> dict:
        """Run the backtest over the supplied (already out-of-sample) panels.

        Returns a dict: equity_curve, returns, trades (per-lot DataFrame),
        and stats.
        """
        close = close.reindex(index=score.index, columns=score.columns)
        dollar_volume = dollar_volume.reindex(index=score.index,
                                              columns=score.columns)
        symbols = list(score.columns)
        times = score.index
        n_bars, n_sym = score.shape

        # Trailing risk/liquidity inputs for the cost model (both lagged one
        # bar inside the simulator -> look-ahead-safe).
        vol = self.impact.rolling_volatility(close.pct_change()).to_numpy(float)
        adv = self.impact.rolling_adv(dollar_volume).to_numpy(float)
        prices = close.to_numpy(float)
        scores = score.to_numpy(float)
        dvol = dollar_volume.to_numpy(float)

        # Overlapping cohorts share the gross budget so total exposure is
        # ~constant regardless of the holding/rebalance ratio.
        overlap = max(1, -(-self.holding_bars // self.rebalance_every))
        cohort_gross = self.gross_exposure / overlap
        per_name_cap = self.max_weight * self.gross_exposure

        book: list[_Position] = []
        closed: list[dict] = []
        realized = 0.0
        traded_notional = 0.0
        equity = np.empty(n_bars)

        for t in range(n_bars):
            # ---- 1. exits --------------------------------------------- #
            survivors: list[_Position] = []
            for pos in book:
                price = prices[t, pos.col]
                if np.isnan(price):
                    price = pos.last_price          # carry over a data gap
                else:
                    pos.last_price = price
                held = t - pos.entry_idx
                favourable = pos.side * (price / pos.entry_fill - 1.0)
                if favourable <= -self.stop_loss:
                    reason = "stop"
                elif held >= self.holding_bars:
                    reason = "horizon"
                elif t == n_bars - 1:
                    reason = "final"
                else:
                    survivors.append(pos)
                    continue
                exit_fill = self._fill(-pos.side, price, pos.qty * price,
                                       adv[t, pos.col], vol[t, pos.col])
                pnl = pos.side * pos.qty * (exit_fill - pos.entry_fill)
                realized += pnl
                traded_notional += pos.qty * exit_fill
                entry_notional = pos.qty * pos.entry_fill
                closed.append({
                    "symbol": pos.symbol,
                    "side": "long" if pos.side > 0 else "short",
                    "entry_time": times[pos.entry_idx],
                    "exit_time": times[t],
                    "bars_held": held,
                    "entry_price": pos.entry_fill,
                    "exit_price": exit_fill,
                    "notional": entry_notional,
                    "pnl": pnl,
                    "return": pnl / entry_notional if entry_notional else np.nan,
                    "exit_reason": reason,
                })
            book = survivors

            # ---- 2. entries ------------------------------------------- #
            if t % self.rebalance_every == 0 and t < n_bars - 1:
                eligible = [
                    i for i in range(n_sym)
                    if np.isfinite(scores[t, i]) and np.isfinite(prices[t, i])
                    and prices[t, i] > 0.0
                    and dvol[t, i] >= self.min_dollar_volume
                    and np.isfinite(vol[t, i]) and np.isfinite(adv[t, i])
                    and adv[t, i] > 0.0
                ]
                if len(eligible) >= 2:
                    eligible.sort(key=lambda i: scores[t, i])
                    n_leg = max(1, int(self.quantile * len(eligible)))
                    n_leg = min(n_leg, len(eligible) // 2)
                    longs = eligible[:n_leg]           # lowest score
                    shorts = eligible[-n_leg:]         # highest score
                    notional = min(per_name_cap, (cohort_gross / 2.0) / n_leg)
                    for col in shorts:
                        self._open(-1, col, t, prices, adv, vol,
                                   notional, symbols, book)
                        traded_notional += notional
                    for col in longs:
                        self._open(+1, col, t, prices, adv, vol,
                                   notional, symbols, book)
                        traded_notional += notional

            # ---- 3. mark to market ------------------------------------ #
            unrealized = 0.0
            for pos in book:
                price = prices[t, pos.col]
                if np.isnan(price):
                    price = pos.last_price
                unrealized += pos.side * pos.qty * (price - pos.entry_fill)
            equity[t] = self.gross_exposure + realized + unrealized

        return self._assemble(times, equity, closed, traded_notional)

    def _open(self, side: int, col: int, t: int, prices, adv, vol,
              notional: float, symbols, book: list) -> None:
        """Open one lot at bar t. Entry slippage is folded into the fill."""
        entry_fill = self._fill(side, prices[t, col], notional,
                                adv[t, col], vol[t, col])
        book.append(_Position(symbol=symbols[col], col=col, side=side,
                              entry_idx=t, entry_fill=entry_fill,
                              qty=notional / entry_fill,
                              last_price=prices[t, col]))

    # ------------------------------------------------------------------ #
    # Reporting                                                          #
    # ------------------------------------------------------------------ #
    def _assemble(self, times, equity: np.ndarray, closed: list,
                  traded_notional: float) -> dict:
        equity_curve = pd.Series(equity, index=times, name="equity")
        bar_return = equity_curve.diff() / self.gross_exposure
        trades = pd.DataFrame(closed)
        return {
            "equity_curve": equity_curve,
            "returns": bar_return,
            "trades": trades,
            "stats": self._stats(equity_curve, bar_return, trades,
                                  traded_notional, times),
        }

    @staticmethod
    def _bars_per_year(times) -> float:
        if len(times) < 2:
            return float("nan")
        step = pd.Series(times).diff().median()
        if pd.isna(step) or step.total_seconds() <= 0:
            return float("nan")
        return pd.Timedelta("365.25D") / step

    def _stats(self, equity: pd.Series, bar_return: pd.Series,
               trades: pd.DataFrame, traded_notional: float, times) -> dict:
        annual = self._bars_per_year(times)
        rets = bar_return.dropna()
        sharpe = float("nan")
        if len(rets) > 1 and rets.std() > 0 and np.isfinite(annual):
            sharpe = rets.mean() / rets.std() * np.sqrt(annual)
        drawdown = (equity / equity.cummax() - 1.0).min()

        stats = {
            "total_return": float((equity.iloc[-1] - equity.iloc[0])
                                  / self.gross_exposure),
            "ann_return": float(rets.mean() * annual) if np.isfinite(annual)
            else float("nan"),
            "sharpe": float(sharpe),
            "max_drawdown": float(drawdown),
            "n_trades": int(len(trades)),
            "turnover_x_gross": float(traded_notional / self.gross_exposure),
        }
        if len(trades):
            pnl = trades["pnl"]
            wins, losses = pnl[pnl > 0], pnl[pnl < 0]
            stats.update({
                "hit_rate": float((pnl > 0).mean()),
                "avg_win": float(wins.mean()) if len(wins) else 0.0,
                "avg_loss": float(losses.mean()) if len(losses) else 0.0,
                "avg_bars_held": float(trades["bars_held"].mean()),
                "pct_stopped_out": float((trades["exit_reason"] == "stop").mean()),
                "short_pnl": float(trades.loc[trades.side == "short", "pnl"].sum()),
                "long_pnl": float(trades.loc[trades.side == "long", "pnl"].sum()),
            })
        return stats


def summarize(result: dict) -> None:
    """Print a compact backtest report."""
    stats = result["stats"]
    print("\n=== Strategy backtest ===")
    for key in ("total_return", "ann_return", "sharpe", "max_drawdown",
                "n_trades", "hit_rate", "avg_win", "avg_loss",
                "avg_bars_held", "pct_stopped_out", "short_pnl", "long_pnl",
                "turnover_x_gross"):
        if key in stats:
            print(f"  {key:18s}: {stats[key]: .6f}")
    short_pnl = stats.get("short_pnl")
    if short_pnl is not None and short_pnl <= 0:
        print("  NOTE: short book P&L is non-positive -> the overextension")
        print("        thesis is not paying on the side it is supposed to.")


__all__ = ["StrategyBacktester", "summarize"]
