"""
Trade Flow Engine  -  the multi-year training backbone
======================================================

Binance Vision publishes only ~11 months of L1 ``bookTicker`` (2023-05 ..
2024-04), but ``aggTrades`` spans the full multi-year history (2019 -> present).
This module extracts the trade-driven cousin of Order Flow Imbalance from
``aggTrades`` so the Latent Overextension Score can be trained on a gapless,
multi-regime dataset.

Trade Flow Imbalance (TFI)
--------------------------
``aggTrades`` records every aggregated taker trade and, via ``is_buyer_maker``,
its aggressor side:

    is_buyer_maker = True  -> the BUYER was the maker -> the taker (aggressor)
                              was the SELLER -> sell-side aggression -> -qty
    is_buyer_maker = False -> the taker was the BUYER -> buy-side aggression
                              -> +qty

TFI over a bar is the net signed taker volume as a fraction of total volume:

    TFI = sum(signed_qty) / sum(qty)        in [-1, 1]

It is the trade-flow analogue of top-of-book imbalance and the *aggressive*
component of OFI - and, unlike OFI, it is computable over Binance's entire
history.

OVEREXTENSION FEATURES
----------------------
The thesis shorts altcoin blow-off peaks, so the model must also see HOW FAR a
coin has run. ``build_trade_flow_panels`` derives, from the trade-price panel,
a market-neutral trailing ``runup`` and its cross-sectional ``rank velocity`` -
the idiosyncratic over-extension signal.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


def _to_bool(series: pd.Series) -> pd.Series:
    """Coerce a Binance ``is_buyer_maker`` column to bool, tolerating native
    bool, 'true'/'false' strings, and 0/1 numerics across archive vintages."""
    if series.dtype == bool:
        return series
    if series.dtype == object:
        return (series.astype(str).str.strip().str.lower()
                .map({"true": True, "false": False,
                      "1": True, "0": False}))
    return series.astype(float) != 0.0


def _cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-bar (row-wise) z-score - removes the universe-wide tilt and strips
    venue-level scale so a Binance-trained model transfers to Bybit."""
    mean = panel.mean(axis=1)
    std = panel.std(axis=1, ddof=0).replace(0.0, np.nan)
    return panel.sub(mean, axis=0).div(std, axis=0)


class TradeFlowEngine:
    """Stateless trade-flow feature calculator over an ``aggTrades`` frame.

    Expects a DataFrame indexed by event timestamp with columns
    ``price, quantity, is_buyer_maker``.
    """

    @staticmethod
    def signed_volume(agg_trades: pd.DataFrame) -> pd.Series:
        r"""Per-trade signed taker volume: +qty for buy-aggressive trades,
        -qty for sell-aggressive. The sign is read off ``is_buyer_maker``."""
        qty = agg_trades["quantity"].to_numpy(dtype=float)
        buyer_maker = _to_bool(agg_trades["is_buyer_maker"]).to_numpy()
        sign = np.where(buyer_maker, -1.0, 1.0)
        return pd.Series(qty * sign, index=agg_trades.index,
                         name="signed_volume")

    @classmethod
    def event_features(cls, agg_trades: pd.DataFrame) -> pd.DataFrame:
        """Per-trade features aligned on the trade index."""
        price = agg_trades["price"].astype(float)
        qty = agg_trades["quantity"].astype(float)
        return pd.DataFrame({
            "signed_volume": cls.signed_volume(agg_trades),
            "quantity": qty,
            "dollar_volume": price * qty,
            "price": price,
        })

    _BAR_COLUMNS = ["tfi", "trade_intensity", "dollar_volume", "close"]

    @classmethod
    def to_bars(cls, event_features: pd.DataFrame, freq: str) -> pd.DataFrame:
        r"""Aggregate per-trade flow onto a regular bar grid.

            tfi             = sum(signed_volume) / sum(quantity)   in [-1, 1]
            trade_intensity = count of aggregated trades in the bar
            dollar_volume   = sum(price * quantity)
            close           = last trade price in the bar (for forward returns)

        TFI normalises by total volume on purpose: the RATIO is bounded and
        size/venue-neutral, where a raw signed-volume sum would not be
        comparable across a $0.0001 coin and a $60k coin.
        """
        if event_features.empty:
            return pd.DataFrame(columns=cls._BAR_COLUMNS)
        grouped = event_features.sort_index().resample(freq)
        signed = grouped["signed_volume"].sum(min_count=1)
        volume = grouped["quantity"].sum(min_count=1).replace(0.0, np.nan)
        return pd.DataFrame({
            "tfi": signed / volume,
            "trade_intensity": grouped["price"].count(),
            "dollar_volume": grouped["dollar_volume"].sum(min_count=1),
            "close": grouped["price"].last(),
        })


def build_trade_flow_panels(loader, symbols, start, end, freq: str = "1h",
                            runup_window: int = 8, velocity_window: int = 4,
                            normalize: str = "zscore",
                            max_workers: int = 8) -> dict[str, pd.DataFrame]:
    """Build the wide multi-year backbone panels for the Latent Overextension
    Score.

    Per symbol: load aggTrades -> event features -> bar aggregation. Two
    PRICE-derived over-extension features are then derived cross-sectionally:

      runup               - trailing `runup_window`-bar log return, market-
                            neutralised by subtracting the cross-sectional
                            MEDIAN return each bar, so it measures
                            IDIOSYNCRATIC over-extension (a coin pumping on its
                            own story) rather than a market-wide rally.
      runup_rank_velocity - change over `velocity_window` bars in an asset's
                            cross-sectional percentile rank of `runup`
                            (rank-migration velocity).

    Every window looks strictly backward, so the panels are look-ahead-safe.

    Returns model-feature panels (cross-sectionally normalised): tfi,
    trade_intensity, runup, runup_rank_velocity. Plus two RAW panels - close
    (price, for forward returns) and dollar_volume (the liquidity / ADV input
    the backtester's cost model needs). Raw panels are never model features.
    """
    engine = TradeFlowEngine()

    def process(symbol: str):
        try:
            trades = loader.load_agg_trades(symbol, start, end)
        except Exception:                       # noqa: BLE001 - skip & continue
            return symbol, None
        if trades.empty:
            return symbol, None
        return symbol, engine.to_bars(engine.event_features(trades), freq)

    results: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for symbol, bars in pool.map(process, list(symbols)):
            if bars is not None and not bars.empty:
                results[symbol] = bars
    if not results:
        raise ValueError("no aggTrades data loaded for any requested symbol")

    panels = {
        col: pd.DataFrame({sym: results[sym][col] for sym in results})
        for col in TradeFlowEngine._BAR_COLUMNS
    }
    close = panels["close"]

    # Idiosyncratic run-up: trailing log return minus the cross-sectional
    # median return (the platform's robust market proxy). This isolates a coin
    # pumping on its OWN narrative from one merely riding a market-wide rally.
    trailing = np.log(close).diff(runup_window)
    runup = trailing.sub(trailing.median(axis=1), axis=0)
    panels["runup"] = runup
    panels["runup_rank_velocity"] = runup.rank(axis=1, pct=True).diff(velocity_window)
    panels["trade_intensity"] = np.log1p(panels["trade_intensity"])
    # close and dollar_volume stay RAW: close prices forward returns, and
    # dollar_volume is the liquidity / ADV input for the backtester's cost
    # model. Neither is a model feature.

    model_features = ["tfi", "trade_intensity", "runup", "runup_rank_velocity"]
    if normalize == "zscore":
        for feature in model_features:
            panels[feature] = _cross_sectional_zscore(panels[feature])
    elif normalize == "rank":
        for feature in model_features:
            panels[feature] = panels[feature].rank(axis=1, pct=True)
    elif normalize is not None:
        raise ValueError("normalize must be 'zscore', 'rank' or None")
    return panels


__all__ = ["TradeFlowEngine", "build_trade_flow_panels"]
