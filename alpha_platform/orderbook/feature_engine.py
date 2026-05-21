"""
Order Book Feature Engine  (Phase 1 - L1 / top-of-book alphas)
==============================================================

Computes the canonical L1 order-book alphas from Binance Vision ``bookTicker``
ticks and aggregates them onto the platform's mid-frequency bar grid.

Primary signal - Order Flow Imbalance (OFI), the Cont-Kukanov-Stoikov measure
of net best-quote order flow. OFI is an L1 quantity *by construction* (it is
defined purely on best bid/ask price and size changes), so ``bookTicker`` is
exactly sufficient - no full L2 book is required.

Sign convention: every feature here is signed so that POSITIVE = buy-side
pressure (bid built up / ask consumed / microprice above mid).

VENUE PORTABILITY
-----------------
These are microstructure features and their RAW levels are venue-specific
(Binance depth != Bybit depth). ``build_orderbook_panels`` therefore
cross-sectionally normalises every feature per bar by default: that strips
venue-level scale, leaving a venue-agnostic feature cloud on which a
Binance-trained model can legitimately be scored against Bybit.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


def _prev(array: np.ndarray) -> np.ndarray:
    """One-step lag; row 0 references itself (its OFI is masked to NaN anyway)."""
    out = np.empty_like(array)
    out[0] = array[0]
    out[1:] = array[:-1]
    return out


class OrderbookFeatureEngine:
    """Stateless L1 order-book feature calculator over a `bookTicker` frame.

    Every method expects a DataFrame indexed by event timestamp with columns
    ``best_bid_price, best_bid_qty, best_ask_price, best_ask_qty`` and returns
    an aligned Series, so the features compose cleanly.
    """

    # ------------------------------------------------------------------ #
    # Order Flow Imbalance                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def order_flow_imbalance(book: pd.DataFrame) -> pd.Series:
        r"""Event-level Order Flow Imbalance (Cont, Kukanov & Stoikov, 2014).

        For consecutive best-quote observations n-1 -> n:

            e^b_n =  q^b_n     · 1{P^b_n >= P^b_{n-1}}
                   - q^b_{n-1} · 1{P^b_n <= P^b_{n-1}}
            e^a_n =  q^a_n     · 1{P^a_n <= P^a_{n-1}}
                   - q^a_{n-1} · 1{P^a_n >= P^a_{n-1}}
            OFI_n =  e^b_n - e^a_n

        Reading the three cases on the bid: a higher bid price contributes
        +q^b_n (fresh buy interest), an unchanged price contributes the size
        delta q^b_n - q^b_{n-1}, a lower bid contributes -q^b_{n-1} (the old
        queue vanished). The ask is the mirror image. Positive OFI_n = net
        buy-side pressure at the touch.

        The bar-level signal is the SUM of OFI_n (order flow is additive).
        """
        if len(book) == 0:
            return pd.Series(dtype=float, name="ofi")
        pb = book["best_bid_price"].to_numpy(dtype=float)
        qb = book["best_bid_qty"].to_numpy(dtype=float)
        pa = book["best_ask_price"].to_numpy(dtype=float)
        qa = book["best_ask_qty"].to_numpy(dtype=float)
        pb0, qb0 = _prev(pb), _prev(qb)
        pa0, qa0 = _prev(pa), _prev(qa)

        e_bid = qb * (pb >= pb0) - qb0 * (pb <= pb0)
        e_ask = qa * (pa <= pa0) - qa0 * (pa >= pa0)
        ofi = e_bid - e_ask
        ofi[0] = np.nan                     # no predecessor for the first tick
        return pd.Series(ofi, index=book.index, name="ofi")

    # ------------------------------------------------------------------ #
    # Static (state) L1 features                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def book_imbalance(book: pd.DataFrame) -> pd.Series:
        r"""Top-of-book imbalance  (q^b - q^a) / (q^b + q^a)  in [-1, 1].

        The instantaneous size asymmetry at the touch; a fast-decaying but
        strong short-horizon directional signal."""
        qb = book["best_bid_qty"].astype(float)
        qa = book["best_ask_qty"].astype(float)
        total = (qb + qa).replace(0.0, np.nan)
        return ((qb - qa) / total).rename("obi")

    @staticmethod
    def microprice_adjustment(book: pd.DataFrame) -> pd.Series:
        r"""Imbalance-weighted fair value, expressed as a deviation from mid.

            I          = q^b / (q^b + q^a)
            microprice = I · P^a + (1 - I) · P^b
            signal     = (microprice - mid) / mid

        Heavy bid size (I -> 1) pulls the fair value up toward the ask; the
        signal is the first-order Stoikov microprice adjustment, scaled by
        mid so it is comparable across assets of different price."""
        pb = book["best_bid_price"].astype(float)
        pa = book["best_ask_price"].astype(float)
        qb = book["best_bid_qty"].astype(float)
        qa = book["best_ask_qty"].astype(float)
        total = (qb + qa).replace(0.0, np.nan)
        imbalance = qb / total
        microprice = imbalance * pa + (1.0 - imbalance) * pb
        mid = 0.5 * (pa + pb)
        return ((microprice - mid) / mid.replace(0.0, np.nan)).rename(
            "microprice_adj")

    @staticmethod
    def relative_spread(book: pd.DataFrame) -> pd.Series:
        r"""Quoted spread as a fraction of mid:  (P^a - P^b) / mid.

        Not directional - a trading-cost / liquidity-regime variable."""
        pb = book["best_bid_price"].astype(float)
        pa = book["best_ask_price"].astype(float)
        mid = 0.5 * (pa + pb)
        return ((pa - pb) / mid.replace(0.0, np.nan)).rename("rel_spread")

    @staticmethod
    def mid_price(book: pd.DataFrame) -> pd.Series:
        """Simple mid (P^a + P^b) / 2 - the reference for forward returns."""
        return (0.5 * (book["best_bid_price"].astype(float)
                       + book["best_ask_price"].astype(float))).rename("mid")

    @classmethod
    def event_features(cls, book: pd.DataFrame) -> pd.DataFrame:
        """All event-level features for one symbol, aligned on the tick index."""
        return pd.DataFrame({
            "ofi": cls.order_flow_imbalance(book),
            "obi": cls.book_imbalance(book),
            "microprice_adj": cls.microprice_adjustment(book),
            "rel_spread": cls.relative_spread(book),
            "mid": cls.mid_price(book),
        })

    # ------------------------------------------------------------------ #
    # Bar aggregation                                                    #
    # ------------------------------------------------------------------ #
    _BAR_COLUMNS = ["ofi", "obi", "microprice_adj", "rel_spread",
                    "update_intensity", "mid_close"]

    @classmethod
    def to_bars(cls, event_features: pd.DataFrame, freq: str) -> pd.DataFrame:
        r"""Aggregate irregular tick-level features onto a regular bar grid.

        Different feature types demand different aggregators:
          * ofi              - a FLOW: summed over the bar.
          * obi / microprice / rel_spread - STATE variables: TIME-weighted
            mean, each tick weighted by how long it stood before the next
            tick. Tick arrival is bursty, so a plain mean over ticks would
            over-weight high-activity micro-bursts; time-weighting is the
            correct expectation of the state across the bar.
          * update_intensity - the tick COUNT (a book-activity regime proxy).
          * mid_close        - the bar-close mid, used downstream for returns.

        The final tick of each bar carries a one-tick boundary approximation
        in its dwell time; negligible at 15m-1h bars over thousands of ticks.
        """
        if event_features.empty:
            return pd.DataFrame(columns=cls._BAR_COLUMNS)

        ef = event_features.sort_index()
        # Dwell time: seconds each observation stood before being superseded.
        dwell = ef.index.to_series().diff().shift(-1).dt.total_seconds()
        dwell = dwell.fillna(dwell.median()).clip(lower=0.0)

        def time_weighted_mean(column: str) -> pd.Series:
            numerator = (ef[column] * dwell).resample(freq).sum(min_count=1)
            denominator = dwell.resample(freq).sum(min_count=1).replace(0.0, np.nan)
            return numerator / denominator

        return pd.DataFrame({
            "ofi": ef["ofi"].resample(freq).sum(min_count=1),
            "obi": time_weighted_mean("obi"),
            "microprice_adj": time_weighted_mean("microprice_adj"),
            "rel_spread": time_weighted_mean("rel_spread"),
            "update_intensity": ef["mid"].resample(freq).count(),
            "mid_close": ef["mid"].resample(freq).last(),
        })


# ---------------------------------------------------------------------- #
# Cross-sectional normalisation + universe panel builder                 #
# ---------------------------------------------------------------------- #
def _cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-bar (row-wise) z-score. De-meaning each bar also removes the
    universe-wide order-book tilt - a crude but effective order-book-beta
    neutralisation - while the /std step strips venue-level scale."""
    mean = panel.mean(axis=1)
    std = panel.std(axis=1, ddof=0).replace(0.0, np.nan)
    return panel.sub(mean, axis=0).div(std, axis=0)


def build_orderbook_panels(loader, symbols, start, end, freq: str = "15min",
                           normalize: str = "zscore",
                           max_workers: int = 8) -> dict[str, pd.DataFrame]:
    """Build wide ``timestamp x symbol`` bar panels for the whole universe.

    For each symbol: load bookTicker -> event features -> bar aggregation.
    Symbols are processed concurrently; a symbol that fails to load is skipped
    so one bad symbol cannot abort the universe scan. Panels are ragged by
    construction (PIT: a symbol is absent before listing / after delisting).

    `normalize` ("zscore" | "rank" | None) is applied to every feature panel
    except ``mid_close`` (a raw price level, needed for forward returns).
    """
    engine = OrderbookFeatureEngine()

    def process(symbol: str):
        try:
            book = loader.load_book_ticker(symbol, start, end)
        except Exception:                       # noqa: BLE001 - skip & continue
            return symbol, None
        if book.empty:
            return symbol, None
        return symbol, engine.to_bars(engine.event_features(book), freq)

    results: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for symbol, bars in pool.map(process, list(symbols)):
            if bars is not None and not bars.empty:
                results[symbol] = bars
    if not results:
        raise ValueError("no order book data loaded for any requested symbol")

    panels = {
        feature: pd.DataFrame({sym: results[sym][feature] for sym in results})
        for feature in OrderbookFeatureEngine._BAR_COLUMNS
    }
    # update_intensity is a heavy-tailed count -> compress before normalising.
    panels["update_intensity"] = np.log1p(panels["update_intensity"])

    to_normalise = ["ofi", "obi", "microprice_adj", "rel_spread",
                    "update_intensity"]
    if normalize == "zscore":
        for feature in to_normalise:
            panels[feature] = _cross_sectional_zscore(panels[feature])
    elif normalize == "rank":
        for feature in to_normalise:
            panels[feature] = panels[feature].rank(axis=1, pct=True)
    elif normalize is not None:
        raise ValueError("normalize must be 'zscore', 'rank' or None")
    return panels


__all__ = ["OrderbookFeatureEngine", "build_orderbook_panels"]
