"""
Microstructure Research Entry Points
=====================================

Four wired paths for investigating the microstructure phenomena described
in the Bouchaud propagator / long-memory literature:

  run_propagator_study    Fits R(l) and C(l) from aggTrades for one symbol.
                          Audits the no-arbitrage condition.  Demonstrates
                          phantom P&L under permanent vs transient impact.

  run_flow_memory_study   Long-memory analysis across a universe.  Measures
                          trade-sign ACF, fits the power-law tail, reports
                          Hurst exponents for flow and prices.  Quantifies
                          the central tension (predictable flow, diffusive
                          prices).

  run_attribution_study   Post-trade markout decomposition for one symbol.
                          Separates permanent alpha from transient impact;
                          estimates adverse selection for passive fills.

  run_metaorder_study     Builds metaorder-signature feature matrix across
                          a universe.  Detects and characterises sustained-
                          pressure windows (TWAP/VWAP footprints).
"""
from __future__ import annotations

import pandas as pd

from alpha_platform.microstructure.propagator  import ImpactPropagator
from alpha_platform.microstructure.flow_memory import FlowMemoryAnalyzer
from alpha_platform.microstructure.metaorder   import MetaorderDetector
from alpha_platform.execution.attribution      import ImpactAttributor


# ──────────────────────────────────────────────────────────────────────────── #
#  Propagator study                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

def run_propagator_study(
    loader,
    symbol: str,
    start,
    end,
    freq: str = "1min",
    max_lag: int = 150,
    n_clips: int = 20,
) -> dict:
    """Propagator study for one symbol.

    Loads aggTrades, fits R(l) and C(l), audits no-arbitrage, and
    demonstrates the phantom arbitrage under permanent vs transient impact.

    Parameters
    ----------
    loader : BinanceVisionLoader
    symbol : str          e.g. 'BTCUSDT'
    start, end : str      date strings e.g. '2022-01-01'
    freq : str            bar frequency for aggregation (default '1min')
    max_lag : int         maximum lag for R(l) and C(l)
    n_clips : int         number of buy clips in the phantom-arb simulation

    Returns
    -------
    dict: response, flow_acf, no_arb_audit, phantom_arb_demo, symbol.
    """
    trades = loader.load_agg_trades(symbol, start, end)
    if trades.empty:
        raise ValueError(f"No aggTrades for {symbol} in [{start}, {end}]")

    prop  = ImpactPropagator(freq=freq, max_lag=max_lag).fit(trades)
    audit = prop.no_arbitrage_audit()
    demo  = prop.phantom_arbitrage_demo(n_clips=n_clips)

    print(f"\n── Propagator study: {symbol} [{start} → {end}] ──")
    print(f"  Hurst prices        : {audit['hurst_prices']:.3f}"
          f"  (diffusive ≈ 0.50)")
    print(f"  Hurst flow          : {audit['hurst_flow']:.3f}"
          f"  (long-memory > 0.60)")
    print(f"  ΣR(l)·C(l)          : {audit['na_product_sum']:.6f}  (no-arb proxy)")
    print(f"  Response plateau    : {audit['response_plateau']:.6f}  (at lag 50)")
    print(f"  Verdict             : {audit['interpretation']}")
    print(f"\n  Phantom arbitrage (buy {n_clips} clips, dump 1 lot):")
    print(demo.to_string())
    print()

    return {
        "symbol":          symbol,
        "response":        prop.response_function(),
        "flow_acf":        prop.flow_acf(),
        "no_arb_audit":    audit,
        "phantom_arb_demo": demo,
    }


# ──────────────────────────────────────────────────────────────────────────── #
#  Flow memory study                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

def run_flow_memory_study(
    loader,
    symbols: list[str],
    start,
    end,
    freq: str = "1min",
    max_lag: int = 500,
) -> dict:
    """Long-memory analysis across a universe of symbols.

    For each symbol: trade-sign ACF, power-law tail fit, Hurst for both
    flow and prices.  The resulting table directly quantifies the
    "predictable flow, diffusive prices" tension.

    Returns
    -------
    dict:
      summaries  pd.DataFrame (one row per symbol, key stats)
      acfs       dict symbol→pd.Series (full sign ACF)
    """
    analyzer  = FlowMemoryAnalyzer(freq=freq, max_lag=max_lag)
    summaries = []
    acfs: dict[str, pd.Series] = {}

    print(f"\n── Flow memory study: {len(symbols)} symbols "
          f"[{start} → {end}] ──")
    for sym in symbols:
        print(f"  {sym:<14} ", end="", flush=True)
        try:
            trades = loader.load_agg_trades(sym, start, end)
            if trades.empty:
                print("no data")
                continue
            summ         = analyzer.memory_summary(trades)
            summ["symbol"] = sym
            summaries.append(summ)
            acfs[sym]    = analyzer.sign_acf(trades)
            pl           = summ["power_law"]
            print(f"H_flow={summ['hurst_flow']:.2f}  "
                  f"H_prices={summ['hurst_prices']:.2f}  "
                  f"γ={pl.get('gamma', float('nan')):.2f}  "
                  f"[{pl.get('interpretation','?')[:40]}]")
        except Exception as exc:
            print(f"ERROR: {exc}")

    df = (
        pd.DataFrame(summaries).set_index("symbol")
        if summaries else pd.DataFrame()
    )
    if not df.empty:
        print(f"\n  Median H_flow  : {df['hurst_flow'].median():.3f}")
        print(f"  Median H_prices: {df['hurst_prices'].median():.3f}")
        print(f"  Median gap     : {(df['hurst_flow'] - df['hurst_prices']).median():.3f}")
    print()

    return {"summaries": df, "acfs": acfs}


# ──────────────────────────────────────────────────────────────────────────── #
#  Attribution study                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

def run_attribution_study(
    loader,
    symbol: str,
    start,
    end,
    freq: str = "1min",
    horizons: list[int] | None = None,
) -> dict:
    """Post-trade markout study for one symbol.

    Returns
    -------
    dict: markout_profile, permanent_transient, adverse_selection, symbol.
    """
    trades = loader.load_agg_trades(symbol, start, end)
    if trades.empty:
        raise ValueError(f"No aggTrades for {symbol}")

    attr    = ImpactAttributor(freq=freq, horizons=horizons)
    profile = attr.markout_profile(trades)
    pt      = attr.permanent_transient_split(trades)
    adv     = attr.adverse_selection(trades)

    print(f"\n── Attribution study: {symbol} [{start} → {end}] ──")
    print(f"  {pt['interpretation']}")
    print(f"  Permanent (alpha)   : {pt['permanent_component']:.2e}")
    print(f"  Transient (impact)  : {pt['transient_component']:.2e}")
    print(f"  Reversion fraction  : {pt['reversion_fraction']:.1%}")
    print("\n  Markout profile (mean markout, t-stat):")
    if not profile.empty:
        print(profile[["mean_markout", "t_stat"]].to_string())
    adv_lags = adv.get("adverse_selection_by_lag", {})
    if adv_lags:
        lag1 = list(adv_lags.values())[0]
        print(f"\n  Adverse selection at lag 1: {lag1:.2e}  "
              f"({'adverse' if lag1 > 0 else 'favourable'} for passive seller)")
    print()

    return {
        "symbol":              symbol,
        "markout_profile":     profile,
        "permanent_transient": pt,
        "adverse_selection":   adv,
    }


# ──────────────────────────────────────────────────────────────────────────── #
#  Metaorder study                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def run_metaorder_study(
    loader,
    symbols: list[str],
    start,
    end,
    freq: str = "5min",
    window_bars: int = 12,
    pressure_threshold: float = 0.60,
    min_sustained: int = 4,
) -> dict:
    """Metaorder-signature feature study across a universe.

    Returns
    -------
    dict:
      feature_panels   dict symbol→DataFrame (full feature matrix per bar)
      detected_windows dict symbol→DataFrame (flagged pressure windows)
      universe_stats   pd.DataFrame (one row per symbol, summary stats)
    """
    detector = MetaorderDetector(
        freq=freq, window_bars=window_bars,
        pressure_threshold=pressure_threshold,
        min_sustained=min_sustained,
    )
    panels:  dict[str, pd.DataFrame] = {}
    windows: dict[str, pd.DataFrame] = {}
    stats: list[dict] = []

    print(f"\n── Metaorder study: {len(symbols)} symbols "
          f"[{start} → {end}] ──")
    for sym in symbols:
        print(f"  {sym:<14} ", end="", flush=True)
        try:
            trades = loader.load_agg_trades(sym, start, end)
            if trades.empty:
                print("no data")
                continue
            feat = detector.metaorder_features(trades)
            wins = detector.sustained_pressure_windows(trades)
            panels[sym]  = feat
            windows[sym] = wins
            avg_w = round(wins["n_bars"].mean(), 1) if len(wins) else 0
            max_w = int(wins["n_bars"].max())        if len(wins) else 0
            stats.append({
                "symbol":               sym,
                "n_bars":               len(feat),
                "n_metaorder_windows":  len(wins),
                "avg_window_bars":      avg_w,
                "max_window_bars":      max_w,
                "mean_pressure_score":  round(float(feat["pressure_score"].mean()), 3),
                "mean_abs_tfi":         round(float(feat["tfi"].abs().mean()), 3),
                "mean_interval_entropy":round(float(feat["interval_entropy"].mean(skipna=True)), 3),
            })
            print(f"{len(wins):3d} windows  "
                  f"avg={avg_w:.1f} bars  "
                  f"max={max_w} bars  "
                  f"pressure={stats[-1]['mean_pressure_score']:.2f}")
        except Exception as exc:
            print(f"ERROR: {exc}")

    df = (
        pd.DataFrame(stats).set_index("symbol")
        if stats else pd.DataFrame()
    )
    print()
    return {
        "feature_panels":  panels,
        "detected_windows": windows,
        "universe_stats":  df,
    }


__all__ = [
    "run_propagator_study",
    "run_flow_memory_study",
    "run_attribution_study",
    "run_metaorder_study",
]
