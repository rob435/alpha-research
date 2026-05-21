"""
Smoke-test / demo runner for the alpha_platform research pipeline.
=================================================================

The repo ships no bundled market data (Binance Vision caches are
.gitignored and regenerable). This script fabricates a small, well-formed
synthetic OHLCV panel so the end-to-end pipeline can be exercised offline.

Run:  python demo.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_platform.pipeline import AlphaResearchPipeline


def synthetic_klines(
    n_bars: int = 400, seed: int = 7
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Build a causal, NaN-free synthetic OHLCV panel + sector map.

    Universe: 12 assets across 3 sectors (4 each) so the leave-one-out
    sector index is always well defined. Returns carry a shared market
    factor, a per-sector factor, and idiosyncratic noise.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n_bars, freq="h")

    sectors = {"L1": 4, "AI": 4, "DEFI": 4}
    sector_map: dict[str, str] = {}
    for sector, count in sectors.items():
        for i in range(count):
            sector_map[f"{sector}{i}"] = sector
    assets = list(sector_map)

    market = rng.normal(0.0, 0.010, n_bars)
    sector_factor = {s: rng.normal(0.0, 0.008, n_bars) for s in sectors}

    close = pd.DataFrame(index=index, columns=assets, dtype=float)
    for asset in assets:
        beta = rng.uniform(0.6, 1.4)
        idio = rng.normal(0.0, 0.012, n_bars)
        log_ret = beta * market + sector_factor[sector_map[asset]] + idio
        close[asset] = 100.0 * np.exp(np.cumsum(log_ret))

    # Derive a coherent OHLCV bar set from the close path.
    open_ = close.shift(1).bfill()
    span = pd.DataFrame(
        rng.uniform(0.002, 0.02, size=close.shape), index=index, columns=assets
    )
    high = pd.concat([open_, close]).groupby(level=0).max() * (1 + span)
    low = pd.concat([open_, close]).groupby(level=0).min() * (1 - span)
    volume = pd.DataFrame(
        rng.lognormal(mean=11.0, sigma=0.6, size=close.shape),
        index=index,
        columns=assets,
    )

    klines = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }
    return klines, sector_map


def main() -> None:
    klines, sector_map = synthetic_klines()
    print(
        f"Synthetic panel: {klines['close'].shape[0]} bars x "
        f"{klines['close'].shape[1]} assets across "
        f"{len(set(sector_map.values()))} sectors"
    )

    pipeline = AlphaResearchPipeline(klines, sector_map, train_frac=0.6)
    result = pipeline.run(forward_n=6, rank_window=6)

    print(f"\nTrain/test split at: {result['train_end']}")

    score = result["latent_extremity_score"]
    print(
        f"Latent extremity score panel: {score.shape}, "
        f"non-null = {int(score.notna().to_numpy().sum())}"
    )

    print("\n=== IC summary (out-of-sample) ===")
    for key, value in result["ic_summary"].items():
        print(f"  {key:14s}: {value}")

    print("\nPipeline ran end-to-end OK.")


if __name__ == "__main__":
    main()
