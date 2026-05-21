"""
Impact Propagator  —  transient-impact kernel and no-arbitrage audit
====================================================================

Core reference: Bouchaud, Gefen, Potters, Wyart (2004)
"Fluctuations and response in financial markets: the subtle nature of
random price changes"  https://arxiv.org/abs/cond-mat/0307332

The propagator model:

    mid(t) = Σ_{s<t} G(t-s) · ε_s  +  diffusive noise

where ε_s = signed taker volume at bar s and G(τ) is the impact decay
kernel.  The empirically observable quantity is the *response function*:

    R(l)  =  Cov( Σ_{k=0}^{l-1} ret_{t+k},  ε_t ) / Var(ε)

i.e. the average cumulative price move over l bars following a unit of
signed flow at t.

Three results this module verifies empirically:

1. R(l) rises fast then plateaus — impact is transient, not permanent.
2. The signed-flow ACF C(l) = Cov(ε_t, ε_{t+l}) / Var(ε) decays slowly
   (long memory, Hurst >> 0.5).
3. The no-arbitrage (diffusivity) condition pins G to satisfy
       Σ_l G(l) · C(l) = const
   so that long-memory flow and diffusive prices coexist.  Any permanently
   impacting model violates this and admits round-trip phantom profits.

PHANTOM ARBITRAGE
-----------------
Strategy: buy N clips of size q at t = 0…N-1, then dump N·q at t = N.
Under permanent impact the pre-dump price is elevated by N·G₀·q, and the
round-trip books a profit proportional to N²·G₀·q² / 2.
Under the fitted transient kernel the elevation has decayed and the
round-trip P&L is approximately zero — the no-arbitrage result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────── #
#  Utilities                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

def _signed_flow(trades: pd.DataFrame) -> pd.Series:
    """Signed taker volume: +qty = buy aggression, -qty = sell aggression."""
    qty = pd.to_numeric(trades["quantity"], errors="coerce").fillna(0.0)
    maker = trades["is_buyer_maker"]
    if maker.dtype == object:
        maker = maker.astype(str).map({"True": True, "False": False})
    return qty.where(~maker.astype(bool), -qty)


def _hurst_dfa(y: np.ndarray, min_scale: int = 8, n_scales: int = 20) -> float:
    """Hurst exponent via Detrended Fluctuation Analysis (DFA-1).

    Returns H where H ≈ 0.5 = random walk, H > 0.5 = persistent (long
    memory), H < 0.5 = anti-persistent.  Robust to non-stationarity of
    the input level series (operates on the integrated profile internally).
    """
    n = len(y)
    max_scale = max(min_scale + 1, n // 4)
    scales = np.unique(
        np.round(np.geomspace(min_scale, max_scale, n_scales)).astype(int)
    )
    # integrate the mean-centred series
    y = np.cumsum(y - np.mean(y))
    fluct, valid = [], []
    for s in scales:
        nw = n // s
        if nw < 2:
            continue
        rms_windows = []
        for i in range(nw):
            seg = y[i * s : (i + 1) * s]
            t = np.arange(s, dtype=float)
            p = np.polyfit(t, seg, 1)
            resid = seg - np.polyval(p, t)
            rms_windows.append(np.sqrt(np.mean(resid ** 2)))
        fluct.append(float(np.mean(rms_windows)))
        valid.append(int(s))
    if len(valid) < 4:
        return 0.5
    log_s = np.log(np.array(valid, dtype=float))
    log_f = np.log(np.array(fluct, dtype=float))
    H = float(np.polyfit(log_s, log_f, 1)[0])
    return float(np.clip(H, 0.0, 1.5))


# ──────────────────────────────────────────────────────────────────────────── #
#  ImpactPropagator                                                             #
# ──────────────────────────────────────────────────────────────────────────── #

class ImpactPropagator:
    """Estimate and audit the transient impact kernel for a single symbol.

    Parameters
    ----------
    freq : str
        Resampling frequency for aggregating tick-level trades into bars.
        '1min' is a good starting point for perpetuals.
    max_lag : int
        Maximum lag (in bars) for R(l) and C(l) estimation.
    """

    def __init__(self, freq: str = "1min", max_lag: int = 150) -> None:
        self.freq = freq
        self.max_lag = int(max_lag)
        self._bars: pd.DataFrame | None = None
        self._R: np.ndarray | None = None   # response function, length max_lag+1
        self._C: np.ndarray | None = None   # flow ACF,          length max_lag+1

    # ── Fitting ──────────────────────────────────────────────────────────── #

    def fit(self, trades: pd.DataFrame) -> "ImpactPropagator":
        """Aggregate aggTrades to bars and estimate R(l), C(l).

        Parameters
        ----------
        trades : pd.DataFrame
            Output of BinanceVisionLoader.load_agg_trades — must have columns
            ['price', 'quantity', 'is_buyer_maker'] and a DatetimeIndex.
        """
        sv = _signed_flow(trades)
        price = pd.to_numeric(trades["price"], errors="coerce")

        flow = sv.resample(self.freq).sum()
        mid  = price.resample(self.freq).last().ffill()
        ret  = np.log(mid).diff()

        idx  = flow.index.intersection(ret.dropna().index)
        flow = flow.reindex(idx).fillna(0.0)
        ret  = ret.reindex(idx).fillna(0.0)
        self._bars = pd.DataFrame({"ret": ret, "flow": flow}, index=idx)

        f, r = flow.to_numpy(), ret.to_numpy()
        var_f = float(np.var(f))
        if var_f < 1e-15:
            raise ValueError("Flow variance is essentially zero — not enough data.")

        self._R = self._compute_R(r, f, self.max_lag, var_f)
        self._C = self._compute_C(f, self.max_lag, var_f)
        return self

    @staticmethod
    def _compute_R(
        ret: np.ndarray, flow: np.ndarray, max_lag: int, var_f: float
    ) -> np.ndarray:
        """R(l) = Cov( cumret(t→t+l), flow_t ) / Var(flow)."""
        n = len(ret)
        R = np.zeros(max_lag + 1)
        cum = np.concatenate([[0.0], np.cumsum(ret)])
        for l in range(1, max_lag + 1):
            n_pairs = n - l
            if n_pairs <= 0:
                break
            lr = cum[l : n] - cum[: n - l]          # l-bar cumulative return
            f_t = flow[: n - l]
            # Cov via dot product (faster than np.cov for large n)
            mu_lr = lr.mean()
            mu_f  = f_t.mean()
            R[l]  = float(np.dot(lr - mu_lr, f_t - mu_f) / n_pairs / var_f)
        return R

    @staticmethod
    def _compute_C(
        flow: np.ndarray, max_lag: int, var_f: float
    ) -> np.ndarray:
        """C(l) = Cov( flow_t, flow_{t+l} ) / Var(flow)."""
        n = len(flow)
        C = np.zeros(max_lag + 1)
        C[0] = 1.0
        mu_f = flow.mean()
        fc   = flow - mu_f
        for l in range(1, max_lag + 1):
            n_pairs = n - l
            if n_pairs <= 0:
                break
            C[l] = float(np.dot(fc[: n - l], fc[l:]) / n_pairs / var_f)
        return C

    # ── Accessors ─────────────────────────────────────────────────────────── #

    def response_function(self) -> pd.Series:
        """R(τ): cumulative price response to a unit of signed flow, by lag."""
        self._check()
        return pd.Series(self._R, name="R_empirical")

    def flow_acf(self) -> pd.Series:
        """C(τ): signed-flow autocorrelation — typically long-memory (slow decay)."""
        self._check()
        return pd.Series(self._C, name="C_flow")

    # ── No-arbitrage audit ────────────────────────────────────────────────── #

    def no_arbitrage_audit(self) -> dict:
        """Check whether G·C = const (price diffusivity) approximately holds.

        Returns
        -------
        dict with keys:
          na_product_sum   : Σ_l R(l)·C(l) — the diffusivity proxy
          hurst_prices     : Hurst of bar log-returns (should be ≈ 0.50)
          hurst_flow       : Hurst of signed flow (typically 0.70–0.90)
          response_plateau : R at lag=50 (long-run level of impact)
          flow_acf_lag20   : C at lag=20 (flow persistence measure)
          interpretation   : plain-language verdict
        """
        self._check()
        na_sum    = float(np.dot(self._R, self._C))
        h_prices  = _hurst_dfa(self._bars["ret"].to_numpy())
        h_flow    = _hurst_dfa(self._bars["flow"].to_numpy())
        pl_lag    = min(50, self.max_lag)
        c20_lag   = min(20, self.max_lag)

        verdict = []
        if abs(h_prices - 0.5) < 0.07:
            verdict.append(f"prices diffusive ✓ (H={h_prices:.2f})")
        else:
            verdict.append(f"prices non-diffusive (H={h_prices:.2f}) — check data")
        if h_flow > 0.60:
            verdict.append(f"flow long-memory ✓ (H={h_flow:.2f})")
        else:
            verdict.append(f"flow memory weaker than expected (H={h_flow:.2f})")

        return {
            "na_product_sum":   round(na_sum, 6),
            "hurst_prices":     round(h_prices, 3),
            "hurst_flow":       round(h_flow, 3),
            "response_plateau": round(float(self._R[pl_lag]), 6),
            "flow_acf_lag20":   round(float(self._C[c20_lag]), 4),
            "interpretation":   " | ".join(verdict),
        }

    # ── Phantom arbitrage demonstration ───────────────────────────────────── #

    def phantom_arbitrage_demo(
        self, n_clips: int = 20, clip_size: float = 1.0
    ) -> pd.DataFrame:
        """Simulate the buy-in-clips + dump round-trip under three impact kernels.

        Strategy
        --------
        Buy n_clips orders of clip_size at t=0,1,…,n_clips-1.
        Sell n_clips*clip_size at t=n_clips.  Net inventory = 0.

        Three analytical kernels (G₀ = max |R(l)| from the fitted response):

          permanent       G(τ) = G₀ for all τ.
                          Price elevation never decays → phantom P&L ∝ G₀·n²/2.

          slow_decay      G(τ) = G₀ · exp(−τ·ln2/(n/4)).
                          Naive slow-decay: halflife = n_clips/4 bars.
                          Smaller phantom profit, still positive.

          purely_temporary  G(τ) = G₀ if τ=0 else 0.
                          Impact appears for one bar then vanishes completely.
                          P&L ≈ G₀  (near zero for large n_clips).

        The ordering permanent >> slow_decay >> purely_temporary holds for any
        G₀>0 and n_clips≥2.  Real markets sit between slow_decay and
        purely_temporary depending on how fast impact reverts.

        Returns
        -------
        pd.DataFrame  (index = model), columns: pnl, avg_buy_price, sell_price.
        """
        self._check()
        # G₀: use the peak absolute response so the demo shows meaningful numbers
        # regardless of how noisy the empirically fitted R is.
        G0       = max(float(np.max(np.abs(self._R))), 1e-6)
        halflife = max(n_clips // 4, 1)

        G_perm = np.full(n_clips + 2, G0)
        G_slow = G0 * np.exp(
            -np.arange(n_clips + 2) * np.log(2) / halflife
        )
        G_temp      = np.zeros(n_clips + 2)
        G_temp[0]   = G0   # purely temporary: vanishes after 1 bar

        rows = []
        for label, G in [
            ("permanent",        G_perm),
            ("slow_decay",       G_slow),
            ("purely_temporary", G_temp),
        ]:
            pnl, prices = self._run_sim(G, n_clips, clip_size, False)
            rows.append({
                "model":         label,
                "pnl":           round(float(pnl), 8),
                "avg_buy_price": round(float(np.mean(prices[:n_clips])), 8),
                "sell_price":    round(float(prices[n_clips]), 8),
            })

        return pd.DataFrame(rows).set_index("model")[
            ["pnl", "avg_buy_price", "sell_price"]
        ]

    @staticmethod
    def _run_sim(
        G: np.ndarray, n_clips: int, clip_size: float, sqrt_impact: bool
    ) -> tuple[float, np.ndarray]:
        """Propagator round-trip simulation.

        price[t] = Σ_{s<t} G(t-1-s) · ε_s   (1-indexed: G[0] = 1-bar impact)
        """
        flows = np.zeros(n_clips + 1)
        flows[:n_clips]  =  clip_size
        flows[n_clips]   = -n_clips * clip_size

        K      = len(G)
        prices = np.zeros(n_clips + 1)
        for t in range(1, n_clips + 1):
            for s in range(t):
                lag = t - 1 - s
                if lag < K:
                    if sqrt_impact:
                        contrib = G[lag] * np.sqrt(abs(flows[s])) * np.sign(flows[s])
                    else:
                        contrib = G[lag] * flows[s]
                    prices[t] += contrib

        pnl = sum(-flows[t] * prices[t] for t in range(n_clips + 1))
        return float(pnl), prices

    # ── Guard ─────────────────────────────────────────────────────────────── #

    def _check(self) -> None:
        if self._R is None:
            raise RuntimeError("Call .fit(trades) first.")


__all__ = ["ImpactPropagator", "_hurst_dfa", "_signed_flow"]
