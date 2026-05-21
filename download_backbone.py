"""
download_backbone.py - bulk-fill the aggTrades cache from Binance Vision.
========================================================================

aggTrades is the multi-year training backbone for the Latent Overextension
Score. This script enumerates the whole USDⓈ-M futures aggTrades universe and
downloads every daily file in [START, END] into a local cache.

Why this is not just a loop over ``BinanceVisionLoader.load``
-------------------------------------------------------------
``loader.load(sym, start, end, "aggTrades")`` downloads each daily file *and*
parses every CSV, then ``pd.concat``-s the whole multi-year history into one
in-memory DataFrame per symbol. For aggTrades that is tens of GB of RAM per
symbol - it will OOM long before the cache is full. For a pure cache-fill the
right primitive is ``loader.download(sym, day, "aggTrades")``: it fetches +
integrity-checks + caches one zip and returns the path, with no CSV parsing
and no concatenation. ``research.run_backbone`` later parses straight from the
cached zips, so nothing is lost.

Also note: ``BinanceVisionLoader``'s download path is *serial* - its
``max_workers`` argument only parallelises the universe scan, not downloads.
This script therefore drives its own thread pool over (symbol, day) tasks.

Safety
------
* Disk guard: stops gracefully once free space on the cache drive falls below
  MIN_FREE_GB, so it can never fill the OS drive. Re-run to resume.
* Idempotent: already-cached files are skipped, so a re-run resumes exactly
  where a crash / disk-guard stop left off.

Run:  python download_backbone.py        (use the repo's .venv interpreter)
"""
from __future__ import annotations

import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from alpha_platform.orderbook.vision_loader import BinanceVisionLoader

# ── CONFIGURE THESE ──────────────────────────────────────────────────────── #
CACHE_DIR   = r"C:\binance_cache"   # where zips land - use your largest drive
START       = "2021-01-01"          # set "2019-12-31" for full BTC/ETH history
END         = "2025-12-31"
DATA_TYPE   = "aggTrades"
MAX_WORKERS = 12                    # concurrent download threads
MIN_FREE_GB = 100                   # graceful-stop threshold (protects C:)
LOG_EVERY   = 200                   # progress-line cadence, in files
# ─────────────────────────────────────────────────────────────────────────── #


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def main() -> None:
    cache = Path(CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)
    loader = BinanceVisionLoader(CACHE_DIR, max_workers=MAX_WORKERS)
    start, end = pd.Timestamp(START), pd.Timestamp(END)

    free0 = _free_gb(cache)
    print(f"[{_ts()}] cache={CACHE_DIR}  free={free0:.0f} GB  "
          f"range={START}..{END}  workers={MAX_WORKERS}", flush=True)
    if free0 < MIN_FREE_GB:
        sys.exit(f"free space {free0:.0f} GB already below "
                 f"MIN_FREE_GB={MIN_FREE_GB} - aborting")

    # -- discover universe -------------------------------------------------- #
    print(f"[{_ts()}] discovering {DATA_TYPE} universe ...", flush=True)
    symbols = loader.list_symbols(DATA_TYPE)
    print(f"[{_ts()}]   {len(symbols)} symbols", flush=True)

    # -- Phase A: enumerate (symbol, day) tasks (parallel S3 listing) ------- #
    print(f"[{_ts()}] enumerating per-symbol date calendars ...", flush=True)

    def dates_for(sym: str):
        days = [d for d in loader.list_dates(sym, DATA_TYPE)
                if start <= d <= end]
        return sym, days

    tasks: list[tuple[str, pd.Timestamp]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for sym, days in pool.map(dates_for, symbols):
            tasks.extend((sym, d) for d in days)
    print(f"[{_ts()}]   {len(tasks)} symbol-days in range", flush=True)

    # -- skip already-cached files (makes re-runs resume cheaply) ----------- #
    def is_cached(sym: str, day: pd.Timestamp) -> bool:
        fname = f"{sym}-{DATA_TYPE}-{day:%Y-%m-%d}.zip"
        return (cache / DATA_TYPE / sym / fname).exists()

    pending = [(s, d) for (s, d) in tasks if not is_cached(s, d)]
    print(f"[{_ts()}]   {len(tasks) - len(pending)} already cached, "
          f"{len(pending)} to download", flush=True)
    if not pending:
        print(f"[{_ts()}] nothing to do - cache is already complete.", flush=True)
        return

    # -- Phase B: parallel download with a disk guard ----------------------- #
    stop = threading.Event()
    lock = threading.Lock()
    t0 = time.time()
    done = 0
    bytes_dl = 0
    failed: list[tuple[str, str, str]] = []
    total = len(pending)

    def worker(task: tuple[str, pd.Timestamp]) -> None:
        nonlocal done, bytes_dl
        if stop.is_set():
            return
        sym, day = task
        try:
            path = loader.download(sym, day, DATA_TYPE)
            size = path.stat().st_size
        except Exception as exc:                       # noqa: BLE001
            with lock:
                failed.append((sym, f"{day:%Y-%m-%d}", repr(exc)))
            return
        with lock:
            done += 1
            bytes_dl += size
            if done % LOG_EVERY == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0.0
                eta_h = (total - done) / rate / 3600 if rate else 0.0
                free_now = _free_gb(cache)
                print(f"[{_ts()}]  {done}/{total}  {bytes_dl / 1e9:.1f} GB  "
                      f"{rate:.1f} files/s  ETA {eta_h:.1f} h  "
                      f"free {free_now:.0f} GB", flush=True)
                if free_now < MIN_FREE_GB:
                    print(f"[{_ts()}] !! free space {free_now:.0f} GB < "
                          f"MIN_FREE_GB={MIN_FREE_GB} - stopping gracefully; "
                          f"re-run to resume.", flush=True)
                    stop.set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(worker, pending))

    elapsed = time.time() - t0
    print(f"\n[{_ts()}] DONE  {done}/{total} files  {bytes_dl / 1e9:.1f} GB  "
          f"in {elapsed / 3600:.2f} h  ({len(failed)} failures)", flush=True)
    if stop.is_set():
        print(f"[{_ts()}] stopped early on disk guard - re-run to resume.",
              flush=True)
    if failed:
        report = cache / "download_failures.txt"
        report.write_text("\n".join(f"{s},{d},{e}" for s, d, e in failed))
        print(f"[{_ts()}] {len(failed)} failures written to {report}", flush=True)


if __name__ == "__main__":
    main()
