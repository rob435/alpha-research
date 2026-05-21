"""
Binance Vision data loader  (Phase 1 of the order book research track)
======================================================================

Binance publishes its entire multi-year futures archive as public files on
``data.binance.vision``. This module enumerates, downloads, integrity-checks,
caches and parses USDⓈ-M futures ``bookTicker`` (L1 tick) and ``aggTrades``
data.

POINT-IN-TIME UNIVERSE
----------------------
Survivorship bias is lethal for an altcoin blow-off/overextension thesis: the
delisted coins are disproportionately the ones that pumped and died. The
archive's own file calendar is the cure - a symbol has data files only for the
dates it actually traded, so *file existence is the historical listing
record*. ``pit_universe()`` reconstructs a date x symbol membership mask from
it; no separate (and itself survivorship-biased) listings database is needed.

DATA REALITY
------------
Binance Vision serves L1 ticks (``bookTicker``) and aggregated trades
(``aggTrades``) historically - NOT the full L2 incremental ``@depth`` stream.
That is sufficient: canonical Order Flow Imbalance is an L1 quantity.
"""
from __future__ import annotations

import hashlib
import io
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

# Column layouts for the headerless (pre-2024) Binance Vision CSV variants.
_BOOK_TICKER_COLS = [
    "update_id", "best_bid_price", "best_bid_qty",
    "best_ask_price", "best_ask_qty", "transaction_time", "event_time",
]
_AGG_TRADES_COLS = [
    "agg_trade_id", "price", "quantity", "first_trade_id",
    "last_trade_id", "transact_time", "is_buyer_maker",
]
# data_type -> (canonical columns, the column to use as the event timestamp)
_DATASETS = {
    "bookTicker": (_BOOK_TICKER_COLS, "transaction_time"),
    "aggTrades": (_AGG_TRADES_COLS, "transact_time"),
}


class BinanceVisionLoader:
    """Download / cache / parse Binance Vision USDⓈ-M futures archives.

    Parameters
    ----------
    cache_dir : path
        Local directory for cached ``.zip`` files (downloaded once, reused).
    market : str
        ``"um"`` = USDⓈ-M futures (default), ``"cm"`` = COIN-M.
    verify_checksum : bool
        Verify each download against its published ``.CHECKSUM`` (SHA-256).
        A silently corrupt file would poison years of backtest, so default on.
    max_workers : int
        Thread-pool width for the symbol-parallel universe scan.
    """

    BASE = "https://data.binance.vision"
    # The CloudFront alias above serves an HTML browser UI for "/" requests,
    # so bucket LISTING must use the raw S3 REST endpoint. Path-style is
    # required: the bucket name contains dots, which breaks the wildcard TLS
    # cert used by virtual-hosted-style addressing.
    LIST_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

    def __init__(self, cache_dir, market: str = "um",
                 verify_checksum: bool = True, max_workers: int = 8,
                 timeout: int = 60, list_base: str | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.market = market
        self.verify_checksum = bool(verify_checksum)
        self.max_workers = int(max_workers)
        self.timeout = int(timeout)
        self.list_base = list_base or self.LIST_BASE

    # ------------------------------------------------------------------ #
    # HTTP + S3 listing                                                  #
    # ------------------------------------------------------------------ #
    def _http_get(self, url: str) -> bytes:
        """GET with bounded exponential-backoff retry. 404 -> FileNotFoundError
        (an expected, non-fatal outcome under a point-in-time universe)."""
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                req = Request(url, headers={"User-Agent": "alpha-platform/0.1"})
                with urlopen(req, timeout=self.timeout) as resp:
                    return resp.read()
            except HTTPError as exc:
                if exc.code == 404:
                    raise FileNotFoundError(url) from exc
                last_err = exc
            except (URLError, TimeoutError) as exc:
                last_err = exc
            time.sleep(1.5 * (attempt + 1))
        raise ConnectionError(f"GET failed after retries: {url} ({last_err})")

    @staticmethod
    def _local_tag(tag: str) -> str:
        """Strip any XML namespace -> bare element name (S3 namespace-agnostic)."""
        return tag.rsplit("}", 1)[-1]

    @classmethod
    def _parse_listing(cls, xml_bytes: bytes):
        """Parse one S3 ListBucketResult page.

        Returns (files, subdirs, truncated, next_marker) where files is a list
        of (key, size_bytes). Pure function - unit-testable without a network.
        """
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            snippet = xml_bytes[:160].decode("utf-8", "replace")
            raise ValueError(
                f"expected S3 listing XML but got a non-XML response "
                f"(wrong list_base?): {snippet!r}") from exc
        files: list[tuple[str, int]] = []
        subdirs: list[str] = []
        truncated, next_marker = False, ""
        for child in root:
            tag = cls._local_tag(child.tag)
            if tag == "Contents":
                key, size = None, "0"
                for node in child:
                    name = cls._local_tag(node.tag)
                    if name == "Key":
                        key = node.text
                    elif name == "Size":
                        size = node.text or "0"
                if key:
                    files.append((key, int(size)))
            elif tag == "CommonPrefixes":
                for node in child:
                    if cls._local_tag(node.tag) == "Prefix" and node.text:
                        subdirs.append(node.text)
            elif tag == "IsTruncated":
                truncated = (child.text or "").strip().lower() == "true"
            elif tag == "NextMarker":
                next_marker = child.text or ""
        return files, subdirs, truncated, next_marker

    def _list_prefix(self, prefix: str):
        """List one S3 prefix, transparently following truncation pages."""
        files: list[tuple[str, int]] = []
        subdirs: list[str] = []
        marker = ""
        while True:
            url = f"{self.list_base}?delimiter=/&prefix={quote(prefix)}"
            if marker:
                url += f"&marker={quote(marker)}"
            page_files, page_dirs, truncated, nxt = self._parse_listing(
                self._http_get(url))
            files += page_files
            subdirs += page_dirs
            if not truncated:
                break
            # S3 omits NextMarker when delimiter is set -> fall back to the
            # last key/prefix seen on this page.
            marker = nxt or (page_files[-1][0] if page_files
                             else page_dirs[-1] if page_dirs else "")
            if not marker:
                break
        return files, subdirs

    # ------------------------------------------------------------------ #
    # Universe discovery                                                 #
    # ------------------------------------------------------------------ #
    def _prefix(self, data_type: str, symbol: str = "",
                interval: str = "daily") -> str:
        base = f"data/futures/{self.market}/{interval}/{data_type}/"
        return f"{base}{symbol}/" if symbol else base

    def list_symbols(self, data_type: str = "bookTicker") -> list[str]:
        """Every symbol that has ever published `data_type` files."""
        _, subdirs = self._list_prefix(self._prefix(data_type))
        return sorted(p.rstrip("/").rsplit("/", 1)[-1] for p in subdirs)

    def list_dates(self, symbol: str,
                   data_type: str = "bookTicker") -> list[pd.Timestamp]:
        """Trading dates for which `symbol` has a `data_type` file."""
        files, _ = self._list_prefix(self._prefix(data_type, symbol))
        dates: list[pd.Timestamp] = []
        for key, _size in files:
            name = key.rsplit("/", 1)[-1]
            if not name.endswith(".zip"):       # skip .CHECKSUM companions
                continue
            try:
                dates.append(pd.Timestamp(
                    datetime.strptime(name[:-4][-10:], "%Y-%m-%d")))
            except ValueError:
                continue
        return sorted(dates)

    def pit_universe(self, data_type: str = "bookTicker") -> pd.DataFrame:
        """Point-in-time membership mask: index = daily dates, columns =
        symbols, value = True when the symbol traded that day.

        Reconstructed purely from archive file existence -> survivorship-bias
        free, including coins that have since been delisted.
        """
        symbols = self.list_symbols(data_type)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            date_lists = pool.map(
                lambda s: self.list_dates(s, data_type), symbols)
        columns = {
            sym: pd.Series(True, index=pd.DatetimeIndex(dates))
            for sym, dates in zip(symbols, date_lists) if dates
        }
        frame = pd.DataFrame(columns).sort_index()
        full = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
        return frame.reindex(full).notna()

    # ------------------------------------------------------------------ #
    # Download + parse                                                   #
    # ------------------------------------------------------------------ #
    def download(self, symbol: str, day, data_type: str = "bookTicker") -> Path:
        """Fetch one daily file into the cache (idempotent) and return its path."""
        day = pd.Timestamp(day)
        fname = f"{symbol}-{data_type}-{day:%Y-%m-%d}.zip"
        local = self.cache_dir / data_type / symbol / fname
        if local.exists():
            return local

        key = self._prefix(data_type, symbol) + fname
        blob = self._http_get(f"{self.BASE}/{key}")
        if self.verify_checksum:
            self._verify_checksum(blob, f"{self.BASE}/{key}.CHECKSUM")
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            if archive.testzip() is not None:
                raise IOError(f"corrupt archive: {key}")

        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(".part")
        tmp.write_bytes(blob)
        tmp.rename(local)                       # atomic publish into the cache
        return local

    def _verify_checksum(self, blob: bytes, checksum_url: str) -> None:
        try:
            published = self._http_get(checksum_url).decode().split()[0].lower()
        except FileNotFoundError:
            return                              # no checksum published -> skip
        if hashlib.sha256(blob).hexdigest() != published:
            raise IOError(f"SHA-256 mismatch: {checksum_url}")

    @staticmethod
    def _read_zip_csv(path, columns: list[str]) -> pd.DataFrame:
        """Read the single CSV inside a Binance Vision zip.

        Binance added header rows to these files around 2024; older files have
        none. The first field of a genuine data row is always a numeric id, so
        a non-numeric first token unambiguously flags a header row.
        """
        with zipfile.ZipFile(path) as archive:
            name = archive.namelist()[0]
            with archive.open(name) as handle:
                first_token = handle.readline().decode(
                    "utf-8", "replace").split(",")[0].strip()
            has_header = not first_token.lstrip("-").replace(".", "", 1).isdigit()
            with archive.open(name) as handle:
                frame = (pd.read_csv(handle) if has_header
                         else pd.read_csv(handle, header=None, names=columns))
        if len(frame.columns) == len(columns):  # pin to canonical names
            frame.columns = columns
        return frame

    @staticmethod
    def _to_datetime(values: pd.Series) -> pd.DatetimeIndex:
        """Epoch integers -> UTC datetimes, auto-detecting ms/us/ns.

        Binance has migrated some futures feeds from millisecond to microsecond
        stamps; magnitude is a robust discriminator (ms~1.7e12, us~1.7e15)."""
        numeric = pd.to_numeric(values, errors="coerce")
        median = float(np.nanmedian(numeric.to_numpy(dtype=float)))
        unit = "ns" if median > 1e17 else "us" if median > 1e14 else "ms"
        return pd.to_datetime(numeric, unit=unit, utc=True)

    def load(self, symbol: str, start, end,
             data_type: str = "bookTicker") -> pd.DataFrame:
        """Load and concatenate a symbol's daily files over [start, end].

        Missing days are skipped silently - under a point-in-time universe an
        absent file simply means the symbol was not listed that day. Note: an
        archive gap leaves one spurious order-flow tick at the seam, which is
        negligible once aggregated to 15m-1h bars.
        """
        columns, time_col = _DATASETS[data_type]
        frames: list[pd.DataFrame] = []
        for day in pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D"):
            try:
                path = self.download(symbol, day, data_type)
            except FileNotFoundError:
                continue
            frames.append(self._read_zip_csv(path, columns))
        if not frames:
            return pd.DataFrame(columns=columns)
        out = pd.concat(frames, ignore_index=True)
        out = out.set_index(self._to_datetime(out[time_col])).sort_index()
        out.index.name = "timestamp"
        return out

    def load_book_ticker(self, symbol: str, start, end) -> pd.DataFrame:
        """L1 best bid/ask tick stream - the input to Order Flow Imbalance."""
        return self.load(symbol, start, end, "bookTicker")

    def load_agg_trades(self, symbol: str, start, end) -> pd.DataFrame:
        """Aggregated public trades (aggressor side via `is_buyer_maker`)."""
        return self.load(symbol, start, end, "aggTrades")


__all__ = ["BinanceVisionLoader"]
