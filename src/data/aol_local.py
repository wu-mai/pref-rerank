"""Local AOL-IA qlog reader — bypasses ir_datasets' slow CDN download path.

Reads the AOL-IA tarball (aol-data.tar.gz) directly from disk and yields
AolQlog-shaped records compatible with the existing pipeline:

    load_aol_sessions(min_session_len=..., max_sessions=...)
    build_doc_click_queries(qlogs_iter)

Format per the AOL README:
    AnonID  Query  QueryTime  ItemRank  ClickURL

Where:
- 3 fields: query NOT followed by a click within the next click-line
- 5 fields: query IS followed by a click; ItemRank is the SERP rank clicked
- Multiple 5-field lines for the same query = multiple clicks on the same SERP

We group consecutive lines by (user, query, time) to form one qlog entry
whose items = [(doc, rank, clicked), ...].

Output shape mirrors ir_datasets' AolQlog:
    user_id, query_id, query, query_orig, time (unix sec), items (tuple)
"""
from __future__ import annotations

import gzip
import hashlib
import logging
import tarfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes — same shape as ir_datasets' AolQlog / LogItem
# ---------------------------------------------------------------------------


@dataclass
class _LogItem:
    """One item in a query's SERP (rank + url + clicked)."""

    doc_id: str
    rank: Optional[int]
    clicked: bool


@dataclass
class _AolQlog:
    """One query event from AOL-IA, ir_datasets-shaped."""

    user_id: str
    query_id: str
    query: str
    query_orig: str
    time: float  # unix timestamp (seconds)
    items: tuple  # tuple[_LogItem]


# ---------------------------------------------------------------------------
# Tarball reader
# ---------------------------------------------------------------------------


def _parse_time(s: str) -> float:
    """Parse 'YYYY-MM-DD HH:MM:SS' into unix timestamp."""
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()
    except (ValueError, TypeError):
        return 0.0


def _iter_tarball(tarball_path: str | Path) -> Iterator[_AolQlog]:
    """Yield _AolQlog records from a local AOL-IA tarball.

    Groups consecutive lines with the same (user, query, time) into a single
    event whose items are the union of all click items in that group.
    """
    p = Path(tarball_path)
    if not p.exists():
        raise FileNotFoundError(f"AOL tarball not found: {p}")

    n_qlog = 0
    n_lines = 0
    with tarfile.open(p, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            if not member.name.endswith(".txt.gz"):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            logger.info("Reading %s ...", member.name)
            with gzip.GzipFile(fileobj=f) as gz:
                prev_key: Optional[tuple] = None
                prev_items: list = []
                prev_query: Optional[str] = None
                prev_query_orig: Optional[str] = None
                prev_user: Optional[str] = None
                prev_ts: float = 0.0
                prev_qid: Optional[str] = None

                def _flush():
                    """Emit a qlog for the accumulated (user, query, time) group."""
                    nonlocal n_qlog
                    if prev_key is None:
                        return
                    yield _AolQlog(
                        user_id=prev_user,
                        query_id=prev_qid,
                        query=prev_query,
                        query_orig=prev_query_orig,
                        time=prev_ts,
                        items=tuple(prev_items),
                    )
                    n_qlog += 1

                for raw in gz:
                    line = raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
                    n_lines += 1
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("AnonID"):
                        # header line
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    anon_id = parts[0]
                    query = parts[1]
                    ts_str = parts[2]
                    rank_str = parts[3] if len(parts) > 3 else "-"
                    url = parts[4] if len(parts) > 4 else "-"

                    # Skip queries that are URL-like (noise)
                    if query.startswith("http") or query.startswith("www."):
                        # flush prev, then start new
                        if prev_key is not None:
                            yield _AolQlog(
                                user_id=prev_user, query_id=prev_qid,
                                query=prev_query, query_orig=prev_query_orig,
                                time=prev_ts, items=tuple(prev_items),
                            )
                            n_qlog += 1
                        prev_key = None
                        continue

                    ts = _parse_time(ts_str)
                    key = (anon_id, query, ts_str)

                    if key != prev_key:
                        # New group: flush previous qlog
                        if prev_key is not None:
                            yield _AolQlog(
                                user_id=prev_user, query_id=prev_qid,
                                query=prev_query, query_orig=prev_query_orig,
                                time=prev_ts, items=tuple(prev_items),
                            )
                            n_qlog += 1
                        # Start new group
                        prev_key = key
                        prev_user = anon_id
                        prev_query = query
                        prev_query_orig = query
                        prev_ts = ts
                        prev_qid = hashlib.md5(f"{anon_id}|{query}|{ts_str}".encode()).hexdigest()[:16]
                        prev_items = []

                    # Append item
                    if rank_str == "-":
                        rank = None
                    else:
                        try:
                            rank = int(rank_str)
                        except ValueError:
                            rank = None
                    if url == "-" or url == "":
                        clicked = False
                        doc_id = ""
                    else:
                        clicked = True
                        doc_id = hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
                    prev_items.append(_LogItem(doc_id=doc_id, rank=rank, clicked=clicked))

                    if n_qlog % 500_000 == 0 and n_qlog > 0:
                        logger.info("  ... %d qlogs so far (lines=%d)", n_qlog, n_lines)

                # flush last group
                if prev_key is not None:
                    yield _AolQlog(
                        user_id=prev_user, query_id=prev_qid,
                        query=prev_query, query_orig=prev_query_orig,
                        time=prev_ts, items=tuple(prev_items),
                    )
                    n_qlog += 1

    logger.info("Iterated %d qlogs from %d raw lines", n_qlog, n_lines)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def iter_local_qlogs(tarball_path: str | Path) -> Iterator[_AolQlog]:
    """Direct iterator over local AOL qlogs."""
    yield from _iter_tarball(tarball_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "/root/gcy/Rerank/data/aol-data.tar.gz"
    n = 0
    for q in iter_local_qlogs(p):
        n += 1
        if n <= 5:
            print(f"  user_id={q.user_id} query={q.query!r} time={q.time} items={len(q.items)}")
            for it in q.items:
                print(f"    doc={it.doc_id} rank={it.rank} clicked={it.clicked}")
        if n >= 10:
            break
    print(f"Read {n} qlogs")