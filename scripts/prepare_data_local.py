"""One-time data preparation using the LOCAL AOL tarball.

Reads /root/gcy/Rerank/data/aol-data.tar.gz directly (no ir_datasets),
groups queries into sessions (gap > 30 min = new session), filters by
min_session_len, and writes cache/data/sessions.pkl + cache/data/doc_click_queries.pkl.

After this runs, all downstream scripts (evaluate.py, train.py, baselines)
load from cache in seconds.

Usage:
    python scripts/prepare_data_local.py \
        --tarball /root/gcy/Rerank/data/aol-data.tar.gz \
        --cache-dir /root/gcy/Rerank/cache/data \
        --min-session-len 4 --seed 42
"""
from __future__ import annotations

import argparse
import logging
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aol_local import iter_local_qlogs  # noqa: E402
from src.data.aol_session import (  # noqa: E402
    Session, QueryEvent, SerpItem,
    SESSION_GAP_SEC, _to_session,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_local")


def build_sessions(qlogs, min_session_len: int, seed: int):
    """Group consecutive queries by user into sessions."""
    by_user: dict[str, list] = defaultdict(list)
    t0 = time.time()
    n_qlog = 0
    for q in qlogs:
        items = [
            SerpItem(doc_id=str(it.doc_id), rank=it.rank, clicked=bool(it.clicked))
            for it in q.items
        ]
        by_user[q.user_id].append((q.query, q.time, items))
        n_qlog += 1
        if n_qlog % 1_000_000 == 0:
            logger.info("  ingested %d qlogs in %.1fs", n_qlog, time.time() - t0)
    logger.info("Ingested %d qlogs across %d users in %.1fs",
                n_qlog, len(by_user), time.time() - t0)

    sessions: list[Session] = []
    rng = random.Random(seed)
    for anon_id, recs in by_user.items():
        recs.sort(key=lambda r: r[1])
        current = []
        last_ts = None
        for r in recs:
            ts = r[1]
            if last_ts is not None and (ts - last_ts) > SESSION_GAP_SEC:
                if len(current) >= min_session_len:
                    sessions.append(_to_session(anon_id, current, len(sessions)))
                current = []
            current.append(r)
            last_ts = ts
        if len(current) >= min_session_len:
            sessions.append(_to_session(anon_id, current, len(sessions)))
    logger.info("Built %d sessions (min_events=%d)", len(sessions), min_session_len)
    return sessions


def build_doc_click_queries(qlogs) -> dict[str, list[str]]:
    """{doc_id: [queries that clicked it]}."""
    out: dict[str, list[str]] = defaultdict(list)
    n = 0
    for q in qlogs:
        for it in q.items:
            if it.clicked:
                out[str(it.doc_id)].append(q.query)
        n += 1
    logger.info("Built doc->queries map: %d docs from %d qlogs", len(out), n)
    return dict(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarball", default="/root/gcy/Rerank/data/aol-data.tar.gz")
    ap.add_argument("--cache-dir", default="/root/gcy/Rerank/cache/data")
    ap.add_argument("--min-session-len", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sessions_file = cache_dir / "sessions.pkl"
    doc_clicks_file = cache_dir / "doc_click_queries.pkl"

    if sessions_file.exists() and doc_clicks_file.exists():
        logger.info("Cache already exists, skipping.")
        return

    if not Path(args.tarball).exists():
        raise FileNotFoundError(
            f"AOL tarball not found at {args.tarball}. "
            f"Download with: curl -L -o {args.tarball} "
            "https://cim.mcgill.ca/~dudek/206/Logs/AOL-user-ct-collection/aol-data.tar.gz"
        )

    logger.info("=== Step 1: parsing qlogs (one pass over tarball) ===")
    t0 = time.time()
    # First pass: build sessions
    qlogs_iter = iter_local_qlogs(args.tarball)
    sessions = build_sessions(qlogs_iter, args.min_session_len, args.seed)
    logger.info("Step 1 done in %.1fs", time.time() - t0)
    sessions_file.write_bytes(pickle.dumps(sessions))
    logger.info("  cached to %s (%.1f MB)", sessions_file, sessions_file.stat().st_size / 1e6)

    logger.info("=== Step 2: building doc->queries map (second pass) ===")
    t1 = time.time()
    qlogs_iter = iter_local_qlogs(args.tarball)
    doc_click_queries = build_doc_click_queries(qlogs_iter)
    logger.info("Step 2 done in %.1fs", time.time() - t1)
    doc_clicks_file.write_bytes(pickle.dumps(doc_click_queries))
    logger.info("  cached to %s (%.1f MB)", doc_clicks_file, doc_clicks_file.stat().st_size / 1e6)

    logger.info("=== Done. Total: %.1fs ===", time.time() - t0)


if __name__ == "__main__":
    main()