"""Build ONLY the doc->queries cache (Step 2 of prepare_data_local.py).

Useful when Step 1 already completed but Step 2 was interrupted.
Skips Step 1 (which would re-pickle all sessions).

Usage:
    python scripts/prepare_data_doc_clicks.py \
        --tarball /root/gcy/Rerank/data/aol-data.tar.gz \
        --cache-dir /root/gcy/Rerank/cache/data
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aol_local import iter_local_qlogs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_doc_clicks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarball", default="/root/gcy/Rerank/data/aol-data.tar.gz")
    ap.add_argument("--cache-dir", default="/root/gcy/Rerank/cache/data")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    doc_clicks_file = cache_dir / "doc_click_queries.pkl"

    if doc_clicks_file.exists():
        logger.info("doc_click_queries.pkl already exists, skipping.")
        return

    if not Path(args.tarball).exists():
        raise FileNotFoundError(f"Tarball not found: {args.tarball}")

    logger.info("Building doc->queries map (one pass over tarball)...")
    t0 = time.time()
    out: dict[str, list[str]] = defaultdict(list)
    n_qlog = 0
    for q in iter_local_qlogs(args.tarball):
        for it in q.items:
            if it.clicked:
                out[str(it.doc_id)].append(q.query)
        n_qlog += 1
        if n_qlog % 1_000_000 == 0:
            logger.info("  ... %d qlogs, %d docs in %.1fs",
                        n_qlog, len(out), time.time() - t0)
    logger.info("Done: %d qlogs, %d unique docs, %.1fs",
                n_qlog, len(out), time.time() - t0)
    doc_clicks_file.write_bytes(pickle.dumps(dict(out)))
    logger.info("Cached to %s (%.1f MB)", doc_clicks_file,
                doc_clicks_file.stat().st_size / 1e6)


if __name__ == "__main__":
    main()