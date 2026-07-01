"""
One-time data preparation: scan AOL qlog ONCE, cache everything to disk.

After this runs, all downstream scripts load from cache in seconds (not minutes).

Outputs:
  cache/data/sessions.pkl      — list[Session] (parsed from qlog)
  cache/data/doc_click_queries.pkl — {doc_id: [queries that clicked it]}

Run once, then everything is fast.
"""
from __future__ import annotations

import logging
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aol_session import load_aol_sessions  # noqa: E402
from src.model.queryflow_backbone import build_doc_click_queries  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare")


def main():
    cache_dir = Path("cache/data")
    cache_dir.mkdir(parents=True, exist_ok=True)

    sessions_file = cache_dir / "sessions.pkl"
    doc_clicks_file = cache_dir / "doc_click_queries.pkl"

    if sessions_file.exists() and doc_clicks_file.exists():
        logger.info("Cache already exists: %s, %s", sessions_file, doc_clicks_file)
        n_sess = len(pickle.loads(sessions_file.read_bytes()))
        n_docs = len(pickle.loads(doc_clicks_file.read_bytes()))
        logger.info("  sessions: %d, doc_click_queries: %d docs", n_sess, n_docs)
        return

    logger.info("=== One-time AOL qlog scan (takes 5-10 min) ===")
    logger.info("This builds the cache that all later scripts will use.")

    # --- Sessions ---
    t0 = time.time()
    logger.info("[1/2] Parsing sessions from qlog...")
    # Load ALL sessions (no max_sessions) so cache is complete
    sessions = load_aol_sessions(min_session_len=4, seed=42)
    logger.info("  %d sessions in %.1fs", len(sessions), time.time() - t0)
    sessions_file.write_bytes(pickle.dumps(sessions))
    logger.info("  cached to %s (%.1f MB)", sessions_file, sessions_file.stat().st_size / 1e6)

    # --- doc_click_queries (needs another qlog pass) ---
    t1 = time.time()
    logger.info("[2/2] Building doc->click queries map (another qlog pass)...")
    import ir_datasets
    ds = ir_datasets.load("aol-ia")
    doc_click_queries = build_doc_click_queries(ds.qlogs_iter())
    logger.info("  %d docs in %.1fs", len(doc_click_queries), time.time() - t1)
    doc_clicks_file.write_bytes(pickle.dumps(doc_click_queries))
    logger.info("  cached to %s (%.1f MB)", doc_clicks_file, doc_clicks_file.stat().st_size / 1e6)

    logger.info("=== Done. Total time: %.1fs ===", time.time() - t0)
    logger.info("All downstream scripts will now load in seconds.")


if __name__ == "__main__":
    main()
