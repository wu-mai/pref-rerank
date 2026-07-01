"""
R001-R003: Sanity checks.

R001: AOL session parsing on 5 sessions.
R002: LLM schema extraction on 3 sessions.
R003: query-flow backbone (BM25 + BGE) inference latency on 12 (q,d) pairs.

Run before any full training.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aol_session import load_aol_sessions
from src.extract.schema_extractor import SchemaExtractor
from src.model.queryflow_backbone import QueryFlowBackbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sanity")


def r001(n=5):
    logger.info("[R001] Parsing %d AOL sessions...", n)
    sessions = load_aol_sessions(min_session_len=4, max_sessions=n)
    assert len(sessions) >= 1, "No sessions parsed"
    for s in sessions[:3]:
        ev0 = s.events[0]
        n_clicked = sum(1 for it in ev0.items if it.clicked)
        logger.info("  session=%s n_events=%d first_q=%r n_clicked=%d", s.session_id, len(s.events), ev0.query[:40], n_clicked)
    logger.info("[R001] PASS: parsed %d sessions", len(sessions))
    return sessions


def r002(sessions, schema_cache):
    logger.info("[R002] Extracting schemas from %d sessions...", len(sessions))
    extractor = SchemaExtractor(cache_dir=schema_cache)
    ok, tot = 0, 0
    for s in sessions[:3]:
        # History = first 3 events' (query, first_clicked_doc)
        history = []
        for ev in s.events[:3]:
            clicked = [it for it in ev.items if it.clicked]
            history.append((ev.query, clicked[0].doc_id if clicked else ""))
        if len([h for h in history if h[1]]) < 2:
            continue
        tot += 1
        schema = extractor.extract(history)
        if schema is not None:
            ok += 1
            logger.info("  session=%s schema=%s", s.session_id, schema.to_dict())
    rate = ok / max(tot, 1)
    logger.info("[R002] rate=%d/%d=%.1f%%", ok, tot, rate * 100)
    if rate >= 0.99:
        logger.info("[R002] PASS")


def r003():
    """Test query-flow backbone (BM25 + BGE) latency."""
    logger.info("[R003] Loading query-flow backbone (BM25 + BGE-base)...")
    # Tiny synthetic doc_click_queries for latency test
    doc_click_queries = {
        "d1": ["best noise canceling headphones", "wireless headphones review"],
        "d2": ["sony wh-1000xm5 review", "sony headphones specs"],
        "d3": ["how to bake sourdough", "sourdough recipe"],
    }
    t0 = time.time()
    backbone = QueryFlowBackbone(doc_click_queries)
    logger.info("[R003] load=%.1fs", time.time() - t0)

    queries = ["headphones", "sony review", "baking bread"] * 4
    docs = ["d1", "d2", "d3"] * 4
    _ = backbone.encode_batch(queries[:2], docs[:2])  # warmup
    t0 = time.time()
    s_base, h = backbone.encode_batch(queries, docs)
    elapsed = time.time() - t0
    ms = elapsed / len(queries) * 1000
    logger.info("[R003] 12 pairs in %.2fs = %.1f ms/pair  s_base=%s h=%s", elapsed, ms, tuple(s_base.shape), tuple(h.shape))
    if ms <= 50:
        logger.info("[R003] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema-cache", default="cache/schema")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-backbone", action="store_true")
    args = ap.parse_args()

    sessions = r001(5)
    if not args.skip_llm:
        r002(sessions, args.schema_cache)
    if not args.skip_backbone:
        r003()
    logger.info("Sanity complete.")


if __name__ == "__main__":
    main()
