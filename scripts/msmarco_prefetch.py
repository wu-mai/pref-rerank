"""MS-MARCO TREC DL 2019 + monoT5 backbone pipeline (offline version).

This version avoids ir_datasets' automatic collection download by:
  1. Loading TREC DL 2019 judged qrels and queries via ir_datasets (small files,
     already cached).
  2. Loading MS-MARCO passage texts directly from the local
     /root/gcy/Rerank/data/msmarco/collectionandqueries.tar.gz.
  3. Building a candidate set per (query, positive) pair using only the
     judged-irrelevant docs in the qrels (TREC DL 2019 judged split has
     ~1000 docs per query with binary relevance labels, so BM25 hard
     negatives are unnecessary).
  4. Precomputing (s_base, h) with monoT5 for every (q, d).
  5. Caching features to disk for reuse across Pref-Rerank + baselines.

The output candidates and features are compatible with msmarco_common.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("msmarco_prefetch")


def load_trec_dl_qrels(ir_dataset: str = "msmarco-passage/trec-dl-2019/judged"):
    """Load qrels and queries via ir_datasets (small, cached)."""
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)
    import ir_datasets
    ds = ir_datasets.load(ir_dataset)
    qrels = list(ds.qrels_iter())
    queries = list(ds.queries_iter())
    return qrels, {q.query_id: q.text for q in queries}


def load_collection_from_tar(tar_path: str) -> dict[str, str]:
    """Stream-parse MS-MARCO collection.tsv from a tar.gz and return doc_id -> text."""
    logger.info("Streaming collection from %s", tar_path)
    doc_text: dict[str, str] = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            # MS-MARCO collection.tsv: doc_id \t text
            for line in f:
                parts = line.decode("utf-8", errors="replace").rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    doc_id, text = parts
                    doc_text[doc_id] = text
                    if len(doc_text) % 1_000_000 == 0:
                        logger.info("  loaded %d docs", len(doc_text))
    logger.info("Loaded %d docs", len(doc_text))
    return doc_text


def build_eval_candidates_from_qrels(qrels, queries, n_negatives: int = 9, seed: int = 42):
    """For each (query, positive) pair, pick n_negatives judged-irrelevant docs."""
    import random
    rng = random.Random(seed)

    pos: dict[str, list[str]] = defaultdict(list)
    neg: dict[str, list[str]] = defaultdict(list)
    for qrel in qrels:
        if qrel.relevance > 0:
            pos[qrel.query_id].append(qrel.doc_id)
        else:
            neg[qrel.query_id].append(qrel.doc_id)

    cands = []
    for qid, pos_docs in pos.items():
        if qid not in queries:
            continue
        for d_pos in pos_docs:
            neg_pool = neg.get(qid, [])
            if len(neg_pool) < n_negatives:
                continue
            negs = rng.sample(neg_pool, n_negatives)
            cands.append({
                "query_id": qid,
                "query_text": queries[qid],
                "pos_doc_id": d_pos,
                "neg_doc_ids": negs,
            })
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir-dataset", default="msmarco-passage/trec-dl-2019/judged")
    ap.add_argument("--collection-tar", default="/root/gcy/Rerank/data/msmarco/collectionandqueries.tar.gz")
    ap.add_argument("--n-negatives", type=int, default=9)
    ap.add_argument("--backbone-path", default="/root/gcy/Rerank/cache/monot5_local")
    ap.add_argument("--feature-cache", default="cache/features_monot5")
    ap.add_argument("--candidate-cache", default="cache/msmarco/candidates.jsonl")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feature_cache_dir = Path(args.feature_cache)
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    cand_cache = Path(args.candidate_cache)
    cand_cache.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Build candidate set ----
    if cand_cache.exists():
        logger.info("Loading cached candidates from %s", cand_cache)
        cands = [json.loads(l) for l in cand_cache.read_text().splitlines()]
    else:
        logger.info("Loading qrels and queries from %s ...", args.ir_dataset)
        qrels, queries = load_trec_dl_qrels(args.ir_dataset)
        logger.info("Qrels: %d   Queries: %d", len(qrels), len(queries))
        cands = build_eval_candidates_from_qrels(
            qrels, queries, n_negatives=args.n_negatives, seed=args.seed,
        )
        logger.info("Built %d evaluation candidates", len(cands))
        cand_cache.write_text("\n".join(json.dumps(c) for c in cands))

    # ---- 2. Load collection texts ----
    doc_text_cache = Path("cache/msmarco/doc_text.pkl")
    if doc_text_cache.exists():
        import pickle
        logger.info("Loading cached doc_text from %s", doc_text_cache)
        doc_text = pickle.loads(doc_text_cache.read_bytes())
    else:
        doc_text = load_collection_from_tar(args.collection_tar)
        import pickle
        doc_text_cache.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving doc_text to %s (%.1f MB)", doc_text_cache,
                    doc_text_cache.stat().st_size / 1e6 if doc_text_cache.exists() else 0)
        doc_text_cache.write_bytes(pickle.dumps(doc_text))
    logger.info("Doc text map: %d entries", len(doc_text))

    # ---- 3. Precompute monoT5 features ----
    from src.model.monot5_backbone import MonoT5Backbone, FeatureCache, precompute_features

    backbone = MonoT5Backbone(model_name=args.backbone_path)
    cache = FeatureCache(args.feature_cache)
    logger.info("Cache has %d pre-existing entries", len(cache))

    pairs = []
    pairs_with_ids = []  # (query_text, doc_text, doc_id)
    seen = set()
    for c in cands:
        for d_id in [c["pos_doc_id"]] + c["neg_doc_ids"]:
            if d_id in doc_text and (c["query_text"], d_id) not in seen:
                seen.add((c["query_text"], d_id))
                pairs.append((c["query_text"], doc_text[d_id]))
                pairs_with_ids.append((c["query_text"], doc_text[d_id], d_id))
    logger.info("Unique (q, d) pairs to featurize: %d", len(pairs))

    # Custom precompute that writes the cache keyed by (query, doc_id),
    # so the downstream msmarco_common.score_with_cache can lookup by doc_id.
    import hashlib
    pending: list[tuple[str, str, str, str]] = []  # (key_hash, query, doc_text, doc_id)

    def flush():
        if not pending:
            return
        qs = [p[1] for p in pending]
        dtexts = [p[2] for p in pending]
        s_base_batch, h_batch = backbone.encode_batch(qs, dtexts)
        for i, (key_hsh, _, _, _) in enumerate(pending):
            cache._mem[key_hsh] = (s_base_batch[i].item(), h_batch[i].tolist())
        pending.clear()

    save_every = 500
    n_new = 0
    for query, doc_text_, doc_id in pairs_with_ids:
        key_hsh = hashlib.sha256(f"{query}\t{doc_id}".encode()).hexdigest()[:16]
        if cache.get(query, doc_id) is not None:
            # cache.get uses pair_hash(query, doc) which won't match our key_hsh
            # so manually check via _mem
            if key_hsh in cache._mem:
                continue
        pending.append((key_hsh, query, doc_text_, doc_id))
        if len(pending) >= 16:
            flush()
            n_new += 16
            if n_new % save_every == 0:
                cache.save()
                logger.info("Precomputed %d new pairs (total: %d)", n_new, len(cache))
    flush()
    cache.save()
    logger.info("Precompute done. Total cached: %d", len(cache))


if __name__ == "__main__":
    main()