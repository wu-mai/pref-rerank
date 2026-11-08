"""TREC-COVID (BEIR) + monoT5 backbone pipeline (offline version).

Mirrors scripts/msmarco_prefetch.py for the BEIR trec-covid collection:
  1. Load TREC-COVID queries (50) and judged qrels (66k) from local JSONL/TSV.
  2. Load corpus from local corpus.jsonl.
  3. Build a candidate set per (query, positive) pair using only the
     judged-irrelevant docs in the qrels (BEIR's qrels cover the full
     corpus, so 1000s of negatives per query — we sample 9 per pair).
  4. Precompute (s_base, h) with monoT5 for every (q, d).
  5. Cache features to disk for reuse across Pref-Rerank + baselines.

Same output format as msmarco_prefetch: candidates.jsonl + features.pt,
compatible with msmarco_common.py (load_candidates + FeatureCache).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trc_prefetch")


def load_trec_covid(corpus_dir: str = "/root/gcy/Rerank/data/trec-covid"):
    """Load TREC-COVID corpus, queries, qrels from local files."""
    base = Path(corpus_dir)

    queries = {}
    with open(base / "queries.jsonl") as f:
        for line in f:
            j = json.loads(line)
            # BEIR JSONL: {"_id": str, "title": str, "text": str, "metadata": {}}
            qid = j["_id"]
            text = j.get("title", "")
            if j.get("text"):
                text = text + " " + j["text"] if text else j["text"]
            queries[qid] = text.strip()

    qrels = []
    with open(base / "qrels" / "test.tsv") as f:
        header = f.readline()  # skip "query-id\tcorpus-id\tscore"
        for line in f:
            parts = line.strip().split("\t")
            # TREC-COVID format: query-id \t corpus-id \t score (3 columns)
            # but tolerate the 4-column TREC-standard format too.
            if len(parts) == 3:
                qid, doc_id, rel = parts[0], parts[1], int(parts[2])
            elif len(parts) >= 4:
                qid, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
            else:
                continue
            qrels.append((qid, doc_id, rel))

    return queries, qrels


def build_eval_candidates(queries, qrels, n_negatives: int = 9, seed: int = 42):
    """For each (query, positive) pair, pick n_negatives judged-irrelevant docs.

    TREC-COVID qrels cover the full corpus, so there are many negatives
    per query. We sample uniformly at random.
    """
    import random
    rng = random.Random(seed)

    pos = defaultdict(list)  # qid -> [doc_id]
    neg = defaultdict(list)  # qid -> [doc_id]
    for qid, doc_id, rel in qrels:
        if rel > 0:
            pos[qid].append(doc_id)
        else:
            neg[qid].append(doc_id)

    cands = []
    for qid, pos_docs in pos.items():
        if qid not in queries or not queries[qid]:
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


def load_corpus_text(corpus_dir: str = None):
    """Stream-load corpus.jsonl into doc_id -> text dict.

    corpus_dir may also be the path to a single corpus.jsonl file. The cache
    pickle is keyed by corpus_dir so different datasets don't collide.
    """
    if corpus_dir is None:
        corpus_dir = "/root/gcy/Rerank/data/trec-covid"
    corpus_dir = Path(corpus_dir)
    if corpus_dir.is_file() and corpus_dir.name == "corpus.jsonl":
        corpus_jsonl = corpus_dir
    else:
        corpus_jsonl = corpus_dir / "corpus.jsonl"
    if not corpus_jsonl.exists():
        raise FileNotFoundError(f"corpus.jsonl not found under {corpus_dir}")
    safe_name = "".join(c if c.isalnum() else "_" for c in str(corpus_dir))
    cache_pkl = Path(f"/root/autodl-tmp/Rerank/data/_doc_text_{safe_name}.pkl")
    if cache_pkl.exists():
        import pickle
        logger.info("Loading cached doc_text from %s", cache_pkl)
        return pickle.loads(cache_pkl.read_bytes())
    logger.info("Streaming %s ...", corpus_jsonl)
    doc_text = {}
    with open(corpus_jsonl) as f:
        for i, line in enumerate(f):
            j = json.loads(line)
            did = j["_id"]
            title = j.get("title", "")
            text = j.get("text", "")
            doc_text[did] = (title + " " + text).strip() if title else text
            if (i + 1) % 10000 == 0:
                logger.info("  loaded %d docs", i + 1)
    logger.info("Loaded %d docs", len(doc_text))
    cache_pkl.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    cache_pkl.write_bytes(pickle.dumps(doc_text))
    logger.info("Saved to %s (%.1f MB)", cache_pkl, cache_pkl.stat().st_size / 1e6)
    return doc_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="/root/gcy/Rerank/data/trec-covid")
    ap.add_argument("--n-negatives", type=int, default=9)
    ap.add_argument("--backbone-path", default="/root/gcy/Rerank/cache/monot5_local")
    ap.add_argument("--feature-cache", default="cache/features_monot5_trc")
    ap.add_argument("--candidate-cache", default="cache/treccovid/candidates.jsonl")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feature_cache_dir = Path(args.feature_cache)
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    cand_cache = Path(args.candidate_cache)
    cand_cache.parent.mkdir(parents=True, exist_ok=True)

    # 1. Build candidate set
    if cand_cache.exists():
        logger.info("Loading cached candidates from %s", cand_cache)
        cands = [json.loads(l) for l in cand_cache.read_text().splitlines()]
    else:
        queries, qrels = load_trec_covid(args.corpus_dir)
        logger.info("Queries: %d   qrels: %d", len(queries), len(qrels))
        cands = build_eval_candidates(queries, qrels, n_negatives=args.n_negatives, seed=args.seed)
        logger.info("Built %d evaluation candidates", len(cands))
        cand_cache.write_text("\n".join(json.dumps(c) for c in cands))

    # 2. Load corpus texts
    doc_text = load_corpus_text(args.corpus_dir)
    logger.info("Doc text map: %d entries", len(doc_text))

    # 3. Precompute monoT5 features
    from src.model.monot5_backbone import MonoT5Backbone, FeatureCache, precompute_features

    backbone = MonoT5Backbone(model_name=args.backbone_path)
    cache = FeatureCache(args.feature_cache)
    logger.info("Cache has %d pre-existing entries", len(cache))

    pairs = []
    seen = set()
    for c in cands:
        for d_id in [c["pos_doc_id"]] + c["neg_doc_ids"]:
            if d_id in doc_text and (c["query_text"], d_id) not in seen:
                seen.add((c["query_text"], d_id))
                pairs.append((c["query_text"], doc_text[d_id]))
    logger.info("Unique (q, d) pairs to featurize: %d", len(pairs))

    precompute_features(backbone, iter(pairs), cache, batch_size=args.batch_size)
    logger.info("Done. Total cached: %d", len(cache))


if __name__ == "__main__":
    main()