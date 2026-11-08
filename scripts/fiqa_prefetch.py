"""FiQA-2018 + monoT5 backbone pipeline (BM25-negative sampling).

Unlike TREC-COVID / TREC DL 2019, FiQA-2018's qrels contain only
positive labels (relevance > 0) — every doc not in qrels is implicitly
non-relevant. We therefore sample 9 BM25-top-100 distractors (excluding
the judged positives) per (query, positive) pair as candidate negatives.

Pipeline:
  1. Load FiQA test split queries, judged positives, and corpus.jsonl.
  2. For each (query, positive) pair, build candidates = positive + 9
     BM25-ranked unjudged distractors.
  3. Precompute (s_base, h) with monoT5 for every (q, d) in candidates.
  4. Cache features to disk for reuse across Pref-Rerank + baselines.
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
logger = logging.getLogger("fiqa_prefetch")


def load_fiqa(corpus_dir: str = "/root/gcy/Rerank/data/fiqa/fiqa", split: str = "test"):
    """Load FiQA-2018 queries, judged positives, and corpus.jsonl."""
    base = Path(corpus_dir)

    queries = {}
    with open(base / "queries.jsonl") as f:
        for line in f:
            j = json.loads(line)
            qid = j["_id"]
            text = j.get("title", "")
            if j.get("text"):
                text = text + " " + j["text"] if text else j["text"]
            queries[qid] = text.strip()

    positives = defaultdict(set)  # qid -> {doc_id}
    qrels_path = base / "qrels" / f"{split}.tsv"
    with open(qrels_path) as f:
        next(f)  # skip header "query-id\tcorpus-id\tscore"
        for line in f:
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, doc_id, rel = parts[0], parts[1], int(parts[2])
            if rel > 0:
                positives[qid].add(doc_id)

    return queries, positives


def load_corpus_text(corpus_dir):
    """Stream-load corpus.jsonl into doc_id -> text dict."""
    if corpus_dir is None:
        corpus_dir = "/root/gcy/Rerank/data/fiqa/fiqa"
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


def build_eval_candidates_bm25(queries, positives, doc_text, n_negatives: int = 9, top_k: int = 100, seed: int = 42):
    """Build candidates: per (query, positive), take 9 BM25-top distractors.

    FiQA qrels contain only positives; we treat all other docs as the
    unjudged pool, rank by BM25, and pick negative samples from below
    the positives in the BM25 ranking.
    """
    import random
    from src.model.queryflow_backbone import BM25, _tokenize

    rng = random.Random(seed)

    # BM25 index over FiQA corpus as pseudo-docs (title+text).
    bm25_docs = {did: [text] for did, text in doc_text.items()}
    logger.info("Building BM25 index over %d docs ...", len(bm25_docs))
    bm25 = BM25(bm25_docs)
    logger.info("BM25 index built.")

    cands = []
    n_skip = 0
    for qid, pos_docs in positives.items():
        if qid not in queries or not queries[qid]:
            continue
        if not pos_docs:
            continue
        # BM25-top-k distractors excluding positives
        bm25_scores = [(did, bm25.score(queries[qid], did))
                        for did in bm25_docs.keys() if did not in pos_docs]
        bm25_scores.sort(key=lambda x: x[1], reverse=True)
        bm25_neg_pool = [did for did, _ in bm25_scores[:top_k]]
        if len(bm25_neg_pool) < n_negatives:
            n_skip += 1
            continue
        for d_pos in pos_docs:
            negs = rng.sample(bm25_neg_pool, n_negatives)
            cands.append({
                "query_id": qid,
                "query_text": queries[qid],
                "pos_doc_id": d_pos,
                "neg_doc_ids": negs,
            })
    logger.info("Built %d evaluation candidates (skipped %d queries with <%d BM25 distractors)",
                len(cands), n_skip, n_negatives)
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="/root/gcy/Rerank/data/fiqa/fiqa")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-negatives", type=int, default=9)
    ap.add_argument("--bm25-top-k", type=int, default=100)
    ap.add_argument("--backbone-path", default="/root/gcy/Rerank/cache/monot5_local")
    ap.add_argument("--feature-cache", default="cache/features_monot5_fiqa")
    ap.add_argument("--candidate-cache", default="cache/fiqa/candidates.jsonl")
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
        queries, positives = load_fiqa(args.corpus_dir, args.split)
        logger.info("Queries: %d   positive (query, doc) pairs: %d",
                    len(queries), sum(len(s) for s in positives.values()))
        doc_text = load_corpus_text(args.corpus_dir)
        logger.info("Doc text map: %d entries", len(doc_text))
        cands = build_eval_candidates_bm25(
            queries, positives, doc_text,
            n_negatives=args.n_negatives, top_k=args.bm25_top_k, seed=args.seed,
        )
        cand_cache.write_text("\n".join(json.dumps(c) for c in cands))

    # 2. Precompute monoT5 features
    from src.model.monot5_backbone import MonoT5Backbone, FeatureCache, precompute_features

    backbone = MonoT5Backbone(model_name=args.backbone_path)
    cache = FeatureCache(args.feature_cache)
    logger.info("Cache has %d pre-existing entries", len(cache))

    pairs = []
    seen = set()
    for c in cands:
        for d_id in [c["pos_doc_id"]] + c["neg_doc_ids"]:
            # FiQA cache key uses query text + doc text (BM25-unjudged distractors
            # are unjudged; we featurise them as the doc text from corpus)
            t = doc_text.get(d_id, "") if "doc_text" in dir() else ""
            if not t:
                continue
            if (c["query_text"], t) not in seen:
                seen.add((c["query_text"], t))
                pairs.append((c["query_text"], t))
    # If doc_text was not loaded in this run (cands reused from cache),
    # load it now.
    if "doc_text" not in dir():
        doc_text = load_corpus_text(args.corpus_dir)
        pairs = []
        seen = set()
        for c in cands:
            for d_id in [c["pos_doc_id"]] + c["neg_doc_ids"]:
                t = doc_text.get(d_id, "")
                if not t:
                    continue
                if (c["query_text"], t) not in seen:
                    seen.add((c["query_text"], t))
                    pairs.append((c["query_text"], t))
    logger.info("Unique (q, d) pairs to featurize: %d", len(pairs))
    precompute_features(backbone, iter(pairs), cache, batch_size=args.batch_size)
    logger.info("Done. Total cached: %d", len(cache))


if __name__ == "__main__":
    main()