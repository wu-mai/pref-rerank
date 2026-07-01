"""
Query-flow backbone (Plan B): replaces monoT5 when AOL docs aren't available.

Idea: a doc_id's "representation" is the bag of queries that led to it being
clicked across the corpus. This is the standard query-flow / co-click
representation in search-log IR (e.g., Craswell & Szummer 2007, Jiang et al.
2016). No doc text needed.

We compute two frozen features per (query, doc) pair:
    s_base(q, d): BM25 score between query and doc's click-query set
    h(q, d):      mean of query-side sentence embeddings over click queries (384-d)

Both are precomputed once and cached, same interface as MonoT5Backbone.
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Embedding dim from BGE-base-en-v1.5 (already cached locally)
EMBED_DIM = 768


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer (no nltk dependency)."""
    return [t.strip(".,!?;:\"'()[]{}") for t in text.lower().split() if t.strip(".,!?;:\"'()[]{}")]


class BM25:
    """In-memory BM25 over a corpus of pseudo-docs.

    Each pseudo-doc is the concatenation of click-queries for one doc_id.
    """

    def __init__(self, docs: dict[str, list[str]], k1: float = 1.5, b: float = 0.75):
        """docs: {doc_id: list of query strings that clicked this doc}."""
        self.k1 = k1
        self.b = b
        self.doc_ids = list(docs.keys())
        self.doc_tokens: dict[str, list[str]] = {d: _tokenize(" ".join(qs)) for d, qs in docs.items()}
        self.doc_len: dict[str, int] = {d: len(toks) for d, toks in self.doc_tokens.items()}
        self.avgdl = float(np.mean(list(self.doc_len.values()))) if self.doc_len else 1.0

        # DF per term
        self.df: dict[str, int] = defaultdict(int)
        for d, toks in self.doc_tokens.items():
            for t in set(toks):
                self.df[t] += 1
        self.N = len(self.doc_ids)
        self.idf: dict[str, float] = {
            t: float(np.log(1 + (self.N - n + 0.5) / (n + 0.5))) for t, n in self.df.items()
        }

    def score(self, query: str, doc_id: str) -> float:
        """BM25 score of (query, doc_id)."""
        q_tokens = _tokenize(query)
        d_tokens = self.doc_tokens.get(doc_id, [])
        if not d_tokens:
            return 0.0
        dl = len(d_tokens)
        tf = defaultdict(int)
        for t in d_tokens:
            tf[t] += 1
        score = 0.0
        for qt in q_tokens:
            if qt not in tf:
                continue
            idf = self.idf.get(qt, 0.0)
            f = tf[qt]
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-6))
            score += idf * (f * (self.k1 + 1)) / denom
        return float(score)


class QueryFlowBackbone:
    """Frozen BM25 + sentence-encoder backbone over click-query pseudo-docs.

    Computes:
        s_base(q, d): BM25 score
        h(q, d):      mean sentence-embedding of click-queries for doc d (384-d)

    All frozen; no gradients.
    """

    def __init__(
        self,
        doc_click_queries: dict[str, list[str]],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        encoder_name: str = "BAAI/bge-base-en-v1.5",  # cached locally
    ):
        """doc_click_queries: {doc_id: list of queries that clicked this doc}.

        Build this from AOL qlog by iterating once and grouping.
        """
        from sentence_transformers import SentenceTransformer

        self.device = device
        logger.info("Building BM25 over %d pseudo-docs...", len(doc_click_queries))
        self.bm25 = BM25(doc_click_queries)
        self.doc_click_queries = doc_click_queries

        logger.info("Loading sentence encoder: %s", encoder_name)
        self.encoder = SentenceTransformer(encoder_name, device=device)

        # Cache doc embeddings (mean of click-query embeddings)
        self._doc_emb_cache: dict[str, np.ndarray] = {}

    @torch.inference_mode()
    def encode_batch(self, queries: list[str], doc_ids: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (s_base, h) for a batch of (query, doc_id) pairs.

        Returns:
            s_base: (B,) BM25 scores
            h:      (B, 384) mean sentence-embedding of each doc's click-queries
        """
        assert len(queries) == len(doc_ids)
        s_bases: list[float] = []
        hs: list[np.ndarray] = []

        # Batch-encode unique doc embeddings
        docs_to_encode = set(doc_ids)
        for d in docs_to_encode:
            if d not in self._doc_emb_cache:
                cqs = self.doc_click_queries.get(d, [])
                cqs = [q for q in cqs if q][:32]  # cap at 32 queries
                if cqs:
                    emb = self.encoder.encode(cqs, convert_to_numpy=True, show_progress_bar=False)
                    self._doc_emb_cache[d] = emb.mean(axis=0)
                else:
                    self._doc_emb_cache[d] = np.zeros(EMBED_DIM, dtype=np.float32)

        for q, d in zip(queries, doc_ids):
            s_bases.append(self.bm25.score(q, d))
            hs.append(self._doc_emb_cache[d])

        return (
            torch.tensor(s_bases, dtype=torch.float32).cpu(),
            torch.tensor(np.stack(hs), dtype=torch.float32).cpu(),
        )


# ---------------------------------------------------------------------------
# Disk cache (same interface as MonoT5Backbone's FeatureCache)
# ---------------------------------------------------------------------------


def pair_hash(query: str, doc: str) -> str:
    raw = f"{query}\t{doc}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class FeatureCache:
    """Append-only on-disk cache for (s_base, h) features."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, tuple[float, list[float]]] = {}
        self._load_into_mem()

    def _cache_file(self) -> Path:
        return self.cache_dir / "features.pt"

    def _load_into_mem(self) -> None:
        f = self._cache_file()
        if f.exists():
            data = torch.load(f, map_location="cpu")
            self._mem = data

    def get(self, query: str, doc: str) -> Optional[tuple[float, list[float]]]:
        return self._mem.get(pair_hash(query, doc))

    def put(self, query: str, doc: str, s_base: float, h: list[float]) -> None:
        self._mem[pair_hash(query, doc)] = (s_base, h)

    def save(self) -> None:
        torch.save(self._mem, self._cache_file())

    def __len__(self) -> int:
        return len(self._mem)


def precompute_features(
    backbone: QueryFlowBackbone,
    pairs: Iterator[tuple[str, str]],
    cache: FeatureCache,
    batch_size: int = 64,
    save_every: int = 500,
) -> None:
    """Precompute (s_base, h) for all (query, doc_id) pairs, cache to disk."""
    pending: list[tuple[str, str, str]] = []

    def flush(pending: list[tuple[str, str, str]]) -> None:
        if not pending:
            return
        queries = [p[1] for p in pending]
        docs = [p[2] for p in pending]
        s_base_batch, h_batch = backbone.encode_batch(queries, docs)
        for i, (hsh, q, d) in enumerate(pending):
            cache.put(q, d, s_base_batch[i].item(), h_batch[i].tolist())
        pending.clear()

    n_new = 0
    for query, doc in pairs:
        if cache.get(query, doc) is not None:
            continue
        pending.append((pair_hash(query, doc), query, doc))
        if len(pending) >= batch_size:
            flush(pending)
            n_new += batch_size
            if n_new % save_every == 0:
                cache.save()
                logger.info("Precomputed %d new pairs (total: %d)", n_new, len(cache))
    flush(pending)
    cache.save()
    logger.info("Precompute done. Total cached: %d", len(cache))


def load_features(query: str, doc: str, cache: FeatureCache) -> tuple[float, list[float]]:
    feat = cache.get(query, doc)
    if feat is None:
        raise KeyError(f"Features not cached for query='{query[:40]}' doc='{doc[:40]}'")
    return feat


def build_doc_click_queries(qlogs_iter) -> dict[str, list[str]]:
    """One pass over qlogs_iter to build {doc_id: [queries that clicked it]}.

    Call this ONCE, cache to disk. qlogs_iter: iterator of AolQlog.
    """
    out: dict[str, list[str]] = defaultdict(list)
    for q in qlogs_iter:
        for it in q.items:
            if it.clicked:
                out[str(it.doc_id)].append(q.query)
    return dict(out)
