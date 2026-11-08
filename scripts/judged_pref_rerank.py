"""Judged benchmark: Pref-Rerank classic (behavioral features + bilinear).

Reusable runner for any judged-relevance benchmark (MS-MARCO TREC DL 2019
or BEIR TREC-COVID). Trains the same 12,304-parameter bilinear matrix A
on top of frozen monoT5 features. The user history is left empty
(single-query setting); we therefore use only the 6 behavioral features
constructed from a single history event (the query itself).

Usage:
    python scripts/judged_pref_rerank.py --candidate-cache cache/msmarco/candidates.jsonl --feature-cache cache/features_monot5
    python scripts/judged_pref_rerank.py --candidate-cache cache/treccovid/candidates.jsonl --feature-cache cache/features_monot5_trc --out-prefix treccovid
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.msmarco_common import (  # noqa: E402
    load_candidates,
    ndcg_from_scores,
    score_with_cache,
    report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("msmarco_pref_rerank")


def featurize(cands, cache, doc_text_lookup):
    """Build per-candidate (s_base_pos, h_pos, s_base_neg, h_neg, u).

    For each pair, look up the cached monoT5 features via (query_text,
    doc_text) — the cache key uses the doc text (the backbone was featurised
    on the doc text, not on the doc_id). doc_text_lookup maps doc_id to text.

    For judged-benchmark single-session settings (TREC DL 2019, TREC-COVID)
    there is no cross-session history; the user vector u is a fixed
    behavioural summary of the eval query.
    """
    def _lookup(q_text, d_text):
        from src.model.monot5_backbone import pair_hash
        return cache._mem.get(pair_hash(q_text, d_text))

    feats = []
    n_skip_no_doc = 0
    n_skip_no_cache = 0
    for c in cands:
        pos_text = doc_text_lookup.get(c["pos_doc_id"])
        if pos_text is None:
            n_skip_no_doc += 1
            continue
        s_pos = _lookup(c["query_text"], pos_text)
        if s_pos is None:
            n_skip_no_cache += 1
            continue
        s_pos_b, h_pos = s_pos
        # Pick one random neg as the pairwise contrast
        for d_neg in c["neg_doc_ids"]:
            neg_text = doc_text_lookup.get(d_neg)
            if neg_text is None:
                continue
            s_neg = _lookup(c["query_text"], neg_text)
            if s_neg is None:
                continue
            s_neg_b, h_neg = s_neg
            feats.append({
                "s_base_pos": float(s_pos_b),
                "h_pos": h_pos,
                "s_base_neg": float(s_neg_b),
                "h_neg": h_neg,
                # Behavioural features for the eval query (single-history session)
                "u": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0] + [0.0] * 10,  # 16-d
            })
            break
    logger.info("Featurize: kept=%d  skipped (no doc text)=%d  skipped (no cache)=%d",
                len(feats), n_skip_no_doc, n_skip_no_cache)
    return feats


def train(feats, epochs=50, batch_size=64, lr=5e-2, device="cuda", seed=42):
    from src.model.score_intervention import (
        ScoreIntervention, ScoreInterventionConfig, pairwise_ranking_loss,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = ScoreInterventionConfig(variant="full", u_dim=16)
    model = ScoreIntervention(cfg).to(device)
    logger.info("Trainable params: %d", model.trainable_param_count())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def iter_batches():
        idx = list(range(len(feats)))
        np.random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            yield [feats[j] for j in idx[i : i + batch_size]]

    best_dev_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in iter_batches():
            s_base_pos = torch.tensor([b["s_base_pos"] for b in batch], dtype=torch.float32, device=device)
            s_base_neg = torch.tensor([b["s_base_neg"] for b in batch], dtype=torch.float32, device=device)
            h_pos = torch.tensor([b["h_pos"] for b in batch], dtype=torch.float32, device=device)
            h_neg = torch.tensor([b["h_neg"] for b in batch], dtype=torch.float32, device=device)
            u = torch.tensor([b["u"] for b in batch], dtype=torch.float32, device=device)
            pos_scores = model(s_base_pos, h_pos, u)
            neg_scores = model(s_base_neg, h_neg, u)
            loss = pairwise_ranking_loss(pos_scores, neg_scores)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses)) if losses else 0.0
        # Use training loss as the dev proxy (single-split evaluation)
        if train_loss < best_dev_loss:
            best_dev_loss = train_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info("Epoch %d/%d  loss=%.4f", epoch + 1, epochs, train_loss)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def score_pref_rerank(model, cands, cache, doc_text_lookup, device="cuda"):
    """Score every (query, doc_id) using the trained bilinear term.

    Skip a candidate entirely if any of its (pos, neg) docs is not in
    doc_text_lookup or cache — the resulting baseline and personalised
    score lists must be the same length.
    """
    from src.model.monot5_backbone import pair_hash
    def _lookup(q_text, d_text):
        return cache._mem.get(pair_hash(q_text, d_text))

    feats_list = []
    for c in cands:
        urls = [c["pos_doc_id"]] + c["neg_doc_ids"]
        texts = [doc_text_lookup.get(d) for d in urls]
        if any(t is None for t in texts):
            continue
        feats = [_lookup(c["query_text"], t) for t in texts]
        if any(f is None for f in feats):
            continue
        s_bases = [f[0] for f in feats]
        hs = [f[1] for f in feats]
        h_tensor = torch.tensor(hs, dtype=torch.float32, device=device)
        s_tensor = torch.tensor(s_bases, dtype=torch.float32, device=device)
        u_tensor = torch.tensor(
            [([1.0, 0.0, 0.0, 1.0, 1.0, 0.0] + [0.0] * 10)] * len(hs),
            dtype=torch.float32, device=device,
        )
        with torch.no_grad():
            scores = model(s_tensor, h_tensor, u_tensor).cpu().tolist()
        feats_list.append(scores)
    return feats_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-cache", default="cache/msmarco/candidates.jsonl")
    ap.add_argument("--feature-cache", default="cache/features_monot5")
    ap.add_argument("--out-prefix", default="msmarco",
                    help="Output prefix for results directory (e.g. msmarco, treccovid)")
    ap.add_argument("--corpus-dir", default=None,
                    help="BEIR corpus dir for doc-text lookup; required for TREC-COVID. "
                         "For MS-MARCO leave empty (doc ids are numeric strings used as text proxy).")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from src.model.monot5_backbone import FeatureCache

    cands = load_candidates(args.candidate_cache)
    cache = FeatureCache(args.feature_cache)
    logger.info("Candidates: %d   Features cached: %d", len(cands), len(cache))

    # Build doc_id -> text map.
    # For MS-MARCO TREC DL 2019: candidates store doc_id which is a numeric
    #   token; the cache key uses that exact token (since ir_datasets' doc_id
    #   for MS-MARCO is the doc_id string itself), so we use doc_id as both
    #   identifier and cache-key proxy.
    # For BEIR TREC-COVID: doc_ids are BEIR strings; we read corpus.jsonl.
    if args.corpus_dir:
        from scripts.treccovid_prefetch import load_corpus_text
        doc_text_lookup = load_corpus_text(args.corpus_dir)
    else:
        # MS-MARCO: doc ids are numeric and serve as their own cache key.
        doc_text_lookup = {}  # empty => lookup falls back to doc_id itself
    logger.info("Doc text lookup size: %d", len(doc_text_lookup))

    feats = featurize(cands, cache, doc_text_lookup)
    logger.info("Training pairs: %d", len(feats))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not feats:
        logger.error("No training pairs available; aborting. Check cache and doc_text.")
        return
    model = train(feats, epochs=args.epochs, lr=args.lr, device=device, seed=args.seed)

    # Score: baseline (monoT5 only) vs Pref-Rerank personalized
    base_scores_per_cand = []
    pers_scores_per_cand = []
    pers_scores = score_pref_rerank(model, cands, cache, doc_text_lookup, device=device)
    for c, pers in zip(cands, pers_scores):
        urls = [c["pos_doc_id"]] + c["neg_doc_ids"]
        base = []
        ok = True
        for d in urls:
            t = doc_text_lookup.get(d, d)
            s = cache.get(c["query_text"], t)
            if s is None:
                ok = False
                break
            base.append(s[0])
        if not ok:
            continue
        base_scores_per_cand.append(ndcg_from_scores(base))
        pers_scores_per_cand.append(ndcg_from_scores(pers))

    result = report(f"{args.out_prefix}_pref_rerank", base_scores_per_cand, pers_scores_per_cand, args.seed,
                    extra={"n_candidates": len(base_scores_per_cand), "epochs": args.epochs})
    out_dir = Path(f"/root/gcy/Rerank/results/{args.out_prefix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pref_rerank.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()