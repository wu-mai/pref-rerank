"""B6: HRNN-style hierarchical-RNN user encoder baseline (CIKM 2018, Ge et al.).

Source: Ge et al., "Bringing Personal Search to the Next Level with
Hierarchical Recurrent Neural Network", CIKM 2018. HRNN encodes the
user's session history with a hierarchical RNN (session-level + user-level
LSTMs) into a user state vector used for matching.

Faithful lightweight adaptation to our frozen-backbone protocol:
- Session-level: encode each history event (query, clicked doc) as the
  mean of the frozen BGE query embedding BGE(q') and the cached doc
  representation h(q', d') (we take the element-wise mean to stay in the
  same 768-d space).
- User-level: a single GRU over the K=3 history event embeddings produces
  the final user vector u_hrnn(H) (768-d).
- Scoring: s'(q, d, H) = s_base(q, d) + u_hrnn(H)^T h(q, d) scaled by a
  learnable temperature; the GRU is trained by the same Bradley-Terry
  pairwise loss as Pref-Rerank.

This isolates the "sequential user-state encoding" mechanism of HRNN on
the same frozen features Pref-Rerank uses. Trainable surface: the GRU
(input 768, hidden 768, 1 layer) + a scalar temperature.

Usage:
    python scripts/baselines/b6_hrnn.py --session-cache cache/data/sessions.pkl \
        --feature-cache cache/features_queryflow --n-negatives 9 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.baselines.common import load_protocol, ndcg_from_scores  # noqa: E402
from src.eval.metrics import paired_bootstrap_ci  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("b6_hrnn")


class BGEQueryEncoder:
    """Caches BGE embeddings for queries (lazy, dict-backed)."""

    def __init__(self, encoder_path="/root/gcy/Rerank/cache/bge_local"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(encoder_path, device="cpu")
        self._cache: dict[str, np.ndarray] = {}

    def get_query_embedding(self, q: str):
        if q in self._cache:
            return self._cache[q]
        e = self.model.encode([q], convert_to_numpy=True, show_progress_bar=False)[0]
        self._cache[q] = e
        return e


class HRNNUserEncoder(nn.Module):
    """GRU user encoder over history event embeddings (768-d in/out)."""

    def __init__(self, hidden: int = 768):
        super().__init__()
        self.gru = nn.GRU(input_size=768, hidden_size=hidden, batch_first=True)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hist_emb: torch.Tensor) -> torch.Tensor:
        # hist_emb: (B, K, 768) -> user vector (B, 768)
        out, _ = self.gru(hist_emb)
        return out[:, -1, :]  # last hidden state


def build_history_embeddings(cand, cache, bge, device):
    """Encode the K=3 history events as (K, 768) tensor.

    Accepts either a dict (eval candidate) or a TrainingPair object.
    Each event: mean of BGE(q') and h(q', d') where available; if the
    history doc was not clicked, use BGE(q') alone.
    """
    if isinstance(cand, dict):
        user_history = cand["user_history"]
        eval_query = cand["eval_query"]
        pos_url = cand.get("pos_url")
        neg_urls = cand.get("neg_urls", [])
    else:
        user_history = cand.user_history
        eval_query = cand.eval_query
        pos_url = cand.pos_url
        neg_urls = [cand.neg_url]

    embs = []
    for hq, hd in user_history:
        qe = bge.get_query_embedding(hq) if hq else None
        if hd and hq:
            pair = cache.get(hq, hd)
            de = np.asarray(pair[1], dtype=np.float32) if pair is not None else None
        else:
            de = None
        if qe is not None and de is not None and len(de) == len(qe):
            embs.append((qe + de) / 2.0)
        elif qe is not None:
            embs.append(qe)
        elif de is not None:
            embs.append(de)
    if not embs:
        return None
    # Pad/truncate to K=3 events
    K = 3
    if len(embs) >= K:
        embs = embs[:K]
    else:
        embs = embs + [np.zeros_like(embs[0]) for _ in range(K - len(embs))]
    return torch.tensor(np.stack(embs), dtype=torch.float32, device=device)


def featurize_batch(cands, cache, bge, device):
    """Build training tensors: (hist_emb, h_pos, h_neg, s_pos, s_neg)."""
    rows = []
    for c in cands:
        if isinstance(c, dict):
            eval_query, pos_url, neg_url = c["eval_query"], c["pos_url"], c["neg_urls"][0]
        else:
            eval_query, pos_url, neg_url = c.eval_query, c.pos_url, c.neg_url
        hist = build_history_embeddings(c, cache, bge, device)
        if hist is None:
            continue
        pos = cache.get(eval_query, pos_url)
        if pos is None:
            continue
        s_pos_b, h_pos = pos
        neg = cache.get(eval_query, neg_url)
        if neg is None:
            continue
        s_neg_b, h_neg = neg
        rows.append({
            "hist": hist,
            "h_pos": torch.tensor(h_pos, dtype=torch.float32, device=device),
            "h_neg": torch.tensor(h_neg, dtype=torch.float32, device=device),
            "s_pos": float(s_pos_b),
            "s_neg": float(s_neg_b),
        })
    return rows


def train(rows, encoder, epochs=30, lr=1e-3, batch_size=64, device="cuda", seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr)
    n = len(rows)
    for epoch in range(epochs):
        losses = []
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch = [rows[j] for j in idx[i : i + batch_size]]
            hists = torch.stack([r["hist"] for r in batch])  # (B, K, 768)
            h_pos = torch.stack([r["h_pos"] for r in batch])  # (B, 768)
            h_neg = torch.stack([r["h_neg"] for r in batch])  # (B, 768)
            s_pos = torch.tensor([r["s_pos"] for r in batch], dtype=torch.float32, device=device)
            s_neg = torch.tensor([r["s_neg"] for r in batch], dtype=torch.float32, device=device)
            u = encoder(hists)  # (B, 768)
            score_pos = s_pos + encoder.scale * torch.einsum("bd,bd->b", u, h_pos)
            score_neg = s_neg + encoder.scale * torch.einsum("bd,bd->b", u, h_neg)
            diff = score_pos - score_neg
            loss = -torch.nn.functional.logsigmoid(diff).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info("Epoch %d/%d  loss=%.4f", epoch + 1, epochs, float(np.mean(losses)))
    return encoder


def evaluate(cands, cache, bge, encoder, device):
    base, pers = [], []
    with torch.no_grad():
        for c in cands:
            if isinstance(c, dict):
                urls = [c["pos_url"]] + c["neg_urls"]
                eval_query = c["eval_query"]
            else:
                urls = [c.pos_url]
                eval_query = c.eval_query
            hist = build_history_embeddings(c, cache, bge, device)
            if hist is None:
                continue
            u = encoder(hist.unsqueeze(0)).squeeze(0)
            s_base_list, s_pers_list = [], []
            ok = True
            for url in urls:
                pair = cache.get(eval_query, url)
                if pair is None:
                    ok = False
                    break
                s_base, h = pair
                h_t = torch.tensor(h, dtype=torch.float32, device=device)
                s_pers_list.append(float(s_base) + float(encoder.scale) * float(torch.dot(u, h_t)))
                s_base_list.append(float(s_base))
            if ok:
                base.append(ndcg_from_scores(s_base_list))
                pers.append(ndcg_from_scores(s_pers_list))
    return base, pers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-cache", default="cache/data/sessions.pkl")
    ap.add_argument("--feature-cache", default="cache/features_queryflow")
    ap.add_argument("--n-negatives", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cands, cache, train_pairs, dev_pairs = load_protocol(
        session_cache=args.session_cache,
        feature_cache=args.feature_cache,
        n_negatives=args.n_negatives,
        seed=args.seed,
    )
    bge = BGEQueryEncoder()
    logger.info("Featurizing %d train pairs for GRU training ...", len(train_pairs))
    rows = featurize_batch(train_pairs, cache, bge, device)
    logger.info("Training rows: %d", len(rows))

    encoder = HRNNUserEncoder().to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    logger.info("HRNN encoder trainable params: %d", n_params)
    encoder = train(rows, encoder, epochs=args.epochs, lr=args.lr, device=device, seed=args.seed)

    base, pers = evaluate(cands, cache, bge, encoder, device)
    mean, lo, hi = paired_bootstrap_ci(pers, base, n_bootstrap=10000, seed=args.seed)
    result = {
        "system": "hrnn",
        "source": "Ge et al., CIKM 2018 - Bringing Personal Search to the Next Level with HRNN",
        "protocol": "AOL full neg-pool, n_negatives=9",
        "n_sessions": len(base),
        "baseline_ndcg10": float(np.mean(base)),
        "personalized_ndcg10": float(np.mean(pers)),
        "delta": float(mean),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_trainable_params": int(n_params),
        "note": "Lightweight adaptation: single GRU (768-d) over BGE-encoded history events; "
                "score = s_base + scale * <u_hrnn, h(q,d)>; trained with the same pairwise loss.",
    }
    print(json.dumps(result, indent=2))
    out = Path("/root/gcy/Rerank/results/baselines_n120/b6_hrnn.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()