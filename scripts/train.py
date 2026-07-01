"""
Unified training entry for Pref-Rerank and all ablation variants.

Handles blocks B2 (main), B3 (prompt/classic/random-schema), B4 (no-user-bias/linear-only).

Usage:
    python scripts/train.py --variant full --seed 42 --output results/r005_full_seed42.json
    python scripts/train.py --variant no_user_bias --seed 42 --output results/r010.json
    ...

For Block 3 "prompt" and "classic" variants, this script loads features the
same way and uses the appropriate u source:
    - full / no_user_bias / linear_only: u = LLM-extracted schema vector
    - random_schema:                      u = random valid schema vector
    - classic:                            u = behavioral features (zero-padded to 32)
    - prompt:                             (handled by train_prompt.py — different architecture)
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aol_session import (  # noqa: E402
    TrainingPair,
    load_aol_sessions_cached_or_fresh,
    build_training_pairs,
    split_sessions,
    save_pairs,
)
from src.extract.schema_extractor import (  # noqa: E402
    PreferenceSchema,
    SchemaExtractor,
    random_schema,
)
from src.model.score_intervention import (  # noqa: E402
    ScoreIntervention,
    ScoreInterventionConfig,
    ScoreVariant,
    pairwise_ranking_loss,
    ClassicFeatures,
    classic_to_vector,
)
from src.model.queryflow_backbone import FeatureCache, load_features  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_user_vector(
    pair: TrainingPair,
    extractor: SchemaExtractor,
    variant: str,
    seed: int,
) -> torch.Tensor | None:
    """Get the 16-d user_vector u for this pair, based on variant."""
    if variant in ("full", "no_user_bias", "linear_only"):
        schema = extractor.extract(pair.user_history)
        if schema is None:
            return None
        return torch.tensor(schema.to_vector(), dtype=torch.float32)
    elif variant == "random_schema":
        # Deterministic random based on session_id hash so it's reproducible
        sid_seed = abs(hash(pair.session_id)) + seed
        return torch.tensor(random_schema(sid_seed).to_vector(), dtype=torch.float32)
    elif variant == "classic":
        # Behavioral features — approximated from history
        feat = _classic_features_from_history(pair)
        return classic_to_vector(feat)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def _classic_features_from_history(pair: TrainingPair) -> ClassicFeatures:
    """Compute behavioral features from a training pair's history.

    AOL has no dwell time, so we proxy with click count.
    """
    history = pair.user_history
    eval_query = pair.eval_query

    # BM25-style lexical overlap (Jaccard on tokens) between eval query and history queries
    eval_tokens = set(eval_query.lower().split())
    overlaps = []
    for h_q, _ in history:
        h_tokens = set(h_q.lower().split())
        if eval_tokens or h_tokens:
            overlaps.append(len(eval_tokens & h_tokens) / len(eval_tokens | h_tokens))
    bm25_proxy = float(np.mean(overlaps)) if overlaps else 0.0

    # Reformulation: 1.0 if any history query shares tokens with eval
    has_reform = 1.0 if any(overlaps) else 0.0

    return ClassicFeatures(
        bm25_click_overlap=bm25_proxy,
        avg_dwell_proxy=float(len(history)),  # proxy: more clicks = more engagement
        avg_rank=3.0,  # AOL doesn't expose rank for unclicked history reliably; use neutral
        history_len=float(len(history)),
        query_len=float(len(eval_tokens)),
        has_reformulation=has_reform,
    )


def featurize_pair(
    pair: TrainingPair,
    cache: FeatureCache,
    extractor: SchemaExtractor,
    variant: str,
    seed: int,
) -> dict | None:
    """Load cached monoT5 features + build user_vector for one pair.

    Returns dict with s_base_pos/neg, h_pos/neg, u — or None if anything missing.
    """
    u = get_user_vector(pair, extractor, variant, seed)
    if u is None:
        return None

    try:
        s_base_pos, h_pos = load_features(pair.eval_query, pair.pos_url, cache)
        s_base_neg, h_neg = load_features(pair.eval_query, pair.neg_url, cache)
    except KeyError as e:
        logger.warning("Missing cached features: %s", e)
        return None

    return {
        "s_base_pos": float(s_base_pos),
        "h_pos": h_pos,
        "s_base_neg": float(s_base_neg),
        "h_neg": h_neg,
        "u": u.tolist(),
        "session_id": pair.session_id,
        "pos_rank": pair.pos_rank,
    }


def featurize_pairs_batch(
    pairs: list,
    cache: FeatureCache,
    extractor: SchemaExtractor,
    variant: str,
    seed: int,
    desc: str = "",
    u_dim: int = 16,
) -> list[dict]:
    """Featurize many pairs at once using batched parallel schema extraction.

    Two-phase: (1) batch parallel LLM schema extraction; (2) load cached
    backbone features and assemble per-pair tensors.
    """
    if not pairs:
        return []

    logger.info("Featurizing %d %s pairs (variant=%s, u_dim=%d)...", len(pairs), desc, variant, u_dim)

    # Phase 1: extract schemas in parallel (with cache) — ONLY for variants
    # that actually consume the LLM schema. classic/concat_linear/film/random
    # build u from behavioral features or a RNG, so skip the API calls entirely.
    needs_schema = variant in ("full", "no_user_bias", "linear_only")
    if needs_schema:
        histories = [p.user_history for p in pairs]
        schemas = extractor.extract_batch(histories, max_workers=10, desc=desc)
    else:
        schemas = [None] * len(pairs)

    # Phase 2: assemble features, dropping pairs that lack schema or features
    out: list[dict] = []
    for i, (pair, schema) in enumerate(zip(pairs, schemas)):
        if schema is None and needs_schema:
            continue  # can't personalize without schema

        # Get user vector per variant
        if variant in ("full", "no_user_bias", "linear_only"):
            u = torch.tensor(schema.to_vector(), dtype=torch.float32)
        elif variant == "random_schema":
            from src.extract.schema_extractor import random_schema
            sid_seed = abs(hash(pair.session_id)) + seed
            u = torch.tensor(random_schema(sid_seed).to_vector(), dtype=torch.float32)
        elif variant in ("classic", "concat_linear", "film"):
            # mechanism baselines reuse classic behavioral features as u
            feat = _classic_features_from_history(pair)
            u = classic_to_vector(feat, u_dim=u_dim)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        try:
            s_base_pos, h_pos = load_features(pair.eval_query, pair.pos_url, cache)
            s_base_neg, h_neg = load_features(pair.eval_query, pair.neg_url, cache)
        except KeyError:
            continue

        out.append({
            "s_base_pos": float(s_base_pos),
            "h_pos": h_pos,
            "s_base_neg": float(s_base_neg),
            "h_neg": h_neg,
            "u": u.tolist(),
            "session_id": pair.session_id,
            "pos_rank": pair.pos_rank,
        })
    return out


def train(
    featurized_train: list[dict],
    featurized_dev: list[dict],
    variant: str,
    seed: int,
    lr: float = 5e-3,
    epochs: int = 10,
    batch_size: int = 64,
    device: str = "cpu",
    u_dim: int = 16,
) -> tuple[ScoreIntervention, dict]:
    """Train the score-intervention module on featurized pairs.

    Returns (trained_module, history_dict).
    """
    cfg = ScoreInterventionConfig(variant=variant, u_dim=u_dim)
    model = ScoreIntervention(cfg).to(device)
    logger.info(
        "Training variant=%s, trainable_params=%d",
        variant,
        model.trainable_param_count(),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {"train_loss": [], "dev_loss": []}

    def batch_iter(data: list[dict], bs: int, shuffle: bool):
        idx = list(range(len(data)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield [data[j] for j in idx[i : i + bs]]

    def run_batch(batch: list[dict], train: bool):
        s_base_pos = torch.tensor([b["s_base_pos"] for b in batch], dtype=torch.float32, device=device)
        s_base_neg = torch.tensor([b["s_base_neg"] for b in batch], dtype=torch.float32, device=device)
        h_pos = torch.tensor([b["h_pos"] for b in batch], dtype=torch.float32, device=device)
        h_neg = torch.tensor([b["h_neg"] for b in batch], dtype=torch.float32, device=device)
        u = torch.tensor([b["u"] for b in batch], dtype=torch.float32, device=device)

        pos_scores = model(s_base_pos, h_pos, u)
        neg_scores = model(s_base_neg, h_neg, u)
        loss = pairwise_ranking_loss(pos_scores, neg_scores)

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item()

    best_dev_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in batch_iter(featurized_train, batch_size, shuffle=True):
            losses.append(run_batch(batch, train=True))
        train_loss = float(np.mean(losses)) if losses else 0.0

        model.eval()
        with torch.no_grad():
            dev_losses = [run_batch(b, train=False) for b in batch_iter(featurized_dev, batch_size, shuffle=False)]
        dev_loss = float(np.mean(dev_losses)) if dev_losses else 0.0

        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  dev_loss=%.4f",
            epoch + 1,
            epochs,
            train_loss,
            dev_loss,
        )
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["full", "no_user_bias", "linear_only", "random_schema", "classic", "concat_linear", "film"])
    ap.add_argument("--u-dim", type=int, default=16, help="User vector dimension (Task 3 dim ablation)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True, help="JSON output path for trained model + history")
    ap.add_argument("--data-cache", default="cache/data", help="Cached sessions/pairs dir")
    ap.add_argument("--feature-cache", default="cache/features_queryflow", help="Cached backbone features dir")
    ap.add_argument("--schema-cache", default="cache/schema", help="Cached LLM schema dir")
    ap.add_argument("--extractor", default="codex", choices=["codex", "phi3"], help="Schema extractor backend")
    ap.add_argument("--max-sessions", type=int, default=None, help="Cap on sessions (for quick tests)")
    ap.add_argument("--history-k", type=int, default=3)
    ap.add_argument("--topk-debias", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)

    # --- Data ---
    logger.info("Loading AOL sessions...")
    sessions = load_aol_sessions_cached_or_fresh(min_session_len=args.history_k + 1, max_sessions=args.max_sessions, seed=args.seed)
    train_sess, dev_sess, test_sess = split_sessions(sessions, seed=args.seed)
    logger.info("Split: train=%d dev=%d test=%d", len(train_sess), len(dev_sess), len(test_sess))

    train_pairs = build_training_pairs(train_sess, history_k=args.history_k, topk_debias=args.topk_debias, seed=args.seed)
    dev_pairs = build_training_pairs(dev_sess, history_k=args.history_k, topk_debias=args.topk_debias, seed=args.seed)
    logger.info("Pairs: train=%d dev=%d", len(train_pairs), len(dev_pairs))

    # --- Feature caches ---
    cache = FeatureCache(args.feature_cache)
    if args.extractor == "phi3":
        from src.extract.phi3_extractor import Phi3SchemaExtractor
        extractor = Phi3SchemaExtractor(cache_dir=args.schema_cache)
        logger.info("Using Phi-3-mini local extractor (no API rate limit)")
    else:
        extractor = SchemaExtractor(cache_dir=args.schema_cache)
        logger.info("Using codex (gpt-5.4-mini) extractor")

    feat_train = featurize_pairs_batch(train_pairs, cache, extractor, args.variant, args.seed, desc="train", u_dim=args.u_dim)
    feat_dev = featurize_pairs_batch(dev_pairs, cache, extractor, args.variant, args.seed, desc="dev", u_dim=args.u_dim)
    logger.info("Featurized: train=%d dev=%d (some pairs dropped if features missing)", len(feat_train), len(feat_dev))

    # --- Train ---
    model, history = train(
        feat_train,
        feat_dev,
        variant=args.variant,
        seed=args.seed,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        u_dim=args.u_dim,
    )

    # --- Save ---
    out = {
        "variant": args.variant,
        "seed": args.seed,
        "trainable_params": model.trainable_param_count(),
        "n_train_pairs": len(feat_train),
        "n_dev_pairs": len(feat_dev),
        "history": history,
        "state_dict": {k: v.tolist() for k, v in model.state_dict().items()},
        "config": vars(args),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    logger.info("Saved trained model to %s", args.output)


if __name__ == "__main__":
    main()
