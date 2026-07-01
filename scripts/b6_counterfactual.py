"""
Block 6: Counterfactual rank-shift inspection.

C3 qualitative evidence: 20-session manual inspection of top-5 evidence
before/after personalization.

For each session:
  - Show user schema (LLM-extracted)
  - Show top-5 docs under user-agnostic baseline (sorted by s_base)
  - Show top-5 docs under Pref-Rerank (sorted by s = s_base + w^T u + h^T A u)
  - Annotator manually verifies: do rank shifts align with schema?

Output: 20 case studies in a markdown table for the paper.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aol_session import load_aol_sessions_cached_or_fresh, split_sessions, build_training_pairs, build_eval_candidates  # noqa: E402
from src.extract.schema_extractor import SchemaExtractor  # noqa: E402
from src.model.queryflow_backbone import FeatureCache, load_features  # noqa: E402
from src.model.score_intervention import ScoreIntervention, ScoreInterventionConfig  # noqa: E402

# Reuse scoring logic from evaluate.py
from scripts.evaluate import score_candidates_baseline, score_candidates_personalized  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("b6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--n-sessions", type=int, default=20)
    ap.add_argument("--output", default="results/b6_counterfactual.md")
    ap.add_argument("--feature-cache", default="cache/features_queryflow")
    ap.add_argument("--schema-cache", default="cache/schema")
    ap.add_argument("--history-k", type=int, default=3)
    ap.add_argument("--topk-debias", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load model
    model_state = json.loads(Path(args.model).read_text())
    cfg = ScoreInterventionConfig(variant=args.variant)
    model = ScoreIntervention(cfg)
    sd = {k: __import__("torch").tensor(v) for k, v in model_state["state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()

    # Load data + sample diverse 20 sessions
    sessions = load_aol_sessions_cached_or_fresh(min_session_len=args.history_k + 1, seed=args.seed)
    train_s, dev_s, test_s = split_sessions(sessions, seed=args.seed)
    test_pairs = build_training_pairs(test_s, history_k=args.history_k, topk_debias=args.topk_debias, seed=args.seed)
    # Candidate negatives also drawn ONLY from test split (user-disjoint from train/dev)
    candidates = build_eval_candidates(test_pairs, test_pairs, n_negatives=9, seed=args.seed)

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    sample = candidates[: args.n_sessions]

    cache = FeatureCache(args.feature_cache)
    extractor = SchemaExtractor(cache_dir=args.schema_cache)

    lines = ["# B6 Counterfactual Rank-Shift Inspection\n"]
    lines.append(f"Sampled {len(sample)} sessions from test split.\n")
    lines.append("For each session: shows user schema, then top-5 docs under baseline vs Pref-Rerank.\n\n")

    intuitive_count = 0
    for i, cand in enumerate(sample):
        schema = extractor.extract(cand["user_history"])
        if schema is None:
            continue
        bs = score_candidates_baseline(cand, cache)
        ps = score_candidates_personalized(cand, cache, model, extractor, args.variant, args.seed)
        if bs is None or ps is None:
            continue

        urls = [cand["pos_url"]] + cand["neg_urls"]
        base_rank = sorted(range(len(bs)), key=lambda k: bs[k], reverse=True)
        pers_rank = sorted(range(len(ps)), key=lambda k: ps[k], reverse=True)

        lines.append(f"## Case {i+1}: session `{cand['session_id']}`\n")
        lines.append(f"**Schema**: `{schema.to_dict()}`\n")
        lines.append(f"**Eval query**: `{cand['eval_query']}`\n\n")
        lines.append("| Rank | Baseline (s_base) | Pref-Rerank (personalized) | Shift |\n")
        lines.append("|------|------------------|-----------------------------|-------|\n")
        for r in range(5):
            b_idx = base_rank[r] if r < len(base_rank) else None
            p_idx = pers_rank[r] if r < len(pers_rank) else None
            b_url = urls[b_idx][:40] + "..." if b_idx is not None else "-"
            p_url = urls[p_idx][:40] + "..." if p_idx is not None else "-"
            shift = ""
            if b_idx is not None and p_idx is not None:
                if b_idx == p_idx:
                    shift = "—"
                else:
                    shift = f"was #{base_rank.index(p_idx)+1}"
            lines.append(f"| {r+1} | {b_url} | {p_url} | {shift} |\n")
        lines.append(f"\n**Click was at rank**: {cand['pos_rank']}\n\n")

        # Ask annotator for intuition (interactive)
        print(f"\n=== Case {i+1}: {cand['session_id']} ===")
        print(f"Schema: {schema.to_dict()}")
        print(f"Query: {cand['eval_query']}")
        print(f"Baseline top-1: {urls[base_rank[0]][:60]}")
        print(f"Pref-Rerank top-1: {urls[pers_rank[0]][:60]}")
        ans = input("Intuitive shift (y/n)? ").strip().lower()
        if ans == "y":
            intuitive_count += 1
            lines.append(f"**Annotator**: intuitive ✓\n\n")
        else:
            lines.append(f"**Annotator**: NOT intuitive ✗\n\n")

    lines.append(f"\n## Summary\n\n- Intuitive shifts: {intuitive_count}/{len(sample)}\n")
    lines.append(f"- Target: ≥15/{len(sample)} for C3 qualitative claim\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("".join(lines))
    logger.info("B6 inspection saved to %s", args.output)
    logger.info("Intuitive: %d/%d", intuitive_count, len(sample))


if __name__ == "__main__":
    main()
