"""
Block 7: LLM schema validation annotation interface.

Anti-claim AC1: LLM schema extraction quality is sufficient.

Protocol:
  1. Sample 200 sessions (stratified by length).
  2. 3 human annotators independently produce schema per session.
  3. Compute Cohen's Kappa between annotators (target ≥0.7).
  4. LLM produces schema for same sessions.
  5. Compute LLM-human agreement rate per dim (target ≥80%).

Usage:
    python scripts/b7_annotation.py --sample 200 --output results/b7_annotation.json

The annotation UI is a simple CLI prompt per session (annotator enters their
labels). Run 3 times with different --annotator-id.
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

from src.data.aol_session import load_aol_sessions, Session  # noqa: E402
from src.extract.schema_extractor import (  # noqa: E402
    SCHEMA_VOCAB,
    SCHEMA_DIMS,
    SchemaExtractor,
)
from src.eval.metrics import cohen_kappa  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("b7")


def sample_sessions(sessions: list[Session], n: int = 200, seed: int = 42) -> list[Session]:
    """Stratified sample: 50 short (3-4 clicks), 100 medium (5-7), 50 long (8+)."""
    rng = random.Random(seed)
    short = [s for s in sessions if 3 <= len(s.clicks) <= 4]
    med = [s for s in sessions if 5 <= len(s.clicks) <= 7]
    long = [s for s in sessions if len(s.clicks) >= 8]
    rng.shuffle(short); rng.shuffle(med); rng.shuffle(long)
    out = short[: n // 4] + med[: n // 2] + long[: n // 4]
    return out[:n]


def annotate_session_cli(session: Session, annotator_id: str) -> dict | None:
    """One-session CLI annotation. Returns schema dict or None to skip."""
    print(f"\n=== Session {session.session_id} ===")
    for i, c in enumerate(session.clicks[:5], 1):
        print(f"  {i}. q='{c.query}' -> {c.clicked_url}")
    print()
    schema = {}
    for dim in SCHEMA_DIMS:
        opts = SCHEMA_VOCAB[dim]
        while True:
            ans = input(f"  {dim} {opts} (or 'skip'): ").strip()
            if ans == "skip":
                return None
            if ans in opts:
                schema[dim] = ans
                break
            print(f"    Invalid. Choose from {opts}")
    schema["annotator"] = annotator_id
    schema["session_id"] = session.session_id
    return schema


def collect_annotations(
    sessions: list[Session],
    annotator_id: str,
    output_file: Path,
    resume: bool = True,
) -> list[dict]:
    """Collect annotations for a single annotator. Resumes from existing file."""
    existing = []
    if resume and output_file.exists():
        existing = [json.loads(l) for l in output_file.read_text().splitlines() if l.strip()]
        done_sids = {e["session_id"] for e in existing}
        logger.info("Resuming: %d sessions already annotated", len(existing))
    else:
        done_sids = set()

    out = list(existing)
    for i, s in enumerate(sessions):
        if s.session_id in done_sids:
            continue
        print(f"\n[{i+1}/{len(sessions)}]")
        schema = annotate_session_cli(s, annotator_id)
        if schema is not None:
            out.append(schema)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(schema, ensure_ascii=False) + "\n")
        cont = input("  Continue? [Y/n]: ").strip().lower()
        if cont == "n":
            break
    return out


def compute_iaa(annotations: dict[str, list[dict]]) -> dict:
    """Compute pairwise Cohen's Kappa between annotators per dim.

    annotations: {annotator_id: [schema_dicts]}
    """
    annotators = list(annotations.keys())
    if len(annotators) < 2:
        return {"error": "need ≥2 annotators for IAA"}

    # Align by session_id
    by_sid: dict[str, dict[str, dict[str, str]]] = {}
    for ann in annotators:
        for s in annotations[ann]:
            by_sid.setdefault(s["session_id"], {})[ann] = s

    common_sids = [sid for sid, anns in by_sid.items() if all(a in anns for a in annotators)]
    logger.info("Common sessions across annotators: %d", len(common_sids))

    kappas = {}
    for dim in SCHEMA_DIMS:
        pairwise = []
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                a1 = [by_sid[sid][annotators[i]][dim] for sid in common_sids]
                a2 = [by_sid[sid][annotators[j]][dim] for sid in common_sids]
                # Encode to int for kappa
                labels = sorted(set(a1 + a2))
                l2i = {l: i for i, l in enumerate(labels)}
                k = cohen_kappa([l2i[x] for x in a1], [l2i[x] for x in a2])
                pairwise.append(k)
        kappas[dim] = float(sum(pairwise) / len(pairwise))
    return {"n_common_sessions": len(common_sids), "kappa_per_dim": kappas}


def compute_llm_agreement(
    annotations: dict[str, list[dict]],
    llm_schemas: dict[str, dict[str, str]],
) -> dict:
    """LLM-vs-human-consensus agreement per dim.

    Human consensus = majority vote across annotators per (session, dim).
    """
    # Build per-session human consensus
    by_sid: dict[str, list[dict]] = {}
    for ann in annotations:
        for s in annotations[ann]:
            by_sid.setdefault(s["session_id"], []).append(s)

    agreement = {dim: {"agree": 0, "total": 0} for dim in SCHEMA_DIMS}
    for sid, anns in by_sid.items():
        if sid not in llm_schemas:
            continue
        for dim in SCHEMA_DIMS:
            vals = [a[dim] for a in anns if dim in a]
            if not vals:
                continue
            # Majority
            from collections import Counter
            consensus = Counter(vals).most_common(1)[0][0]
            llm_val = llm_schemas[sid].get(dim)
            agreement[dim]["total"] += 1
            if llm_val == consensus:
                agreement[dim]["agree"] += 1

    return {
        dim: {
            "agree": v["agree"],
            "total": v["total"],
            "rate": v["agree"] / max(v["total"], 1),
        }
        for dim, v in agreement.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["annotate", "analyze"], required=True)
    ap.add_argument("--annotator-id", default="A1")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--sessions-cache", default="cache/data/sampled_sessions.jsonl")
    ap.add_argument("--annotation-dir", default="cache/annotations")
    ap.add_argument("--llm-cache", default="cache/schema")
    ap.add_argument("--output", default="results/b7_annotation.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ann_dir = Path(args.annotation_dir)
    ann_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "annotate":
        # Load or sample sessions
        sess_file = Path(args.sessions_cache)
        if sess_file.exists():
            logger.info("Loading sampled sessions from %s", sess_file)
            sessions = [json.loads(l) for l in sess_file.read_text().splitlines()]
        else:
            logger.info("Sampling %d sessions from AOL...", args.sample)
            all_sessions = load_aol_sessions(min_session_len=3, seed=args.seed)
            sampled = sample_sessions(all_sessions, n=args.sample, seed=args.seed)
            sess_file.parent.mkdir(parents=True, exist_ok=True)
            with open(sess_file, "w", encoding="utf-8") as f:
                for s in sampled:
                    f.write(json.dumps({"session_id": s.session_id, "clicks": [c.__dict__ for c in s.clicks]}, ensure_ascii=False) + "\n")
            sessions = sampled

        # Convert dicts back to Session if needed
        from src.data.aol_session import SessionClick
        if sessions and isinstance(sessions[0], dict):
            sessions = [
                Session(
                    session_id=s["session_id"],
                    anon_id=s["session_id"].split("_")[0],
                    clicks=[SessionClick(**c) for c in s["clicks"]],
                )
                for s in sessions
            ]

        out_file = ann_dir / f"{args.annotator_id}.jsonl"
        annotations = collect_annotations(sessions, args.annotator_id, out_file)
        logger.info("Annotator %s: %d annotations saved to %s", args.annotator_id, len(annotations), out_file)

    elif args.mode == "analyze":
        # Load all annotators
        annotations = {}
        for f in sorted(ann_dir.glob("*.jsonl")):
            ann_id = f.stem
            annotations[ann_id] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        logger.info("Loaded annotators: %s", list(annotations.keys()))

        # Compute IAA
        iaa = compute_iaa(annotations) if len(annotations) >= 2 else {"error": "need ≥2 annotators"}

        # LLM extraction on sampled sessions
        sess_file = Path(args.sessions_cache)
        sessions_raw = [json.loads(l) for l in sess_file.read_text().splitlines()]
        from src.data.aol_session import Session, SessionClick
        sessions = [
            Session(
                session_id=s["session_id"],
                anon_id=s["session_id"].split("_")[0],
                clicks=[SessionClick(**c) for c in s["clicks"]],
            )
            for s in sessions_raw
        ]

        extractor = SchemaExtractor(cache_dir=args.llm_cache)
        llm_schemas: dict[str, dict[str, str]] = {}
        for s in sessions:
            history = [(c.query, c.clicked_url) for c in s.clicks[:3]]
            if len(history) < 2:
                continue
            schema = extractor.extract(history)
            if schema is not None:
                llm_schemas[s.session_id] = schema.to_dict()

        agreement = compute_llm_agreement(annotations, llm_schemas)

        out = {
            "n_annotators": len(annotations),
            "annotators": list(annotations.keys()),
            "iaa": iaa,
            "llm_human_agreement": agreement,
            "success_iaa_kappa_ge_07": all(
                v >= 0.7 for v in iaa.get("kappa_per_dim", {}).values()
            ) if "kappa_per_dim" in iaa else False,
            "success_agreement_ge_80": all(
                v["rate"] >= 0.8 for v in agreement.values()
            ) if agreement else False,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        logger.info("B7 analysis saved to %s", args.output)
        logger.info("IAA Kappa per dim: %s", iaa.get("kappa_per_dim", {}))
        logger.info("LLM agreement per dim: %s", {k: v["rate"] for k, v in agreement.items()})


if __name__ == "__main__":
    main()
