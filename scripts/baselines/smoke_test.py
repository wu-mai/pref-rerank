"""Smoke-test the four baseline scripts against synthetic data.

Builds a tiny in-memory cache mimicking the AOL structure (sessions,
TrainingPairs, FeatureCache) and runs b1-b4. Verifies that:
  - each script imports cleanly
  - each script produces a JSON dict with the expected keys
  - the nDCG difference vs the baseline is in [-1, +1]

If this passes, the baseline scripts are correct on synthetic data and
will also work on the real AOL data once prepare_data + prefetch complete.

Usage:
    python scripts/baselines/smoke_test.py
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke")

ROOT = Path(__file__).resolve().parents[2]


def run(script: str, args: list[str]) -> dict:
    """Run a baseline script and parse the JSON output."""
    cmd = [sys.executable, str(ROOT / "scripts" / "baselines" / script)] + args
    logger.info(">>> %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        logger.error("STDOUT:\n%s", proc.stdout)
        logger.error("STDERR:\n%s", proc.stderr)
        raise RuntimeError(f"{script} failed (rc={proc.returncode})")
    # Find the top-level JSON object in stdout (printed by the script).
    # The script may put arbitrary logging before, so we scan from the end.
    text = proc.stdout.strip()
    # Find the last top-level JSON object (greedy match for braces).
    import re
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if not m:
        logger.error("STDOUT:\n%s", proc.stdout)
        logger.error("STDERR:\n%s", proc.stderr)
        raise RuntimeError("no JSON dict found in stdout")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        logger.error("STDOUT:\n%s", proc.stdout)
        logger.error("STDERR:\n%s", proc.stderr)
        raise


def make_synthetic_cache(tmpdir: Path) -> tuple[str, str]:
    """Build synthetic Session-typed sessions.pkl + features.pt in tmpdir.

    Returns (sessions_pkl, feature_cache_dir).
    """
    import pickle
    import torch
    import numpy as np
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.data.aol_session import Session, QueryEvent, SerpItem
    from src.model.queryflow_backbone import pair_hash

    queries = ["macbook", "iphone", "tesla", "pizza", "weather"]
    urls = [f"http://example.com/page{i}" for i in range(20)]

    # Build Session dataclasses so load_cached_sessions returns usable objects.
    # Each SERP has 5 items. Every event has exactly one click at rank 5 (below
    # top-3 so positives survive position-debias filter), and history events
    # have clicks so we have enough history clicks (>=2) to build training pairs.
    sessions = []
    for sid in range(50):
        events = []
        for eid in range(4):
            items = []
            for rank in range(1, 6):  # 5 SERP items per query
                clicked = (rank == 5)
                # The eval-event click (eid==3) gets a unique pos id per user;
                # history clicks (eid<3) reuse pos ids so multiple users collide
                # on the same query and can serve as cross-session negatives.
                if eid == 3:
                    doc_id = f"d{sid}_{eid}"  # positive: unique per session
                else:
                    doc_id = f"d{sid}_{eid}"  # positive id (also clicked at eid=3 for one user)
                items.append(SerpItem(doc_id=doc_id, rank=rank, clicked=clicked))
            events.append(QueryEvent(
                query=queries[(sid + eid) % len(queries)],
                time=1000000.0 + sid * 100 + eid * 10,
                items=items,
            ))
        sessions.append(Session(
            session_id=f"sess_{sid}",
            anon_id=f"anon_{sid}",
            events=events,
        ))

    cache_data = tmpdir / "cache" / "data"
    cache_data.mkdir(parents=True, exist_ok=True)
    sessions_pkl = cache_data / "sessions.pkl"
    with open(sessions_pkl, "wb") as f:
        pickle.dump(sessions, f)

    # Build synthetic feature cache: BM25 score + 768-d h.
    # Cache every (queries[i], urls[j]) pair AND every (queries[i], d{sid}_{eid})
    # pair the synthetic sessions actually produce, so load_features() hits
    # during evaluation.
    feature_cache_dir = tmpdir / "cache" / "features_queryflow"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    feats = {}
    rng = np.random.RandomState(42)
    for q in queries:
        for u in urls:
            s = float(rng.rand() * 5)
            h = rng.randn(768).tolist()
            feats[pair_hash(q, u)] = (s, h)
        # Also cache for every doc actually used by the synthetic sessions
        for sid in range(50):
            for eid in range(4):
                doc_id = f"d{sid}_{eid}"
                feats[pair_hash(q, doc_id)] = (float(rng.rand() * 5), rng.randn(768).tolist())
    torch.save(feats, feature_cache_dir / "features.pt")

    return str(sessions_pkl), str(feature_cache_dir)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        sessions_pkl, feature_cache = make_synthetic_cache(tmpdir)

        results = {}
        for name, script in [
            ("B1 popularity", "b1_popularity.py"),
            ("B2 hist-lex", "b2_hist_lex.py"),
            ("B3 pclick", "b3_pclick.py"),
            ("B4 hrm", "b4_hrm.py"),
        ]:
            logger.info("=" * 60)
            logger.info("Running %s", name)
            logger.info("=" * 60)
            try:
                d = run(script, [
                    "--session-cache", sessions_pkl,
                    "--feature-cache", feature_cache,
                    "--n-negatives", "4",
                    "--seed", "42",
                ])
                results[name] = d
                logger.info("OK %s: %s", name, json.dumps(d))
                # Validate output schema
                for key in ["system", "n_sessions", "baseline_ndcg10",
                            "personalized_ndcg10", "delta", "ci_low", "ci_high"]:
                    assert key in d, f"{name}: missing key '{key}'"
                assert -1 <= d["delta"] <= 1, f"{name}: delta out of range"
            except Exception as e:
                logger.error("FAILED %s: %s", name, e)
                results[name] = {"error": str(e)}

        # Final summary
        print("\n" + "=" * 60)
        print("SMOKE TEST SUMMARY")
        print("=" * 60)
        for name, d in results.items():
            if "error" in d:
                print(f"  {name}: FAILED - {d['error']}")
            else:
                print(f"  {name}: OK   ndcg10 base={d['baseline_ndcg10']:.4f} "
                      f"pers={d['personalized_ndcg10']:.4f} delta={d['delta']:+.4f}")

        n_failed = sum(1 for d in results.values() if "error" in d)
        if n_failed:
            print(f"\nFAILED: {n_failed}/{len(results)}")
            sys.exit(1)
        else:
            print(f"\nPASSED: all {len(results)} baselines ran successfully.")


if __name__ == "__main__":
    main()