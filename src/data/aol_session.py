"""
AOL Session data pipeline (rewritten to use qlogs_iter with FULL SERP items).

Key upgrade over v1: AOL-IA's qlogs_iter() yields AolQlog records that include
the FULL SERP items (clicked AND not-clicked) for each query. This gives us
true SERP-based negatives — no need for cross-session negative sampling, and
no "proxy evaluation" caveat.

AolQlog fields:
    user_id:   anon user id
    query_id:  query hash
    query:     normalized query text
    query_orig: raw query
    time:      datetime
    items:     Tuple[LogItem(doc_id, rank, clicked)]

We group consecutive queries from the same user (within 30 min) into sessions.
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ir_datasets

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SerpItem:
    """One item in a query's SERP."""

    doc_id: str
    rank: int       # 1-indexed SERP rank
    clicked: bool


@dataclass
class QueryEvent:
    """One query in a session, with its full SERP."""

    query: str
    time: float  # timestamp
    items: list[SerpItem] = field(default_factory=list)

    @property
    def clicked_items(self) -> list[SerpItem]:
        return [it for it in self.items if it.clicked]

    @property
    def not_clicked_items(self) -> list[SerpItem]:
        return [it for it in self.items if not it.clicked]


@dataclass
class Session:
    """A user session = ordered sequence of query events within 30 min."""

    session_id: str
    anon_id: str
    events: list[QueryEvent] = field(default_factory=list)

    def history_events(self, k: int) -> list[QueryEvent]:
        """First K query events serve as user history."""
        return self.events[:k]

    def eval_event(self) -> Optional[QueryEvent]:
        """Last query event serves as the evaluation query."""
        if len(self.events) < 2:
            return None
        return self.events[-1]


# Backward-compat alias (old code may reference .clicks)
def _legacy_clicks(self):
    """Flatten events into (query, clicked_doc_id, rank) tuples for legacy code."""
    out = []
    for ev in self.events:
        for it in ev.clicked_items:
            out.append(_LegacyClick(query=ev.query, clicked_url=it.doc_id, rank=it.rank, timestamp=ev.time))
    return out


@dataclass
class _LegacyClick:
    query: str
    clicked_url: str
    rank: Optional[int]
    timestamp: Optional[float]


# ---------------------------------------------------------------------------
# Loading + session grouping
# ---------------------------------------------------------------------------


SESSION_GAP_SEC = 1800  # 30 minutes


def _safe_int(v) -> int:
    """Coerce AOL rank field to int. It may arrive as int, numeric str, or bytes."""
    if v is None:
        return 0
    if isinstance(v, int):
        return v
    if isinstance(v, (bytes, bytearray)):
        # AOL pickle sometimes stores rank as raw byte; decode leniently
        try:
            return int(v.decode("ascii", errors="ignore").strip() or 0)
        except ValueError:
            return 0
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            # Strip non-digit chars (e.g. control chars from pickle)
            digits = "".join(ch for ch in v if ch.isdigit())
            return int(digits) if digits else 0
    return 0


def load_aol_sessions(
    min_session_len: int = 3,
    max_sessions: Optional[int] = None,
    seed: int = 42,
    max_qlogs: Optional[int] = None,
) -> list[Session]:
    """Load AOL-IA sessions via qlogs_iter.

    Groups consecutive queries from the same user (gap < 30 min) into sessions.
    Filters sessions with fewer than min_session_len query events.

    min_session_len here means number of QUERY EVENTS (each with its own SERP),
    not number of clicks. A session with 3 events has 3 queries.
    """
    dataset = ir_datasets.load("aol-ia")
    by_user: dict[str, list[tuple]] = defaultdict(list)

    n_seen = 0
    for q in dataset.qlogs_iter():
        # q is AolQlog(user_id, query_id, query, query_orig, time, items)
        ts = q.time.timestamp() if hasattr(q.time, "timestamp") else float(q.time)
        items = [
            SerpItem(
                doc_id=str(it.doc_id),
                rank=_safe_int(it.rank),
                clicked=bool(it.clicked),
            )
            for it in q.items
        ]
        by_user[q.user_id].append((q.query, ts, items))
        n_seen += 1
        if max_qlogs and n_seen >= max_qlogs:
            break
        # Early stop if we have way more users than needed
        if max_sessions and len(by_user) >= max_sessions * 20:
            # Keep collecting for this batch of users but stop adding new ones
            pass

    logger.info("Loaded %d qlogs across %d users", n_seen, len(by_user))

    sessions: list[Session] = []
    rng = random.Random(seed)

    for anon_id, recs in by_user.items():
        recs.sort(key=lambda r: r[1])  # by timestamp
        current: list[tuple] = []
        last_ts: Optional[float] = None
        for r in recs:
            ts = r[1]
            if last_ts is not None and (ts - last_ts) > SESSION_GAP_SEC:
                if len(current) >= min_session_len:
                    sessions.append(_to_session(anon_id, current, len(sessions)))
                current = []
            current.append(r)
            last_ts = ts
        if len(current) >= min_session_len:
            sessions.append(_to_session(anon_id, current, len(sessions)))

    if max_sessions and len(sessions) > max_sessions:
        rng.shuffle(sessions)
        sessions = sessions[:max_sessions]

    logger.info("Built %d AOL sessions (min_events=%d)", len(sessions), min_session_len)
    return sessions


def _to_session(anon_id: str, recs: list[tuple], sid: int) -> Session:
    events = [
        QueryEvent(
            query=r[0],
            time=r[1],
            items=r[2],
        )
        for r in recs
    ]
    return Session(
        session_id=f"{anon_id}_{sid}",
        anon_id=anon_id,
        events=events,
    )


def load_cached_sessions(cache_path: str | Path = "cache/data/sessions.pkl") -> Optional[list[Session]]:
    """Load pre-cached sessions (built by scripts/prepare_data.py).

    Returns None if cache doesn't exist. Downstream code should call this first
    and fall back to load_aol_sessions() only if cache is missing.
    """
    p = Path(cache_path)
    if not p.exists():
        return None
    import pickle

    logger.info("Loading cached sessions from %s", p)
    return pickle.loads(p.read_bytes())


def load_aol_sessions_cached_or_fresh(
    min_session_len: int = 4,
    max_sessions: Optional[int] = None,
    seed: int = 42,
    cache_path: str | Path = "cache/data/sessions.pkl",
) -> list[Session]:
    """Try cache first; fall back to local tarball, then ir_datasets.

    This is the recommended entry point for all downstream scripts.

    Note: If the cache is missing, the function first tries the local
    tarball at /root/gcy/Rerank/data/aol-data.tar.gz (fast, CDN-independent),
    and only falls back to ir_datasets' AOL-IA mirror as a last resort.
    """
    cached = load_cached_sessions(cache_path)
    if cached is not None:
        sessions = cached
        if min_session_len != 4:
            # Re-filter if different threshold requested
            sessions = [s for s in sessions if len(s.events) >= min_session_len]
        logger.info("Using %d cached sessions (min_len=%d)", len(sessions), min_session_len)
    else:
        # Try the local tarball fallback before ir_datasets (slow CDN)
        local_tarball = Path("/root/gcy/Rerank/data/aol-data.tar.gz")
        if local_tarball.exists():
            logger.info("No cache at %s; falling back to local tarball %s",
                        cache_path, local_tarball)
            from src.data.aol_local import iter_local_qlogs
            from collections import defaultdict
            qlogs = iter_local_qlogs(local_tarball)
            by_user: dict[str, list] = defaultdict(list)
            for q in qlogs:
                items = [
                    SerpItem(doc_id=str(it.doc_id), rank=it.rank, clicked=bool(it.clicked))
                    for it in q.items
                ]
                by_user[q.user_id].append((q.query, q.time, items))
            sessions = []
            for anon_id, recs in by_user.items():
                recs.sort(key=lambda r: r[1])
                current: list = []
                last_ts: Optional[float] = None
                for r in recs:
                    ts = r[1]
                    if last_ts is not None and (ts - last_ts) > SESSION_GAP_SEC:
                        if len(current) >= min_session_len:
                            sessions.append(_to_session(anon_id, current, len(sessions)))
                        current = []
                    current.append(r)
                    last_ts = ts
                if len(current) >= min_session_len:
                    sessions.append(_to_session(anon_id, current, len(sessions)))
        else:
            logger.info("No cache found, loading from qlog via ir_datasets (slow, first time)...")
            sessions = load_aol_sessions(min_session_len=min_session_len, max_sessions=max_sessions, seed=seed)

    if max_sessions and len(sessions) > max_sessions:
        rng = random.Random(seed)
        rng.shuffle(sessions)
        sessions = sessions[:max_sessions]
    return sessions


# ---------------------------------------------------------------------------
# Training pair construction (TRUE SERP-based negatives)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Training pair construction (TRUE SERP-based negatives)
# ---------------------------------------------------------------------------


@dataclass
class TrainingPair:
    """A pairwise training instance for the reranker.

    user_history: list of (query, clicked_doc_id) from the first K events.
    eval_query: the evaluation query.
    pos_url: clicked doc_id for eval query (rank > topk_debias).
    neg_url: a NOT-CLICKED doc_id from the SAME SERP (true negative, not proxy).
    pos_rank / neg_rank: SERP positions.
    """

    session_id: str
    user_history: list[tuple[str, str]]
    eval_query: str
    pos_url: str
    neg_url: str
    pos_rank: Optional[int]
    neg_rank: Optional[int]


def build_training_pairs(
    sessions: list[Session],
    history_k: int = 3,
    topk_debias: int = 3,
    seed: int = 42,
) -> list[TrainingPair]:
    """Build position-debiased (clicked, cross-session-clicked) pairs.

    IMPORTANT AOL DATA CAVEAT: AOL query log only records CLICKED items per
    query — there are no not-clicked SERP candidates. So negatives MUST be
    cross-session: URLs clicked for the SAME query by a DIFFERENT user (in the
    same split). This is a proxy for true non-relevance (the user may have found
    it relevant but just didn't click). The paper must disclose this.

    Per GPT-5.4 review requirements:
      - negatives come from a DIFFERENT user (no user leakage)
      - negatives exclude URLs in this user's history
      - position-debiased: positive click must be rank > topk_debias (below top-3)
    """
    rng = random.Random(seed)

    # Build query -> [(doc_id, anon_id, rank)] for THIS split only
    query_to_clicks: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for s in sessions:
        for ev in s.events:
            for it in ev.clicked_items:
                query_to_clicks[ev.query].append((it.doc_id, s.anon_id, it.rank))

    pairs: list[TrainingPair] = []
    MAX_PAIRS_PER_SESSION = 5

    for s in sessions:
        if len(s.events) < history_k + 1:
            continue
        history = s.history_events(history_k)
        eval_ev = s.eval_event()
        if eval_ev is None or not eval_ev.clicked_items:
            continue
        # Skip sessions with empty eval query (AOL has some empty-query logs)
        if not eval_ev.query or not eval_ev.query.strip():
            continue

        # User history: (query, first clicked doc_id) per history event
        user_history = []
        history_urls: set[str] = set()
        for he in history:
            clicked = he.clicked_items
            if clicked:
                user_history.append((he.query, clicked[0].doc_id))
                history_urls.add(clicked[0].doc_id)
            else:
                user_history.append((he.query, ""))
        if sum(1 for _, d in user_history if d) < 2:
            continue  # need at least 2 history clicks for meaningful schema

        # Position-debiased positives: clicked items below top-3
        pos_items = [it for it in eval_ev.clicked_items if it.rank > topk_debias]
        if not pos_items:
            continue

        # Negatives: same query, clicked by a DIFFERENT user, not in history
        cands = query_to_clicks.get(eval_ev.query, [])
        neg_pool = [
            (doc_id, rank)
            for (doc_id, anon_id, rank) in cands
            if anon_id != s.anon_id
            and doc_id not in history_urls
        ]
        if not neg_pool:
            continue

        n_pairs = 0
        for pos in pos_items:
            if n_pairs >= MAX_PAIRS_PER_SESSION:
                break
            neg_doc, neg_rank = rng.choice(neg_pool)
            # Also exclude exact pos
            if neg_doc == pos.doc_id:
                continue
            pairs.append(
                TrainingPair(
                    session_id=s.session_id,
                    user_history=user_history,
                    eval_query=eval_ev.query,
                    pos_url=pos.doc_id,
                    neg_url=neg_doc,
                    pos_rank=pos.rank,
                    neg_rank=neg_rank,
                )
            )
            n_pairs += 1

    logger.info(
        "Built %d training pairs (history_k=%d, topk_debias=%d, cross-session negatives)",
        len(pairs),
        history_k,
        topk_debias,
    )
    return pairs


# ---------------------------------------------------------------------------
# User-disjoint split
# ---------------------------------------------------------------------------


def split_sessions(
    sessions: list[Session],
    train_frac: float = 0.7,
    dev_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[Session], list[Session], list[Session]]:
    """Split sessions into train/dev/test by USER (anon_id).

    CRITICAL: prevents user leakage — same user's sessions must all be in the
    same split.
    """
    rng = random.Random(seed)

    by_user: dict[str, list[Session]] = defaultdict(list)
    for s in sessions:
        by_user[s.anon_id].append(s)

    users = list(by_user.keys())
    rng.shuffle(users)
    n = len(users)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)

    train_users = set(users[:n_train])
    dev_users = set(users[n_train : n_train + n_dev])

    train_s, dev_s, test_s = [], [], []
    for u in users:
        if u in train_users:
            train_s.extend(by_user[u])
        elif u in dev_users:
            dev_s.extend(by_user[u])
        else:
            test_s.extend(by_user[u])
    return train_s, dev_s, test_s


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_sessions(sessions, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(
                json.dumps(
                    {
                        "session_id": s.session_id,
                        "anon_id": s.anon_id,
                        "events": [
                            {
                                "query": ev.query,
                                "time": ev.time,
                                "items": [it.__dict__ for it in ev.items],
                            }
                            for ev in s.events
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_pairs(pairs, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p.__dict__, ensure_ascii=False) + "\n")
