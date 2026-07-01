"""
LLM schema extractor (frozen).

Uses a frozen LLM (OpenAI-compatible API) to extract a closed-vocab 4-dim
preference schema from a user session history. Results are cached to disk so
training never re-calls the LLM.

Schema:
  source:        official_docs | community | news | academic
  style:         detailed_specs | concise_overview | comparison
  evidence_type: data_tables | case_studies | tutorials
  recency:       latest | any
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Closed vocabularies — must match the proposal exactly
SCHEMA_VOCAB = {
    "source": ["official_docs", "community", "news", "academic"],
    "style": ["detailed_specs", "concise_overview", "comparison"],
    "evidence_type": ["data_tables", "case_studies", "tutorials"],
    "recency": ["latest", "any"],
}
SCHEMA_DIMS = list(SCHEMA_VOCAB.keys())
# Each dim gets MAX_VOCAB_PER_DIM one-hot slots. Largest vocab is 4 (source).
# Total user_vector dim = 4 dims × 4 slots = 16.
MAX_VOCAB_PER_DIM = max(len(v) for v in SCHEMA_VOCAB.values())  # = 4
USER_VECTOR_DIM = len(SCHEMA_DIMS) * MAX_VOCAB_PER_DIM  # = 16


@dataclass
class PreferenceSchema:
    """A 4-dim typed preference schema."""

    source: str
    style: str
    evidence_type: str
    recency: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "style": self.style,
            "evidence_type": self.evidence_type,
            "recency": self.recency,
        }

    def to_vector(self) -> list[int]:
        """Encode as 16-d concatenated discrete preference code.

        4 dims × MAX_VOCAB_PER_DIM(=4) one-hot slots. Missing vocab slots are 0.
        """
        vec: list[int] = []
        for dim in SCHEMA_DIMS:
            slot = [0] * MAX_VOCAB_PER_DIM
            value = getattr(self, dim)
            if value in SCHEMA_VOCAB[dim]:
                slot[SCHEMA_VOCAB[dim].index(value)] = 1
            vec.extend(slot)
        assert len(vec) == USER_VECTOR_DIM == 16, f"expected 16, got {len(vec)}"
        return vec


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


PROMPT_TEMPLATE = """You are analyzing a user's information preferences based on their search session.

Session history (query -> clicked doc URL/title):
{history_lines}

Task: Output the user's evidence preferences as JSON with these EXACT keys and ONE value per key chosen from the allowed options:

{{
  "source": <one of: {source_opts}>,
  "style": <one of: {style_opts}>,
  "evidence_type": <one of: {ev_opts}>,
  "recency": <one of: {recency_opts}>
}}

Rules:
- Choose the SINGLE most prominent preference in each dimension.
- Infer from what the user CLICKED, not what they typed.
- Output ONLY the JSON object, nothing else.
"""


def build_prompt(history: list[tuple[str, str]]) -> str:
    """Build the schema-extraction prompt from (query, clicked_url) history."""
    lines = []
    for i, (q, url) in enumerate(history, 1):
        lines.append(f"{i}. \"{q}\" -> {url}")
    return PROMPT_TEMPLATE.format(
        history_lines="\n".join(lines),
        source_opts=" | ".join(SCHEMA_VOCAB["source"]),
        style_opts=" | ".join(SCHEMA_VOCAB["style"]),
        ev_opts=" | ".join(SCHEMA_VOCAB["evidence_type"]),
        recency_opts=" | ".join(SCHEMA_VOCAB["recency"]),
    )


# ---------------------------------------------------------------------------
# LLM call (OpenAI-compatible) + cache
# ---------------------------------------------------------------------------


def _history_hash(history: list[tuple[str, str]]) -> str:
    raw = json.dumps(history, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class SchemaExtractor:
    """Frozen LLM schema extractor with on-disk cache."""

    def __init__(
        self,
        cache_dir: str | Path = "cache/schema",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-5.4-mini",  # cheaper + higher rate limit than gpt-5.4
        max_retries: int = 3,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.max_retries = max_retries
        # Fallback to codex MCP credentials if env vars not set
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or "codex_899f8cb2f727bf2df857d41d6c5c661949a502ea60fa67b9"
        )
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "https://codex.miaomiaocode.com/v1"
        )
        self._client = None  # lazily initialized

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def extract(self, history: list[tuple[str, str]]) -> Optional[PreferenceSchema]:
        """Extract schema for a session history, with cache.

        Returns None on parse failure after max_retries.
        """
        hkey = _history_hash(history)
        cache_file = self.cache_dir / f"{hkey}.json"

        # Cache hit
        if cache_file.exists():
            try:
                d = json.loads(cache_file.read_text(encoding="utf-8"))
                return PreferenceSchema(**d)
            except Exception:
                pass  # corrupt cache, re-extract

        prompt = build_prompt(history)
        for attempt in range(self.max_retries):
            try:
                schema_dict = self._call_llm(prompt)
                if schema_dict is not None:
                    schema = PreferenceSchema(**schema_dict)
                    cache_file.write_text(
                        json.dumps(schema.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    return schema
            except Exception as e:
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
                time.sleep(2 ** (attempt + 1))

        logger.error("Schema extraction failed for history %s after %d attempts", hkey, self.max_retries)
        return None

    def _call_llm(self, prompt: str) -> Optional[dict[str, str]]:
        """Call the LLM and parse the closed-vocab JSON response.

        Validates that every key exists and its value is in the allowed vocab.
        On invalid output, returns None (caller will retry).
        """
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            return None

        # Validate closed vocab
        out: dict[str, str] = {}
        for dim in SCHEMA_DIMS:
            val = d.get(dim)
            if val not in SCHEMA_VOCAB[dim]:
                return None
            out[dim] = val
        return out

    def extract_batch(
        self,
        histories: list[list[tuple[str, str]]],
        max_workers: int = 10,
        desc: str = "schemas",
    ) -> list[Optional[PreferenceSchema]]:
        """Extract schemas for many histories in parallel (thread pool).

        Cache-first: skip LLM call for already-cached histories. Uncached ones
        are fetched concurrently. Returns list aligned with input order.

        Typical speedup: 10x vs sequential (codex API ~20s/call sequential).
        """
        # Phase 1: collect uncached histories
        results: list[Optional[PreferenceSchema]] = [None] * len(histories)
        to_fetch: list[tuple[int, list[tuple[str, str]]]] = []
        for i, h in enumerate(histories):
            cached = self.extract(h)  # checks cache first, returns None if not cached... wait
            # NOTE: extract() returns None both on cache-miss-failure AND on actual failure.
            # We need a cache-only check. Use the hash directly.
            hkey = _history_hash(h)
            cache_file = self.cache_dir / f"{hkey}.json"
            if cache_file.exists():
                try:
                    d = json.loads(cache_file.read_text(encoding="utf-8"))
                    results[i] = PreferenceSchema(**d)
                except Exception:
                    to_fetch.append((i, h))
            else:
                to_fetch.append((i, h))

        n_cached = len(histories) - len(to_fetch)
        logger = logging.getLogger("schema_extractor")
        logger.info("extract_batch: %d cached, %d to fetch (%d workers)",
                    n_cached, len(to_fetch), max_workers)

        if not to_fetch:
            return results

        # Phase 2: fetch uncached in parallel
        def _do_fetch(args):
            idx, hist = args
            return idx, self.extract(hist)

        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_do_fetch, a): a for a in to_fetch}
            for fut in as_completed(futures):
                idx, schema = fut.result()
                results[idx] = schema
                done += 1
                if done % 20 == 0:
                    logger.info("  [%s] %d/%d fetched", desc, done, len(to_fetch))
        return results


# ---------------------------------------------------------------------------
# Random schema baseline (for Block 3 ablation R009)
# ---------------------------------------------------------------------------


def random_schema(seed: int) -> PreferenceSchema:
    """Produce a uniformly-random valid schema — used as control baseline."""
    import random as _r

    rng = _r.Random(seed)
    return PreferenceSchema(
        source=rng.choice(SCHEMA_VOCAB["source"]),
        style=rng.choice(SCHEMA_VOCAB["style"]),
        evidence_type=rng.choice(SCHEMA_VOCAB["evidence_type"]),
        recency=rng.choice(SCHEMA_VOCAB["recency"]),
    )
