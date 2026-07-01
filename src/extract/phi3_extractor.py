"""
Local Phi-3-mini schema extractor (replaces codex when rate-limited).

Uses microsoft/Phi-3-mini-4k-instruct (cached locally) for closed-vocab
schema extraction. Same interface as SchemaExtractor but no API calls.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Reuse schema vocab and PreferenceSchema from main extractor
from src.extract.schema_extractor import (
    SCHEMA_VOCAB,
    SCHEMA_DIMS,
    MAX_VOCAB_PER_DIM,
    PreferenceSchema,
    PROMPT_TEMPLATE,
    _history_hash,
)


class Phi3SchemaExtractor:
    """Local Phi-3-mini schema extractor with on-disk cache.

    Same cache format as SchemaExtractor (cache/{history_hash}.json),
    so caches are interchangeable. Uses Phi-3-mini-4k-instruct locally.
    """

    def __init__(self, cache_dir: str | Path = "cache/schema"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = "microsoft/Phi-3-mini-4k-instruct"
            logger.info("Loading Phi-3-mini locally (first call, ~10s)...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self._model.eval()
            logger.info("Phi-3-mini loaded on GPU")

    def extract(self, history: list[tuple[str, str]]) -> Optional[PreferenceSchema]:
        """Extract schema for a session history, with cache."""
        hkey = _history_hash(history)
        cache_file = self.cache_dir / f"{hkey}.json"

        # Cache hit
        if cache_file.exists():
            try:
                d = json.loads(cache_file.read_text(encoding="utf-8"))
                return PreferenceSchema(**d)
            except Exception:
                pass

        # Build prompt
        lines = []
        for i, (q, url) in enumerate(history, 1):
            lines.append(f'{i}. "{q}" -> {url}')
        prompt = PROMPT_TEMPLATE.format(
            history_lines="\n".join(lines),
            source_opts=" | ".join(SCHEMA_VOCAB["source"]),
            style_opts=" | ".join(SCHEMA_VOCAB["style"]),
            ev_opts=" | ".join(SCHEMA_VOCAB["evidence_type"]),
            recency_opts=" | ".join(SCHEMA_VOCAB["recency"]),
        )

        # Generate
        schema_dict = self._generate(prompt)
        if schema_dict is not None:
            schema = PreferenceSchema(**schema_dict)
            cache_file.write_text(
                json.dumps(schema.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return schema

        logger.warning("Phi-3 schema extraction failed for history %s", hkey)
        return None

    def _generate(self, prompt: str) -> Optional[dict[str, str]]:
        """Call Phi-3 and parse JSON. Returns None on failure."""
        import torch

        self._load_model()
        messages = [{"role": "user", "content": prompt}]
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        text = self._tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        # Strip markdown fences
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

        # Find JSON object
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            d = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
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
        max_workers: int = 1,  # Phi-3 is single-GPU, no parallelism
        desc: str = "schemas",
    ) -> list[Optional[PreferenceSchema]]:
        """Extract schemas sequentially (Phi-3 is GPU-bound, no thread parallelism)."""
        results: list[Optional[PreferenceSchema]] = [None] * len(histories)
        n_cached = 0
        to_fetch: list[tuple[int, list]] = []

        for i, h in enumerate(histories):
            hkey = _history_hash(h)
            cache_file = self.cache_dir / f"{hkey}.json"
            if cache_file.exists():
                try:
                    d = json.loads(cache_file.read_text(encoding="utf-8"))
                    results[i] = PreferenceSchema(**d)
                    n_cached += 1
                except Exception:
                    to_fetch.append((i, h))
            else:
                to_fetch.append((i, h))

        logger.info("extract_batch (Phi-3): %d cached, %d to fetch", n_cached, len(to_fetch))

        for done, (idx, hist) in enumerate(to_fetch):
            results[idx] = self.extract(hist)
            if (done + 1) % 20 == 0:
                logger.info("  [%s] %d/%d fetched", desc, done + 1, len(to_fetch))

        return results
