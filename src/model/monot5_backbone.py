"""
Frozen monoT5-220M cross-encoder backbone.

Computes and caches two frozen features for every (query, doc) pair:
    s_base(q, d): scalar relevance logit
    h(q, d):      768-d mean-pooled encoder last-hidden-state

These features are precomputed once and reused across all training/ablation runs.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# monoT5 input format (from Castellino et al. 2021):
#   "Query: {q} Document: {d}"
MONOT5_PREFIX = "Query: "
MONOT5_DOC_PREFIX = " Document: "
MONOT5_SUFFIX = " Relevant:"

# T5-220M hidden size (t5-base encoder)
T5_HIDDEN_SIZE = 768


class MonoT5Backbone:
    """Frozen monoT5-220M cross-encoder.

    Loads once, exposes batch inference for s_base and h.
    All weights are frozen; no gradients flow.
    """

    def __init__(
        self,
        model_name: str = "castorini/monot5-base-msmarco",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
    ):
        from transformers import T5ForConditionalGeneration, T5TokenizerFast

        self.device = device
        self.max_length = max_length
        logger.info("Loading monoT5 backbone: %s on %s", model_name, device)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.model: nn.Module = T5ForConditionalGeneration.from_pretrained(
            model_name
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # vocab ids for "true"/"false" tokens (monoT5 outputs these)
        self._true_id = self.tokenizer.encode("true", add_special_tokens=False)[0]
        self._false_id = self.tokenizer.encode("false", add_special_tokens=False)[0]

    def _format_input(self, query: str, doc: str) -> str:
        return f"{MONOT5_PREFIX}{query}{MONOT5_DOC_PREFIX}{doc}{MONOT5_SUFFIX}"

    @torch.inference_mode()
    def encode_batch(self, queries: list[str], docs: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (s_base, h) for a batch of (query, doc) pairs.

        Returns:
            s_base: (B,) scalar logits (log p(true) - log p(false))
            h:      (B, 768) mean-pooled encoder last-hidden-state
        """
        assert len(queries) == len(docs)
        inputs = [self._format_input(q, d) for q, d in zip(queries, docs)]
        enc = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        # Encoder forward to get hidden states for h
        encoder_outputs = self.model.encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            return_dict=True,
        )
        last_hidden = encoder_outputs.last_hidden_state  # (B, L, H)

        # Mean-pool over non-pad tokens
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        summed = (last_hidden * mask).sum(dim=1)  # (B, H)
        counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        h = summed / counts  # (B, H)

        # Decode to get s_base (true/false logits)
        # monoT5: feed decoder with just the EOS/BOS, read logits at first step
        decoder_input_ids = torch.full(
            (enc["input_ids"].size(0), 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )
        out = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )
        logits = out.logits[:, 0, :]  # (B, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)
        s_base = log_probs[:, self._true_id] - log_probs[:, self._false_id]  # (B,)

        return s_base.cpu(), h.cpu()


# ---------------------------------------------------------------------------
# Disk cache (precompute once, reuse across runs)
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
    backbone: MonoT5Backbone,
    pairs: Iterator[tuple[str, str]],  # (query, doc) iterator
    cache: FeatureCache,
    batch_size: int = 32,
    save_every: int = 500,
) -> None:
    """Precompute (s_base, h) for all (query, doc) pairs, cache to disk."""
    pending: list[tuple[str, str, str]] = []  # (hash, query, doc)

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
            continue  # cached
        pending.append((pair_hash(query, doc), query, doc))
        if len(pending) >= batch_size:
            flush(pending)
            n_new += batch_size
            if n_new % save_every == 0:
                cache.save()
                logger.info("Precomputed %d new pairs (total cached: %d)", n_new, len(cache))
    flush(pending)
    cache.save()
    logger.info("Precompute done. Total cached: %d", len(cache))


def load_features(query: str, doc: str, cache: FeatureCache) -> tuple[float, list[float]]:
    """Retrieve cached features for a (query, doc) pair.

    Raises KeyError if not cached. Call precompute_features first.
    """
    feat = cache.get(query, doc)
    if feat is None:
        raise KeyError(f"Features not cached for query='{query[:40]}...' doc='{doc[:40]}...'")
    return feat
