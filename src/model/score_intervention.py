"""
Pref-Rerank score intervention module + ablation variants.

Core formula (full Pref-Rerank):
    s(q, d, u) = s_base(q, d) + w^T u + h(q, d)^T A u

where:
    s_base(q, d): frozen monoT5 cross-encoder logit (scalar)
    h(q, d):      frozen monoT5 encoder last-layer mean-pooled rep (768-d)
    u:            16-d user preference code (from LLM-extracted schema)
    w:            16-d trainable user-bias vector
    A:            768 x 16 trainable bilinear interaction matrix

Total trainable: 16 + 768*16 = 12,304 params.

Ablation variants:
    - "no_user_bias":   s = s_base + h^T A u       (no w^T u)
    - "linear_only":    s = s_base + w^T u          (no h^T A u)
    - "random_schema":  full formula but u is random per session
    - "prompt":         (handled separately — uses raw history in prompt)
    - "classic":        (handled separately — uses behavioral features)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn


ScoreVariant = Literal[
    "full",              # Pref-Rerank: s_base + w^T u + h^T A u
    "no_user_bias",      # s_base + h^T A u
    "linear_only",       # s_base + w^T u
    "random_schema",     # full formula, random u (control)
    "classic",           # full formula, u = classic features (6-d zero-padded to 16-d)
    # --- mechanism baselines (Task 2: lightweight alternatives to bilinear) ---
    "concat_linear",     # s_base + Linear([h; u]): concat + linear projection
    "film",              # s_base + sum(gamma*h + beta), gamma/beta = Linear(u)
]


@dataclass
class ScoreInterventionConfig:
    variant: ScoreVariant = "full"
    h_dim: int = 768      # monoT5 encoder hidden size
    u_dim: int = 16       # user preference code size (4 schema dims × 4 one-hot slots)
    init_std: float = 0.02


class ScoreIntervention(nn.Module):
    """Lightweight trainable score intervention on a frozen cross-encoder.

    Inputs are PRE-COMPUTED frozen features:
        s_base: (B,)         scalar base scores from monoT5
        h:      (B, 768)     mean-pooled monoT5 encoder rep
        u:      (B, 32)      user preference code (one-hot concat)

    Output:
        s: (B,)  personalized score
    """

    def __init__(self, cfg: ScoreInterventionConfig):
        super().__init__()
        self.cfg = cfg

        # Only create the parameters each variant actually uses, so
        # trainable_param_count() reflects the true budget per variant.
        if cfg.variant in ("full", "no_user_bias", "linear_only", "random_schema", "classic"):
            # bilinear variants need A (and w for those with user bias)
            self.A = nn.Parameter(torch.zeros(cfg.h_dim, cfg.u_dim))  # bilinear interaction
            nn.init.normal_(self.A, std=cfg.init_std)
            if cfg.variant in ("full", "linear_only", "random_schema", "classic"):
                self.w = nn.Parameter(torch.zeros(cfg.u_dim))  # user-only bias
                nn.init.normal_(self.w, std=cfg.init_std)

        # --- mechanism baselines (Task 2) ---
        if cfg.variant == "concat_linear":
            # Linear([h; u]) -> 1 ; params: (h_dim+u_dim)*1 + 1 bias
            self.concat_proj = nn.Linear(cfg.h_dim + cfg.u_dim, 1)
            nn.init.normal_(self.concat_proj.weight, std=cfg.init_std)
            nn.init.zeros_(self.concat_proj.bias)
        elif cfg.variant == "film":
            # gamma, beta = Linear(u -> 2*h_dim); params: u_dim*2*h_dim + 2*h_dim
            self.film_proj = nn.Linear(cfg.u_dim, 2 * cfg.h_dim)
            nn.init.normal_(self.film_proj.weight, std=cfg.init_std)
            nn.init.zeros_(self.film_proj.bias)

    def forward(
        self,
        s_base: torch.Tensor,   # (B,)
        h: torch.Tensor,        # (B, 768)
        u: torch.Tensor,        # (B, 32)
    ) -> torch.Tensor:
        """Compute personalized score per variant.

        s_base and h are FROZEN features (precomputed by MonoT5Backbone under
        inference_mode). We detach explicitly to guarantee no gradient flows back
        into the backbone, even if a future caller forgets to precompute.
        """
        s_base = s_base.detach().float()
        h = h.detach().float()
        u = u.float()
        s = s_base

        if self.cfg.variant in ("full", "linear_only", "random_schema", "classic"):
            # w^T u: (B,) = einsum('b,bd->b', w, u)
            user_bias = u @ self.w  # (B,)
            s = s + user_bias

        if self.cfg.variant in ("full", "no_user_bias", "random_schema", "classic"):
            # h^T A u: (B,) = einsum('bd,de,be->b', h, A, u)
            # = sum over d of h[b,d] * (A @ u)[b,d]
            # Compute (B, u_dim) = h @ A, then dot with u per row
            h_A = h @ self.A  # (B, u_dim)
            bilinear = (h_A * u).sum(dim=-1)  # (B,)
            s = s + bilinear

        if self.cfg.variant == "concat_linear":
            # concat [h; u] then linear project to scalar
            concat = torch.cat([h, u], dim=-1)  # (B, h_dim+u_dim)
            s = s + self.concat_proj(concat).squeeze(-1)  # (B,)

        if self.cfg.variant == "film":
            # gamma, beta = Linear(u); modulate h elementwise, sum to scalar
            gb = self.film_proj(u)              # (B, 2*h_dim)
            gamma, beta = gb.chunk(2, dim=-1)   # each (B, h_dim)
            s = s + (gamma * h + beta).sum(dim=-1)  # (B,)

        # linear_only: only w^T u added above; no bilinear
        # no_user_bias: only bilinear added above; no w^T u
        return s

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Pairwise ranking loss (session-level)
# ---------------------------------------------------------------------------


def pairwise_ranking_loss(
    pos_scores: torch.Tensor,  # (B,)
    neg_scores: torch.Tensor,  # (B,)
) -> torch.Tensor:
    """L = -log sigmoid(s_pos - s_neg).

    This is the Bradley-Terry pairwise loss; equivalent to RankNet.
    """
    diff = pos_scores - neg_scores
    # numerically stable log-sigmoid
    return -torch.nn.functional.logsigmoid(diff).mean()


# ---------------------------------------------------------------------------
# Classic features encoder (for Block 3 ablation R008)
# ---------------------------------------------------------------------------


@dataclass
class ClassicFeatures:
    """Behavioral session features (NOT semantic).

    Used by Pref-Rerank-classic ablation to show typed schema > behavioral.
    """

    bm25_click_overlap: float    # avg lexical overlap between eval query and past clicks
    avg_dwell_proxy: float       # proxy: number of clicks in history (no real dwell in AOL)
    avg_rank: float              # avg SERP rank of past clicks
    history_len: float           # number of past clicks
    query_len: float             # eval query token count
    has_reformulation: float     # 1.0 if eval query shares tokens with history query


CLASSIC_FEATURE_DIM = 6


def classic_to_vector(f: ClassicFeatures, u_dim: int = 16) -> torch.Tensor:
    """6-d raw features -> u_dim-d user_vector via zero-pad.

    Note: classic ablation uses the SAME score-intervention architecture as
    Pref-Rerank so the trainable budget matches the bilinear baseline. We
    zero-pad the 6 classic features to u_dim. For u_dim < 6 the vector is
    truncated (used only by the dim ablation).
    """
    v = torch.tensor(
        [
            f.bm25_click_overlap,
            f.avg_dwell_proxy,
            f.avg_rank,
            f.history_len,
            f.query_len,
            f.has_reformulation,
        ],
        dtype=torch.float32,
    )
    if u_dim >= CLASSIC_FEATURE_DIM:
        pad = torch.zeros(u_dim - CLASSIC_FEATURE_DIM)
        return torch.cat([v, pad])
    else:
        return v[:u_dim]
