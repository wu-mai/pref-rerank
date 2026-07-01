# Pref-Rerank: Personalized Evidence Reranking for Web RAG

A lightweight bilinear score intervention (12,288 trainable parameters) on top of a frozen BM25+BGE query-flow backbone, making Web-RAG evidence reranking user-conditioned.

**Paper**: WISE 2026 submission.
**Result on AOL-IA (200k sessions, n=201 test)**: session-level nDCG@10 from 0.459 → 0.543 (Δ=+0.084, paired bootstrap 95% CI [+0.034, +0.131]).

---

## What this repo contains

| Directory | Purpose |
|-----------|---------|
| `src/` | Core model and data code |
| `scripts/` | Training, evaluation, sanity checks |
| `configs/` | Hyperparameters |
| `paper/` | LaTeX source (LNCS) for the paper |
| `paper/figures/` | All LaTeX tables (table1-6) + hero figure PDF |
| `results/` | Evaluation outputs only (no model weights) |

---

## Installation

```bash
pip install ir_datasets torch transformers openai numpy
export OPENAI_API_KEY=sk-...   # for LLM schema extraction
export HF_HUB_OFFLINE=1        # if running offline
export TRANSFORMERS_OFFLINE=1
```

---

## Reproducing the results

All commands assume `--max-sessions 200000` (the paper setting). Use a smaller value for quick smoke tests.

### Step 1: Sanity check (R001-R003)
```bash
python scripts/sanity_r001_r003.py --skip-monot5 --skip-llm
```

### Step 2: Precompute backbone features (one-time, ~30 min on A100)
```bash
python scripts/prefetch.py --max-sessions 200000
# cached to cache/features_queryflow/features.pt
```
**Note**: the `cache/` directory is gitignored. Rebuild it after cloning.

### Step 3: Train the main result (Table 1)
```bash
python scripts/train.py --variant classic --seed 42 \
    --max-sessions 200000 --lr 0.05 --epochs 50 \
    --output results/r_classic_seed42.json
python scripts/evaluate.py --model results/r_classic_seed42.json \
    --variant classic --max-sessions 200000 \
    --output results/r_classic_seed42_eval.json
```

### Step 4: Mechanism ablation (Table 2)
```bash
for V in no_user_bias linear_only random_schema full; do
  python scripts/train.py --variant $V --seed 42 \
      --max-sessions 200000 --lr 0.05 --epochs 50 \
      --output results/r_${V}_seed42.json
  python scripts/evaluate.py --model results/r_${V}_seed42.json \
      --variant $V --max-sessions 200000 \
      --output results/r_${V}_seed42_eval.json
done
```

### Step 5: Multi-seed stability (Table 4)
```bash
for SEED in 1 2 42; do
  python scripts/train.py --variant classic --seed $SEED \
      --max-sessions 200000 --lr 0.05 --epochs 50 \
      --output results/r_seed_${SEED}_classic.json
  python scripts/evaluate.py --model results/r_seed_${SEED}_classic.json \
      --variant classic --max-sessions 200000 \
      --output results/r_seed_${SEED}_eval.json
done
```

### Step 6: Mechanism comparison (Table 5)
```bash
for V in concat_linear film; do
  python scripts/train.py --variant $V --seed 42 \
      --max-sessions 200000 --lr 0.05 --epochs 50 \
      --output results/r_${V}_seed42.json
  python scripts/evaluate.py --model results/r_${V}_seed42.json \
      --variant $V --max-sessions 200000 \
      --output results/r_${V}_eval.json
done
```

### Step 7: User-vector dimension ablation (Table 6)
```bash
for UDIM in 4 8 16 32 64; do
  python scripts/train.py --variant classic --seed 42 --u-dim $UDIM \
      --max-sessions 200000 --lr 0.05 --epochs 50 \
      --output results/r_classic_udim${UDIM}.json
  python scripts/evaluate.py --model results/r_classic_udim${UDIM}.json \
      --variant classic --u-dim $UDIM --max-sessions 200000 \
      --output results/r_classic_udim${UDIM}_eval.json
done
```

---

## Variants

| Variant | Score formula | Trainable params |
|---------|--------------|------------------|
| `full` | $s_{\mathrm{base}} + w^\top u + h^\top A u$ | 12,304 |
| `no_user_bias` | $s_{\mathrm{base}} + h^\top A u$ | 12,288 |
| `linear_only` | $s_{\mathrm{base}} + w^\top u$ | 16 |
| `random_schema` | full formula, $u$ = random per session | 12,304 |
| `classic` | full formula, $u$ = 6 behavioral features zero-padded to 16 | 12,304 |
| `concat_linear` | $s_{\mathrm{base}} + \mathrm{Linear}([h;u])$ | 785 |
| `film` | $s_{\mathrm{base}} + \sum(\gamma h + \beta)$, $\gamma,\beta = Wu$ | 26,112 |

---

## Project structure

```
src/
├── data/aol_session.py            # AOL-IA loader, position-debiased pairs
├── extract/schema_extractor.py    # LLM schema extraction (frozen, cached)
├── model/
│   ├── queryflow_backbone.py      # BM25+BGE query-flow features
│   └── score_intervention.py      # Trainable score intervention + all variants
└── eval/metrics.py                # Session-level nDCG, paired bootstrap CI

scripts/
├── sanity_r001_r003.py            # Pipeline sanity check
├── prefetch.py                    # Precompute backbone features
├── prepare_data.py                # Build training pairs
├── prewarm_schemas.py             # Pre-fetch LLM schemas (rate-limit-friendly)
├── train.py                       # Train any variant
├── evaluate.py                    # Evaluate + paired bootstrap CI
├── b6_counterfactual.py           # Counterfactual rank-shift inspection
└── b7_annotation.py               # Schema validation annotation interface

paper/                             # LaTeX source for the paper
paper/figures/                     # All 6 tables + hero figure PDF

results/                           # Trained models + eval JSONs (~1.5MB)
```

---

## Compute and budget

- ~3 A100-hours for full reproduction
- ~$10 LLM API cost for schema extraction (cached after first run)

---

## License

TBD (anonymous submission; license to be set after de-anonymization).