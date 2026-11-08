# Personalization baselines

Four published personalization baselines for the click-log ranking
stage, run against the same 201-session test set as `\textsc{Pref-Rerank}`.

| ID | Source | Citation | CCF venue |
|---|---|---|---|
| B1 `popularity` | craswell2007 | "Random Walks on the Click Graph" | SIGIR 2007 (A) |
| B2 `hist-lex` | craswell2008 | "An Experimental Comparison of Click Position-Bias Models" | WSDM 2008 (A) |
| B3 `pclick` | joachims2017 | "Unbiased Learning-to-Rank with Biased Feedback" | WSDM 2017 (A) |
| B4 `hrm` | bennett2012 | "Modeling the impact of short- and long-term behavior on search personalization" | SIGIR 2012 (A) |

All four are **non-SOTA** by design: they are widely cited personalization
methods chosen because each one isolates a single design choice that
`\textsc{Pref-Rerank}` subsumes. None of them is presented as a current
state-of-the-art system; the role of these baselines is to test whether
the additional 12{,}288 bilinear parameters deliver any observable
improvement beyond the published click-log personalization literature.

## Layout

```
b1_popularity.py     # zero-training global click-count prior
b2_hist_lex.py       # zero-training query-history lexicalization
b3_pclick.py         # learned per-(query, URL) bias (Joachims 2017 style)
b4_hrm.py            # learned per-history-query weight (Bennett 2012)
run_all_baselines.sh # one-shot driver
```

## Run

```bash
bash scripts/baselines/run_all_baselines.sh \
    --session-cache cache/aol_sessions.json \
    --feature-cache cache/features_queryflow
```

Each script prints a JSON line with the per-session nDCG@10 of the
baseline (alpha=0, BM25 only) and the personalized version, plus the
paired bootstrap 95% CI on the difference.

## Output

JSON results land in `results/baselines/`. Copy the four numbers into
`paper_repo/tables/table5_baselines.tex` to fill the previously
`pending` rows.

## Trainable parameters

| System | Trainable params |
|---|---|
| B1 `popularity` | 1 (alpha) |
| B2 `hist-lex` | 0 |
| B3 `pclick` | O(\|V_Q\| * \|V_U\|) queries x URLs seen in training |
| B4 `hrm` | \|V_H\| history-query vocabulary |
| Pref-Rerank | 12{,}288 bilinear matrix A |