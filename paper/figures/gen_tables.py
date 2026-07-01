"""Generate Tables 1-3 as LaTeX from results/r100k_eval_v2_*.json."""
import json


def load(name):
    with open(f"results/r100k_eval_v2_{name}.json") as f:
        return json.load(f)


def fmt_ci(d):
    pb = d["paired_bootstrap_ci"]
    return f"[{pb['ci_low']:+.3f}, {pb['ci_high']:+.3f}]"


def fmt_delta(d):
    return f"{d['delta_ndcg10']:+.3f}"


# ---- Table 1: Main result (Baseline vs Pref-Rerank no_user_bias) ----
no_ub = load("no_user_bias")
t1 = r"""\begin{table}[t]
\centering
\caption{Main result: user-conditioned bilinear reranking (Pref-Rerank, \texttt{no\_user\_bias} variant) vs.\ user-agnostic baseline on 100k AOL sessions. $\Delta$ is the per-session nDCG@10 difference; CI from paired bootstrap (10k resamples).}
\label{tab:main}
\begin{tabular}{lrrrr}
\toprule
System & nDCG@10 & MRR & Recall@10 & $\Delta$ nDCG@10 (95\% CI) \\
\midrule
Baseline (user-agnostic) & """ + f"{no_ub['baseline']['ndcg10']:.3f}" + r""" & """ + f"{no_ub['baseline']['mrr']:.3f}" + r""" & """ + f"{no_ub['baseline']['recall10']:.3f}" + r""" & --- \\
\textbf{Pref-Rerank (ours)} & \textbf{""" + f"{no_ub['personalized']['ndcg10']:.3f}" + r"""} & \textbf{""" + f"{no_ub['personalized']['mrr']:.3f}" + r"""} & \textbf{""" + f"{no_ub['personalized']['recall10']:.3f}" + r"""} & """ + f"{fmt_delta(no_ub)} {fmt_ci(no_ub)}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
with open("figures/table1_main.tex", "w") as f:
    f.write(t1)
print("Saved figures/table1_main.tex")


# ---- Table 2: Mechanism ablation (full / no_user_bias / linear_only) ----
full = load("full")
lin = load("linear_only")
t2 = r"""\begin{table}[t]
\centering
\caption{Mechanism ablation: which score term drives the gain? Only the bilinear term $h^T A u$ contributes; the user-only bias $w^T u$ alone is indistinguishable from baseline. $\Delta$ vs.\ user-agnostic baseline; CI from paired bootstrap.}
\label{tab:mechanism}
\begin{tabular}{lcrr}
\toprule
Variant & Score formula & nDCG@10 & $\Delta$ (95\% CI) \\
\midrule
Baseline & $s_{base}$ & """ + f"{full['baseline']['ndcg10']:.3f}" + r""" & --- \\
\texttt{linear\_only} & $s_{base} + w^\top u$ & """ + f"{lin['personalized']['ndcg10']:.3f}" + r""" & """ + f"{fmt_delta(lin)} {fmt_ci(lin)}" + r""" \\
\texttt{full} & $s_{base} + w^\top u + h^\top A u$ & """ + f"{full['personalized']['ndcg10']:.3f}" + r""" & """ + f"{fmt_delta(full)} {fmt_ci(full)}" + r""" \\
\texttt{no\_user\_bias} & $s_{base} + h^\top A u$ & \textbf{""" + f"{no_ub['personalized']['ndcg10']:.3f}" + r"""} & \textbf{""" + f"{fmt_delta(no_ub)} {fmt_ci(no_ub)}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""
with open("figures/table2_mechanism.tex", "w") as f:
    f.write(t2)
print("Saved figures/table2_mechanism.tex")


# ---- Table 3: User-representation ablation (LLM schema / classic / random) ----
rand = load("random_schema")
classic = load("classic")
t3 = r"""\begin{table}[t]
\centering
\caption{User-representation ablation: all three constructions of $u$ yield statistically indistinguishable gains. The bilinear interaction, not the user representation, drives the improvement. LLM-extracted typed schema retains an interpretability advantage (Sec.~\ref{sec:discussion}).}
\label{tab:representation}
\begin{tabular}{llrr}
\toprule
$u$ construction & Dim & nDCG@10 & $\Delta$ (95\% CI) \\
\midrule
Baseline & --- & """ + f"{no_ub['baseline']['ndcg10']:.3f}" + r""" & --- \\
Random vector & 16 & """ + f"{rand['personalized']['ndcg10']:.3f}" + r""" & """ + f"{fmt_delta(rand)} {fmt_ci(rand)}" + r""" \\
Behavioral features & 6 $\to$ 16 & """ + f"{classic['personalized']['ndcg10']:.3f}" + r""" & """ + f"{fmt_delta(classic)} {fmt_ci(classic)}" + r""" \\
\textbf{LLM typed schema (ours)} & 16 & \textbf{""" + f"{no_ub['personalized']['ndcg10']:.3f}" + r"""} & \textbf{""" + f"{fmt_delta(no_ub)} {fmt_ci(no_ub)}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""
with open("figures/table3_representation.tex", "w") as f:
    f.write(t3)
print("Saved figures/table3_representation.tex")

print("\nAll tables generated. n_eval =", no_ub["n_eval_sessions"])
