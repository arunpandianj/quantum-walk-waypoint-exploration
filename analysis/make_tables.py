"""Generate the LaTeX tables of the manuscript (steps on glued trees and hypercubes, exact probabilities, corridor/grid/room results).

Usage (from the repository root):  python analysis/make_tables.py
"""
import argparse, math, os
import pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--glued", default="results_glued"); ap.add_argument("--grid", default="results_v2")
ap.add_argument("--reduced", default="results_reduced/table_reduced_chain.csv"); ap.add_argument("--out", default="figures")
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
def sci(x, d=1):
    e = int(math.floor(math.log10(abs(x)))); return f"${x / 10 ** e:.{d}f}\\times10^{{{e}}}$"
def pfmt(p): return "$<0.001$" if p < 0.001 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}")
T = pd.read_csv(f"{args.glued}/table_main_distributional.csv"); S = pd.read_csv(f"{args.glued}/table_significance.csv")
def row(en, m): return T[(T.env == en) & (T.method == m)].iloc[0]
def table(fam, depths, label, caption, ntxt):
    L = [r"\begin{table}[!ht]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\footnotesize", r"\begin{tabular}{rrrrrrrrrrr}", r"\toprule",
         r" & & & \multicolumn{3}{c}{DQW, $p=0$} & \multicolumn{3}{c}{L\'evy waypoint} & CRW & \\", r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
         f"{ntxt} & $n$ & path & median & mean & Hit10 & median & mean & Hit10 & median & $p_{{\\max}}$ \\\\", r"\midrule"]
    for d in depths:
        en = f"{fam}-{d}"; q, l, c = row(en, "DQW p=0 (auto T)"), row(en, "Levy-waypoint"), row(en, "CRW-waypoint"); pm = S[(S.env == en) & (S.role == "walker")].p_holm.max()
        nn = {"Glued": 2 ** (d + 2) - 2, "Cube": 2 ** d}[fam]; path = {"Glued": 2 * d + 1, "Cube": d}[fam]
        L.append(f"{d} & {nn} & {path} & {q['median']:.0f} & {q['mean']:.0f} & {q.hit10:.2f} & {l['median']:.0f} & {l['mean']:.0f} & {l.hit10:.2f} & {c['median']:.0f} & {pfmt(pm)} \\\\")
    return "\n".join(L + [r"\botrule", r"\end{tabular}", r"\end{table}"])
out = [table("Glued", range(4, 10), "tab:glued", "Glued trees: steps to the first goal visit, 20 seeds.", "$d$"), table("Cube", range(4, 10), "tab:cube", "Hypercubes: steps to the first goal visit, 20 seeds.", "$n_c$")]
A = pd.read_csv(args.reduced).set_index("depth")
L = [r"\begin{table}[!ht]", r"\centering", r"\caption{Exact goal probability of one waypoint walk.}", r"\label{tab:exact}", r"\footnotesize", r"\begin{tabular}{lrrrrr}", r"\toprule", r"Graph & nodes $n$ & $T^{*}$ & coherent & classical, any $t$ & ratio \\", r"\midrule"]
for d in [4, 6, 8, 10, 12, 14, 64]:
    n = 2 ** (d + 2) - 2; pq = A.loc[d, "P_quantum_max"]; L.append(f"Glued trees, $d={d}$ & {n if d <= 14 else '$2^{%d}-2$' % (d + 2)} & {int(A.loc[d, 'T_opt'])} & {pq:.3f} & {sci(1 / n)} & {sci(pq * n)} \\\\")
pq = A.loc[1000, "P_quantum_max"]; L.append(f"Glued trees, $d=1000$ & $2^{{1002}}-2$ & {int(A.loc[1000, 'T_opt'])} & {pq:.3f} & $2^{{-1002}}$ & $2^{{{1002 + math.log2(pq):.1f}}}$ \\\\")
for nn in [8, 13, 17]:
    pq = math.sin(1.5) ** (2 * nn); pc = 2.0 ** (-nn); L.append(f"Hypercube, $n_c={nn}$ & {2 ** nn} & 3 & {pq:.3f} & {sci(pc)} & {sci(pq / pc)} \\\\")
out.append("\n".join(L + [r"\botrule", r"\end{tabular}", r"\end{table}"]))
V = pd.read_csv(f"{args.grid}/table4_verdict_vs_best_classical.csv")
L = [r"\begin{table}[!ht]", r"\centering", r"\caption{Corridors, open grids and rooms: tuned DQW versus the L\'evy waypoint.}", r"\label{tab:grids}", r"\footnotesize", r"\begin{tabular}{lrrrrrl}", r"\toprule", r"Environment & path & DQW median & L\'evy median & Cliff's $\delta$ & Holm $p$ & outcome \\", r"\midrule"]
for _, r in V.iterrows():
    dl = r["Cliff delta"]; L.append(f"{r.Env} & {r.path} & {r['median DQW']:.0f} & {r['median best classical']:.0f} & {('$-$' if dl < 0 else '') + format(abs(dl), '.2f')} & {pfmt(r['p (Holm)'])} & {'L\\\'evy faster' if r['verdict'] == 'classical faster' else 'no difference'} \\\\")
out.append("\n".join(L + [r"\botrule", r"\end{tabular}", r"\end{table}"]))
open(f"{args.out}/tables_generated.tex", "w").write("\n\n".join(out)); print("tables written to", f"{args.out}/tables_generated.tex")
