"""Generate the seven figures of the manuscript from the result files of the experiment scripts.

Usage (from the repository root, after running experiments 01, 02, 04 and 05):
    python analysis/make_figures.py
Optional arguments set the input folders and the output folder.
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ap = argparse.ArgumentParser()
ap.add_argument("--glued", default="results_glued", help="output folder of experiments/01_glued_trees_hypercubes.py")
ap.add_argument("--grid", default="results_v2", help="output folder of experiments/02_corridors_grids_rooms.py")
ap.add_argument("--reduced", default="results_reduced/table_reduced_chain.csv", help="table written by experiments/04_exact_reduced_chain.py")
ap.add_argument("--spread", default="results_spreading/spread.npz", help="variance curves written by experiments/05_spreading_law.py")
ap.add_argument("--out", default="figures", help="folder for the figures (PDF and PNG)")
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

W5 = 5.4                                       # figure width in inches, close to a single-column text width
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9.5, "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "lines.linewidth": 1.5, "lines.markersize": 4.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans", "axes.linewidth": 0.9})
CL = {"DQW": "#D55E00", "LEVY": "#0072B2", "CRW": "#009E73", "NBRW": "#56B4E9", "Uniform": "#E69F00", "eps": "#666666", "Far": "#000000"}
M = {"eps-greedy": ("epsilon-greedy", CL["eps"], "x", "-"), "Uniform-waypoint": ("Uniform waypoint", CL["Uniform"], "D", "-"), "CRW-waypoint": ("Lazy random walk", CL["CRW"], "^", "-"),
     "NBRW-waypoint": ("Non-backtracking walk", CL["NBRW"], "v", "-"), "Levy-waypoint": ("L\u00e9vy waypoint", CL["LEVY"], "s", "-"),
     "DQW p=0 (auto T)": ("DQW, $p=0$, automatic $T$", CL["DQW"], "o", "-"), "Farthest-first (reference)": ("Farthest-first reference", CL["Far"], None, "--")}
ORDER = ["eps-greedy", "Uniform-waypoint", "CRW-waypoint", "NBRW-waypoint", "Levy-waypoint", "Farthest-first (reference)", "DQW p=0 (auto T)"]
def let(ax, s, dx=-0.13, dy=1.04): ax.text(dx, dy, s, transform=ax.transAxes, fontsize=11, fontweight="bold")
def save(fig, name): fig.savefig(f"{args.out}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{args.out}/{name}.png", dpi=250, bbox_inches="tight"); plt.close(fig)
GL = lambda f: pd.read_csv(f"{args.glued}/{f}"); GR = lambda f: pd.read_csv(f"{args.grid}/{f}")

# ---------------- Fig 1: method schematic
fig, ax = plt.subplots(figsize=(W5, 3.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 78); ax.axis("off")
def box(x, y, w, h, txt, fc="#FFFFFF", fs=7.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec="#333333", lw=0.9)); ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, linespacing=1.45)
def arrow(pts, label=None, lx=0, ly=0):
    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]): ax.plot([x1, x2], [y1, y2], color="#333333", lw=1.0)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=9, lw=1.0, color="#333333"))
    if label: ax.text(lx, ly, label, ha="center", va="center", fontsize=7.6)
box(1, 52, 26, 24, "Known graph\n$G=(V,E)$,\nstart $s_0$,\nhidden goal $g$"); box(38, 52, 26, 24, "Decision at\nstate $s$")
box(75, 52, 24, 24, "Dephased walk,\n$T$ steps,\nmeasured node\nis the waypoint $w$", "#FFF1E6"); box(75, 20, 24, 24, "Shortest path\nto $w$, then\n$B=3$ random\nactions")
box(38, 20, 26, 24, "Greedy action\non $Q(s,\\cdot)$"); box(1, 20, 26, 24, "Environment step,\nreward $+1$ at $g$,\nQ-learning\nupdate")
arrow([(27.4, 64), (37.6, 64)]); arrow([(64.4, 64), (74.6, 64)], "$\\epsilon_e$", 69.5, 67.5); arrow([(51, 51.6), (51, 44.4)], "otherwise", 60.5, 48)
arrow([(87, 51.6), (87, 44.4)]); arrow([(37.6, 32), (27.4, 32)]); arrow([(87, 19.6), (87, 10), (14, 10), (14, 19.6)])
ax.text(50, 3.2, "Sampler step: $\\rho\\mapsto(1-p)\\,U\\rho U^{\\dagger}+p\\,\\Delta(U\\rho U^{\\dagger})$;  $p=0$ coherent,  $p=1$ classical", ha="center", va="center", fontsize=7.8)
save(fig, "fig1_method")

# ---------------- Fig 2: spreading law
S = np.load(args.spread); G = 0.5; t = np.arange(1, 129); ps = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
fig = plt.figure(figsize=(W5, 5.6)); gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.42, wspace=0.38)
a = fig.add_subplot(gs[0, :]); cm = plt.cm.viridis(np.linspace(0, 0.9, 7))
for c, p in zip(cm, [0] + ps): a.loglog(t, S[f"p{p}"], color=c, label=f"$p={p:g}$")
a.loglog(t, 0.25 * t ** 2, "k--", lw=1.0, label="$\\propto t^2$ (ballistic)"); a.loglog(t, 0.6 * t, "k:", lw=1.2, label="$\\propto t$ (diffusive)")
a.set_xlabel("step $t$"); a.set_ylabel("position variance $\\sigma^2(t)$"); a.legend(ncol=3, loc="upper left", frameon=False, columnspacing=1.0, handlelength=1.6); let(a, "a", -0.07); a.set_ylim(0.3, 3e4)
cross, D = [], []
for p in ps:
    v = S[f"p{p}"]; sl = np.gradient(np.log(v + 1e-12), np.log(t)); idx = np.flatnonzero((sl < 1.5) & (t >= 2)); cross.append(t[idx[0]] if len(idx) else np.nan); D.append(np.polyfit(t[64:], v[64:], 1)[0] / 2)
cross, D = np.array(cross), np.array(D)
b = fig.add_subplot(gs[1, 0]); b.loglog(1 / np.array(ps), cross, "o", color=CL["DQW"], label="measured"); xx = np.logspace(0, 1.8, 20); b.loglog(xx, 2.1 * xx, "k--", lw=1.0, label="$t^*=2.1/p$")
b.set_xlabel("$1/p$"); b.set_ylabel("crossover step $t^*$"); b.legend(frameon=False, loc="upper left"); let(b, "b")
c = fig.add_subplot(gs[1, 1]); c.loglog(ps, D, "o", color=CL["DQW"], label="measured"); pp = np.logspace(-1.8, 0, 20); c.loglog(pp, G ** 2 * (2 - pp) / pp, "k--", lw=1.0, label="$\\gamma^2(2-p)/p$")
c.set_xlabel("dephasing probability $p$"); c.set_ylabel("diffusion constant $D(p)$"); c.legend(frameon=False, loc="upper right"); let(c, "c")
save(fig, "fig2_spreading")

# ---------------- Fig 3: exact asymptotics
A = pd.read_csv(args.reduced); fig, axs = plt.subplots(2, 2, figsize=(W5, 5.2)); fig.subplots_adjust(hspace=0.42, wspace=0.38)
a = A[A.depth <= 64]; ax = axs[0, 0]
ax.semilogy(a.depth, a.P_quantum_max, "o-", color=CL["DQW"], label="coherent walk"); ax.semilogy(a.depth, 1 / (2.0 ** (a.depth + 2) - 2), "s-", color=CL["LEVY"], label="classical, bound $1/n$")
ax.set_xlabel("tree depth $d$"); ax.set_ylabel("goal probability"); ax.legend(frameon=False, loc="lower left"); ax.set_ylim(1e-32, 5); ax.set_yticks([1e-30, 1e-20, 1e-10, 1]); let(ax, "a")
ax = axs[0, 1]; x = np.log(A.depth); y = np.log(A.P_quantum_max); k = np.polyfit(x, y, 1); ax.loglog(A.depth, A.P_quantum_max, "o", color=CL["DQW"], label="exact values"); dd = np.logspace(0.6, 3, 20)
ax.loglog(dd, np.exp(k[1]) * dd ** k[0], "k--", lw=1.0, label=f"fit $d^{{{k[0]:.2f}}}$"); ax.set_xlabel("tree depth $d$"); ax.set_ylabel("coherent goal probability"); ax.legend(frameon=False, loc="upper right"); let(ax, "b")
ax = axs[1, 0]; ax.semilogx(A.depth, A.T_opt / A.depth, "o-", color=CL["DQW"], label="best walk time $T^*/d$"); ax.axhline(np.sqrt(2), color="k", ls="--", lw=1.0, label="limit $\\sqrt{2}$")
ax.set_xlabel("tree depth $d$"); ax.set_ylabel("$T^*/d$"); ax.legend(frameon=False, loc="upper right"); let(ax, "c")
ax = axs[1, 1]; n = np.arange(3, 18); ax.semilogy(n, np.sin(G * 3) ** (2 * n), "o-", color=CL["DQW"], label="coherent, $T=3$"); ax.semilogy(n, 2.0 ** (-n.astype(float)), "s-", color=CL["LEVY"], label="classical, bound $2^{-n}$")
ax.set_xlabel("hypercube dimension $n$"); ax.set_ylabel("antipode probability"); ax.legend(frameon=False, loc="lower left"); ax.set_ylim(1e-10, 5); ax.set_yticks([1e-8, 1e-5, 1e-2, 1]); let(ax, "d")
save(fig, "fig3_asymptotics")

# ---------------- Fig 4: agent step counts
T = GL("table_main_distributional.csv"); fig, axs = plt.subplots(2, 2, figsize=(W5, 5.6)); fig.subplots_adjust(hspace=0.42, wspace=0.38, bottom=0.2)
for j, fam in enumerate(["Glued", "Cube"]):
    for i, (stat, lo, hi, yl) in enumerate([("median", "med_lo", "med_hi", "median steps to first goal"), ("mean", "mean_lo", "mean_hi", "mean steps to first goal")]):
        ax = axs[i, j]
        for mn in ORDER:
            s = T[(T.method == mn) & T.env.str.startswith(fam)].copy(); s["d"] = s.env.str.split("-").str[1].astype(int); s = s[s.d >= 4].sort_values("d"); lab, col, mk, ls = M[mn]
            ax.plot(s.d, s[stat], ls, color=col, marker=mk, label=lab, lw=1.7 if mn.startswith("DQW") else 1.2, zorder=6 if mn.startswith("Far") else (5 if mn.startswith("DQW") else 3))
            if mn == "DQW p=0 (auto T)": ax.fill_between(s.d, s[lo], s[hi], color=col, alpha=.18, lw=0)
        ax.set_yscale("log"); ax.set_xlabel("tree depth $d$" if fam == "Glued" else "hypercube dimension $n$"); ax.set_ylabel(yl); let(ax, ["a", "c", "b", "d"][2 * j + i])
h, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l_, loc="lower center", ncol=3, frameon=False, fontsize=8.2, columnspacing=1.2)
save(fig, "fig4_agent_steps")

# ---------------- Fig 5: empirical distributions
R = GL("raw_per_run.csv"); sel = ["Glued-7", "Glued-9", "Cube-7", "Cube-9"]; fig, axs = plt.subplots(2, 2, figsize=(W5, 5.3), sharey=True); fig.subplots_adjust(hspace=0.4, wspace=0.12, bottom=0.2)
for ax, en, l in zip(axs.ravel(), sel, "abcd"):
    cap = R[R.env == en].cap.iloc[0]
    for mn in ORDER:
        v = np.sort(R[(R.env == en) & (R.method == mn)].first_goal.values); lab, col, mk, ls = M[mn]
        ax.step(np.r_[1, v, cap], np.r_[0, np.arange(1, len(v) + 1) / len(v), 1], where="post", color=col, ls=ls, label=lab, lw=1.8 if mn.startswith("DQW") else 1.2)
    ax.set_xscale("log"); ax.set_xlim(3, 1e5); ax.set_title(en.replace("Glued-", "Glued trees, $d=$").replace("Cube-", "Hypercube, $n=$"), fontsize=9); let(ax, l, -0.12)
for ax in axs[1]: ax.set_xlabel("environment steps")
for ax in axs[:, 0]: ax.set_ylabel("fraction found")
h, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l_, loc="lower center", ncol=3, frameon=False, fontsize=8.2, columnspacing=1.2)
save(fig, "fig5_ecdf")

# ---------------- Fig 6: walk-duration sensitivity and dephasing tolerance
Wd = GL("table_walk_time_sensitivity.csv"); N = GL("table_noise_budget.csv"); H = GL("table_theory_hitting.csv")
fig, axs = plt.subplots(2, 2, figsize=(W5, 5.6)); fig.subplots_adjust(hspace=0.45, wspace=0.38, bottom=0.2)
for ax, en, l in zip(axs[0], ["Glued-6", "Cube-8"], "ab"):
    for nm, col, lab in [("DQW p=0", CL["DQW"], "DQW, $p=0$"), ("CRW", CL["CRW"], "Lazy random walk"), ("NBRW", CL["NBRW"], "Non-backtracking walk")]:
        s = Wd[(Wd.env == en) & (Wd.method == nm)].sort_values("multiplier"); ax.plot(s.multiplier, s.median_steps, "o-", color=col, label=lab, lw=1.8 if nm.startswith("DQW") else 1.2)
    sc = 6 if en == "Glued-6" else 8; ta = H[H.env == en].T_auto.iloc[0]; ax.axvline(ta / sc, color="k", ls=":", lw=1.2, label="automatic $T$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([0.5, 1, 2, 3, 5, 8]); ax.set_xticklabels(["0.5", "1", "2", "3", "5", "8"]); ax.minorticks_off()
    ax.set_yticks([5, 10, 20, 50, 100, 200, 500]); ax.set_yticklabels(["5", "10", "20", "50", "100", "200", "500"]); ax.set_ylim(5, 900)
    ax.set_xlabel("walk time divided by $d$" if en == "Glued-6" else "walk time divided by $n$"); ax.set_ylabel("median steps to first goal"); ax.set_title(en.replace("Glued-", "Glued trees, $d=$").replace("Cube-", "Hypercube, $n=$"), fontsize=9); let(ax, l)
cols = {"Glued-5": "#D55E00", "Glued-6": "#E69F00", "Cube-6": "#0072B2", "Cube-8": "#009E73"}; mk = {"Glued-5": "o", "Glued-6": "s", "Cube-6": "^", "Cube-8": "D"}
for en in N.env.unique():
    s = N[N.env == en].sort_values("p"); lab = en.replace("Glued-", "Glued $d=$").replace("Cube-", "Cube $n=$")
    axs[1, 0].errorbar(s.p.replace(0, 0.001), s.hit10, [s.hit10 - s.hit10_lo, s.hit10_hi - s.hit10], fmt=mk[en] + "-", color=cols[en], capsize=2, label=lab, lw=1.2, elinewidth=0.8)
    axs[1, 1].errorbar(s.pT.replace(0, 0.003), s.hit10, [s.hit10 - s.hit10_lo, s.hit10_hi - s.hit10], fmt=mk[en] + "-", color=cols[en], capsize=2, lw=1.2, elinewidth=0.8)
for ax, xl, l in zip(axs[1], ["dephasing probability $p$", "expected measurements per walk $pT$"], "cd"):
    ax.set_xscale("log"); ax.set_xlabel(xl); ax.set_ylabel("Hit10"); ax.set_ylim(0, 1.05); let(ax, l)
h, l_ = axs[0, 0].get_legend_handles_labels(); h2, l2 = axs[1, 0].get_legend_handles_labels(); fig.legend(h + h2, l_ + l2, loc="lower center", ncol=4, frameon=False, fontsize=8.0, columnspacing=1.0)
save(fig, "fig6_sensitivity_noise")

# ---------------- Fig 7: corridors, open grids, rooms
V = GR("raw_per_run_metrics.csv"); Sg = GR("table3_significance.csv")
fig, axs = plt.subplots(2, 1, figsize=(W5, 6.4), gridspec_kw={"height_ratios": [1, 1.05], "hspace": 0.42}); path = {"Corridor-S": 18, "Corridor-M": 31, "Corridor-L": 50, "Corridor-XL": 73, "Corridor-XXL": 112}
for mn, lab, col, mk in [("Levy-waypoint", "L\u00e9vy waypoint", CL["LEVY"], "s"), ("Uniform-waypoint", "Uniform waypoint", CL["Uniform"], "D"), ("NBRW-waypoint", "Non-backtracking walk", CL["NBRW"], "v"),
                         ("CRW-waypoint", "Lazy random walk", CL["CRW"], "^"), ("DQW (tuned p,T)", "DQW, tuned $p$ and $T$", CL["DQW"], "o")]:
    ys = [V[(V.env == e) & (V.method == mn)].first_goal.median() for e in path]; axs[0].loglog([path[e] for e in path], ys, "-", color=col, marker=mk, label=lab, lw=1.8 if mn.startswith("DQW") else 1.3)
axs[0].set_xticks([18, 31, 50, 73, 112]); axs[0].set_xticklabels(["18", "31", "50", "73", "112"]); axs[0].minorticks_off()
axs[0].set_xlabel("shortest-path length $\\ell$"); axs[0].set_ylabel("median steps to first goal"); axs[0].legend(frameon=False, loc="upper left"); axs[0].set_title("Serpentine corridors", fontsize=9.5); let(axs[0], "a", -0.1)
rows = Sg[Sg.vs == "Levy-waypoint"].reset_index(drop=True); assert len(rows) == 9, len(rows); ax = axs[1]
for i, r in rows.iterrows():
    dlt = r["Cliff delta (+ = DQW faster)"]; ax.errorbar(dlt, i, xerr=[[dlt - r.delta_lo], [r.delta_hi - dlt]], fmt="o", color=CL["DQW"], capsize=3, lw=1.3, markersize=6)
ax.axvline(0, color="k", lw=1.0); ax.set_yticks(range(len(rows))); ax.set_yticklabels(list(rows.Env)); ax.invert_yaxis(); ax.set_xlim(-1.08, 0.4)
ax.set_xlabel("Cliff's $\\delta$ of DQW against the L\u00e9vy waypoint (negative: L\u00e9vy faster)"); ax.set_title("Effect size with 95% bootstrap interval", fontsize=9.5); let(ax, "b", -0.22)
save(fig, "fig7_grids_corridors_rooms"); print("figures written to", args.out)
