# %% [markdown]
# # Exact asymptotics: coherent vs classical hitting probabilities on glued trees and hypercubes
# No agents and no sampling: the state of a continuous-time quantum walk (H = -gamma*A) and of the classical
# continuous-time walk (generator -gamma*L) is computed exactly with sparse time evolution (`expm_multiply`),
# up to glued-tree depth 14 (65,534 nodes) and hypercube dimension 17 (131,072 nodes). This replaces the agent-based
# exponential-vs-polynomial classification, which six depths cannot support, by an exact statement.
# Outputs: `results_theory/` and `results_theory.zip`. Runtime: about a minute.

# %% Setup
import os, math, time, zipfile, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply
from scipy.sparse.csgraph import shortest_path
warnings.filterwarnings("ignore")
GAMMA = 0.5                                   # hopping rate (same as the agent notebooks; time t = number of walk steps T)
QUICK = bool(int(os.environ.get("QUICK", "0")))
GLUED = list(range(3, 9 if QUICK else 15)); CUBE = list(range(3, 11 if QUICK else 18))
OUT = "results_theory"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
def savefig(name):
    plt.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight"); plt.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); plt.show()

# %% Graphs (sparse adjacency)
def glued_adj(d, seed=0):
    rng = np.random.default_rng(seed); m = 2 ** (d + 1) - 1; n = 2 * m; rows, cols = [], []
    for off in (0, m):
        i = np.arange(2 ** d - 1); rows += [off + i, off + i]; cols += [off + 2 * i + 1, off + 2 * i + 2]
    L = rng.permutation(np.arange(2 ** d - 1, m)); R = m + rng.permutation(np.arange(2 ** d - 1, m))
    cyc = np.empty(2 * len(L), int); cyc[0::2], cyc[1::2] = L, R; rows.append(cyc); cols.append(np.roll(cyc, -1))
    r, c = np.concatenate(rows), np.concatenate(cols); A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr(); A = ((A + A.T) > 0).astype(float)
    return A, 0, m
def cube_adj(n):
    N = 2 ** n; idx = np.arange(N); rows = np.repeat(idx, n); cols = (idx[:, None] ^ (1 << np.arange(n))[None, :]).ravel()
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N)), 0, N - 1

# %% Exact evolution
def analyse(A, s, g, name, family, size):
    n = A.shape[0]; dist = shortest_path(A, unweighted=True, indices=s).ravel(); ecc = int(dist.max()); Tmax = 6 * ecc + 10; Ts = np.arange(1, Tmax + 1)
    e0 = np.zeros(n); e0[s] = 1.0
    psi = expm_multiply(1j * GAMMA * A, e0.astype(complex), start=1, stop=Tmax, num=Tmax, endpoint=True)         # exp(-iHt) with H = -gamma A
    Pq = np.abs(psi[:, g]) ** 2; Eq = (np.abs(psi) ** 2) @ dist
    Lp = sp.diags(np.asarray(A.sum(1)).ravel()) - A; pc = expm_multiply(-GAMMA * Lp, e0, start=1, stop=Tmax, num=Tmax, endpoint=True); Pc = pc[:, g]
    T_auto = int(np.argmax(Eq)) + 1
    for i in range(1, len(Eq) - 1):                                           # first local maximum of E[d(T)] within 5% of its maximum
        if Eq[i] >= Eq[i - 1] and Eq[i] > Eq[i + 1] and Eq[i] >= 0.95 * Eq.max(): T_auto = i + 1; break
    stat = float(A[g].sum() / A.sum())                                       # stationary probability at the goal (classical long-time limit)
    return dict(family=family, size=size, states=n, path=int(dist[g]), T_auto=T_auto, P_quantum_at_T_auto=Pq[T_auto - 1], P_quantum_max=Pq.max(), T_argmax_quantum=int(Ts[np.argmax(Pq)]),
                P_classical_at_T_auto=Pc[T_auto - 1], P_classical_max=Pc.max(), P_classical_stationary=stat, ratio_at_T_auto=Pq[T_auto - 1] / max(Pc[T_auto - 1], 1e-300),
                samples_to_hit_quantum=1 / Pq[T_auto - 1], samples_to_hit_classical=1 / max(Pc.max(), 1e-300))
t0 = time.time(); rows = []
for d in GLUED:
    A, s, g = glued_adj(d); rows.append(analyse(A, s, g, f"Glued-{d}", "Glued", d)); print(rows[-1]["family"], d, f"n={rows[-1]['states']}", f"Pq={rows[-1]['P_quantum_at_T_auto']:.4f}", f"Pc_max={rows[-1]['P_classical_max']:.2e}", f"{time.time()-t0:.0f}s")
for n_ in CUBE:
    A, s, g = cube_adj(n_); rows.append(analyse(A, s, g, f"Cube-{n_}", "Cube", n_)); print(rows[-1]["family"], n_, f"n={rows[-1]['states']}", f"Pq={rows[-1]['P_quantum_at_T_auto']:.4f}", f"Pc_max={rows[-1]['P_classical_max']:.2e}", f"{time.time()-t0:.0f}s")
tab = pd.DataFrame(rows); tab.to_csv(f"{OUT}/table_exact_hitting.csv", index=False)

# %% Instance variability: random gluings of the glued trees (exact, no sampling)
INST_DEPTHS = [4, 5, 6] if QUICK else [4, 6, 8, 10, 12]; N_INST = 4 if QUICK else 10; rows = []
for d in INST_DEPTHS:
    for k in range(N_INST):
        A, s, g = glued_adj(d, seed=k); r = analyse(A, s, g, f"Glued-{d}@{k}", "Glued", d); r["instance"] = k; rows.append(r)
inst = pd.DataFrame(rows); inst.to_csv(f"{OUT}/table_gluing_instances_raw.csv", index=False)
summ = inst.groupby("size").agg(states=("states", "first"), T_auto_min=("T_auto", "min"), T_auto_max=("T_auto", "max"), Pq_mean=("P_quantum_at_T_auto", "mean"), Pq_min=("P_quantum_at_T_auto", "min"),
                                Pq_max=("P_quantum_at_T_auto", "max"), Pc_best_mean=("P_classical_max", "mean"), Pc_best_min=("P_classical_max", "min"), Pc_best_max=("P_classical_max", "max")).reset_index()
print(summ.round(6).to_string(index=False)); summ.to_csv(f"{OUT}/table_gluing_instances_summary.csv", index=False)
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.errorbar(summ["size"], summ.Pq_mean, [summ.Pq_mean - summ.Pq_min, summ.Pq_max - summ.Pq_mean], fmt="o-", capsize=3, label="coherent walk (min-max over instances)")
ax.errorbar(summ["size"], summ.Pc_best_mean, [summ.Pc_best_mean - summ.Pc_best_min, summ.Pc_best_max - summ.Pc_best_mean], fmt="s--", capsize=3, label="best classical walk")
ax.set_yscale("log"); ax.set_xlabel("depth"); ax.set_ylabel("goal probability"); ax.legend(fontsize=7); savefig("fig_gluing_instance_variability")

# %% Closed-form check on hypercubes: P_q(t) = sin^{2n}(gamma t), P_c(t) = ((1 - exp(-2 gamma t)) / 2)^n
chk = []
for n_ in CUBE:
    r = tab[(tab.family == "Cube") & (tab["size"] == n_)].iloc[0]; T = int(r.T_auto)
    chk.append(dict(dim=n_, numeric_quantum=r.P_quantum_at_T_auto, closed_form_quantum=math.sin(GAMMA * T) ** (2 * n_), numeric_classical=r.P_classical_at_T_auto, closed_form_classical=((1 - math.exp(-2 * GAMMA * T)) / 2) ** n_))
chk = pd.DataFrame(chk); chk["max_rel_err"] = np.maximum(abs(chk.numeric_quantum / chk.closed_form_quantum - 1), abs(chk.numeric_classical / chk.closed_form_classical - 1)); print(chk.round(6).to_string(index=False))
assert chk.max_rel_err.max() < 1e-3; chk.to_csv(f"{OUT}/table_cube_closed_form_check.csv", index=False)

# %% Model comparison on exact quantities (11-16 points: discriminating)
def aicc(y, yhat, k=3):
    n = len(y); rss = max(((y - yhat) ** 2).sum(), 1e-24); return n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)
def compare(x, y, label, fam):
    x = np.asarray(x, float); y = np.log(np.asarray(y, float)); pe = np.polyfit(x, y, 1); pp = np.polyfit(np.log(x), y, 1)
    ae, ap = aicc(y, np.polyval(pe, x)), aicc(y, np.polyval(pp, np.log(x)))
    return dict(family=fam, quantity=label, exp_rate_per_level=pe[0], power_exponent=pp[0], AICc_exp=ae, AICc_power=ap, dAICc_exp_minus_power=ae - ap, preferred="power-law" if ae - ap > 2 else ("exponential" if ap - ae > 2 else "indistinguishable"))
fits = []
for fam, dmin in [("Glued", 5), ("Cube", 5)]:
    s_ = tab[(tab.family == fam) & (tab["size"] >= dmin)]
    fits += [compare(s_["size"], s_.samples_to_hit_quantum, "1 / P_quantum(T_auto)  (expected waypoint samples)", fam), compare(s_["size"], s_.samples_to_hit_classical, "1 / max_t P_classical(t)", fam)]
fits = pd.DataFrame(fits); pd.set_option("display.width", 220, "display.max_colwidth", 60); print(fits.round(3).to_string(index=False)); fits.to_csv(f"{OUT}/table_exact_model_comparison.csv", index=False)

# %% Figures
fig, axs = plt.subplots(1, 3, figsize=(13, 3.7))
for fam, mk in [("Glued", "o"), ("Cube", "s")]:
    s_ = tab[tab.family == fam]; axs[0].semilogy(s_["size"], s_.P_quantum_at_T_auto, mk + "-", label=f"{fam}: coherent walk"); axs[0].semilogy(s_["size"], s_.P_classical_max, mk + "--", label=f"{fam}: best classical walk")
    axs[1].semilogy(s_["size"], s_.samples_to_hit_quantum, mk + "-", label=f"{fam}: coherent"); axs[1].semilogy(s_["size"], s_.samples_to_hit_classical, mk + "--", label=f"{fam}: classical")
    axs[2].plot(s_["size"], s_.T_auto, mk + "-", label=fam)
axs[0].set_ylabel("goal probability of one waypoint walk"); axs[1].set_ylabel("expected waypoint samples to reach the goal"); axs[2].set_ylabel("reward-free walk time T_auto")
for a in axs: a.set_xlabel("depth / dimension")
axs[0].legend(fontsize=6); axs[1].legend(fontsize=6); axs[2].legend(); savefig("fig_exact_asymptotics")

# %% Package
with zipfile.ZipFile("results_theory.zip", "w") as z:
    for f in sorted(os.listdir(OUT)): z.write(f"{OUT}/{f}")
print("saved results_theory.zip,", len(os.listdir(OUT)), "files")
