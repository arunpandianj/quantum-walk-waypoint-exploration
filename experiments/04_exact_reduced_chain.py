# %% [markdown]
# # Exact glued-tree asymptotics from the symmetry reduction (Proposition 3)
# The goal probability of one waypoint walk on a glued tree is computed on the reduced chain of 2d+2 columns
# (off-diagonal elements sqrt(2)*gamma, and 2*gamma at the gluing link). The reduction is first checked against the full-graph
# evolution at depths 4, 6 and 14 (two random gluings each). The script then produces: the coherent probability at the best walk
# time up to depth 1000, its power-law fit, the classical bound 1/n, the ratio T*/d, and the behaviour of the reward-independent
# walk-time rule at large depth. Runtime: about one minute. Outputs: `results_reduced/` and `results_reduced.zip`.

# %% Setup
import os, math, zipfile, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import expm_multiply
warnings.filterwarnings("ignore")
G = 0.5; OUT = "results_reduced"; os.makedirs(OUT, exist_ok=True)

# %% Reduced chains
def P_quantum(d, Ts):
    off = np.full(2 * d + 1, np.sqrt(2.0)); off[d] = 2.0; w, V = eigh_tridiagonal(np.zeros(2 * d + 2), -G * off)
    amp = (V[-1][:, None] * (np.exp(-1j * np.outer(w, Ts)) * V[0][:, None])).sum(0); return np.abs(amp) ** 2
def E_dist(d, Ts):
    off = np.full(2 * d + 1, np.sqrt(2.0)); off[d] = 2.0; w, V = eigh_tridiagonal(np.zeros(2 * d + 2), -G * off)
    psi = V @ (np.exp(-1j * np.outer(w, Ts)) * V[0][:, None]); return (np.abs(psi) ** 2 * np.arange(2 * d + 2)[:, None]).sum(0)
def P_classical(d, Ts):
    n = 2 * d + 2; Q = np.zeros((n, n))
    for j in range(n - 1):
        up, dn = (2.0, 1.0) if j < d else ((2.0, 2.0) if j == d else (1.0, 2.0))
        Q[j + 1, j] += G * up; Q[j, j] -= G * up; Q[j, j + 1] += G * dn; Q[j + 1, j + 1] -= G * dn
    e0 = np.zeros(n); e0[0] = 1; return expm_multiply(sp.csr_matrix(Q), e0, start=Ts[0], stop=Ts[-1], num=len(Ts), endpoint=True)[:, -1]
def auto_T(E):
    T = int(np.argmax(E)) + 1
    for i in range(1, len(E) - 1):
        if E[i] >= E[i - 1] and E[i] > E[i + 1] and E[i] >= 0.95 * E.max(): T = i + 1; break
    return T

# %% Check against the full glued-tree graph (random gluing)
def glued_adj(d, seed=0):
    rng = np.random.default_rng(seed); m = 2 ** (d + 1) - 1; n = 2 * m; rows, cols = [], []
    for off in (0, m):
        i = np.arange(2 ** d - 1); rows += [off + i, off + i]; cols += [off + 2 * i + 1, off + 2 * i + 2]
    L = rng.permutation(np.arange(2 ** d - 1, m)); R = m + rng.permutation(np.arange(2 ** d - 1, m))
    cyc = np.empty(2 * len(L), int); cyc[0::2], cyc[1::2] = L, R; rows.append(cyc); cols.append(np.roll(cyc, -1))
    r, c = np.concatenate(rows), np.concatenate(cols); A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr(); A = ((A + A.T) > 0).astype(float); return A, 0, m
rows = []
for d in (4, 6, 14):
    for seed in (0, 1):
        A, s, g = glued_adj(d, seed); n = A.shape[0]; Ts = np.arange(1, 6 * (2 * d + 1) + 11); e0 = np.zeros(n); e0[s] = 1
        full_q = np.abs(expm_multiply(1j * G * A, e0.astype(complex), start=1, stop=int(Ts[-1]), num=len(Ts), endpoint=True)[:, g]) ** 2
        Lp = sp.diags(np.asarray(A.sum(1)).ravel()) - A; full_c = expm_multiply(-G * Lp, e0, start=1, stop=int(Ts[-1]), num=len(Ts), endpoint=True)[:, g]
        rows.append(dict(depth=d, gluing_seed=seed, nodes=n, max_abs_diff_quantum=np.abs(full_q - P_quantum(d, Ts)).max(), max_abs_diff_classical=np.abs(full_c - P_classical(d, Ts)).max()))
chk = pd.DataFrame(rows); print(chk.to_string(index=False)); chk.to_csv(f"{OUT}/table_reduction_check.csv", index=False)
assert chk.max_abs_diff_quantum.max() < 1e-8 and chk.max_abs_diff_classical.max() < 1e-8

# %% Exact probabilities up to depth 1000
rows = []
for d in [4, 6, 8, 10, 12, 14, 16, 20, 24, 30, 40, 64, 100, 160, 256, 400, 640, 1000]:
    Ts = np.arange(1, 6 * (2 * d + 1) + 11); q = P_quantum(d, Ts); Ta = auto_T(E_dist(d, Ts))
    r = dict(depth=d, T_opt=int(Ts[np.argmax(q)]), P_quantum_max=q.max(), T_auto=Ta, P_quantum_at_T_auto=q[Ta - 1])
    if d <= 30: r["P_classical_best"] = P_classical(d, Ts).max(); r["one_over_n"] = 1.0 / (2.0 ** (d + 2) - 2)
    rows.append(r)
tab = pd.DataFrame(rows); pd.set_option("display.width", 220); print(tab.round(6).to_string(index=False)); tab.to_csv(f"{OUT}/table_reduced_chain.csv", index=False)
c = tab.dropna(subset=["P_classical_best"]); print("max relative difference classical best vs 1/n:", float(np.abs(c.P_classical_best / c.one_over_n - 1).max()))

# %% Fits (power law vs exponential) and the reward-independent rule at large depth
def aicc(y, yh, k=3):
    n = len(y); rss = ((y - yh) ** 2).sum(); return n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
fit_rows = []
for dmin in (4, 16):
    a = tab[tab.depth >= dmin]; y = np.log(a.P_quantum_max); pw = np.polyfit(np.log(a.depth), y, 1); ex = np.polyfit(a.depth, y, 1)
    fit_rows.append(dict(depths=f"{dmin}-1000", points=len(a), power_exponent=pw[0], exp_rate_per_level=ex[0], dAICc_exp_minus_power=aicc(y, np.polyval(ex, a.depth)) - aicc(y, np.polyval(pw, np.log(a.depth)))))
fit = pd.DataFrame(fit_rows); print(fit.round(4).to_string(index=False)); fit.to_csv(f"{OUT}/table_fits.csv", index=False)
print("T_opt/d at d=1000:", round(tab.T_opt.iloc[-1] / 1000, 4), "| sqrt(2) =", round(math.sqrt(2), 4))
rule = []
for d in (60, 120, 480):
    Ts = np.arange(1, 6 * (2 * d + 1) + 11); q = P_quantum(d, Ts); Ta = auto_T(E_dist(d, Ts)); rule.append(dict(depth=d, T_auto=Ta, P_at_T_auto=q[Ta - 1], T_best=int(Ts[np.argmax(q)]), P_best=q.max()))
rule = pd.DataFrame(rule); print(rule.round(4).to_string(index=False)); rule.to_csv(f"{OUT}/table_rule_large_depth.csv", index=False)

# %% Package
with zipfile.ZipFile("results_reduced.zip", "w") as z:
    for f in sorted(os.listdir(OUT)): z.write(f"{OUT}/{f}")
print("saved results_reduced.zip")
