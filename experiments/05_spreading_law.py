# %% [markdown]
# # Spreading law of the dephased walk on a line (variance, crossover step, diffusion constant)
# Density-matrix simulation of rho -> (1-p) U rho U^+ + p diag(U rho U^+) on a line of 281 sites, 128 steps.
# Outputs: `results_spreading/spread.npz` (variance curves) and `results_spreading/table_spreading.csv`.
# Compares the fitted diffusion constant with the closed form D(p) = gamma^2 (2 - p) / p. Runtime: about two minutes.

# %% Run
import os, numpy as np, pandas as pd
G, R, T = 0.5, 140, 128; n = 2 * R + 1; OUT = "results_spreading"; os.makedirs(OUT, exist_ok=True)
A = np.zeros((n, n))
for i in range(n - 1): A[i, i + 1] = A[i + 1, i] = 1
lam, V = np.linalg.eigh(G * (np.diag(A.sum(1)) - A)); U = (V * np.exp(-1j * lam)) @ V.T; Ud = U.conj().T
x = np.arange(n) - R; out = {}
for p in [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    rho = np.zeros((n, n), complex); rho[R, R] = 1; var = []
    for t in range(T):
        rho = U @ rho @ Ud; d = np.diag(rho).copy(); rho *= (1 - p); rho[np.arange(n), np.arange(n)] += p * d
        pr = np.diag(rho).real; var.append((pr * x ** 2).sum() - ((pr * x).sum()) ** 2)
    out[f"p{p}"] = np.array(var); print("p =", p, flush=True)
np.savez(f"{OUT}/spread.npz", **out)
t = np.arange(1, T + 1); rows = []
for p in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    v = out[f"p{p}"]; sl = np.gradient(np.log(v + 1e-12), np.log(t)); idx = np.flatnonzero((sl < 1.5) & (t >= 2))
    rows.append(dict(p=p, crossover_step=int(t[idx[0]]) if len(idx) else np.nan, D_measured=np.polyfit(t[64:], v[64:], 1)[0] / 2, D_closed_form=G ** 2 * (2 - p) / p))
tab = pd.DataFrame(rows); print(tab.round(4).to_string(index=False)); tab.to_csv(f"{OUT}/table_spreading.csv", index=False)
