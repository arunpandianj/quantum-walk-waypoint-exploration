# %% [markdown]
# # Decoherence-Controlled Quantum-Walk Exploration, v2 (journal-grade protocol)
# **Question.** In goal search with a hidden sparse reward, does *coherence* (dephasing probability p < 1) help
# beyond what a classical walker with the same spreading can do?
#
# **What changed vs v1.** (1) Every waypoint method now gets the same map knowledge and the same machinery; we add
# classical walkers that reproduce ballistic spreading (non-backtracking walk, Levy waypoints) and a
# **distance-matched classical control** that copies the quantum walk's waypoint-distance distribution but destroys
# its coherent structure. (2) Scaling families (corridor length up to ~73, open grids, large rooms) plus glued
# trees, the graph family with a *proven* exponential quantum-walk speedup. (3) Cost is counted in environment
# steps, with right-censoring, bootstrap CIs, Holm-corrected tests, Cliff's delta, mean ranks and fitted scaling
# exponents. (4) Ablation with a *learned* map (no privileged topology) and a no-map baseline group.
# (5) Matching hyper-parameter tuning for every method on separate seeds.
#
# **Setting (state it in the paper).** Known map, hidden reward: the agent knows the state graph but not where the
# reward is. This is goal search / exploration with a map, not map-free RL. The learned-map ablation shows what
# happens without the map. The quantum walk is simulated classically (density matrix and single-shot
# trajectories); claims are about exploration behaviour, not computational speed-up.
#
# Set `QUICK=1` for a smoke test. Outputs: `results_v2/` and `results_v2.zip`.

# %% [markdown]
# ## Pre-registered hypotheses and decision rules (fix BEFORE looking at results)
# * **H1 (theory-anchored regime, glued trees, known map).** The coherent walk (p=0, tuned walk time) has the lowest
#   median steps-to-first-goal at every depth >= 5 (Holm p < 0.05 vs every other method) and the smallest growth rate
#   of log(steps) with depth (its bootstrap CI lies below every classical walker's CI).
# * **H2 (the dial).** (a) The dephasing crossover time scales as 1/p (fitted log-log slope in [0.8, 1.2]);
#   (b) on glued trees a small dephasing (p = 0.05) is significantly slower than p = 0.
# * **H3 (coherence beyond spreading, corridors / open / rooms).** Per environment, DQW(tuned) vs the distance-matched
#   classical control: Holm p < 0.05 with Cliff delta > 0 means "coherence helps beyond spreading"; otherwise report parity
#   (or the control being faster). No claim is made beyond what each environment shows.
# The notebook prints a `table_hypotheses` summary with the status of each rule.

# %% [markdown]
# ## Protocol v2.1: changes made AFTER inspecting the first full run (disclose these in the paper)
# The first full run (v2) met H2 but not H1, and showed three protocol weaknesses, now fixed *for all methods alike*:
# (i) tuning used 3-5 seeds and the mean of heavy-tailed step counts, which picked an outlying walk time at glued-tree depth 7;
# (ii) several tuned values sat on the edge of their grids (walk time 24, Levy alpha 1.2);
# (iii) glued trees stopped at depth 7, too shallow to separate polynomial from exponential growth.
# v2.1 therefore uses 8 tuning seeds (6 for glued trees), the mean of log(steps) as the tuning criterion, wider grids,
# glued-tree depths up to 9 (the fully coherent walk is sampled from a cached eigendecomposition, exact and fast),
# a walk-time sensitivity experiment, and a censoring flag on corridor scaling exponents. Report the v2 outcome of H1
# as a preliminary run in the supplement, and the v2.1 outcome as the main result with this deviation stated.

# %% [markdown]
# ## Protocol v2.2: wider tuning grids
# In the v2.1 run several tuned values sat at the edge of their grids: the walk time T=64 for the lazy random walk (all nine
# environments) and for the dephased sampler (five), and the Levy exponent 0.6 (eight environments). v2.2 extends the grids
# to T up to 128 and the Levy exponent down to 0.3, for all methods alike, and nothing else changes. Re-run, then compare the
# tuned-value table, the verdict table and the scaling exponents with v2.1. If a tuned value is still at an edge, extend again.

# %% Setup
import time as _time; T_START = _time.time()
import os, math, time, warnings, zipfile, types
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse.csgraph import shortest_path
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")

QUICK   = bool(int(os.environ.get("QUICK", "0")))
N_SEEDS = 4 if QUICK else 20       # evaluation seeds (main comparison)
N_TUNE  = 2 if QUICK else 8        # separate tuning seeds (per configuration)
N_SIDE  = 3 if QUICK else 20       # seeds for sweeps / ablations / glued trees
N_EP    = 60 if QUICK else 500     # training episodes per run
GAMMA_RL, ALPHA, GAMMA_H, BURST, EVAL_EVERY, EVAL_ROLL = 0.99, 0.5, 0.5, 3, 15, 3
ENVS = ["Corridor-S", "Corridor-M", "Open-12", "Rooms-S"] if QUICK else \
       ["Corridor-S", "Corridor-M", "Corridor-L", "Corridor-XL", "Corridor-XXL", "Open-12", "Open-20", "Rooms-S", "Rooms-L"]
CORR_ENVS = [e for e in ENVS if e.startswith("Corridor")]
SWEEP_ENVS = ["Corridor-M", "Open-12", "Rooms-S"]
GLUED_DEPTHS = [3, 4] if QUICK else [4, 5, 6, 7, 8, 9]
GLUED_TRAJ_MAX = 7                  # coherent p>0 trajectories only up to this depth (cost O(T n^2) per waypoint)
N_GT = 3 if QUICK else 20          # seeds for glued-tree agents (dense n up to 510)
P_SWEEP = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
GRID = dict(dqw_p=[0, 0.05, 0.2, 0.5, 1.0], T=[8, 24, 64, 128], levy=[0.3, 0.45, 0.6, 0.9, 1.2, 1.6, 2.2], count=[0.02, 0.05, 0.2])   # v2.2: grids extended beyond the former edges (T=64, alpha=0.6)
OUT = "results_v2"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
def savefig(name):
    plt.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight"); plt.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); plt.show()

# %% Environments: scalable families on an explicit state graph
class TabEnv:
    def __init__(self, name, NB, s0, g, H, inv=None):
        self.name, self.NB, self.s0, self.g, self.H, self.inv = name, NB, s0, g, H, inv
        n, A = NB.shape; self.n, self.A = n, A
        adj = np.zeros((n, n), bool); act = -np.ones((n, n), int)
        for s in range(n):
            for a in range(A):
                s2 = NB[s, a]
                if s2 != s:
                    adj[s, s2] = adj[s2, s] = True
                    if act[s, s2] < 0: act[s, s2] = a
        self.adj, self.act_to = adj, act
        self.D = shortest_path(adj.astype(float), unweighted=True)
        assert np.isfinite(self.D).all(), "graph must be connected"
        self.path_len = int(self.D[s0, g])

def grid_env(name, rows):
    cells = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch != "#": cells[(r, c)] = len(cells)
            if ch == "S": st = (r, c)
            if ch == "G": gl = (r, c)
    NB = np.zeros((len(cells), 4), int)
    for (r, c), i in cells.items():
        for a, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):   # inverse of a is a^1
            NB[i, a] = cells.get((r + dr, c + dc), i)
    e = TabEnv(name, NB, cells[st], cells[gl], 100, inv=lambda a: a ^ 1)
    e.H = max(100, 4 * e.path_len); e.pos = np.array(list(cells.keys())); e.shape = (len(rows), len(rows[0])); return e

def serpentine(seg, nseg):
    W = seg + 2; rows = ["#" * W]
    for i in range(nseg):
        row = ["#"] + ["."] * seg + ["#"]
        if i == 0: row[1] = "S"
        if i == nseg - 1: row[seg if i % 2 == 0 else 1] = "G"
        rows.append("".join(row))
        if i < nseg - 1:
            r = ["#"] * W; r[seg if i % 2 == 0 else 1] = "."; rows.append("".join(r))
    return rows + ["#" * W]

def four_rooms(r):
    N = 2 * r + 3; g = [["."] * N for _ in range(N)]
    for i in range(N):
        g[0][i] = g[N - 1][i] = g[i][0] = g[i][N - 1] = "#"; g[r + 1][i] = "#"; g[i][r + 1] = "#"
    for rr, cc in [(1 + r // 2, r + 1), (r + 2 + r // 2, r + 1), (r + 1, 1 + r // 2), (r + 1, r + 2 + r // 2)]: g[rr][cc] = "."
    g[1][1] = "S"; g[N - 2][N - 2] = "G"; return ["".join(x) for x in g]

def glued_trees(d, seed=0):
    """Two binary trees of depth d whose leaves are joined by a random alternating cycle (Childs et al. 2003)."""
    rng = np.random.default_rng(seed); m = 2 ** (d + 1) - 1; n = 2 * m; nbr = [[] for _ in range(n)]
    for off in (0, m):
        for i in range(2 ** d - 1):
            for ch in (2 * i + 1, 2 * i + 2): nbr[off + i].append(off + ch); nbr[off + ch].append(off + i)
    L = list(rng.permutation(np.arange(2 ** d - 1, m))); R = list(m + rng.permutation(np.arange(2 ** d - 1, m)))
    cyc = [x for pair in zip(L, R) for x in pair]
    for i in range(len(cyc)): a, b = cyc[i], cyc[(i + 1) % len(cyc)]; nbr[a].append(b); nbr[b].append(a)
    NB = np.array([sorted(set(x)) + [i] * (3 - len(set(x))) for i, x in enumerate(nbr)])
    return TabEnv(f"Glued-{d}", NB, 0, m, 30 * d)

_ENV = {}
def make_env(name):
    if name in _ENV: return _ENV[name]
    if name.startswith("Corridor-"): e = grid_env(name, serpentine(*{"S": (9, 2), "M": (10, 3), "L": (12, 4), "XL": (14, 5), "XXL": (18, 6)}[name[9:]]))
    elif name.startswith("Open-"):
        n = int(name[5:]); rows = ["." * n for _ in range(n)]; rows[0] = "S" + rows[0][1:]; rows[-1] = rows[-1][:-1] + "G"; e = grid_env(name, rows)
    elif name.startswith("Rooms-"): e = grid_env(name, four_rooms({"S": 5, "L": 7}[name[6:]]))
    elif name.startswith("Glued-"): e = glued_trees(int(name[6:]))
    else: raise ValueError(name)
    _ENV[name] = e; return e

fig, axs = plt.subplots(1, len(ENVS), figsize=(2.6 * len(ENVS), 2.8))
for ax, en in zip(np.atleast_1d(axs), ENVS):
    e = make_env(en); img = np.zeros(e.shape)
    for (r, c) in e.pos: img[r, c] = 1
    ax.imshow(img, cmap="gray_r", vmin=-.2, vmax=1.6); ax.scatter(*e.pos[e.s0][::-1], c="tab:blue", s=25); ax.scatter(*e.pos[e.g][::-1], c="tab:red", marker="*", s=60)
    ax.set_title(f"{en}\n{e.n} states, path {e.path_len}", fontsize=8); ax.axis("off")
savefig("fig_env_overview")
print({en: (make_env(en).n, make_env(en).path_len, make_env(en).H) for en in ENVS})

# %% Walkers on a graph: coherent (dephased) quantum walk by single-shot trajectories + classical comparators
def make_U(Ak, ham="lap"):
    Hm = GAMMA_H * (np.diag(Ak.sum(1)) - Ak) if ham == "lap" else -GAMMA_H * Ak
    lam, V = np.linalg.eigh(Hm); return (V * np.exp(-1j * lam)) @ V.T

def make_eig(Ak, ham="lap"):
    Hm = GAMMA_H * (np.diag(Ak.sum(1)) - Ak) if ham == "lap" else -GAMMA_H * Ak
    return np.linalg.eigh(Hm)

def traj_sample(U, s, p, T, rng):
    """One sample of a measurement-interrupted quantum walk. Equivalent to rho -> (1-p) U rho U^+ + p diag(U rho U^+)
    per step, but needs one device shot per waypoint."""
    n = len(U); psi = np.zeros(n, complex); psi[s] = 1.0
    for _ in range(T):
        psi = U @ psi
        if p > 0 and rng.random() < p:
            pr = np.abs(psi) ** 2; x = min(n - 1, int(np.searchsorted(np.cumsum(pr / pr.sum()), rng.random())))
            psi = np.zeros(n, complex); psi[x] = 1.0
    pr = np.abs(psi) ** 2
    return min(n - 1, int(np.searchsorted(np.cumsum(pr / pr.sum()), rng.random())))

_CACHE = {}                                   # per-process cache of known-map graph structures (deterministic given the env)
def build_cache(known, Adj, need_eig, need_traj, ham):
    K = np.flatnonzero(known); nk = len(K); g2l = -np.ones(len(known), int); g2l[K] = np.arange(nk)
    Ak = Adj[np.ix_(K, K)].astype(float)
    Dk = shortest_path(Ak, unweighted=True) if nk > 1 else np.zeros((1, 1))
    C = types.SimpleNamespace(K=K, nk=nk, g2l=g2l, Ak=Ak, Dk=Dk, nbrs=[np.flatnonzero(Ak[i]) for i in range(nk)], U=None)
    C.lam = C.V = None; C.p0 = {}
    if need_eig and nk > 1:
        C.lam, C.V = make_eig(Ak, ham)
        if need_traj: C.U = (C.V * np.exp(-1j * C.lam)) @ C.V.T
    return C

def qsample(C, ls, p, T, rng):
    """Waypoint from the (dephased) quantum walk. p = 0: exact Born distribution via the cached eigendecomposition;
    p > 0: single-shot measurement-interrupted trajectory."""
    if p == 0:
        cdf = C.p0.get((ls, T))
        if cdf is None:
            psi = C.V @ (np.exp(-1j * C.lam * T) * C.V[ls]); pr = np.abs(psi) ** 2; cdf = np.cumsum(pr / pr.sum()); C.p0[(ls, T)] = cdf
        return min(len(cdf) - 1, int(np.searchsorted(cdf, rng.random())))
    return traj_sample(C.U, ls, p, T, rng)

def pick_waypoint(kind, s, C, rng, p, T, alpha):
    ls = C.g2l[s]
    if ls < 0 or C.nk < 2: return -1
    nk = C.nk
    if kind == "uniform": l = int(rng.integers(nk - 1)); l += l >= ls
    elif kind == "crw":                                   # lazy simple random walk, T steps
        l = ls
        for _ in range(T):
            if rng.random() < 0.5: continue
            nb = C.nbrs[l]
            if len(nb): l = int(nb[int(rng.random() * len(nb))])
    elif kind == "nbrw":                                  # non-backtracking walk: classical ballistic baseline
        l, prev = ls, -1
        for _ in range(T):
            nb = C.nbrs[l]
            if len(nb) == 0: break
            ch = nb[nb != prev] if len(nb) > 1 else nb
            nxt = int(ch[int(rng.random() * len(ch))]); prev, l = l, nxt
    elif kind == "levy":                                  # heavy-tailed graph distance
        dmax = int(C.Dk[ls].max()); d = min(dmax, int(rng.pareto(alpha)) + 1)
        cand = np.flatnonzero(C.Dk[ls] == d); l = int(cand[int(rng.random() * len(cand))])
    elif kind == "dqw": l = qsample(C, ls, p, T, rng)
    elif kind == "distmatch":                             # same distance as a quantum sample, random identity
        l0 = qsample(C, ls, p, T, rng); d = int(C.Dk[ls, l0])
        if d <= 0: return -1
        cand = np.flatnonzero(C.Dk[ls] == d); l = int(cand[int(rng.random() * len(cand))])
    else: raise ValueError(kind)
    return -1 if l == ls else int(C.K[l])

# %% Agents: tabular Q-learning; exploration = waypoint (walker) + shortest-path navigation + short random burst
STRUCT = {"uniform", "crw", "nbrw", "levy", "dqw", "distmatch"}
def run(cfg, env_name, seed, n_ep=N_EP):
    n_ep = cfg.get("n_ep", n_ep); rng = np.random.default_rng(seed); env = make_env(env_name); n, A, s0, g, H, NB = env.n, env.A, env.s0, env.g, env.H, env.NB
    kind = cfg["kind"]; oracle = cfg.get("oracle", True); ham = cfg.get("ham", "lap"); T = cfg.get("T", 16); p = cfg.get("p", 0.0)
    Q = np.zeros((n, A)); N = np.zeros((n, A)); visited = np.zeros(n, bool)
    if oracle: known = np.ones(n, bool); Adj = env.adj.copy(); act_to = env.act_to.copy()
    else: known = np.zeros(n, bool); Adj = np.zeros((n, n), bool); act_to = -np.ones((n, n), int); known[s0] = True
    need_eig = kind in ("dqw", "distmatch"); need_traj = need_eig and p > 0; C = None; edges_built = -1
    succ, steps, cover, ev, ev_steps = np.zeros(n_ep), np.zeros(n_ep, int), np.zeros(n_ep), [], []
    n_wp, wps = 0, np.zeros(n_ep)                      # walker (quantum-device) queries, cumulative per episode
    cum = 0
    def greedy(q):
        idx = np.flatnonzero(q == q.max()); return int(idx[0]) if len(idx) == 1 else int(idx[int(rng.random() * len(idx))])
    def explore_action(s):
        un = np.flatnonzero(N[s] == 0); return int(un[int(rng.random() * len(un))]) if len(un) else int(rng.integers(A))
    def nav_action(s, tgt):
        ls, lt = C.g2l[s], C.g2l[tgt]
        if ls < 0 or lt < 0: return -1
        nb = C.nbrs[ls]
        if len(nb) == 0: return -1
        l2 = int(nb[np.argmin(C.Dk[nb, lt])]); return int(act_to[C.K[ls], C.K[l2]])
    for ep in range(n_ep):
        eps = max(0.05, cfg.get("decay", 0.98) ** ep)
        if kind in STRUCT:
            ne = int(Adj.sum())
            if oracle:
                if C is None:
                    key = (env_name, ham, need_eig, need_traj); C = _CACHE.get(key)
                    if C is None:
                        C = build_cache(known, Adj, need_eig, need_traj, ham); _CACHE[key] = C
                        while len(_CACHE) > 4: _CACHE.pop(next(iter(_CACHE)))
            elif C is None or (ne > 1.05 * edges_built and (ep % 3 == 0)): C = build_cache(known, Adj, need_eig, need_traj, ham); edges_built = ne
        s = s0; visited[s] = True; seg, seg_tgt, seg_left, rep_a, done = 0, -1, 0, -1, False
        for t in range(H):
            a = -1
            if seg == 1:
                if s == seg_tgt: seg, seg_left = 2, BURST
                else:
                    a = nav_action(s, seg_tgt); seg_left -= 1
                    if a < 0 or seg_left < 0: seg, a = 0, -1
            if a < 0 and seg == 2:
                if seg_left > 0: a = explore_action(s); seg_left -= 1
                else: seg = 0
            if a < 0 and seg == 3:
                if seg_left > 0: a = rep_a; seg_left -= 1
                else: seg = 0
            if a < 0:
                if kind in STRUCT:
                    if rng.random() < eps:
                        n_wp += 1; tgt = pick_waypoint(kind, s, C, rng, p, T, cfg.get("alpha", 1.6))
                        if tgt >= 0:
                            a = nav_action(s, tgt)
                            if a >= 0: seg, seg_tgt, seg_left = 1, tgt, 2 * int(C.Dk[C.g2l[s], C.g2l[tgt]]) + 5
                        if a < 0: a = explore_action(s); seg, seg_left = 2, BURST - 1
                    else: a = greedy(Q[s])
                elif kind == "epsz":
                    if rng.random() < eps:
                        rep_a = int(rng.integers(A)); z = min(50, int(1 / (1 - rng.random()))); seg, seg_left = 3, z - 1; a = rep_a
                    else: a = greedy(Q[s])
                elif kind == "eps": a = int(rng.integers(A)) if rng.random() < eps else greedy(Q[s])
                elif kind == "count": a = int(rng.integers(A)) if rng.random() < 0.05 else greedy(Q[s] + cfg["beta"] / np.sqrt(N[s] + 1))
            s2 = int(NB[s, a]); N[s, a] += 1
            if not oracle:
                known[s2] = True
                if s2 != s and not Adj[s, s2]:
                    Adj[s, s2] = Adj[s2, s] = True; act_to[s, s2] = a
                    if env.inv is not None: act_to[s2, s] = env.inv(a)
            done = (s2 == g); r = 1.0 if done else 0.0
            Q[s, a] += ALPHA * (r + (0.0 if done else GAMMA_RL * Q[s2].max()) - Q[s, a])
            s = s2; visited[s] = True; cum += 1
            if done: break
        succ[ep] = done; steps[ep] = t + 1; cover[ep] = visited.mean(); wps[ep] = n_wp
        if (ep + 1) % EVAL_EVERY == 0:
            ok = 0.0
            for _ in range(EVAL_ROLL):
                ss = s0
                for _ in range(H):
                    ss = int(NB[ss, greedy(Q[ss])])
                    if ss == g: ok += 1; break
            ev.append(ok / EVAL_ROLL); ev_steps.append(int(steps[:ep + 1].sum()))
    return dict(succ=succ, steps=steps, cover=cover, ev=np.array(ev), ev_steps=np.array(ev_steps), cap=n_ep * H, wps=wps)

def metrics(o):
    cs = np.cumsum(o["steps"]); cap = o["cap"]; hit = o["succ"] > 0; ev = o["ev"]
    sol = np.flatnonzero(ev >= 0.9)
    return dict(first_goal=cs[np.argmax(hit)] if hit.any() else cap, censored=int(not hit.any()),
                solve=o["ev_steps"][sol[0]] if len(sol) else cap, final_success=ev[-3:].mean(),
                cover=o["cover"][-1], goals=hit.sum(), wp_first=(o["wps"][np.argmax(hit)] if hit.any() else o["wps"][-1]))

def run_batch(methods, seeds, envs, n_jobs=-1):
    jobs = [(en, mn, s) for en in envs for mn in methods for s in seeds]
    outs = Parallel(n_jobs=n_jobs)(delayed(run)((methods[mn](en) if callable(methods[mn]) else methods[mn]), en, s) for en, mn, s in jobs)
    res = {}
    for (en, mn, s), o in zip(jobs, outs): res.setdefault((en, mn), []).append((s, o))
    return res
def fg(res, en, mn): return np.array([metrics(o)["first_goal"] for _, o in res[(en, mn)]], float)

# %% Validation (1): trajectory sampler equals the density-matrix dephasing channel
e = make_env("Corridor-M"); Ak = e.adj.astype(float); Uv = make_U(Ak); rng_v = np.random.default_rng(0); n_s = 1500 if QUICK else 4000
def rho_dist(p, T=16):
    n = e.n; rho = np.zeros((n, n), complex); rho[e.s0, e.s0] = 1
    for _ in range(T):
        rho = Uv @ rho @ Uv.conj().T; d = np.diag(rho).copy(); rho *= (1 - p); rho[np.arange(n), np.arange(n)] += p * d
    return np.diag(rho).real
rows = []
for p in [0, 0.1, 0.5, 1.0]:
    ref = rho_dist(p); emp = np.bincount([traj_sample(Uv, e.s0, p, 16, rng_v) for _ in range(n_s)], minlength=e.n) / n_s
    fl = np.bincount(rng_v.choice(e.n, n_s, p=ref / ref.sum()), minlength=e.n) / n_s
    rows.append(dict(p=p, TVD_traj_vs_density=0.5 * np.abs(emp - ref).sum(), TVD_noise_floor=0.5 * np.abs(fl - ref).sum()))
tab_val = pd.DataFrame(rows); print(tab_val.round(4)); tab_val.to_csv(f"{OUT}/table_validation_trajectory.csv", index=False)

# %% Experiment A: spreading law of the dial (cite: Kendon & Tregenna 2003 for the qualitative phenomenon)
R = 70 if QUICK else 140; T_MSD = 60 if QUICK else 128; n_line = 2 * R + 1
A_line = np.zeros((n_line, n_line))
for i in range(n_line - 1): A_line[i, i + 1] = A_line[i + 1, i] = 1
Ul = make_U(A_line); x = np.arange(n_line) - R; P_MSD = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]; msd = {}
for p in P_MSD:
    rho = np.zeros((n_line, n_line), complex); rho[R, R] = 1; var = []
    for t in range(T_MSD):
        rho = Ul @ rho @ Ul.conj().T; d = np.diag(rho).copy(); rho *= (1 - p); rho[np.arange(n_line), np.arange(n_line)] += p * d
        pr = np.diag(rho).real; var.append((pr * x ** 2).sum() - ((pr * x).sum()) ** 2)
    msd[p] = np.array(var)
tt = np.arange(1, T_MSD + 1); cross, Dco = {}, {}
for p in P_MSD:
    if p == 0: continue
    sl = np.gradient(np.log(msd[p] + 1e-12), np.log(tt)); idx = np.flatnonzero((sl < 1.5) & (tt >= 2)); cross[p] = tt[idx[0]] if len(idx) else np.nan
    h = slice(T_MSD // 2, T_MSD); Dco[p] = np.polyfit(tt[h], msd[p][h], 1)[0] / 2
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6)); cm = plt.cm.viridis(np.linspace(0, .95, len(P_MSD)))
for c, p in zip(cm, P_MSD): axs[0].loglog(tt, msd[p], color=c, label=f"p={p:g}")
axs[0].loglog(tt, 0.3 * tt ** 2, "k--", lw=.8, label=r"$t^2$"); axs[0].loglog(tt, 0.5 * tt, "k:", lw=.8, label=r"$t$")
axs[0].set_xlabel("t"); axs[0].set_ylabel(r"$\sigma^2(t)$"); axs[0].legend(fontsize=7, ncol=2); axs[0].set_title("Ballistic to diffusive")
pp = np.array([q for q in cross if np.isfinite(cross[q])]); ts = np.array([cross[q] for q in pp])
axs[1].loglog(1 / pp, ts, "o-"); sl_c = np.polyfit(np.log(1 / pp), np.log(ts), 1)[0] if len(pp) > 2 else np.nan
axs[1].set_title(f"crossover time $t^*$ vs 1/p (slope {sl_c:.2f})"); axs[1].set_xlabel("1/p")
pD = np.array([q for q in Dco if q <= 0.2]); sl_D = np.polyfit(np.log(pD), np.log([Dco[q] for q in pD]), 1)[0]
axs[2].loglog(list(Dco), list(Dco.values()), "o-"); axs[2].set_title(f"late-time diffusion constant D(p) (slope {sl_D:.2f} for p<=0.2)"); axs[2].set_xlabel("p")
savefig("figA_spreading_law"); pd.DataFrame({"p": list(cross), "t_star": list(cross.values()), "D": [Dco[q] for q in cross]}).to_csv(f"{OUT}/tableA_spreading.csv", index=False)

# %% Experiment B: glued trees, the graph family with a proven exponential quantum-walk speedup (theory anchor)
rows = []
for d in ([3, 4] if QUICK else [3, 4, 5, 6, 7]):
    e = glued_trees(d); n = e.n; Ad = e.adj.astype(float); Lp = np.diag(Ad.sum(1)) - Ad
    lam, V = np.linalg.eigh(-GAMMA_H * 2 * Ad); lam2, V2 = np.linalg.eigh(Lp); ts_ = np.linspace(0, 6 * d, 120)
    e0 = np.zeros(n); e0[e.s0] = 1; pq, pc = [], []
    for t in ts_:
        psi = V @ (np.exp(-1j * lam * t) * (V.T @ e0)); pq.append(abs(psi[e.g]) ** 2)
        pc.append((V2 @ (np.exp(-GAMMA_H * 2 * lam2 * t) * (V2.T @ e0)))[e.g])
    rows.append(dict(depth=d, n=n, quantum_max_hit=max(pq), classical_max_hit=max(pc), t_best=ts_[int(np.argmax(pq))]))
tab_gt = pd.DataFrame(rows); print(tab_gt.round(5)); tab_gt.to_csv(f"{OUT}/tableB_glued_tree_hitting.csv", index=False)
fig, ax = plt.subplots(figsize=(5, 3.4)); ax.semilogy(tab_gt.depth, tab_gt.quantum_max_hit, "o-", label="coherent quantum walk")
ax.semilogy(tab_gt.depth, tab_gt.classical_max_hit, "s-", label="classical continuous-time walk"); ax.set_xlabel("tree depth d"); ax.set_ylabel("max probability at exit root")
ax.legend(); savefig("figB_glued_tree_hitting")

# %% Tuning on separate seeds (every method gets its own grid; same number of tuning seeds)
t0 = time.time(); TS = [10_000 + i for i in range(N_TUNE)]; Tg = GRID["T"]
stage = {}
for T in Tg:
    stage[f"crw|T={T}"] = dict(kind="crw", T=T); stage[f"nbrw|T={T}"] = dict(kind="nbrw", T=T)
    for pq in GRID["dqw_p"]: stage[f"dqw|p={pq:g}|T={T}"] = dict(kind="dqw", p=pq, T=T)
for al in GRID["levy"]: stage[f"levy|a={al}"] = dict(kind="levy", alpha=al)
for b in GRID["count"]: stage[f"count|b={b}"] = dict(kind="count", beta=b)
rt = run_batch(stage, TS, ENVS)
def best(en, prefix): return min([k for k in stage if k.startswith(prefix)], key=lambda k: np.mean(np.log(fg(rt, en, k))))
TUNED = {}
for en in ENVS:
    kb = best(en, "dqw|"); pbest, Tbest = float(kb.split("|")[1][2:]), int(kb.split("|")[2][2:])
    TUNED[en] = dict(dqw=(pbest, Tbest), crw=int(best(en, "crw|").split("T=")[1]), nbrw=int(best(en, "nbrw|").split("T=")[1]),
                     levy=float(best(en, "levy|").split("a=")[1]), count=float(best(en, "count|").split("b=")[1]))
print(f"tuning done in {time.time()-t0:.0f}s"); tab_tuned = pd.DataFrame(TUNED).T; print(tab_tuned); tab_tuned.to_csv(f"{OUT}/table_tuned_hyperparameters.csv")

# %% Main experiment: map-known exploration, all methods, N_SEEDS seeds, steps-to-first-goal with censoring
t0 = time.time()
MAIN = {
 "eps-greedy":            lambda en: dict(kind="eps"),
 "eps-z-greedy":          lambda en: dict(kind="epsz"),
 "Count-UCB":             lambda en: dict(kind="count", beta=TUNED[en]["count"]),
 "Uniform-waypoint":      lambda en: dict(kind="uniform"),
 "CRW-waypoint":          lambda en: dict(kind="crw", T=TUNED[en]["crw"]),
 "NBRW-waypoint":         lambda en: dict(kind="nbrw", T=TUNED[en]["nbrw"]),
 "Levy-waypoint":         lambda en: dict(kind="levy", alpha=TUNED[en]["levy"]),
 "DistMatched-classical": lambda en: dict(kind="distmatch", p=TUNED[en]["dqw"][0], T=TUNED[en]["dqw"][1]),
 "DQW p=1 (decohered)":   lambda en: dict(kind="dqw", p=1.0, T=TUNED[en]["dqw"][1]),
 "DQW p=0 (coherent)":    lambda en: dict(kind="dqw", p=0.0, T=TUNED[en]["dqw"][1]),
 "DQW (tuned p,T)":       lambda en: dict(kind="dqw", p=TUNED[en]["dqw"][0], T=TUNED[en]["dqw"][1]),
}
res = run_batch(MAIN, list(range(N_SEEDS)), ENVS); print(f"main done in {time.time()-t0:.0f}s")
ORDER = list(MAIN); GROUP = {"eps-greedy": "no map", "eps-z-greedy": "no map", "Count-UCB": "no map"}
COL = {m: c for m, c in zip(ORDER, ["#999999", "#777777", "#555555", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#74c476", "#fd8d3c", "#e6550d", "#d62728"])}
rows = [dict(env=en, method=mn, seed=s, **metrics(o)) for (en, mn), lst in res.items() for s, o in lst]
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/raw_per_run_metrics.csv", index=False)
CAP = {en: N_EP * make_env(en).H for en in ENVS}

# %% Statistics: bootstrap CIs, Holm-corrected Mann-Whitney, Cliff's delta, mean ranks, scaling exponents
rngb = np.random.default_rng(123)
def boot_ci(x, fn=np.median, B=4000):
    x = np.asarray(x, float); bs = [fn(x[rngb.integers(0, len(x), len(x))]) for _ in range(B)]; return np.percentile(bs, [2.5, 97.5])
def cliffs(a, b):
    """P(a<b) - P(a>b): positive means 'a' (proposed) needs fewer steps."""
    a, b = np.asarray(a), np.asarray(b); return ((a[:, None] < b[None, :]).mean() - (a[:, None] > b[None, :]).mean())
def holm(p):
    p = np.array(p); o = np.argsort(p); adj = np.empty_like(p); run_ = 0
    for r, i in enumerate(o): run_ = max(run_, (len(p) - r) * p[i]); adj[i] = min(1, run_)
    return adj
def logrank_p(a, ca, b, cb):
    """Log-rank test for right-censored times (scipy >= 1.11); NaN if unavailable."""
    try:
        x = stats.CensoredData(uncensored=a[~ca], right=a[ca]); y = stats.CensoredData(uncensored=b[~cb], right=b[cb]); return float(stats.logrank(x, y).pvalue)
    except Exception: return float("nan")
REF = "DQW (tuned p,T)"; t_main, t_sig = [], []
for en in ENVS:
    for mn in ORDER:
        x = df[(df.env == en) & (df.method == mn)]; lo, hi = boot_ci(x.first_goal); slo, shi = boot_ci(x.solve)
        t_main.append({"Env": en, "Method": mn, "Median steps to 1st goal [95% CI]": f"{np.median(x.first_goal):,.0f} [{lo:,.0f}, {hi:,.0f}]",
                       "Median steps to solve [95% CI]": f"{np.median(x.solve):,.0f} [{slo:,.0f}, {shi:,.0f}]",
                       "Median walker queries to 1st goal": f"{np.median(x.wp_first):,.0f}", "Never found goal": f"{int(x.censored.sum())}/{len(x)}", "Final greedy success": f"{x.final_success.mean():.2f}", "Coverage": f"{x.cover.mean():.2f}"})
    a = df[(df.env == en) & (df.method == REF)].first_goal.values; tmp = []
    for mn in ORDER:
        if mn == REF: continue
        b = df[(df.env == en) & (df.method == mn)].first_goal.values
        try: pv = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        except ValueError: pv = 1.0
        bd = [cliffs(a[rngb.integers(0, len(a), len(a))], b[rngb.integers(0, len(b), len(b))]) for _ in range(1000)]
        ca = df[(df.env == en) & (df.method == REF)].censored.values.astype(bool); cb = df[(df.env == en) & (df.method == mn)].censored.values.astype(bool)
        tmp.append({"Env": en, "vs": mn, "median ratio (DQW/other)": np.median(a) / max(np.median(b), 1), "Cliff delta (+ = DQW faster)": cliffs(a, b),
                    "delta 95% CI": f"[{np.percentile(bd, 2.5):.2f}, {np.percentile(bd, 97.5):.2f}]", "delta_lo": np.percentile(bd, 2.5), "delta_hi": np.percentile(bd, 97.5),
                    "p (MWU)": pv, "p (log-rank, censored)": logrank_p(a, ca, b, cb)})
    for r_, pa in zip(tmp, holm([r_["p (MWU)"] for r_ in tmp])): r_["p (Holm)"] = pa
    t_sig += tmp
t_main, t_sig = pd.DataFrame(t_main), pd.DataFrame(t_sig)
med = df.groupby(["env", "method"]).first_goal.median().unstack()[ORDER]; ranks = med.rank(axis=1, method="average"); t_rank = ranks.mean().sort_values().to_frame("mean rank (lower = faster)")
print(t_main.to_string(index=False)); print(); print(t_rank)

# Verdict table: DQW against the strongest *classical walker* in each environment
CLASSICAL_W = ["Uniform-waypoint", "CRW-waypoint", "NBRW-waypoint", "Levy-waypoint", "DistMatched-classical", "DQW p=1 (decohered)"]
vr = []
for en in ENVS:
    bc = min(CLASSICAL_W, key=lambda m: med.loc[en, m]); r_ = t_sig[(t_sig.Env == en) & (t_sig.vs == bc)].iloc[0]
    verdict = "DQW faster" if (r_["p (Holm)"] < 0.05 and r_["Cliff delta (+ = DQW faster)"] > 0) else ("classical faster" if (r_["p (Holm)"] < 0.05) else "no significant difference")
    vr.append({"Env": en, "path": make_env(en).path_len, "best classical walker": bc, "median DQW": med.loc[en, REF], "median best classical": med.loc[en, bc],
               "Cliff delta": r_["Cliff delta (+ = DQW faster)"], "p (Holm)": r_["p (Holm)"], "verdict": verdict})
t_verdict = pd.DataFrame(vr); print(t_verdict.to_string(index=False))

# scaling exponents on the corridor family (median steps vs path length, bootstrap over seeds)
t_scale = []; L_ = np.array([make_env(en).path_len for en in CORR_ENVS], float)
if len(CORR_ENVS) >= 3:
    for mn in ORDER:
        data = [df[(df.env == en) & (df.method == mn)].first_goal.values for en in CORR_ENVS]
        def fit(dd): return np.polyfit(np.log(L_), np.log([max(np.median(v), 1) for v in dd]), 1)[0]
        bs = [fit([v[rngb.integers(0, len(v), len(v))] for v in data]) for _ in range(1000)]
        t_scale.append({"Method": mn, "scaling exponent": fit(data), "95% CI": f"[{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]", "sizes with censored median": int(sum(np.median(v) >= CAP[en] for v, en in zip(data, CORR_ENVS)))})
t_scale = pd.DataFrame(t_scale); print(t_scale.to_string(index=False) if len(t_scale) else "need >=3 corridor sizes")

# %% Figures: discovery curves, medians with CIs, scaling, summary
fig, axs = plt.subplots(1, len(ENVS), figsize=(3.3 * len(ENVS), 3.2), sharey=True)
for ax, en in zip(np.atleast_1d(axs), ENVS):
    xs = np.linspace(0, CAP[en], 300)
    for mn in ORDER:
        v = df[(df.env == en) & (df.method == mn)].first_goal.values; ax.plot(xs, [(v <= q).mean() for q in xs], color=COL[mn], lw=2.2 if mn == REF else 1.1, label=mn)
    ax.set_title(f"{en}", fontsize=9); ax.set_xlabel("environment steps")
np.atleast_1d(axs)[0].set_ylabel("P(goal found by step t)"); np.atleast_1d(axs)[0].legend(fontsize=5); plt.tight_layout(); savefig("fig_discovery_curves")

fig, ax = plt.subplots(figsize=(1.6 * len(ENVS) + 3, 4)); wdt = 0.85 / len(ORDER)
for j, mn in enumerate(ORDER):
    m, lo, hi = [], [], []
    for en in ENVS:
        v = df[(df.env == en) & (df.method == mn)].first_goal.values; c = boot_ci(v); m.append(np.median(v)); lo.append(np.median(v) - c[0]); hi.append(c[1] - np.median(v))
    ax.bar(np.arange(len(ENVS)) + j * wdt, m, wdt, yerr=[lo, hi], color=COL[mn], label=mn, capsize=1.5)
ax.set_yscale("log"); ax.set_xticks(np.arange(len(ENVS)) + .42); ax.set_xticklabels(ENVS, fontsize=8); ax.set_ylabel("median steps to first goal (95% CI)"); ax.legend(fontsize=5, ncol=2)
savefig("fig_median_steps")

if len(CORR_ENVS) >= 3:
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for mn in ORDER:
        yy = [np.median(df[(df.env == en) & (df.method == mn)].first_goal) for en in CORR_ENVS]
        ex = t_scale[t_scale.Method == mn]["scaling exponent"].iloc[0]; ax.loglog(L_, yy, "o-", color=COL[mn], lw=2.2 if mn == REF else 1, label=f"{mn} ({ex:.2f})")
    ax.set_xlabel("shortest-path length"); ax.set_ylabel("median steps to first goal"); ax.legend(fontsize=5); savefig("fig_scaling_corridors")

t_rank.to_csv(f"{OUT}/table_mean_ranks.csv")

# %% Experiment C: waypoint-distance distributions (mechanism) on the corridor graph
en = "Corridor-M"; e = make_env(en); C = build_cache(np.ones(e.n, bool), e.adj, True, True, "lap"); rng_m = np.random.default_rng(5); tun = TUNED[en]
samplers = {"Uniform": dict(kind="uniform"), "CRW": dict(kind="crw", T=tun["crw"]), "NBRW": dict(kind="nbrw", T=tun["nbrw"]), "Levy": dict(kind="levy", alpha=tun["levy"]),
            "DQW p=1": dict(kind="dqw", p=1.0, T=tun["dqw"][1]), "DQW p=0": dict(kind="dqw", p=0.0, T=tun["dqw"][1]), f"DQW tuned (p={tun['dqw'][0]:g})": dict(kind="dqw", p=tun["dqw"][0], T=tun["dqw"][1])}
start = int(C.g2l[e.s0]); mid = C.nk // 2; fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, (ls, ttl) in zip(axs, [(start, "start of corridor"), (mid, "middle of corridor")]):
    for nm, c in samplers.items():
        d = []
        for _ in range(1500 if not QUICK else 300):
            w = pick_waypoint(c["kind"], int(C.K[ls]), C, rng_m, c.get("p", 0), c.get("T", 16), c.get("alpha", 1.6))
            if w >= 0: d.append(C.Dk[ls, C.g2l[w]])
        ax.hist(d, bins=np.arange(0, C.Dk[ls].max() + 2) - .5, histtype="step", density=True, label=nm, lw=1.5)
    ax.set_xlabel("graph distance of sampled waypoint"); ax.set_title(f"agent at {ttl}", fontsize=9)
axs[0].legend(fontsize=6); savefig("figC_waypoint_distance_distributions")

# %% Experiment D: p-sweep (dial sensitivity) vs classical walkers
sweep = {f"DQW p={q:g}": (lambda en, q=q: dict(kind="dqw", p=q, T=TUNED[en]["dqw"][1])) for q in P_SWEEP}
rs = run_batch(sweep, list(range(N_SIDE)), SWEEP_ENVS); rows = []
fig, axs = plt.subplots(1, len(SWEEP_ENVS), figsize=(4.4 * len(SWEEP_ENVS), 3.4))
for ax, en in zip(np.atleast_1d(axs), SWEEP_ENVS):
    M = np.array([fg(rs, en, f"DQW p={q:g}") for q in P_SWEEP]); m = np.median(M, 1); ci = np.array([boot_ci(r) for r in M])
    ax.errorbar(range(len(P_SWEEP)), m, [m - ci[:, 0], ci[:, 1] - m], fmt="o-", color="tab:red", capsize=3, label="DQW fixed p")
    for mn, ls_ in [("CRW-waypoint", "--"), ("NBRW-waypoint", "-."), ("DistMatched-classical", ":")]:
        v = df[(df.env == en) & (df.method == mn)].first_goal.values; ax.axhline(np.median(v), color=COL[mn], ls=ls_, label=mn)
    ax.set_xticks(range(len(P_SWEEP))); ax.set_xticklabels([f"{q:g}" for q in P_SWEEP], rotation=45); ax.set_yscale("log"); ax.set_xlabel("dephasing p"); ax.set_title(en, fontsize=9)
    rows += [dict(env=en, p=q, median_steps=m[i], ci_lo=ci[i, 0], ci_hi=ci[i, 1]) for i, q in enumerate(P_SWEEP)]
np.atleast_1d(axs)[0].set_ylabel("median steps to first goal"); np.atleast_1d(axs)[0].legend(fontsize=6); plt.tight_layout(); savefig("figD_p_sensitivity")
pd.DataFrame(rows).to_csv(f"{OUT}/tableD_p_sweep.csv", index=False)

# %% Experiment E: learned map (no privileged topology) vs known map
learned = {f"{k} (learned map)": (lambda en, k=k: {**MAIN[k](en), "oracle": False}) for k in ["CRW-waypoint", "NBRW-waypoint", "Levy-waypoint", "DQW (tuned p,T)"]}
rl = run_batch(learned, list(range(N_SIDE)), ENVS); rows = []
for en in ENVS:
    for k in ["CRW-waypoint", "NBRW-waypoint", "Levy-waypoint", "DQW (tuned p,T)"]:
        a = df[(df.env == en) & (df.method == k)].first_goal.values; b = fg(rl, en, f"{k} (learned map)")
        rows.append(dict(env=en, method=k, known_map_median=np.median(a), learned_map_median=np.median(b), never_found_learned=f"{int((b >= CAP[en]).sum())}/{len(b)}"))
tab_E = pd.DataFrame(rows); print(tab_E.to_string(index=False)); tab_E.to_csv(f"{OUT}/tableE_learned_map.csv", index=False)
# no-map baselines are in the main table (eps-greedy, eps-z-greedy, Count-UCB)

# %% Experiment F: glued trees (known map) - regime where coherence is provably useful; every walker's time is tuned
GL_ENVS = [f"Glued-{d}" for d in GLUED_DEPTHS]; MULT = [0.5, 1, 1.5, 1.75, 2, 2.5, 3, 5]; GREF = "DQW p=0 (coherent)"; dep = lambda en: int(en[6:])
Tm = lambda en, m: max(1, int(round(m * dep(en))))
stageF = {}
for m in MULT:
    stageF[f"crw|m={m}"] = (lambda en, m=m: dict(kind="crw", T=Tm(en, m), n_ep=200))
    stageF[f"nbrw|m={m}"] = (lambda en, m=m: dict(kind="nbrw", T=Tm(en, m), n_ep=200))
    stageF[f"dqw|m={m}"] = (lambda en, m=m: dict(kind="dqw", p=0.0, T=Tm(en, m), ham="adj", n_ep=200))
for al in GRID["levy"]: stageF[f"levy|a={al}"] = (lambda en, al=al: dict(kind="levy", alpha=al, n_ep=200))
rtF = run_batch(stageF, [20_000 + i for i in range(2 if QUICK else 6)], GL_ENVS)
def bestF(en, k): return min([n_ for n_ in stageF if n_.startswith(k + "|")], key=lambda n_: np.mean(np.log(fg(rtF, en, n_))))
MF = {en: dict(crw=float(bestF(en, "crw").split("=")[1]), nbrw=float(bestF(en, "nbrw").split("=")[1]), dqw=float(bestF(en, "dqw").split("=")[1]), levy=float(bestF(en, "levy").split("=")[1])) for en in GL_ENVS}
GT = lambda en, k: Tm(en, MF[en][k]); print("glued-tree tuned walk-time multipliers (T = m*depth):", MF)
core = {"eps-greedy": lambda en: dict(kind="eps", n_ep=200), "Uniform-waypoint": lambda en: dict(kind="uniform", n_ep=200),
        "CRW-waypoint": lambda en: dict(kind="crw", T=GT(en, "crw"), n_ep=200), "NBRW-waypoint": lambda en: dict(kind="nbrw", T=GT(en, "nbrw"), n_ep=200),
        "Levy-waypoint": lambda en: dict(kind="levy", alpha=MF[en]["levy"], n_ep=200),
        "DQW p=0 (coherent)": lambda en: dict(kind="dqw", p=0.0, T=GT(en, "dqw"), ham="adj", n_ep=200)}
extra = {nm: (lambda en, q=q: dict(kind="dqw", p=q, T=GT(en, "dqw"), ham="adj", n_ep=200)) for q, nm in [(1.0, "DQW p=1 (decohered)"), (0.05, "DQW p=0.05"), (0.01, "DQW p=0.01")]}
gmeth = {**core, **extra}; small = [en for en in GL_ENVS if dep(en) <= GLUED_TRAJ_MAX]; deep = [en for en in GL_ENVS if dep(en) > GLUED_TRAJ_MAX]
rg = run_batch(gmeth, list(range(N_GT)), small)
if deep: rg.update(run_batch(core, list(range(N_GT)), deep))
capF = {en: 200 * make_env(en).H for en in GL_ENVS}; rows = []
for en in GL_ENVS:
    for mn in gmeth:
        if (en, mn) not in rg: continue
        v = fg(rg, en, mn); lo, hi = boot_ci(v); rows.append(dict(depth=dep(en), method=mn, median_steps=np.median(v), ci_lo=lo, ci_hi=hi, never_found=int((v >= capF[en]).sum())))
tab_F = pd.DataFrame(rows); print(tab_F.pivot(index="method", columns="depth", values="median_steps").round(0).to_string()); tab_F.to_csv(f"{OUT}/tableF_glued_trees_agents.csv", index=False)
fig, ax = plt.subplots(figsize=(5.8, 3.9))
for mn in gmeth:
    s_ = tab_F[tab_F.method == mn]; ax.errorbar(s_.depth, s_.median_steps, [s_.median_steps - s_.ci_lo, s_.ci_hi - s_.median_steps], fmt="o-", capsize=3, label=mn)
ax.set_yscale("log"); ax.set_xlabel("glued-tree depth"); ax.set_ylabel("median steps to first goal"); ax.legend(fontsize=6); savefig("figF_glued_trees_agents")
t_expF = []
for mn in gmeth:
    ens = [en for en in GL_ENVS if (en, mn) in rg]
    if len(ens) < 3: continue
    dd = np.array([dep(en) for en in ens], float); data = [fg(rg, en, mn) for en in ens]
    fit = lambda D_: np.polyfit(dd, np.log([max(np.median(v), 1) for v in D_]), 1)[0]; bs = [fit([v[rngb.integers(0, len(v), len(v))] for v in data]) for _ in range(1000)]
    t_expF.append({"Method": mn, "depths used": f"{int(dd.min())}-{int(dd.max())}", "growth rate of ln(steps) per depth level": fit(data), "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
t_expF = pd.DataFrame(t_expF); print(t_expF.round(3).to_string(index=False)); t_expF.to_csv(f"{OUT}/tableF_growth_rates.csv", index=False)
t_sigF = []
for en in GL_ENVS:
    a = fg(rg, en, GREF); ca = a >= capF[en]; tmp = []
    for mn in gmeth:
        if mn == GREF or (en, mn) not in rg: continue
        b = fg(rg, en, mn); cb = b >= capF[en]
        try: pv = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        except ValueError: pv = 1.0
        tmp.append({"Env": en, "Reference": GREF, "vs": mn, "median ref": np.median(a), "median other": np.median(b), "Cliff delta (+ = ref faster)": cliffs(a, b), "p (MWU)": pv, "p (log-rank, censored)": logrank_p(a, ca, b, cb)})
    for r_, pa in zip(tmp, holm([r_["p (MWU)"] for r_ in tmp])): r_["p (Holm)"] = pa
    t_sigF += tmp
t_sigF = pd.DataFrame(t_sigF); t_sigF.to_csv(f"{OUT}/tableF_glued_trees_significance.csv", index=False)
# walk-time sensitivity: the coherent walk has a sharp optimal time (first hitting peak); classical walkers do not
sm = [d for d in GLUED_DEPTHS if d <= GLUED_TRAJ_MAX]; GP = f"Glued-{sm[-2] if len(sm) > 1 else sm[0]}"; MS = [0.5, 1, 1.5, 1.75, 2, 2.5, 3, 5, 8]
rts = run_batch({**{f"DQW p=0|m={m}": (lambda en, m=m: dict(kind="dqw", p=0.0, T=Tm(en, m), ham="adj", n_ep=200)) for m in MS},
                 **{f"CRW|m={m}": (lambda en, m=m: dict(kind="crw", T=Tm(en, m), n_ep=200)) for m in MS},
                 **{f"NBRW|m={m}": (lambda en, m=m: dict(kind="nbrw", T=Tm(en, m), n_ep=200)) for m in MS}}, list(range(N_GT)), [GP])
fig, ax = plt.subplots(figsize=(5.4, 3.6)); rows = []
for nm in ["DQW p=0", "CRW", "NBRW"]:
    M_ = np.array([fg(rts, GP, f"{nm}|m={m}") for m in MS]); md = np.median(M_, 1); ci_ = np.array([boot_ci(r) for r in M_])
    ax.errorbar(MS, md, [md - ci_[:, 0], ci_[:, 1] - md], fmt="o-", capsize=3, label=nm); rows += [dict(method=nm, multiplier=m, median_steps=md[i]) for i, m in enumerate(MS)]
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("walk time T / depth"); ax.set_ylabel("median steps to first goal"); ax.set_title(GP); ax.legend()
savefig("figF3_walk_time_sensitivity"); pd.DataFrame(rows).to_csv(f"{OUT}/tableF3_walk_time_sensitivity.csv", index=False)
# dial sensitivity on glued trees
PG = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0]
rgs = run_batch({f"DQW p={q:g}": (lambda en, q=q: dict(kind="dqw", p=q, T=GT(en, "dqw"), ham="adj", n_ep=200)) for q in PG}, list(range(N_GT)), [GP])
Mg = np.array([fg(rgs, GP, f"DQW p={q:g}") for q in PG]); mg = np.median(Mg, 1); cig = np.array([boot_ci(r) for r in Mg])
fig, ax = plt.subplots(figsize=(5.2, 3.5)); ax.errorbar(range(len(PG)), mg, [mg - cig[:, 0], cig[:, 1] - mg], fmt="o-", color="tab:red", capsize=3)
ax.set_xticks(range(len(PG))); ax.set_xticklabels([f"{q:g}" for q in PG], rotation=45); ax.set_yscale("log"); ax.set_xlabel("dephasing p"); ax.set_ylabel("median steps to first goal"); ax.set_title(GP)
savefig("figF2_glued_p_sweep"); pd.DataFrame({"p": PG, "median_steps": mg, "ci_lo": cig[:, 0], "ci_hi": cig[:, 1]}).to_csv(f"{OUT}/tableF2_glued_p_sweep.csv", index=False)

# %% Hypothesis check, forest plot, quantum-resource table, run information
H = []
# H1
ok1, notes = True, []
for en in GL_ENVS:
    if dep(en) < min(5, max(GLUED_DEPTHS)): continue
    r_ = t_sigF[t_sigF.Env == en]; best_ = (r_["median ref"] < r_["median other"]).all(); sig_ = (r_["p (Holm)"] < 0.05).all()
    ok1 &= bool(best_ and sig_); notes.append(f"{en}: lowest={best_}, all Holm<0.05={sig_}")
if len(t_expF):
    ref_ = t_expF[t_expF.Method == GREF].iloc[0]; oth = t_expF[(t_expF.Method != GREF) & ~t_expF.Method.str.startswith("DQW")]
    rate_ok = bool((ref_.ci_hi < oth.ci_lo).all()); notes.append(f"growth-rate CI below all classical walkers: {rate_ok}"); ok1 &= rate_ok
H.append(dict(Hypothesis="H1 glued trees", Rule="coherent DQW lowest + Holm<0.05 at depth>=5 and smallest growth rate", Observed="; ".join(notes), Status="PASS" if ok1 else "NOT MET (see detail)"))
# H2
okc = 0.8 <= sl_c <= 1.2; p0, p05 = Mg[PG.index(0)], Mg[PG.index(0.05)]
try: pdeg = stats.mannwhitneyu(p0, p05, alternative="less").pvalue
except ValueError: pdeg = 1.0
H.append(dict(Hypothesis="H2 dial", Rule="crossover slope in [0.8,1.2]; p=0.05 slower than p=0 on glued trees", Observed=f"crossover slope {sl_c:.2f}; D(p) slope {sl_D:.2f}; one-sided p(p=0 faster than p=0.05)={pdeg:.3g}",
              Status="PASS" if (okc and pdeg < 0.05) else "NOT MET (see detail)"))
# H3
cl = []
for en in ENVS:
    r_ = t_sig[(t_sig.Env == en) & (t_sig.vs == "DistMatched-classical")].iloc[0]
    cl.append(f"{en}: " + ("coherence beyond spread" if (r_["p (Holm)"] < 0.05 and r_["Cliff delta (+ = DQW faster)"] > 0) else ("control faster" if r_["p (Holm)"] < 0.05 else "parity")))
H.append(dict(Hypothesis="H3 grids/corridors/rooms", Rule="DQW(tuned) vs distance-matched control, Holm<0.05 & delta>0", Observed="; ".join(cl), Status="per-environment classification"))
t_hyp = pd.DataFrame(H); print(t_hyp.to_string(index=False)); t_hyp.to_csv(f"{OUT}/table_hypotheses.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 0.55 * len(ENVS) + 1.5)); ypos = 0; labels = []
for en in ENVS:
    for comp, col in [(t_verdict[t_verdict.Env == en]["best classical walker"].iloc[0], "tab:blue"), ("DistMatched-classical", "tab:orange")]:
        r_ = t_sig[(t_sig.Env == en) & (t_sig.vs == comp)].iloc[0]; d_ = r_["Cliff delta (+ = DQW faster)"]
        ax.errorbar(d_, ypos, xerr=[[d_ - r_.delta_lo], [r_.delta_hi - d_]], fmt="o", color=col, capsize=3); labels.append(f"{en} vs {comp}"); ypos += 1
ax.axvline(0, color="k", lw=.8); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=6); ax.set_xlim(-1.05, 1.05); ax.set_xlabel("Cliff's delta (+ = DQW needs fewer steps), 95% bootstrap CI"); ax.invert_yaxis()
savefig("fig_forest_cliff_delta")

rows = []
for en in ENVS:
    e = make_env(en); T_ = TUNED[en]["dqw"][1]; x = df[(df.env == en) & (df.method == REF)]
    rows.append(dict(env=en, states=e.n, qubits_needed=int(np.ceil(np.log2(e.n))), walk_steps_T=T_, device_shots_per_waypoint=1, median_waypoint_queries_to_goal=np.median(x.wp_first), classical_sim_cost_per_waypoint=f"O(T n^2) = {T_ * e.n ** 2:,}"))
for en in GL_ENVS:
    e = make_env(en); T_ = GT(en, "dqw"); x = fg(rg, en, GREF)
    rows.append(dict(env=en, states=e.n, qubits_needed=int(np.ceil(np.log2(e.n))), walk_steps_T=T_, device_shots_per_waypoint=1, median_waypoint_queries_to_goal=float("nan"), classical_sim_cost_per_waypoint=f"O(T n^2) = {T_ * e.n ** 2:,}"))
t_res = pd.DataFrame(rows); print(t_res.to_string(index=False)); t_res.to_csv(f"{OUT}/table_quantum_resources.csv", index=False)

import json, platform, sys
json.dump(dict(python=sys.version, numpy=np.__version__, scipy=__import__("scipy").__version__, pandas=pd.__version__, matplotlib=__import__("matplotlib").__version__,
               platform=platform.platform(), QUICK=QUICK, N_SEEDS=N_SEEDS, N_TUNE=N_TUNE, N_SIDE=N_SIDE, N_GT=N_GT, N_EP=N_EP, GAMMA_RL=GAMMA_RL, ALPHA=ALPHA, GAMMA_H=GAMMA_H, BURST=BURST,
               ENVS=ENVS, GLUED_DEPTHS=GLUED_DEPTHS, GRID={k: [float(x) for x in v] for k, v in GRID.items()}, TUNED={k: {a: (list(b) if isinstance(b, tuple) else b) for a, b in v.items()} for k, v in TUNED.items()},
               GLUED_TUNED=MF, protocol='v2.1', seeds_main=list(range(N_SEEDS)), seeds_tuning=[10_000 + i for i in range(N_TUNE)], runtime_seconds=round(_time.time() - T_START)),
          open(f"{OUT}/run_info.json", "w"), indent=1)

# %% Tables to CSV / LaTeX and package
t1 = pd.DataFrame([("Setting", "known map, hidden sparse reward (learned-map ablation in Exp. E)"), ("RL", f"tabular Q-learning alpha={ALPHA}, gamma={GAMMA_RL}"),
    ("Exploration trigger", "eps_ep = max(0.05, 0.98^ep), identical for all methods"), ("Waypoint execution", f"shortest path then {BURST} random (untried-first) actions"),
    ("Training episodes / horizon", f"{N_EP} / max(100, 4*path)"), ("Hopping rate gamma*dt", GAMMA_H), ("Tuning grids", str(GRID)),
    ("Seeds (main / tuning / side)", f"{N_SEEDS} / {N_TUNE} / {N_SIDE}"), ("Censoring", "never-found runs are set to the step budget"),
    ("Reference method", REF)], columns=["Item", "Value"])
for name, t in [("table1_protocol", t1), ("table2_main_results", t_main), ("table3_significance", t_sig), ("table4_verdict_vs_best_classical", t_verdict), ("table5_scaling_exponents", t_scale)]:
    t.to_csv(f"{OUT}/{name}.csv", index=False)
    try: t.to_latex(f"{OUT}/{name}.tex", index=False, float_format="%.3f")
    except Exception: open(f"{OUT}/{name}.tex", "w").write(t.to_string())
with zipfile.ZipFile("results_v2.zip", "w") as z:
    for f in sorted(os.listdir(OUT)): z.write(f"{OUT}/{f}")
print("saved results_v2.zip with", len(os.listdir(OUT)), "files")
