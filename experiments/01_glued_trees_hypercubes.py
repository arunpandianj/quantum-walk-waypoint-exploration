# %% [markdown]
# # Glued trees and hypercubes: coherent quantum-walk exploration vs matched classical strategies (focused notebook)
# **Purpose.** The v2.1 study showed that, with a known map, a fully coherent walk finds a hidden goal on glued trees in
# about 2*depth+1 steps while classical walkers need exponentially more, and that coherence adds nothing on grids,
# corridors and rooms. This notebook hardens that finding for publication:
# 1. **Second family (confirmation set).** Hypercubes, where a coherent walk has *perfect* state transfer to the antipode
#    (classical hitting is exponential). Hypotheses below were written after seeing glued-tree results, so the
#    hypercube result is the held-out confirmation.
# 2. **Distributional reporting.** Per-run data are saved; we report medians, means (restricted), Hit10 = P(goal found within 10 x path length)
#    with Wilson CIs, ECDFs, queries (device shots) and censoring. v2.1 medians were unstable near a 50% success rate.
# 3. **Reward-free walk time.** A rule that needs no reward and no tuning: choose the first local maximum of the
#    expected waypoint distance (within 5% of its global maximum). It is compared with a reward-tuned T.
# 4. **Structure-aware reference.** `Farthest-first` goes to the farthest known node. It uses structural knowledge that the
#    walkers do not have, so it is a reference, not a competitor: it shows what the quantum walk matches *without* being told the structure.
# 5. **Noise budget.** Success versus p*T (probability of at least one measurement during a waypoint walk).
# 6. **Exact theory table.** Quantum and classical hitting probabilities and the chosen T for every graph.
#
# ## Hypotheses (decision rules fixed here)
# * **H1 (glued trees, cube).** `DQW p=0 (auto T)` has a lower median steps-to-goal than every classical *walker* (eps-greedy, Uniform, CRW, NBRW, Levy), Holm p < 0.05, at every depth >= 5.
# * **H2 (scaling).** For DQW(auto T), AICc prefers the power-law model over the exponential model (dAICc > 2) in both families; for CRW the exponential model is preferred.
# * **H3 (reward-free T).** Median steps of auto-T are within 1.25x of reward-tuned-T at every depth.
# * **H4 (noise).** Hit10 falls below 0.5 once p*T exceeds about 1 (report the measured threshold).
# * **H5 (reference).** DQW(auto T) median is within 2x of Farthest-first (matches the structure-aware optimum without being told the structure).
# **Gluing instances.** Every seed uses its own random gluing of the two trees (instance = seed mod N_INST), so results do not depend on one lucky graph.
# Limitations to state in the paper: known map; classically simulated; the speed-up is the known glued-tree / hypercube phenomenon.

# %% Setup
import os, math, time, warnings, zipfile, types, json, platform, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse.csgraph import shortest_path
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
T_START = time.time()
QUICK   = bool(int(os.environ.get("QUICK", "0")))
N_SEEDS = 4 if QUICK else 20       # evaluation seeds
N_TUNE  = 2 if QUICK else 6        # tuning seeds (separate)
N_SIDE  = 3 if QUICK else 20       # seeds for sensitivity / noise experiments
N_EP    = 200                      # training episodes per run
N_INST  = 4 if QUICK else 20       # number of distinct random gluings of the glued trees (instance = seed % N_INST)
GAMMA_RL, ALPHA, GAMMA_H, BURST, EVAL_EVERY, EVAL_ROLL = 0.99, 0.5, 0.5, 3, 15, 3
DEPTHS  = {"Glued": [3, 4] if QUICK else [4, 5, 6, 7, 8, 9], "Cube": [3, 4] if QUICK else [4, 5, 6, 7, 8, 9]}
ENVS    = [f"{f}-{d}" for f, ds in DEPTHS.items() for d in ds]
MULT    = [0.5, 1, 1.5, 1.75, 2, 2.5, 3, 5]
GRID_LEVY = [0.6, 0.9, 1.2, 1.6, 2.2]
OUT = "results_glued"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
def savefig(name):
    plt.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight"); plt.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); plt.show()

# %% Environments: glued trees (Childs et al. 2003) and hypercubes; exit / antipode is the hidden goal
class TabEnv:
    def __init__(self, name, NB, s0, g, H, scale):
        self.name, self.NB, self.s0, self.g, self.H, self.scale, self.inv = name, NB, s0, g, H, scale, None
        n, A = NB.shape; self.n, self.A = n, A
        adj = np.zeros((n, n), bool); act = -np.ones((n, n), int)
        for s in range(n):
            for a in range(A):
                s2 = NB[s, a]
                if s2 != s:
                    adj[s, s2] = adj[s2, s] = True
                    if act[s, s2] < 0: act[s, s2] = a
        self.adj, self.act_to = adj, act
        self.D = shortest_path(adj.astype(float), unweighted=True); assert np.isfinite(self.D).all()
        self.path_len = int(self.D[s0, g])

def glued_trees(d, seed=0):
    rng = np.random.default_rng(seed); m = 2 ** (d + 1) - 1; n = 2 * m; nbr = [[] for _ in range(n)]
    for off in (0, m):
        for i in range(2 ** d - 1):
            for ch in (2 * i + 1, 2 * i + 2): nbr[off + i].append(off + ch); nbr[off + ch].append(off + i)
    L = list(rng.permutation(np.arange(2 ** d - 1, m))); R = list(m + rng.permutation(np.arange(2 ** d - 1, m)))
    cyc = [x for pair in zip(L, R) for x in pair]
    for i in range(len(cyc)): a, b = cyc[i], cyc[(i + 1) % len(cyc)]; nbr[a].append(b); nbr[b].append(a)
    NB = np.array([sorted(set(x)) + [i] * (3 - len(set(x))) for i, x in enumerate(nbr)])
    return TabEnv(f"Glued-{d}", NB, 0, m, 30 * d, d)

def hypercube(n):
    N = 2 ** n; NB = np.array([[s ^ (1 << a) for a in range(n)] for s in range(N)]); return TabEnv(f"Cube-{n}", NB, 0, N - 1, 30 * n, n)

_ENV, _ENV_INST = {}, {}
def make_env(name):
    """'Glued-d' = gluing instance 0; 'Glued-d@k' = random gluing with seed k (kept in a tiny cache to bound memory)."""
    if "@" in name:
        if name not in _ENV_INST:
            _ENV_INST[name] = glued_trees(int(name[6:name.index("@")]), seed=int(name[name.index("@") + 1:]))
            while len(_ENV_INST) > 2: _ENV_INST.pop(next(iter(_ENV_INST)))
        return _ENV_INST[name]
    if name not in _ENV: _ENV[name] = glued_trees(int(name[6:])) if name.startswith("Glued-") else hypercube(int(name[5:]))
    return _ENV[name]
print({en: (make_env(en).n, make_env(en).path_len) for en in ENVS})

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
    elif kind == "farthest":                              # structure-aware reference (NOT a walker): farthest known node
        cand = np.flatnonzero(C.Dk[ls] == C.Dk[ls].max()); l = int(cand[int(rng.random() * len(cand))])
    else: raise ValueError(kind)
    return -1 if l == ls else int(C.K[l])

def auto_T(C, ls):
    """Reward-free walk-time rule: first local maximum of the expected waypoint distance E[d(T)] (within 5% of its global maximum over T <= 6*ecc+10)."""
    key = ("autoT", int(ls)); 
    if key in C.p0: return C.p0[key]
    Dls = C.Dk[ls]; Ts = np.arange(1, int(6 * Dls.max()) + 11)
    Psi = C.V @ (np.exp(-1j * np.outer(C.lam, Ts)) * C.V[ls][:, None]); E = Dls @ (np.abs(Psi) ** 2)
    T_ = int(np.argmax(E)) + 1                           # fallback: global maximum
    for i in range(1, len(E) - 1):                         # first local maximum within 5% of the global maximum
        if E[i] >= E[i - 1] and E[i] > E[i + 1] and E[i] >= 0.95 * E.max(): T_ = i + 1; break
    C.p0[key] = T_; return T_

# %% Agents: tabular Q-learning; exploration = waypoint (walker) + shortest-path navigation + short random burst
STRUCT = {"uniform", "crw", "nbrw", "levy", "dqw", "distmatch", "farthest"}
def run(cfg, env_name, seed, n_ep=N_EP):
    n_ep = cfg.get("n_ep", n_ep); rng = np.random.default_rng(seed)
    if env_name.startswith("Glued-") and "@" not in env_name and cfg.get("rand_inst", True): env_name = f"{env_name}@{seed % N_INST}"   # every seed gets its own random gluing
    env = make_env(env_name); n, A, s0, g, H, NB = env.n, env.A, env.s0, env.g, env.H, env.NB
    kind = cfg["kind"]; auto = bool(cfg.get("auto", False)); oracle = cfg.get("oracle", True); ham = cfg.get("ham", "lap"); T = cfg.get("T", 16); p = cfg.get("p", 0.0)
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
                        while len(_CACHE) > 2: _CACHE.pop(next(iter(_CACHE)))
                    if auto: T = auto_T(C, int(C.g2l[s0]))
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


# %% Statistics helpers
rngb = np.random.default_rng(123)
def boot_ci(x, fn=np.median, B=3000):
    x = np.asarray(x, float); bs = [fn(x[rngb.integers(0, len(x), len(x))]) for _ in range(B)]; return np.percentile(bs, [2.5, 97.5])
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den; return (c - h, c + h)
def cliffs(a, b):
    a, b = np.asarray(a), np.asarray(b); return ((a[:, None] < b[None, :]).mean() - (a[:, None] > b[None, :]).mean())
def holm(p):
    p = np.array(p); o = np.argsort(p); adj = np.empty_like(p); run_ = 0
    for r, i in enumerate(o): run_ = max(run_, (len(p) - r) * p[i]); adj[i] = min(1, run_)
    return adj
def logrank_p(a, ca, b, cb):
    try:
        x = stats.CensoredData(uncensored=a[~ca], right=a[ca]); y = stats.CensoredData(uncensored=b[~cb], right=b[cb]); return float(stats.logrank(x, y).pvalue)
    except Exception: return float("nan")
def aicc(y, yhat, k=3):
    n = len(y); rss = max(((y - yhat) ** 2).sum(), 1e-12); return n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)
def fg(res, en, mn): return np.array([metrics(o)["first_goal"] for _, o in res[(en, mn)]], float)

# %% Exact theory table: quantum (coherent) vs classical continuous-time hitting, and the reward-free walk time
rows, THEORY = [], {}
for en in ENVS:
    e = make_env(en); C = build_cache(np.ones(e.n, bool), e.adj, True, False, "adj"); ls = int(C.g2l[e.s0]); T_auto = auto_T(C, ls)
    Ts = np.arange(1, int(6 * C.Dk[ls].max()) + 11); Psi = C.V @ (np.exp(-1j * np.outer(C.lam, Ts)) * C.V[ls][:, None]); Pq = np.abs(Psi[int(C.g2l[e.g])]) ** 2
    Lp = np.diag(e.adj.sum(1).astype(float)) - e.adj; lam2, V2 = np.linalg.eigh(Lp)
    Pc = np.array([(V2[e.g] * np.exp(-GAMMA_H * lam2 * t) * V2[e.s0]).sum() for t in Ts])
    THEORY[en] = dict(Ts=Ts, Pq=Pq, Pc=Pc, T_auto=T_auto)
    rows.append(dict(env=en, states=e.n, path=e.path_len, T_auto=T_auto, P_goal_quantum_at_T_auto=Pq[T_auto - 1], max_P_goal_quantum=Pq.max(), argmax_T_quantum=int(Ts[np.argmax(Pq)]),
                     P_goal_classical_at_T_auto=Pc[T_auto - 1], max_P_goal_classical=Pc.max()))
    _CACHE.clear()
tab_theory = pd.DataFrame(rows); print(tab_theory.round(4).to_string(index=False)); tab_theory.to_csv(f"{OUT}/table_theory_hitting.csv", index=False)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for fam, ls_, mk in [("Glued", "-", "o"), ("Cube", "--", "s")]:
    s_ = tab_theory[tab_theory.env.str.startswith(fam)]; d_ = [int(x.split("-")[1]) for x in s_.env]
    axs[0].semilogy(d_, s_.P_goal_quantum_at_T_auto, mk + ls_, label=f"{fam}: coherent walk at T_auto"); axs[0].semilogy(d_, s_.max_P_goal_classical, mk + ":", label=f"{fam}: best classical (any t)")
    axs[1].plot(d_, s_.T_auto, mk + ls_, label=fam)
axs[0].set_xlabel("depth / dimension"); axs[0].set_ylabel("probability at goal"); axs[0].legend(fontsize=6); axs[1].set_xlabel("depth / dimension"); axs[1].set_ylabel("reward-free walk time T_auto"); axs[1].legend()
pick = lambda fam, pref: f"{fam}-{pref if pref in DEPTHS[fam] else DEPTHS[fam][len(DEPTHS[fam]) // 2]}"
for en, c_ in [(pick("Glued", 6), "tab:blue"), (pick("Cube", 8), "tab:orange")]:
    th = THEORY[en]; axs[2].plot(th["Ts"], th["Pq"], color=c_, label=f"{en} P(goal)"); axs[2].axvline(th["T_auto"], color=c_, ls=":")
axs[2].set_xlabel("walk time T"); axs[2].set_xlim(0, 60); axs[2].legend(fontsize=7); axs[2].set_title("dotted = reward-free T_auto", fontsize=8); savefig("fig1_theory_hitting_and_T_rule")

# %% Tuning of the classical walkers (and reward-tuned T for the coherent walk) on separate seeds
t0 = time.time(); Tm = lambda en, m: max(1, int(round(m * make_env(en).scale)))
stageT = {}
for m in MULT:
    stageT[f"crw|m={m}"] = (lambda en, m=m: dict(kind="crw", T=Tm(en, m), n_ep=N_EP)); stageT[f"nbrw|m={m}"] = (lambda en, m=m: dict(kind="nbrw", T=Tm(en, m), n_ep=N_EP))
    stageT[f"dqw|m={m}"] = (lambda en, m=m: dict(kind="dqw", p=0.0, T=Tm(en, m), ham="adj", n_ep=N_EP))
for al in GRID_LEVY: stageT[f"levy|a={al}"] = (lambda en, al=al: dict(kind="levy", alpha=al, n_ep=N_EP))
rtT = run_batch(stageT, [20_000 + i for i in range(N_TUNE)], ENVS)
def bestT(en, k): return min([n_ for n_ in stageT if n_.startswith(k + "|")], key=lambda n_: np.mean(np.log(fg(rtT, en, n_))))
TUN = {en: dict(crw=float(bestT(en, "crw").split("=")[1]), nbrw=float(bestT(en, "nbrw").split("=")[1]), dqw=float(bestT(en, "dqw").split("=")[1]), levy=float(bestT(en, "levy").split("=")[1])) for en in ENVS}
print(f"tuning done in {time.time()-t0:.0f}s"); tab_tun = pd.DataFrame(TUN).T; print(tab_tun); tab_tun.to_csv(f"{OUT}/table_tuned.csv")

# %% Main experiment
t0 = time.time()
METH = {
 "eps-greedy":              lambda en: dict(kind="eps", n_ep=N_EP),
 "Uniform-waypoint":        lambda en: dict(kind="uniform", n_ep=N_EP),
 "CRW-waypoint":            lambda en: dict(kind="crw", T=Tm(en, TUN[en]["crw"]), n_ep=N_EP),
 "NBRW-waypoint":           lambda en: dict(kind="nbrw", T=Tm(en, TUN[en]["nbrw"]), n_ep=N_EP),
 "Levy-waypoint":           lambda en: dict(kind="levy", alpha=TUN[en]["levy"], n_ep=N_EP),
 "DQW p=0 (tuned T)":       lambda en: dict(kind="dqw", p=0.0, T=Tm(en, TUN[en]["dqw"]), ham="adj", n_ep=N_EP),
 "DQW p=0 (auto T)":        lambda en: dict(kind="dqw", p=0.0, auto=True, ham="adj", n_ep=N_EP),
 "DistMatched (control)":   lambda en: dict(kind="distmatch", p=0.0, auto=True, ham="adj", n_ep=N_EP),
 "Farthest-first (reference)": lambda en: dict(kind="farthest", n_ep=N_EP),
}
REF = "DQW p=0 (auto T)"; WALKERS = ["eps-greedy", "Uniform-waypoint", "CRW-waypoint", "NBRW-waypoint", "Levy-waypoint"]
res = run_batch(METH, list(range(N_SEEDS)), ENVS); print(f"main done in {time.time()-t0:.0f}s")
rows = []
for (en, mn), lst in res.items():
    e = make_env(en)
    for s, o in lst:
        m = metrics(o); rows.append(dict(env=en, family=en.split("-")[0], depth=int(en.split("-")[1]), method=mn, seed=s, path=e.path_len, cap=o["cap"], hit10=int(m["first_goal"] <= 10 * e.path_len), **m))
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/raw_per_run.csv", index=False)

# %% Tables: distributional summary, significance, scaling model comparison, hypotheses
tm = []
for en in ENVS:
    for mn in METH:
        x = df[(df.env == en) & (df.method == mn)]; v = x.first_goal.values; lo, hi = boot_ci(v); mlo, mhi = boot_ci(v, np.mean); wl, wh = wilson(int(x.hit10.sum()), len(x))
        tm.append(dict(env=en, method=mn, median=np.median(v), med_lo=lo, med_hi=hi, mean=v.mean(), mean_lo=mlo, mean_hi=mhi, hit10=x.hit10.mean(), hit10_lo=wl, hit10_hi=wh,
                       median_queries=np.median(x.wp_first), never_found=int(x.censored.sum()), n=len(x)))
t_main = pd.DataFrame(tm); t_main.to_csv(f"{OUT}/table_main_distributional.csv", index=False)
CLASS = {"eps-greedy": "walker", "Uniform-waypoint": "walker", "CRW-waypoint": "walker", "NBRW-waypoint": "walker", "Levy-waypoint": "walker", "DQW p=0 (tuned T)": "quantum variant", "DistMatched (control)": "control", "Farthest-first (reference)": "reference"}
ts = []
for en in ENVS:
    a = df[(df.env == en) & (df.method == REF)]; av = a.first_goal.values; ca = a.censored.values.astype(bool); tmp = []
    for mn in METH:
        if mn == REF: continue
        b = df[(df.env == en) & (df.method == mn)]; bv = b.first_goal.values; cb = b.censored.values.astype(bool)
        try: pv = stats.mannwhitneyu(av, bv, alternative="two-sided").pvalue
        except ValueError: pv = 1.0
        bd = [cliffs(av[rngb.integers(0, len(av), len(av))], bv[rngb.integers(0, len(bv), len(bv))]) for _ in range(600)]
        tmp.append(dict(env=en, vs=mn, role=CLASS[mn], median_ref=np.median(av), median_other=np.median(bv), ratio=np.median(av) / max(np.median(bv), 1), cliff=cliffs(av, bv), cliff_lo=np.percentile(bd, 2.5), cliff_hi=np.percentile(bd, 97.5), p_mwu=pv, p_logrank=logrank_p(av, ca, bv, cb)))
    for r_, pa in zip(tmp, holm([r_["p_mwu"] for r_ in tmp])): r_["p_holm"] = pa
    ts += tmp
t_sig = pd.DataFrame(ts); t_sig.to_csv(f"{OUT}/table_significance.csv", index=False)

# scaling: exponential vs power-law model (AICc) on median and mean steps vs depth, bootstrap CIs on the rate / exponent
tsc = []
for fam in DEPTHS:
    ens = [f"{fam}-{d}" for d in DEPTHS[fam]]; dd = np.array(DEPTHS[fam], float)
    for mn in METH:
        data = [df[(df.env == en) & (df.method == mn)].first_goal.values for en in ens]
        for stat_name, fn in [("median", np.median), ("mean", np.mean)]:
            y = np.log([max(fn(v), 1) for v in data]); pe = np.polyfit(dd, y, 1); pp = np.polyfit(np.log(dd), y, 1)
            ae, ap = aicc(y, np.polyval(pe, dd)), aicc(y, np.polyval(pp, np.log(dd)))
            bs = [(np.polyfit(dd, np.log([max(fn(w[rngb.integers(0, len(w), len(w))]), 1) for w in data]), 1)[0]) for _ in range(500)]
            bp = [(np.polyfit(np.log(dd), np.log([max(fn(w[rngb.integers(0, len(w), len(w))]), 1) for w in data]), 1)[0]) for _ in range(500)]
            tsc.append(dict(family=fam, method=mn, statistic=stat_name, exp_rate=pe[0], exp_lo=np.percentile(bs, 2.5), exp_hi=np.percentile(bs, 97.5), power_exponent=pp[0], pow_lo=np.percentile(bp, 2.5), pow_hi=np.percentile(bp, 97.5),
                            AICc_exp=ae, AICc_power=ap, dAICc_exp_minus_power=ae - ap, preferred="power-law" if ae - ap > 2 else ("exponential" if ap - ae > 2 else "indistinguishable")))
t_scale = pd.DataFrame(tsc); t_scale.to_csv(f"{OUT}/table_scaling_models.csv", index=False)
print(t_scale[t_scale.statistic == "median"][["family", "method", "exp_rate", "power_exponent", "dAICc_exp_minus_power", "preferred"]].round(3).to_string(index=False))

# %% Figures: scaling, ECDFs, forest plot
fig, axs = plt.subplots(2, 2, figsize=(11, 7)); cols = plt.cm.tab10(np.arange(len(METH))); CM = dict(zip(METH, cols))
for j, fam in enumerate(DEPTHS):
    for i, (stat_name, fn) in enumerate([("median", np.median), ("mean", np.mean)]):
        ax = axs[i, j]
        for mn in METH:
            ys = [fn(df[(df.env == f"{fam}-{d}") & (df.method == mn)].first_goal.values) for d in DEPTHS[fam]]
            ax.semilogy(DEPTHS[fam], ys, "o-", color=CM[mn], lw=2.4 if mn == REF else 1.1, label=mn)
        ax.set_xlabel("depth" if fam == "Glued" else "dimension"); ax.set_ylabel(f"{stat_name} steps to first goal"); ax.set_title(fam)
axs[0, 0].legend(fontsize=6); plt.tight_layout(); savefig("fig2_scaling")
sel = [pick("Glued", 7), pick("Glued", DEPTHS["Glued"][-1]), pick("Cube", 7), pick("Cube", DEPTHS["Cube"][-1])]; sel = list(dict.fromkeys(sel))
fig, axs = plt.subplots(1, len(sel), figsize=(3.4 * len(sel), 3.2), sharey=True)
for ax, en in zip(np.atleast_1d(axs), sel):
    cap_ = N_EP * make_env(en).H; xs = np.unique(np.concatenate([[1], np.logspace(0, np.log10(cap_), 200)]))
    for mn in METH:
        v = df[(df.env == en) & (df.method == mn)].first_goal.values; ax.step(xs, [(v <= q).mean() for q in xs], color=CM[mn], lw=2.4 if mn == REF else 1.1, label=mn)
    ax.set_xscale("log"); ax.set_title(en, fontsize=9); ax.set_xlabel("environment steps")
np.atleast_1d(axs)[0].set_ylabel("P(goal found by step t)"); np.atleast_1d(axs)[0].legend(fontsize=5); plt.tight_layout(); savefig("fig3_ecdf")
fig, ax = plt.subplots(figsize=(6.5, 0.25 * len(ENVS) * 2 + 1.5)); ypos, labels = 0, []
for en in ENVS:
    for mn, col in [("Levy-waypoint", "tab:purple"), ("Uniform-waypoint", "tab:orange")]:
        r_ = t_sig[(t_sig.env == en) & (t_sig.vs == mn)].iloc[0]; ax.errorbar(r_.cliff, ypos, xerr=[[r_.cliff - r_.cliff_lo], [r_.cliff_hi - r_.cliff]], fmt="o", color=col, capsize=2); labels.append(f"{en} vs {mn.split('-')[0]}"); ypos += 1
ax.axvline(0, color="k", lw=.8); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=5); ax.set_xlim(-1.05, 1.05); ax.invert_yaxis(); ax.set_xlabel("Cliff's delta (+ = DQW auto-T needs fewer steps)"); savefig("fig4_forest")

# %% Walk-time sensitivity and noise budget (p*T)
GP = list(dict.fromkeys([pick("Glued", 6), pick("Cube", 8)])); MS = [0.5, 1, 1.5, 1.75, 2, 2.5, 3, 5, 8]
rts = run_batch({**{f"DQW p=0|m={m}": (lambda en, m=m: dict(kind="dqw", p=0.0, T=Tm(en, m), ham="adj", n_ep=N_EP)) for m in MS},
                 **{f"CRW|m={m}": (lambda en, m=m: dict(kind="crw", T=Tm(en, m), n_ep=N_EP)) for m in MS},
                 **{f"NBRW|m={m}": (lambda en, m=m: dict(kind="nbrw", T=Tm(en, m), n_ep=N_EP)) for m in MS}}, list(range(N_SIDE)), GP)
fig, axs = plt.subplots(1, len(GP), figsize=(5 * len(GP), 3.5)); rows = []
for ax, en in zip(np.atleast_1d(axs), GP):
    for nm in ["DQW p=0", "CRW", "NBRW"]:
        M_ = np.array([fg(rts, en, f"{nm}|m={m}") for m in MS]); md = np.median(M_, 1); ci_ = np.array([boot_ci(r) for r in M_])
        ax.errorbar(MS, md, [md - ci_[:, 0], ci_[:, 1] - md], fmt="o-", capsize=3, label=nm); rows += [dict(env=en, method=nm, multiplier=m, median_steps=md[i]) for i, m in enumerate(MS)]
    ax.axvline(THEORY[en]["T_auto"] / make_env(en).scale, color="k", ls=":", label="auto T"); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("walk time T / scale"); ax.set_title(en); ax.legend(fontsize=7)
np.atleast_1d(axs)[0].set_ylabel("median steps to first goal"); savefig("fig5_walk_time_sensitivity"); pd.DataFrame(rows).to_csv(f"{OUT}/table_walk_time_sensitivity.csv", index=False)

NOISE_ENVS = list(dict.fromkeys([pick("Glued", 5), pick("Glued", 6), pick("Cube", 6), pick("Cube", 8)])); PN = [0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0]
rn = run_batch({f"p={q:g}": (lambda en, q=q: dict(kind="dqw", p=q, auto=True, ham="adj", n_ep=N_EP)) for q in PN}, list(range(N_SIDE)), NOISE_ENVS); rows = []
for en in NOISE_ENVS:
    for q in PN:
        v = fg(rn, en, f"p={q:g}"); pth = 10 * make_env(en).path_len; k = int((v <= pth).sum()); lo, hi = wilson(k, len(v))
        rows.append(dict(env=en, p=q, T=THEORY[en]["T_auto"], pT=q * THEORY[en]["T_auto"], median=np.median(v), hit10=k / len(v), hit10_lo=lo, hit10_hi=hi))
t_noise = pd.DataFrame(rows); t_noise.to_csv(f"{OUT}/table_noise_budget.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for en in NOISE_ENVS:
    s_ = t_noise[t_noise.env == en]; axs[0].semilogx(s_.p.replace(0, 1e-3), s_.hit10, "o-", label=en); axs[1].semilogx(s_.pT.replace(0, 1e-2), s_.hit10, "o-", label=en)
axs[0].set_xlabel("dephasing p (p=0 plotted at 1e-3)"); axs[1].set_xlabel("p x T (expected measurements per waypoint walk)"); axs[0].set_ylabel("Hit10 = P(found within 10 x path)"); axs[1].legend(fontsize=7)
savefig("fig6_noise_budget")

# %% Hypothesis check
H = []
ok1, det = True, []
for fam in DEPTHS:
    for d in [x for x in DEPTHS[fam] if x >= min(5, max(DEPTHS[fam]))]:
        r_ = t_sig[(t_sig.env == f"{fam}-{d}") & (t_sig.role == "walker")]; cond = bool((r_.median_ref < r_.median_other).all() and (r_.p_holm < 0.05).all()); ok1 &= cond
        if not cond: det.append(f"{fam}-{d} fails vs {list(r_[(r_.p_holm >= 0.05) | (r_.median_ref >= r_.median_other)].vs)}")
H.append(dict(H="H1 faster than every classical walker (depth>=5)", Status="PASS" if ok1 else "NOT MET", Detail="; ".join(det) or "all depths"))
sc = t_scale[t_scale.statistic == "median"]; ok2 = True; det = []
for fam in DEPTHS:
    q = sc[(sc.family == fam) & (sc.method == REF)].iloc[0]; c = sc[(sc.family == fam) & (sc.method == "CRW-waypoint")].iloc[0]
    cond = (q.dAICc_exp_minus_power > 2) and (c.dAICc_exp_minus_power < -2); ok2 &= bool(cond); det.append(f"{fam}: DQW dAICc={q.dAICc_exp_minus_power:.1f} ({q.preferred}), CRW dAICc={c.dAICc_exp_minus_power:.1f} ({c.preferred})")
H.append(dict(H="H2 power-law (DQW) vs exponential (CRW)", Status="PASS" if ok2 else "NOT MET", Detail="; ".join(det)))
r3 = [t_main[(t_main.env == en) & (t_main.method == "DQW p=0 (auto T)")]["median"].iloc[0] / max(t_main[(t_main.env == en) & (t_main.method == "DQW p=0 (tuned T)")]["median"].iloc[0], 1) for en in ENVS]
H.append(dict(H="H3 auto-T within 1.25x of tuned-T", Status="PASS" if max(r3) <= 1.25 else "NOT MET", Detail=f"median ratio auto/tuned: min {min(r3):.2f}, max {max(r3):.2f}"))
thr = []
for en in NOISE_ENVS:
    s_ = t_noise[t_noise.env == en]; low = s_[s_.hit10 < 0.5]; thr.append(f"{en}: Hit10<0.5 first at pT={low.pT.min():.2f}" if len(low) else f"{en}: Hit10>=0.5 for all p")
H.append(dict(H="H4 noise budget (report threshold)", Status="reported", Detail="; ".join(thr)))
r5 = [t_main[(t_main.env == en) & (t_main.method == REF)]["median"].iloc[0] / max(t_main[(t_main.env == en) & (t_main.method == "Farthest-first (reference)")]["median"].iloc[0], 1) for en in ENVS]
H.append(dict(H="H5 within 2x of Farthest-first reference", Status="PASS" if max(r5) <= 2 else "NOT MET", Detail=f"median ratio DQW/Farthest: min {min(r5):.2f}, max {max(r5):.2f}"))
t_h = pd.DataFrame(H); pd.set_option("display.width", 250, "display.max_colwidth", 300); print(t_h.to_string(index=False)); t_h.to_csv(f"{OUT}/table_hypotheses.csv", index=False)

# %% Package
t_main.round(3).to_latex(f"{OUT}/table_main_distributional.tex", index=False) if False else None
json.dump(dict(python=sys.version, numpy=np.__version__, scipy=__import__("scipy").__version__, pandas=pd.__version__, platform=platform.platform(), QUICK=QUICK, N_SEEDS=N_SEEDS, N_TUNE=N_TUNE, N_SIDE=N_SIDE,
               N_EP=N_EP, GAMMA_H=GAMMA_H, DEPTHS=DEPTHS, TUNED=TUN, AUTO_T={en: int(THEORY[en]["T_auto"]) for en in ENVS}, runtime_seconds=round(time.time() - T_START)), open(f"{OUT}/run_info.json", "w"), indent=1)
with zipfile.ZipFile("results_glued.zip", "w") as z:
    for f in sorted(os.listdir(OUT)): z.write(f"{OUT}/{f}")
print("saved results_glued.zip with", len(os.listdir(OUT)), "files")
