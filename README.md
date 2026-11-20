# Dephased quantum-walk waypoint sampling for hidden-goal search on graphs with a known map

Reference implementation and analysis scripts for the article *"Dephasing-controlled quantum-walk waypoint sampling for hidden-goal search on graphs with a known map: conditions for a coherent advantage"* by Arun Pandian J and Chidambaram S (Vellore Institute of Technology, Vellore, India).

The repository contains the simulation code for every experiment reported in the article, the exact-theory calculations, and the scripts that generate the figures and tables. All computations run on a CPU with plain NumPy and SciPy; no GPU and no quantum hardware or quantum software development kit are required.

## 1. What the work proposes

An agent searches a graph whose structure is known and whose reward location (the goal) is hidden. At each decision point it draws a **waypoint** from a walk on the graph, travels to the waypoint along a shortest path, takes a few exploratory actions, and learns by tabular Q-learning. Every method in the study shares this agent. Only the waypoint sampler differs, so any difference in performance is attributable to the sampler.

The proposed sampler is a **dephased continuous-time quantum walk**. One walk step with dephasing probability `p` is the channel

    Phi_p(rho) = (1 - p) U rho U^dagger + p Delta(U rho U^dagger),     U = exp(-i H),  H = -gamma A  (gamma = 1/2)

where `Delta` removes all coherences in the node basis. The single parameter `p` interpolates between a coherent quantum walk (`p = 0`) and a classical Markov chain (`p = 1`). Main elements:

- **Single-shot form.** Apply `U`; with probability `p` measure the position and collapse. The final position measurement has exactly the law of the channel (Proposition 1 of the article), so one waypoint costs one measurement shot when `p = 0`.
- **Diffusion constant.** On a line the dephased walk has `D(p) = gamma^2 (2 - p) / p`, and the ballistic-to-diffusive crossover occurs at about `t* = 2.1 / p` steps.
- **Reward-independent walk duration.** `T_auto` is the first local maximum of the expected waypoint distance of the coherent walk. It uses only the graph and the start node.
- **Exact results.** On hypercubes the coherent antipode probability is `sin^(2n)(gamma t)` and the classical probability is below `2^(-n)`. On glued trees a symmetry reduction to a line of `2d + 2` sites gives exact goal probabilities up to depth 1000; the classical probability is bounded by `1 / n` with `n = 2^(d+2) - 2`.
- **Matched comparison.** Five tuned classical samplers (lazy random walk, non-backtracking walk, Levy waypoint, uniform waypoint, epsilon-greedy), a distance-matched control and a structure-aware farthest-first reference, with separate tuning seeds, right-censored step counts, bootstrap intervals, Holm-corrected tests and Cliff's delta.

Main findings reproduced by this code (20 evaluation seeds per method and environment):

| Setting | Result |
|---|---|
| Glued trees, depth 4 to 9 | coherent median steps equal the shortest path (9 to 19); the strongest classical walker (Levy waypoint) needs 69 to 321; Holm-adjusted p <= 0.021 against all five classical walkers |
| Hypercubes, dimension 4 to 9 | coherent median equals the dimension (4 to 9) in every run; Levy waypoint needs 9 to 448; Holm-adjusted p <= 0.0007 |
| Exact glued-tree probabilities | coherent probability decays as d^(-0.53) up to depth 1000; classical probability <= 1/n |
| Corridors, open grids, rooms | Levy waypoints need fewer steps than the tuned dephased sampler in all nine environments; no difference from the distance-matched control |
| Dephasing tolerance | glued-tree advantage degrades once p*T exceeds about 0.5 to 1 |

## 2. Repository layout

    experiments/
      01_glued_trees_hypercubes.py     agent experiments on glued trees and hypercubes, tuning, statistics, hypotheses
      02_corridors_grids_rooms.py      agent experiments on serpentine corridors, open grids and four-room layouts
      03_exact_theory_full_graphs.py   exact hitting probabilities on the full graphs (sparse time evolution), closed-form checks
      04_exact_reduced_chain.py        exact glued-tree asymptotics on the symmetry-reduced chain, depth up to 1000
      05_spreading_law.py              variance, crossover step and diffusion constant of the dephased walk on a line
    analysis/
      make_figures.py                  the seven figures (PDF and PNG)
      make_tables.py                   the result tables as LaTeX
    run_all.sh                         runs everything in order
    requirements.txt  LICENSE  CITATION.cff

The experiment scripts are ordinary Python files with `# %%` cell markers, so they also open as cells in editors that support them. They write their results to folders in the current working directory (`results_glued/`, `results_v2/`, `results_theory/`, `results_reduced/`, `results_spreading/`) and to a zip file of the same name.

## 3. Installation

Python 3.10 or newer is required. The reported results were produced with Python 3.13, NumPy 2.1, SciPy 1.16, pandas 2.2 to 2.3 and matplotlib 3.10; the scripts were also tested with Python 3.12, NumPy 2.5, SciPy 1.18, pandas 3.0 and matplotlib 3.11.

    git clone https://github.com/arunpandianj/quantum-walk-waypoint-exploration.git
    cd quantum-walk-waypoint-exploration
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

SciPy 1.11 or newer is needed for the log-rank test of censored data; with an older SciPy the log-rank column is left empty and everything else is unaffected.

## 4. Quick test

Every experiment script has a smoke-test mode, selected with the environment variable `QUICK=1`. It uses few seeds, short budgets and small environments, so it finishes in a few minutes and checks that the installation works. The numbers it produces are not meaningful.

    QUICK=1 python experiments/01_glued_trees_hypercubes.py
    QUICK=1 python experiments/02_corridors_grids_rooms.py
    QUICK=1 python experiments/03_exact_theory_full_graphs.py
    # or all at once
    bash run_all.sh quick

## 5. Reproducing the reported results

Run the scripts from the repository root, without `QUICK`:

| Step | Command | Produces | Article content | Measured runtime |
|---|---|---|---|---|
| 1 | `python experiments/01_glued_trees_hypercubes.py` | `results_glued/` | Tables on glued trees and hypercubes, Figures 4 to 6, hypotheses | 5538 s |
| 2 | `python experiments/02_corridors_grids_rooms.py` | `results_v2/` | Corridor, grid and room table, Figure 7 | 8046 s on four cores |
| 3 | `python experiments/03_exact_theory_full_graphs.py` | `results_theory/` | Full-graph exact probabilities and closed-form checks | a few minutes |
| 4 | `python experiments/04_exact_reduced_chain.py` | `results_reduced/` | Exact probability table to depth 1000, Figure 3 | about 1 minute |
| 5 | `python experiments/05_spreading_law.py` | `results_spreading/` | Figure 2 and the diffusion-constant comparison | about 2 minutes |
| 6 | `python analysis/make_figures.py` | `figures/*.pdf`, `figures/*.png` | Figures 1 to 7 | seconds |
| 7 | `python analysis/make_tables.py` | `figures/tables_generated.tex` | LaTeX tables of the article | seconds |

Runtimes are those of the reported runs on the machines used; they scale roughly linearly with the number of seeds. Memory use is about 0.5 GB per worker process for glued trees of depth 9 in step 1, and several gigabytes at the largest sizes of step 3 (hypercube dimension 17 and glued-tree depth 14); reduce the sizes at the top of the script if memory is limited.

Steps 6 and 7 read the outputs of steps 1, 2, 4 and 5. Their input folders can be changed with command-line arguments, for example `python analysis/make_figures.py --glued path/to/results_glued --grid path/to/results_v2`.

## 6. Configuration

The constants at the top of each experiment script control the study. The most useful ones:

| Constant | Script | Meaning |
|---|---|---|
| `QUICK` (environment variable) | 01, 02, 03 | smoke-test mode |
| `N_SEEDS` | 01, 02 | evaluation seeds per method and environment (20 in the article) |
| `N_TUNE` | 01, 02 | tuning seeds per configuration, disjoint from the evaluation seeds |
| `N_EP` | 01, 02 | training episodes per run |
| `GAMMA_H` | 01, 02 | hopping rate `gamma` of the walk |
| `GRID` / `MULT`, `GRID_LEVY` | 02 / 01 | tuning grids for walk durations and Levy exponents |
| `DEPTHS` | 01 | glued-tree depths and hypercube dimensions |

Seeds are fixed in the code, so a rerun with the same constants and library versions reproduces the reported numbers up to library-level floating-point differences. In experiment 01 each evaluation seed uses its own random gluing of the trees; the exact goal probabilities do not depend on the gluing (Proposition 3).

## 7. Outputs

Each experiment folder contains CSV tables (per-run metrics, summaries with bootstrap intervals, significance tests, tuned hyper-parameters), PNG and PDF figures of the exploratory analysis, `run_info.json` (library versions, seeds, tuned values, runtime) and a hypotheses table in which the pre-specified decision rules of the article are evaluated automatically.

Key files: `results_glued/table_main_distributional.csv` (medians, means, Hit10, queries), `results_glued/table_significance.csv`, `results_glued/table_hypotheses.csv`, `results_v2/table3_significance.csv`, `results_v2/table4_verdict_vs_best_classical.csv`, `results_reduced/table_reduced_chain.csv`.

## 8. Metrics

- **Steps to first goal** (`first_goal`): environment steps until the first visit of the hidden goal, right-censored at the budget `N_EP x horizon` for runs that never find it.
- **Hit10**: probability that the goal is found within 10 times the shortest-path length.
- **Waypoint queries**: number of sampler calls until the first goal visit (one measurement shot each on a quantum device when `p = 0`).
- **Statistics**: percentile bootstrap intervals, two-sided Mann-Whitney tests with Holm correction within each environment, Cliff's delta with bootstrap intervals, log-rank tests for censored times, small-sample Akaike criterion for exponential versus power-law growth.

## 9. Scope and limitations

The graph is known to the agent (a learned-map ablation is included in experiment 02). Learning is tabular. The quantum walk is simulated on a classical computer, so no computational speed-up is claimed; the results concern exploration behaviour. The coherent advantage appears on graph families with a unique distant goal (glued trees, hypercubes) and not on corridors, open grids and rooms, where a classical Levy sampler is faster. The reward-independent walk-duration rule is reliable for the sizes tested in the agent experiments and loses probability at depths of about 60 and above (see `results_reduced/table_rule_large_depth.csv`). A structure-aware planner with the same map matches the coherent sampler's median; see the article for the full discussion.

## 10. Citation and license

Please cite the associated article; the reference will be added here after publication. `CITATION.cff` contains the software citation metadata. The code is released under the MIT License (see `LICENSE`).

## 11. Contact

Arun Pandian J (corresponding author), Centre for Artificial Intelligence Research (C-FAIR), Vellore Institute of Technology, Vellore, India. E-mail: arunpandian.j@vit.ac.in

Chidambaram S, School of Computer Science Engineering and Information Systems, Vellore Institute of Technology, Vellore, India. E-mail: chidambaram.s2024@vitstudent.ac.in
