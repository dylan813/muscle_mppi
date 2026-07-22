"""
CMA-ES optimizer for Hill muscle parameters of the walk task.

Optimizes x = [lce_min, lce_max, FVmax, pFLmax] (scalar per param, broadcast
to all 12 joints) to minimize the mean per-step locomotion cost from tasks.yaml.

Usage:
  python cmaes_walk.py               # run full optimization
  python cmaes_walk.py --test        # single eval at initial point (smoke test)
  python cmaes_walk.py --maxiter 20 --workers 6
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")   # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import yaml

# Ensure this directory is on the path for gait_generator import in workers
sys.path.insert(0, os.path.dirname(__file__))

import cma
from objective import evaluate, BASE_YAML

# ── search space (12D: hip/thigh/calf × lce_min/lce_max/FVmax/pFLmax) ────────
# All 4 legs share the same value per joint type.
PARAM_NAMES = [
    "lce_min_hip",  "lce_min_thigh",  "lce_min_calf",
    "lce_max_hip",  "lce_max_thigh",  "lce_max_calf",
    "FVmax_hip",    "FVmax_thigh",    "FVmax_calf",
    "pFLmax_hip",   "pFLmax_thigh",   "pFLmax_calf",
]
# Start from the 4D-optimized values (all joint types same)
X0 = [
    0.8474, 0.8474, 0.8474,   # lce_min: hip, thigh, calf
    1.1637, 1.1637, 1.1637,   # lce_max
    1.4391, 1.4391, 1.4391,   # FVmax
    0.8050, 0.8050, 0.8050,   # pFLmax
]
SIGMA0 = [
    0.05, 0.05, 0.05,   # lce_min
    0.10, 0.10, 0.10,   # lce_max
    0.25, 0.25, 0.25,   # FVmax
    0.30, 0.30, 0.30,   # pFLmax
]
LO = [0.50, 0.50, 0.50,  1.00, 1.00, 1.00,  1.00, 1.00, 1.00,  0.20, 0.20, 0.20]
HI = [0.90, 0.90, 0.90,  1.60, 1.60, 1.60,  3.00, 3.00, 3.00,  2.50, 2.50, 2.50]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _eval_worker(args):
    """Top-level function required for ProcessPoolExecutor pickling."""
    x, worker_id = args
    return evaluate(x, worker_id=worker_id, verbose=True)


def run_parallel(solutions, n_workers):
    """Evaluate a list of candidate solutions in parallel."""
    args = [(x, i) for i, x in enumerate(solutions)]
    costs = [None] * len(solutions)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_eval_worker, a): i for i, a in enumerate(args)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                costs[idx] = fut.result()
            except Exception as e:
                print(f"  Worker {idx} raised: {e}")
                costs[idx] = 1e9
    return costs


def save_best(x, fitness, generation):
    """Write best params found so far to results/best_params.yaml."""
    from objective import _broadcast_per_type
    with open(BASE_YAML) as f:
        base = yaml.safe_load(f)
    quad = base["default_muscle_quad"]

    best = {
        "generation": generation,
        "fitness": float(fitness),
        "params": {name: float(val) for name, val in zip(PARAM_NAMES, x)},
        # Per-joint-type scalars
        "by_type": {
            "hip":   {"lce_min": float(x[0]), "lce_max": float(x[3]), "FVmax": float(x[6]), "pFLmax": float(x[9])},
            "thigh": {"lce_min": float(x[1]), "lce_max": float(x[4]), "FVmax": float(x[7]), "pFLmax": float(x[10])},
            "calf":  {"lce_min": float(x[2]), "lce_max": float(x[5]), "FVmax": float(x[8]), "pFLmax": float(x[11])},
        },
        # Full 12-element arrays ready to paste into tasks.yaml
        "lce_min":  _broadcast_per_type(x[0:3]),
        "lce_max":  _broadcast_per_type(x[3:6]),
        "FVmax":    _broadcast_per_type(x[6:9]),
        "pFLmax":   _broadcast_per_type(x[9:12]),
        "vmax":       quad["vmax"],
        "phi_min":    quad["phi_min"],
        "phi_max":    quad["phi_max"],
        "peak_force": quad["peak_force"],
    }
    out = os.path.join(RESULTS_DIR, "best_params.yaml")
    with open(out, "w") as f:
        yaml.dump(best, f, default_flow_style=False)
    return out


# ── palette (reference palette slots 1/2/3 for hip/thigh/calf) ───────────────
_C_HIP   = "#2a78d6"   # slot 1 — blue
_C_THIGH = "#008300"   # slot 2 — green
_C_CALF  = "#e87ba4"   # slot 3 — magenta
_C_BEST  = "#2a78d6"   # blue — best fitness line
_C_MEAN  = "#52514e"   # secondary ink — mean fitness line
_SURFACE = "#fcfcfb"
_GRID    = "#e1e0d9"
_INK     = "#0b0b0b"
_INK2    = "#52514e"

_PARAM_LABELS = {
    "lce_min": "lce_min  (norm. fiber length)",
    "lce_max": "lce_max  (norm. fiber length)",
    "FVmax":   "FVmax  (eccentric amplification)",
    "pFLmax":  "pFLmax  (passive elasticity scale)",
}


def plot_progress(results_dir):
    """
    Read history.csv and save results/progress.png.
    Top row: 2 subplots — best feasible fitness, mean ± std feasible fitness.
    Bottom row: 4 subplots — one per param type, each with hip/thigh/calf lines.
    """
    hist_path = os.path.join(results_dir, "history.csv")
    if not os.path.exists(hist_path):
        return

    import pandas as pd
    df = pd.read_csv(hist_path)
    if df.empty:
        return

    gens = df["generation"].values

    fig = plt.figure(figsize=(14, 8), facecolor=_SURFACE)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.38,
                            top=0.91, bottom=0.09, left=0.06, right=0.98)

    # ── row 0: best fitness | mean ± std fitness (side by side) ──────────────
    ax_best = fig.add_subplot(gs[0, 0:2])
    ax_best.set_facecolor(_SURFACE)

    best = df["best_cost"].values

    # Exclude infeasible-penalty values so the y-axis stays readable
    feasible = best[best < 1e5]
    best_clip = feasible.max() * 1.1 if len(feasible) else best.max()
    best_plot = np.where(best < 1e5, best, np.nan)

    ax_best.plot(gens, best_plot, color=_C_BEST, linewidth=2.0, zorder=3)

    # Marker at overall best feasible point
    if len(feasible):
        best_gen_idx = int(np.nanargmin(best_plot))
        ax_best.scatter([gens[best_gen_idx]], [best_plot[best_gen_idx]],
                        color=_C_BEST, s=60, zorder=5, clip_on=False)

    ax_best.set_ylim(bottom=0, top=best_clip)
    ax_best.set_title("Best feasible fitness over generations", fontsize=10,
                      color=_INK, fontweight="semibold", loc="left", pad=6)
    ax_best.set_xlabel("Generation", fontsize=8, color=_INK2)
    ax_best.set_ylabel("Cost (feasible only)", fontsize=8, color=_INK2)
    _style_ax(ax_best)

    ax_mean = fig.add_subplot(gs[0, 2:4])
    ax_mean.set_facecolor(_SURFACE)

    if "mean_cost" in df.columns:
        mean_raw = df["mean_cost"].values
        std_raw  = df["std_cost"].values if "std_cost" in df.columns else np.zeros_like(mean_raw)
        mean_plot = np.where(mean_raw < 1e5, mean_raw, np.nan)
        std_plot  = np.where(mean_raw < 1e5, std_raw, np.nan)

        finite_hi = mean_plot + std_plot
        finite_hi = finite_hi[~np.isnan(finite_hi)]
        mean_clip = finite_hi.max() * 1.1 if len(finite_hi) else best_clip

        ax_mean.fill_between(gens, mean_plot - std_plot, mean_plot + std_plot,
                             color=_C_MEAN, alpha=0.15, linewidth=0, zorder=1)
        ax_mean.plot(gens, mean_plot, color=_C_MEAN, linewidth=1.8, zorder=2)
        ax_mean.set_ylim(bottom=0, top=mean_clip)

    ax_mean.set_title("Mean ± std fitness over generations", fontsize=10,
                      color=_INK, fontweight="semibold", loc="left", pad=6)
    ax_mean.set_xlabel("Generation", fontsize=8, color=_INK2)
    ax_mean.set_ylabel("Cost (feasible only)", fontsize=8, color=_INK2)
    _style_ax(ax_mean)

    # ── row 1: one subplot per param type ────────────────────────────────────
    param_keys  = ["lce_min", "lce_max", "FVmax", "pFLmax"]
    joint_types = ["hip", "thigh", "calf"]
    colors      = [_C_HIP, _C_THIGH, _C_CALF]

    for col, pk in enumerate(param_keys):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor(_SURFACE)

        for jt, color in zip(joint_types, colors):
            col_name = f"{pk}_{jt}"
            if col_name in df.columns:
                ax.plot(gens, df[col_name].values,
                        color=color, linewidth=1.8, label=jt)

        ax.set_title(_PARAM_LABELS[pk], fontsize=7.5, color=_INK,
                     fontweight="semibold", loc="left", pad=4)
        ax.set_xlabel("Generation", fontsize=7, color=_INK2)
        if col == 0:
            ax.legend(fontsize=7, frameon=False, labelcolor=_INK2,
                      loc="best", handlelength=1.2)
        _style_ax(ax, tick_size=7)

    fig.suptitle("CMA-ES — muscle parameter optimization",
                 fontsize=11, color=_INK, x=0.06, ha="left", fontweight="semibold")

    out = os.path.join(results_dir, "progress.png")
    fig.savefig(out, dpi=140, facecolor=_SURFACE)
    plt.close(fig)
    return out


def _style_ax(ax, tick_size=8):
    """Apply recessive grid/axes style consistent with reference palette."""
    ax.set_facecolor(_SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=tick_size, length=3)
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def append_history(generation, costs, solutions):
    """Append one row to results/history.csv."""
    path = os.path.join(RESULTS_DIR, "history.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["generation", "best_cost", "mean_cost", "std_cost"] + PARAM_NAMES)
        best_idx = int(np.argmin(costs))
        # Mean/std over feasible candidates only — matches the plot's
        # feasibility filter so infeasible-penalty outliers don't swamp them.
        feasible_costs = [c for c in costs if c < 1e5]
        mean_cost = float(np.mean(feasible_costs)) if feasible_costs else float("nan")
        std_cost  = float(np.std(feasible_costs))  if feasible_costs else float("nan")
        w.writerow([
            generation,
            f"{min(costs):.4f}",
            f"{mean_cost:.4f}",
            f"{std_cost:.4f}",
        ] + [f"{v:.6f}" for v in solutions[best_idx]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",    action="store_true",
                        help="Evaluate initial point only (smoke test)")
    parser.add_argument("--maxiter", type=int,   default=50)
    parser.add_argument("--workers", type=int,   default=8)
    parser.add_argument("--sigma",   type=float, default=None,
                        help="Override initial sigma (scalar); default: per-param")
    parser.add_argument("--popsize", type=int,   default=None,
                        help="CMA-ES population size (default: auto)")
    args = parser.parse_args()

    print(f"CMA-ES walk optimizer")
    print(f"  x0:      {dict(zip(PARAM_NAMES, X0))}")
    print(f"  bounds:  lo={LO}  hi={HI}")
    print(f"  results: {RESULTS_DIR}")
    print()

    if args.test:
        print("── SMOKE TEST (initial point) ──")
        t0 = time.time()
        fitness = evaluate(X0, worker_id=0, verbose=True)
        print(f"\nFitness = {fitness:.4f}  ({time.time()-t0:.1f} s)")
        return

    # CMA-ES uses a single scalar sigma; we scale x0 to unit-sigma space
    # by dividing by SIGMA0 elementwise, then CMA-ES searches in that space.
    # Alternatively, use CMA's built-in scaling via 'CMA_stds'.
    sigma0 = args.sigma if args.sigma is not None else 1.0

    opts = {
        "bounds":   [LO, HI],
        "maxiter":  args.maxiter,
        "tolx":     1e-3,
        "tolfun":   1e-3,
        "verbose":  3,
        "CMA_stds": SIGMA0,   # per-parameter step size scaling
    }
    if args.popsize:
        opts["popsize"] = args.popsize

    es = cma.CMAEvolutionStrategy(X0, sigma0, opts)
    cma_log_dir = os.path.join(RESULTS_DIR, "cmaes_log")
    os.makedirs(cma_log_dir, exist_ok=True)
    logger = cma.CMADataLogger(os.path.join(cma_log_dir, "run")).register(es)

    best_cost = float("inf")
    best_x    = list(X0)
    generation = 0

    print(f"Starting CMA-ES: popsize={es.popsize}, workers={args.workers}")
    print("─" * 60)

    while not es.stop():
        generation += 1
        solutions = es.ask()

        t0 = time.time()
        costs = run_parallel(solutions, n_workers=args.workers)
        elapsed = time.time() - t0

        es.tell(solutions, costs)
        logger.add()

        gen_best_idx  = int(np.argmin(costs))
        gen_best_cost = costs[gen_best_idx]

        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_x    = list(solutions[gen_best_idx])
            out = save_best(best_x, best_cost, generation)
            print(f"  ★ new best saved → {out}")

        append_history(generation, costs, solutions)
        plot_progress(RESULTS_DIR)

        n_feasible = sum(c < 1e5 for c in costs)
        print(f"\nGen {generation:3d} | best={gen_best_cost:.2f} "
              f"| feasible={n_feasible}/{len(costs)} | {elapsed:.0f}s")
        print(f"  best x: {dict(zip(PARAM_NAMES, best_x))}")
        print(f"  overall best fitness: {best_cost:.4f}")
        print("─" * 60)

    print("\n═══ Optimization complete ═══")
    print(f"Best fitness : {best_cost:.4f}")
    print(f"Best params  : {dict(zip(PARAM_NAMES, best_x))}")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
