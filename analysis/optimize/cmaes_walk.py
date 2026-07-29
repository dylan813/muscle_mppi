"""
CMA-ES optimizer for Hill muscle parameters of the walk task.

Optimizes x = [lce_min, lce_max, pFLmax] (scalar per param, broadcast to all
12 joints) to minimize the mean per-step locomotion cost from tasks.yaml.
FVmax is held fixed per joint type (objective.FVMAX_HIP/THIGH/CALF, each
overridable via env var) rather than searched.

Usage:
  python cmaes_walk.py               # run full optimization
  python cmaes_walk.py --test        # single eval at initial point (smoke test)
  python cmaes_walk.py --maxiter 20 --workers 6
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import wandb

# Ensure this directory is on the path for gait_generator import in workers
sys.path.insert(0, os.path.dirname(__file__))

import cma
from objective import evaluate, render_rollout, RENDER_FPS

# ── search space (9D: hip/thigh/calf × lce_min/lce_max/pFLmax) ───────────────
# FVmax is fixed at objective.FVMAX_FIXED (not searched).
# All 4 legs share the same value per joint type.
PARAM_NAMES = [
    "lce_min_hip",  "lce_min_thigh",  "lce_min_calf",
    "lce_max_hip",  "lce_max_thigh",  "lce_max_calf",
    "pFLmax_hip",   "pFLmax_thigh",   "pFLmax_calf",
]
# Start from the 4D-optimized values (all joint types same)
X0 = [
    0.8474, 0.8474, 0.8474,   # lce_min: hip, thigh, calf
    1.1637, 1.1637, 1.1637,   # lce_max
    0.8050, 0.8050, 0.8050,   # pFLmax
]
SIGMA0 = [
    0.05, 0.05, 0.05,   # lce_min
    0.10, 0.10, 0.10,   # lce_max
    0.30, 0.30, 0.30,   # pFLmax
]
LO = [0.50, 0.50, 0.50,  1.00, 1.00, 1.00,  0.20, 0.20, 0.20]
HI = [0.90, 0.90, 0.90,  1.60, 1.60, 1.60,  2.50, 2.50, 2.50]

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
    parser.add_argument("--wandb-project", type=str, default="muscle-mppi-cmaes",
                        help="wandb project to log runs to")
    parser.add_argument("--run-name", type=str, default=None,
                        help="wandb run name (default: wandb auto-generates one)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results dir (default: ./results). Use a "
                             "distinct dir per process when running sweeps in "
                             "parallel, so cmaes_log files don't collide.")
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir else RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    print(f"CMA-ES walk optimizer")
    print(f"  x0:      {dict(zip(PARAM_NAMES, X0))}")
    print(f"  bounds:  lo={LO}  hi={HI}")
    print(f"  results: {results_dir}")
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
    cma_log_dir = os.path.join(results_dir, "cmaes_log")
    os.makedirs(cma_log_dir, exist_ok=True)
    logger = cma.CMADataLogger(os.path.join(cma_log_dir, "run")).register(es)

    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config={
            "x0": dict(zip(PARAM_NAMES, X0)),
            "sigma0": dict(zip(PARAM_NAMES, SIGMA0)),
            "lo": dict(zip(PARAM_NAMES, LO)),
            "hi": dict(zip(PARAM_NAMES, HI)),
            "maxiter": args.maxiter,
            "workers": args.workers,
            "sigma": sigma0,
            "popsize": args.popsize,
        },
    )

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
            print(f"  ★ new best: {best_cost:.4f}")
            wandb.run.summary["best_cost"] = best_cost
            wandb.run.summary["best_generation"] = generation
            for name, val in zip(PARAM_NAMES, best_x):
                wandb.run.summary[f"best/{name}"] = val

            frames = render_rollout(best_x)
            if frames is not None:
                wandb.log({"best_rollout": wandb.Video(frames, fps=RENDER_FPS, format="gif")},
                          step=generation)
            else:
                print("  (rollout render failed, skipping gif)")

        # Mean/std over feasible candidates only, so infeasible-penalty
        # outliers don't swamp them.
        feasible_costs = [c for c in costs if c < 1e5]
        mean_cost = float(np.mean(feasible_costs)) if feasible_costs else float("nan")
        std_cost  = float(np.std(feasible_costs))  if feasible_costs else float("nan")
        n_feasible = len(feasible_costs)

        log = {
            "generation": generation,
            "gen_best_cost": gen_best_cost,
            "overall_best_cost": best_cost,
            "mean_cost": mean_cost,
            "std_cost": std_cost,
            "n_feasible": n_feasible,
            "elapsed_s": elapsed,
        }
        for name, val in zip(PARAM_NAMES, solutions[gen_best_idx]):
            log[f"params/{name}"] = val
        wandb.log(log, step=generation)

        print(f"\nGen {generation:3d} | best={gen_best_cost:.2f} "
              f"| feasible={n_feasible}/{len(costs)} | {elapsed:.0f}s")
        print(f"  best x: {dict(zip(PARAM_NAMES, best_x))}")
        print(f"  overall best fitness: {best_cost:.4f}")
        print("─" * 60)

    print("\n═══ Optimization complete ═══")
    print(f"Best fitness : {best_cost:.4f}")
    print(f"Best params  : {dict(zip(PARAM_NAMES, best_x))}")
    print(f"Results saved to {results_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
