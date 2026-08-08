"""
CMA-ES optimizer for Hill muscle parameters of the walk task.

Optimizes x = [lce_min, lce_max, pFLmax, FVmax] (scalar per param, broadcast
to all 12 joints) to minimize the mean per-step locomotion cost from
tasks.yaml.

Usage:
  python cmaes_walk.py               # run full optimization
  python cmaes_walk.py --test        # single eval at initial point (smoke test)
  python cmaes_walk.py --maxiter 20 --workers 6
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

# Ensure this directory is on the path for gait_generator import in workers
sys.path.insert(0, os.path.dirname(__file__))

import cma
from objective import evaluate, render_rollout, curve_area_mean, RENDER_FPS
from curve_plots import plot_fl_curves, plot_fv_curves

# ── search space (12D: hip/thigh/calf × lce_min/lce_max/pFLmax/FVmax) ────────
# All 4 legs share the same value per joint type.
PARAM_NAMES = [
    "lce_min_hip",  "lce_min_thigh",  "lce_min_calf",
    "lce_max_hip",  "lce_max_thigh",  "lce_max_calf",
    "pFLmax_hip",   "pFLmax_thigh",   "pFLmax_calf",
    "FVmax_hip",    "FVmax_thigh",    "FVmax_calf",
]

X0 = [
    0.85, 0.82, 0.80,   # lce_min: hip, thigh, calf
    1.09, 1.06, 1.19,   # lce_max
    0.54, 0.94, 0.21,   # pFLmax
    1.2, 1.3, 1.1,   # FVmax
]
# ~15% of each param's (HI - LO) range, recomputed for the current bounds.
SIGMA0 = [
    0.07, 0.07, 0.07,   # lce_min
    0.07, 0.07, 0.07,   # lce_max
    0.25, 0.25, 0.25,   # pFLmax
    0.11, 0.11, 0.11,   # FVmax
]
# FVmax lower bound is kept strictly above 1.0: gait_generator._force_vel
# divides by (FVmax - 1.0), so a candidate at or below 1.0 is a singularity.
LO = [0.50, 0.50, 0.50,  1.05, 1.05, 1.05,  0.10, 0.10, 0.10,  1.05, 1.05, 1.05]
HI = [0.95, 0.95, 0.95,  1.50, 1.50, 1.50,  1.80, 1.80, 1.80,  1.80, 1.80, 1.80]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _eval_worker(args):
    """Top-level function required for ProcessPoolExecutor pickling."""
    x, worker_id = args
    return evaluate(x, worker_id=worker_id, verbose=True)


def _write_best_result(results_dir, best_cost, best_generation, best_x, cli_args, done=False):
    """Write the best-so-far result to results_dir/best_params.json, so it
    survives even if the run is interrupted before completing."""
    payload = {
        "best_cost": best_cost,
        "best_area_mean": curve_area_mean(best_x),
        "best_generation": best_generation,
        "best_params": dict(zip(PARAM_NAMES, best_x)),
        "run_complete": done,
        "run_args": {
            "maxiter": cli_args.maxiter,
            "popsize": cli_args.popsize,
            "workers": cli_args.workers,
            "seed": cli_args.seed,
            "cmd_vel_x": cli_args.cmd_vel_x,
        },
    }
    path = os.path.join(results_dir, "best_params.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


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
    parser.add_argument("--maxiter", type=int,   default=15)
    parser.add_argument("--workers", type=int,   default=8)
    parser.add_argument("--sigma",   type=float, default=None,
                        help="Override initial sigma (scalar); default: per-param")
    parser.add_argument("--popsize", type=int,   default=100,
                        help="CMA-ES population size (default: 100)")
    parser.add_argument("--seed",    type=int,   default=None,
                        help="Random seed for CMA-ES's sampling (default: "
                             "unseeded/random each run)")
    parser.add_argument("--cmd-vel-x", type=float, default=None,
                        help="Override the walk task's forward cmd_vel "
                             "(phases[0].cmd_vel[0]) for this run. Default: "
                             "unset, uses the value from tasks.yaml.")
    parser.add_argument("--wandb-project", type=str, default="muscle-mppi-cmaes",
                        help="wandb project to log runs to")
    parser.add_argument("--run-name", type=str, default=None,
                        help="wandb run name (default: wandb auto-generates one)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results dir (default: ./results). Use a "
                             "distinct dir per process when running sweeps in "
                             "parallel, so cmaes_log files don't collide.")
    args = parser.parse_args()

    # objective.evaluate() reads CMD_VEL_X fresh from the environment on
    # every call (not cached at import time), so setting it here — before
    # any worker pool is spawned — is picked up by every worker process,
    # and also applies to the --test smoke-test path below.
    if args.cmd_vel_x is not None:
        os.environ["CMD_VEL_X"] = str(args.cmd_vel_x)

    results_dir = args.results_dir if args.results_dir else RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    print(f"CMA-ES walk optimizer")
    print(f"  x0:          {dict(zip(PARAM_NAMES, X0))}")
    print(f"  bounds:      lo={LO}  hi={HI}")
    print(f"  cmd_vel_x:   {args.cmd_vel_x if args.cmd_vel_x is not None else '(tasks.yaml default)'}")
    print(f"  results:     {results_dir}")
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
    if args.seed is not None:
        opts["seed"] = args.seed

    es = cma.CMAEvolutionStrategy(X0, sigma0, opts)
    print(f"  seed:    {es.opts.get('seed')}")
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
            "seed": es.opts.get("seed"),  # resolved seed, even if unspecified
            "cmd_vel_x": args.cmd_vel_x,
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
            _write_best_result(results_dir, best_cost, generation, best_x, args)
            wandb.run.summary["best_cost"] = best_cost
            wandb.run.summary["best_area_mean"] = curve_area_mean(best_x)
            wandb.run.summary["best_generation"] = generation
            for name, val in zip(PARAM_NAMES, best_x):
                wandb.run.summary[f"best/{name}"] = val

        # Mean/std over feasible candidates only, so infeasible-penalty
        # outliers don't swamp them.
        feasible_costs = [c for c in costs if c < 1e5]
        mean_cost = float(np.mean(feasible_costs)) if feasible_costs else float("nan")
        std_cost  = float(np.std(feasible_costs))  if feasible_costs else float("nan")
        n_feasible = len(feasible_costs)

        # area is a cheap closed-form function of x (no simulation), kept as
        # an informational diagnostic alongside fitness (= locomotion cost).
        gen_best_area  = curve_area_mean(solutions[gen_best_idx])
        best_area      = curve_area_mean(best_x)
        log = {
            "generation": generation,
            "gen_best_fitness": gen_best_cost,
            "gen_best_area_mean": gen_best_area,
            "overall_best_fitness": best_cost,
            "overall_best_area_mean": best_area,
            "mean_cost": mean_cost,
            "std_cost": std_cost,
            "n_feasible": n_feasible,
            "elapsed_s": elapsed,
            # kept for backward compatibility with existing dashboards
            "gen_best_cost": gen_best_cost,
            "overall_best_cost": best_cost,
        }
        for name, val in zip(PARAM_NAMES, solutions[gen_best_idx]):
            log[f"params/{name}"] = val

        # Curve plots for this generation's best candidate — cheap (no
        # simulation, pure closed-form curves) so safe to render every
        # generation, letting you scrub through wandb's media history to
        # watch the shape evolve across generations.
        gen_best_x = solutions[gen_best_idx]
        fig_fl = plot_fl_curves(gen_best_x, title_suffix=f" (gen {generation})")
        fig_fv = plot_fv_curves(gen_best_x, title_suffix=f" (gen {generation})")
        log["fl_curves"] = wandb.Image(fig_fl)
        log["fv_curves"] = wandb.Image(fig_fv)
        plt.close(fig_fl)
        plt.close(fig_fv)

        # Rollout video of this generation's best candidate — unlike the
        # curve plots above, this re-runs gait generation + mppi_sim +
        # MuJoCo rendering (real simulation cost, not free), but rendering
        # every generation (rather than only on new-best) lets you watch
        # gait behavior evolve generation-by-generation in wandb, not just
        # at improvement points.
        frames = render_rollout(gen_best_x)
        if frames is not None:
            log["gen_best_rollout"] = wandb.Video(frames, fps=RENDER_FPS, format="gif")
        else:
            print("  (rollout render failed, skipping gif)")

        wandb.log(log, step=generation)

        print(f"\nGen {generation:3d} | best={gen_best_cost:.2f} "
              f"| feasible={n_feasible}/{len(costs)} | {elapsed:.0f}s")
        print(f"  best x: {dict(zip(PARAM_NAMES, best_x))}")
        print(f"  overall best fitness: {best_cost:.4f}  (area_mean={best_area:.4f})")
        print("─" * 60)

    best_path = _write_best_result(results_dir, best_cost, generation, best_x, args, done=True)
    final_area = curve_area_mean(best_x)
    print("\n═══ Optimization complete ═══")
    print(f"Best fitness (cost)  : {best_cost:.4f}")
    print(f"  area_mean          : {final_area:.4f}")
    print(f"Best params  : {dict(zip(PARAM_NAMES, best_x))}")
    print(f"Best params saved to {best_path}")
    print(f"Results saved to {results_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
