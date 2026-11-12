"""
CMA-ES optimizer for Hill muscle parameters of the walk task.

Optimizes x = [lce_min, lce_max, pFLmax, FVmax] (one value per joint type,
shared by all 4 legs) and the co-contraction level stiffness (one value for
all joints) to minimize the mean per-step locomotion cost of the walk task,
run with the task settings in objective.py.

Usage:
  python cmaes_walk.py               # run full optimization
  python cmaes_walk.py --test        # single eval at initial point (smoke test)
  python cmaes_walk.py --maxiter 20 --workers 6
  python cmaes_walk.py --resume --results-dir results/X --maxiter 30
                                     # continue run X to 30 generations
"""

import argparse
import json
import os
import pickle
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
from objective import evaluate, render_rollout
from curve_plots import plot_fl_curves, plot_fv_curves

# ── search space (13D: hip/thigh/calf × lce_min/lce_max/pFLmax/FVmax, stiffness) ─
# All 4 legs share the same value per joint type. X0, SIGMA0, LO and HI are
# those of the Aug 4 2026 run that produced tasks.yaml's muscle values
# (run name area_0, commit b0c904a), except the calf lce bounds and the
# stiffness entries below.
PARAM_NAMES = [
    "lce_min_hip",  "lce_min_thigh",  "lce_min_calf",
    "lce_max_hip",  "lce_max_thigh",  "lce_max_calf",
    "pFLmax_hip",   "pFLmax_thigh",   "pFLmax_calf",
    "FVmax_hip",    "FVmax_thigh",    "FVmax_calf",
    "stiffness",
]

X0 = [
    0.85, 0.82, 0.80,   # lce_min: hip, thigh, calf
    1.09, 1.06, 1.19,   # lce_max
    0.54, 0.94, 0.21,   # pFLmax
    1.2, 1.3, 1.1,   # FVmax
    0.75,               # stiffness (tasks.yaml's value)
]
# ~15% of each param's (HI - LO) range, recomputed for the current bounds.
SIGMA0 = [
    0.07, 0.07, 0.05,   # lce_min
    0.07, 0.07, 0.03,   # lce_max
    0.25, 0.25, 0.25,   # pFLmax
    0.11, 0.11, 0.11,   # FVmax
    0.04,               # stiffness
]
# FVmax lower bound is kept strictly above 1.0: gait_generator._force_vel
# divides by (FVmax - 1.0), so a candidate at or below 1.0 is a singularity.
# Calf lce_min >= 0.60 and lce_max <= 1.25: outside them walk_fast is almost
# never feasible (4000 uniform samples of the earlier 0.50-0.95 / 1.05-1.50
# calf bounds: 1% feasible with lce_min < 0.59, 0% with lce_max > 1.32).
# Stiffness 0.50-0.775: where both activations of a joint stay at least
# 2 * noise_sigma_act from 0 and 1 for >= 85% of walk_fast, for 96% (at 0.50)
# to 100% (0.60-0.775) of 400 feasible parameter sets sampled within these
# bounds; at 0.80 for 0.2%, as one activation nears 1 - 2 * 0.1.
LO = [0.50, 0.50, 0.60,  1.05, 1.05, 1.05,  0.10, 0.10, 0.10,  1.05, 1.05, 1.05,  0.50]
HI = [0.90, 0.90, 0.90,  1.50, 1.50, 1.25,  1.80, 1.80, 1.80,  1.80, 1.80, 1.80,  0.775]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Everything --resume needs to continue a run, written in its results dir
# after every generation.
STATE_FILE = "cmaes_state.pkl"


def _eval_worker(args):
    """Top-level function required for ProcessPoolExecutor pickling."""
    x, worker_id = args
    return evaluate(x, worker_id=worker_id, verbose=True)


def _write_best_result(results_dir, best_cost, best_generation, best_x, cli_args, done=False):
    """Write the best-so-far result to results_dir/best_params.json, so it
    survives even if the run is interrupted before completing."""
    payload = {
        "best_cost": best_cost,
        "best_generation": best_generation,
        "best_params": dict(zip(PARAM_NAMES, best_x)),
        "run_complete": done,
        "run_args": {
            "maxiter": cli_args.maxiter,
            "popsize": cli_args.popsize,
            "workers": cli_args.workers,
            "seed": cli_args.seed,
        },
    }
    path = os.path.join(results_dir, "best_params.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _save_state(results_dir, es, best_cost, best_generation, best_x, wandb_id):
    """Write the CMA-ES object (mean, step size, covariance, evolution paths,
    random number generator) and the best so far to results_dir/STATE_FILE.
    Written to a temporary file first, so an interrupted write never leaves
    a broken state behind."""
    state = {
        "es": es.pickle_dumps(),
        "best_cost": best_cost,
        "best_generation": best_generation,
        "best_x": best_x,
        "wandb_id": wandb_id,
    }
    path = os.path.join(results_dir, STATE_FILE)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(state, f)
    os.replace(path + ".tmp", path)


def _load_state(results_dir):
    """The state _save_state() wrote in results_dir, with the CMA-ES object
    restored."""
    path = os.path.join(results_dir, STATE_FILE)
    if not os.path.exists(path):
        sys.exit(f"Cannot resume: no {STATE_FILE} in {results_dir} "
                 f"(runs started before --resume existed did not save one).")
    with open(path, "rb") as f:
        state = pickle.load(f)
    state["es"] = pickle.loads(state["es"])
    return state


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
    parser.add_argument("--maxiter", type=int,   default=15,
                        help="Generations in total, counting those of a "
                             "resumed run")
    parser.add_argument("--resume",  action="store_true",
                        help="Continue the run saved in --results-dir up to "
                             "--maxiter generations, in the same logs and "
                             "wandb run. --sigma, --popsize and --seed are "
                             "the saved run's.")
    parser.add_argument("--workers", type=int,   default=8)
    parser.add_argument("--sigma",   type=float, default=None,
                        help="Override initial sigma (scalar); default: per-param")
    parser.add_argument("--popsize", type=int,   default=100,
                        help="CMA-ES population size (default: 100)")
    parser.add_argument("--seed",    type=int,   default=None,
                        help="Random seed for CMA-ES's sampling (default: "
                             "unseeded/random each run)")
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
    print(f"  x0:          {dict(zip(PARAM_NAMES, X0))}")
    print(f"  bounds:      lo={LO}  hi={HI}")
    print(f"  results:     {results_dir}")
    print()

    if args.test:
        print("── SMOKE TEST (initial point) ──")
        t0 = time.time()
        cost = evaluate(X0, worker_id=0, verbose=True)
        print(f"\nCost = {cost:.4f}  ({time.time()-t0:.1f} s)")
        return

    if args.resume:
        state = _load_state(results_dir)
        es = state["es"]
        es.opts.set("maxiter", args.maxiter)
        # pycma re-checks its stop conditions once per generation unless told
        # to: without this, the saved run's own maxiter stop still holds.
        if es.stop(check_in_same_iteration=True):
            sys.exit(f"Nothing to resume: the run stops at generation "
                     f"{es.countiter} ({dict(es.stop())}).")
        print(f"Resuming at generation {es.countiter + 1} of {args.maxiter}")

        wandb.init(project=args.wandb_project, id=state["wandb_id"], resume="must")
        wandb.config.update({"maxiter": args.maxiter, "workers": args.workers},
                            allow_val_change=True)
    else:
        state = {"best_cost": float("inf"), "best_generation": 0, "best_x": list(X0)}

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
            },
        )

    print(f"  seed:    {es.opts.get('seed')}")
    cma_log_dir = os.path.join(results_dir, "cmaes_log")
    os.makedirs(cma_log_dir, exist_ok=True)
    logger = cma.CMADataLogger(os.path.join(cma_log_dir, "run"), append=args.resume).register(es)

    best_cost       = state["best_cost"]
    best_generation = state["best_generation"]
    best_x          = state["best_x"]
    generation      = es.countiter

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
            best_cost       = gen_best_cost
            best_generation = generation
            best_x          = [float(v) for v in solutions[gen_best_idx]]
            print(f"  ★ new best: {best_cost:.4f}")
            _write_best_result(results_dir, best_cost, best_generation, best_x, args)
            wandb.run.summary["best_cost"] = best_cost
            wandb.run.summary["best_generation"] = best_generation
            for name, val in zip(PARAM_NAMES, best_x):
                wandb.run.summary[f"best/{name}"] = val

        # Saved before this generation's slow wandb logging (its rollout is a
        # simulation): a resume after a crash there continues with the next
        # generation instead of repeating this one in the logs.
        _save_state(results_dir, es, best_cost, best_generation, best_x, wandb.run.id)

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

        # Curve plots for this generation's best candidate — cheap (no
        # simulation, pure closed-form curves) so safe to render every
        # generation, letting you scrub through wandb's media history to
        # watch the shape evolve.
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
        gif = render_rollout(gen_best_x)
        if gif is not None:
            log["gen_best_rollout"] = wandb.Video(gif, format="gif")
        else:
            print("  (rollout render failed, skipping gif)")

        wandb.log(log, step=generation)

        print(f"\nGen {generation:3d} | best={gen_best_cost:.2f} "
              f"| feasible={n_feasible}/{len(costs)} | {elapsed:.0f}s")
        print(f"  best x: {dict(zip(PARAM_NAMES, best_x))}")
        print(f"  overall best cost: {best_cost:.4f}")
        print("─" * 60)

    best_path = _write_best_result(results_dir, best_cost, best_generation, best_x, args, done=True)
    print("\n═══ Optimization complete ═══")
    print(f"Best cost    : {best_cost:.4f}")
    print(f"Best params  : {dict(zip(PARAM_NAMES, best_x))}")
    print(f"Best params saved to {best_path}")
    print(f"Results saved to {results_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
