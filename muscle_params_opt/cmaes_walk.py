"""
CMA-ES optimizer for Hill muscle parameters of the walk task.

Optimizes x = [lce_min, lce_max, pFLmax, FVmax] (one value per joint type,
shared by all 4 legs) and the co-contraction level stiffness (one value for
all joints) to minimize the mean per-step locomotion cost of the walk task,
run with the task settings in objective.py. Starting point, step sizes and
bounds: config.yaml.

Usage:
  python cmaes_walk.py               # run full optimization
  python cmaes_walk.py --test        # single eval at initial point (smoke test)
  python cmaes_walk.py --maxiter 20 --workers 6
  python cmaes_walk.py --resume --results-dir results/X --maxiter 30
                                     # continue run X to 30 generations
"""

import argparse
import csv
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
import yaml

# Ensure this directory is on the path for gait_generator import in workers
sys.path.insert(0, os.path.dirname(__file__))

import cma
from objective import FAILED, FELL, INFEASIBLE, OUTCOMES, evaluate, render_rollout
from curve_plots import plot_fl_curves, plot_fv_curves

# ── settings: CONFIG_PATH's run settings and search space ────────────────────
# The search space is 13D: hip/thigh/calf × lce_min/lce_max/pFLmax/FVmax, and
# stiffness. Its values (x0, step sizes, bounds) come from the config; the
# layout of x is fixed here: objective.py and curve_plots.py index x by position.
JOINT_TYPES = ["hip", "thigh", "calf"]
PER_JOINT   = ["lce_min", "lce_max", "pFLmax", "FVmax"]   # one value per joint type
PARAM_NAMES = [f"{p}_{j}" for p in PER_JOINT for j in JOINT_TYPES] + ["stiffness"]
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def _require_keys(path, where, entry, keys):
    if not isinstance(entry, dict) or set(entry) != set(keys):
        found = sorted(entry) if isinstance(entry, dict) else entry
        raise ValueError(f"{path}: {where} needs exactly {sorted(keys)}, has {found}")


def _positive(path, name, value, kind):
    """value, which must be a kind (int or float) above 0."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not is_number or (kind is int and not isinstance(value, int)) or value <= 0:
        # PyYAML follows YAML 1.1, which reads 1e-3 as text: an exponent needs
        # a '.' in the number and a sign, as in 1.0e-3.
        hint = " (write 0.001 or 1.0e-3: YAML reads 1e-3 as text)" if isinstance(value, str) else ""
        raise ValueError(f"{path}: {name} must be {'an integer' if kind is int else 'a number'} "
                         f"above 0, is {value!r}{hint}")
    return kind(value)


def _parse_run(path, run):
    """The run settings, checked."""
    _require_keys(path, "run", run, ["maxiter", "popsize", "workers", "seed", "sigma",
                                     "tolx", "tolfun", "wandb_project"])
    seed = run["seed"]
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError(f"{path}: run.seed must be an integer or null, is {seed!r}")
    if not isinstance(run["wandb_project"], str) or not run["wandb_project"]:
        raise ValueError(f"{path}: run.wandb_project must be a name, is {run['wandb_project']!r}")

    return {
        "maxiter":       _positive(path, "run.maxiter", run["maxiter"], int),
        "popsize":       _positive(path, "run.popsize", run["popsize"], int),
        "workers":       _positive(path, "run.workers", run["workers"], int),
        "seed":          seed,
        "sigma":         _positive(path, "run.sigma",   run["sigma"],   float),
        "tolx":          _positive(path, "run.tolx",    run["tolx"],    float),
        "tolfun":        _positive(path, "run.tolfun",  run["tolfun"],  float),
        "wandb_project": run["wandb_project"],
    }


def _parse_search(path, search):
    """x0, sigma0, lo and hi, each a list in PARAM_NAMES order, checked."""
    _require_keys(path, "search", search, ["x0", "sigma0", "lo", "hi"])
    space = {}
    for key, entry in search.items():
        _require_keys(path, f"search.{key}", entry, PER_JOINT + ["stiffness"])
        values = []
        for p in PER_JOINT:
            if not isinstance(entry[p], list) or len(entry[p]) != len(JOINT_TYPES):
                raise ValueError(f"{path}: search.{key}.{p} needs one value per joint type {JOINT_TYPES}")
            values += [float(v) for v in entry[p]]
        space[key] = values + [float(entry["stiffness"])]

    for name, x0, s, lo, hi in zip(PARAM_NAMES, space["x0"], space["sigma0"], space["lo"], space["hi"]):
        if not (lo < hi and lo <= x0 <= hi and s > 0):
            raise ValueError(f"{path}: {name} needs lo < hi, x0 within them and sigma0 > 0 "
                             f"(lo {lo}, hi {hi}, x0 {x0}, sigma0 {s})")
        if name.startswith("FVmax") and lo <= 1.0:
            raise ValueError(f"{path}: {name} lo must stay above 1.0, the force-velocity "
                             f"curve's singularity (FVmax - 1 is a divisor)")
    return space


def load_config(path):
    """(run, space) from the config at path: the run settings as a dict, and
    the search space's x0, sigma0, lo and hi, each a list in PARAM_NAMES order.
    Raises ValueError on a missing or extra entry, or values a run can't use."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _require_keys(path, "the file", cfg, ["run", "search"])
    return _parse_run(path, cfg["run"]), _parse_search(path, cfg["search"])


RUN, _SPACE = load_config(CONFIG_PATH)
X0, SIGMA0, LO, HI = _SPACE["x0"], _SPACE["sigma0"], _SPACE["lo"], _SPACE["hi"]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Everything --resume needs to continue a run, written in its results dir
# after every generation.
STATE_FILE = "cmaes_state.pkl"

# How each generation's evaluations ended (objective.OUTCOMES), one row per
# generation, in the run's results dir.
OUTCOMES_FILE = "outcomes.csv"


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


def _append_outcomes(results_dir, generation, outcomes, gen_best_outcome):
    """Append generation's outcome counts, and how its best candidate's
    evaluation ended, to results_dir/OUTCOMES_FILE (header on first write)."""
    path = os.path.join(results_dir, OUTCOMES_FILE)
    first = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if first:
            writer.writerow(["generation", *OUTCOMES, "gen_best"])
        writer.writerow([generation, *(outcomes.count(o) for o in OUTCOMES), gen_best_outcome])


def run_parallel(solutions, n_workers):
    """Evaluate a list of candidate solutions in parallel. Returns (costs,
    outcomes), in the order of solutions."""
    args = [(x, i) for i, x in enumerate(solutions)]
    costs    = [None] * len(solutions)
    outcomes = [None] * len(solutions)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_eval_worker, a): i for i, a in enumerate(args)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                costs[idx], outcomes[idx] = fut.result()
            except Exception as e:
                print(f"  Worker {idx} raised: {e}")
                costs[idx], outcomes[idx] = 1e9, FAILED
    return costs, outcomes


def main():
    # Run-setting flags default to None, so that one left out takes the
    # config's value (RUN) below.
    parser = argparse.ArgumentParser(
        description=f"Run settings default to the run section of {CONFIG_PATH}.")
    parser.add_argument("--test",    action="store_true",
                        help="Evaluate initial point only (smoke test)")
    parser.add_argument("--maxiter", type=int,   default=None,
                        help=f"Generations in total, counting those of a "
                             f"resumed run (config: {RUN['maxiter']})")
    parser.add_argument("--resume",  action="store_true",
                        help="Continue the run saved in --results-dir up to "
                             "--maxiter generations, in the same logs and "
                             "wandb run. Its popsize, sigma, seed, tolx and "
                             "tolfun are the saved run's.")
    parser.add_argument("--workers", type=int,   default=None,
                        help=f"Sims run in parallel (config: {RUN['workers']})")
    parser.add_argument("--sigma",   type=float, default=None,
                        help=f"Scale on every per-parameter step size "
                             f"(config: {RUN['sigma']})")
    parser.add_argument("--popsize", type=int,   default=None,
                        help=f"CMA-ES population size (config: {RUN['popsize']})")
    parser.add_argument("--seed",    type=int,   default=None,
                        help=f"Random seed for CMA-ES's sampling (config: "
                             f"{RUN['seed']}; none: random each run)")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help=f"wandb project to log runs to (config: {RUN['wandb_project']})")
    parser.add_argument("--run-name", type=str, default=None,
                        help="wandb run name (default: wandb auto-generates one)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results dir (default: ./results). Use a "
                             "distinct dir per process when running sweeps in "
                             "parallel, so cmaes_log files don't collide.")
    args = parser.parse_args()
    for key in ("maxiter", "workers", "sigma", "popsize", "seed", "wandb_project"):
        if getattr(args, key) is None:
            setattr(args, key, RUN[key])

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
        cost, outcome = evaluate(X0, worker_id=0, verbose=True)
        print(f"\nCost = {cost:.4f}, {outcome}  ({time.time()-t0:.1f} s)")
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
        sigma0 = args.sigma

        opts = {
            "bounds":   [LO, HI],
            "maxiter":  args.maxiter,
            "tolx":     RUN["tolx"],
            "tolfun":   RUN["tolfun"],
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
                "tolx": RUN["tolx"],
                "tolfun": RUN["tolfun"],
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

    print(f"Starting CMA-ES: popsize={es.popsize}, workers={args.workers}, maxiter={args.maxiter}")
    print("─" * 60)

    while not es.stop():
        generation += 1
        solutions = es.ask()

        t0 = time.time()
        costs, outcomes = run_parallel(solutions, n_workers=args.workers)
        elapsed = time.time() - t0

        es.tell(solutions, costs)
        logger.add()

        gen_best_idx  = int(np.argmin(costs))
        gen_best_cost = costs[gen_best_idx]
        n_outcome     = {o: outcomes.count(o) for o in OUTCOMES}
        _append_outcomes(results_dir, generation, outcomes, outcomes[gen_best_idx])

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
            # Outcome counts (objective.OUTCOMES); all of them are in OUTCOMES_FILE.
            "n_fell": n_outcome[FELL],
            "n_infeasible": n_outcome[INFEASIBLE],
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

        print(f"\nGen {generation:3d} | best={gen_best_cost:.2f} ({outcomes[gen_best_idx]}) "
              f"| feasible={n_feasible}/{len(costs)} | {elapsed:.0f}s")
        print("  outcomes: " + ", ".join(f"{o} {n_outcome[o]}" for o in OUTCOMES))
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
