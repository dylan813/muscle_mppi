# Muscle-Inspired Torque-Level Control for Legged Locomotion with Sampling-Based MPC

# Installation

```bash
git clone https://github.com/dylan813/muscle_mppi.git
git submodule update --init --recursive
```

After following documentation from [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco/tree/main) and [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) for C++ simulation, run their tests. Below are the spcific tests you should verify from the unitree_mujoco repository.
```bash
cd muscle_mppi/unitree_mujoco/simulate/build
./unitree_mujoco -r go2 -s scene_terrain.xml
```

```bash
cd muscle_mppi/unitree_mujoco/example/cpp/build
./stand_go2
```

# DDS Configuration

The simulator and controller communicate over CycloneDDS. By default it uses shared memory transport, which can cause silent message drops between processes on the same machine. Disabling it forces traffic over the loopback network which is reliable for local simulation.

Set this in **every terminal** before running the sim or any controller:
```bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><SharedMemory><Enable>false</Enable></SharedMemory></Domain></CycloneDDS>'
```

# Running the MPPI Controller Implementations

Controller code lives in `controllers/`: `muscle/` is the muscle-actuated controller, `pd/` is the PD-actuated baseline, and `common/` holds code shared by both. One CMake project builds every binary into `controllers/build/`.

The binaries can be run from any directory. CMake compiles in the repo's absolute path, and the default task YAML, the gait files, the default output CSVs and every `model_path` inside a task YAML are resolved against it (`controllers/common/task_config.h`). Paths you pass on the command line stay relative to your current directory. If you move or re-clone the repo, re-run `cmake` so the compiled-in path is updated.

Build implementations
```bash
cd muscle_mppi/controllers/
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Terminal 1: Open the MuJoCo simulation with the Unitree Go2
```bash
cd muscle_mppi/unitree_mujoco/simulate/build
./unitree_mujoco -r go2 -s scene.xml
```

For recording rollout,
```bash
sudo apt install ffmpeg
cd muscle_mppi/unitree_mujoco/simulate/build
./unitree_mujoco -r go2 -s scene.xml -o ../../../analysis/data/videos/run.mp4
```

Terminal 2: Run muscle-inspired torque-level MPPI controller
```bash
cd muscle_mppi/controllers/build
./muscle_mppi_controller
```

<!-- record headless
```bash
sudo apt install xvfb
Xvfb :99 -screen 0 1280x720x24 &
DISPLAY=:99 ./unitree_mujoco -r go2w -s scene_terrain.xml -o ../../../analysis/data/videos/run.mp4
``` -->

# Standalone Sims

`mppi_sim` (muscle) and `pd_mppi_sim` (PD) run MPPI against a local MuJoCo sim, no DDS needed:
```bash
cd muscle_mppi/controllers/build
./mppi_sim                              # task "walk"
./mppi_sim walk_rough                   # any task in controllers/muscle/utils/tasks.yaml
./pd_mppi_sim walk                      # tasks in controllers/pd/utils/tasks_pd.yaml
```

Output goes to `analysis/data/mppi_sim/` (or `pd_mppi_sim/`): `<name>.csv`, `<name>_qpos.csv` and a rollout `<name>.gif`, named `mppi_sim` by default. A re-run with the same name replaces all three.

| Flag | Effect |
|---|---|
| `--name <run>` | name the output files, e.g. `--name rough/test1` → `analysis/data/mppi_sim/rough/test1.*` |
| `--no-gif` | skip the GIF (and delete any old GIF of that name) |
| `--save <name>` | also copy the run into `analysis/log/trials/<name>/trial_NNN/` |
| `--dataset <prefix>` | `mppi_sim` only: also write the learned policy's training data, one row per CSV row, to `<prefix>_obs.npy` and `<prefix>_plan.npy` |

Flags go in any order. A third positional argument sets a full output path instead of `--name`: `./mppi_sim walk <yaml> /tmp/x.csv`.

GIFs are rendered by `analysis/render_gif.py`, which needs `mujoco`, `numpy`, `pyyaml` and `Pillow` (headless EGL by default). To use a specific interpreter, or re-render a run by hand (from the repo root):
```bash
export MUSCLE_MPPI_PYTHON=/path/to/python3
MUJOCO_GL=egl python3 analysis/render_gif.py <qpos.csv> <out.gif> <task>
```

# Batch Trials

`run_trials.sh` runs each sim N times sequentially with `--save`, and also keeps each run's console output:
```bash
./run_trials.sh                    # 100 muscle + 100 PD runs of guinea_fowl, under trials/workshop/
./run_trials.sh -n 20 -t walk -N flat
./run_trials.sh --muscle-only      # or --pd-only; --tee mirrors output; --gif renders GIFs
```

```
analysis/log/trials/<name>/
  muscle/trial_NNN/{mppi_sim.csv, mppi_sim_qpos.csv, console.log}
  pd/trial_NNN/{pd_mppi_sim.csv, pd_mppi_sim_qpos.csv, console.log}
  batch_<timestamp>/{batch.log, summary.csv, failed/}
```

`summary.csv` has one row per run; `reached_goal` is the success column. Trial indices continue across batches, and a lock stops two batches from running at once.

# Plots

Plotting scripts are in `analysis/plot/` and save to `analysis/plot/figures/`:
```bash
cd muscle_mppi/analysis/plot
python3 plot_force_length.py
python3 plot_force_velocity.py
python3 plot_walk_leg.py [csv] [name] [--outdir DIR]   # defaults to the latest mppi_sim run
```

# CMA-ES

`muscle_params_opt/` tunes the Hill muscle parameters on the `walk` task with CMA-ES: lce_min, lce_max, pFLmax and FVmax per joint type (hip, thigh, calf; shared by all 4 legs), plus `stiffness`. Each candidate runs `mppi_sim` once and is scored by the controller's own logged cost; candidates whose gait is infeasible skip the sim and get a penalty. The search space and bounds are in `cmaes_walk.py`; the walk settings it overrides (goal, command velocity, duration) are in `objective.py`.

Needs a built `controllers/build/mppi_sim` and the `muscle` conda env (`environment.yml`), logged in to wandb.
```bash
cd muscle_mppi/muscle_params_opt
python3 cmaes_walk.py --run-name <name> --results-dir results/<name>                     # new run
python3 cmaes_walk.py --resume --results-dir results/<name> --maxiter <total generations>  # continue it
python3 cmaes_walk.py --test                                                             # one sim at the starting point
```

| Flag | Default | Effect |
|---|---|---|
| `--maxiter` | 15 | generations in total, counting those of a resumed run |
| `--popsize` | 100 | candidates per generation |
| `--workers` | 8 | sims run in parallel |
| `--seed` | random | seed for CMA-ES's sampling |
| `--sigma` | 1 | scale on the per-parameter step sizes (`SIGMA0`) |
| `--run-name` | wandb's | wandb run name |
| `--wandb-project` | `muscle-mppi-cmaes` | wandb project |
| `--results-dir` | `results/` | output folder; use one per run, or runs overwrite each other |
| `--resume` | | continue the run saved in `--results-dir` (same logs and wandb run); `--popsize`, `--seed` and `--sigma` are the saved run's |

```
results/<name>/
  best_params.json   best candidate so far: cost, generation, parameters
  cmaes_log/         pycma's per-generation logs (mean, step sizes, costs)
  cmaes_state.pkl    everything --resume needs, saved after every generation
```

wandb logs each generation's costs, its best candidate's force-length and force-velocity curves, and a rollout GIF. To use a result, copy its parameters into `default_muscle_quad` in `controllers/muscle/utils/tasks.yaml`.
