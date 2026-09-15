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
./unitree_mujoco -r go2 -s scene.xml -o ../../../analysis/videos/run.mp4
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
DISPLAY=:99 ./unitree_mujoco -r go2w -s scene_terrain.xml -o ../../../analysis/videos/run.mp4
``` -->

# Plots

Plotting scripts live in `analysis/plot/` and save their figures to `analysis/plot/figures/` (created if missing).

```bash
cd muscle_mppi/analysis/plot
python3 plot_force_length.py
python3 plot_force_velocity.py
```

```bash
cd muscle_mppi/controllers/build
./mppi_sim                       # task "walk"; also renders the rollout GIF
./mppi_sim walk_rough            # any task from tasks.yaml
./mppi_sim walk --no-gif         # skip the GIF
./mppi_sim walk_rough --name rough_test1   # name this run's output files
python3 ../../analysis/plot/plot_walk_leg.py <name>
```

Working sim output (CSVs, GIFs) goes to `analysis/data/`: `mppi_sim` writes `analysis/data/mppi_sim/mppi_sim.csv` + `mppi_sim_qpos.csv` + `mppi_sim.gif`, and `pd_mppi_sim` writes `analysis/data/pd_mppi_sim/pd_mppi_sim.csv` + `pd_mppi_sim_qpos.csv` + `pd_mppi_sim.gif`. The directories are created on first run if missing.

## Naming runs

A run's three files are always named after its output CSV: `<name>.csv`, `<name>_qpos.csv`, `<name>.gif`. Re-running with the same name replaces them; a different name leaves other runs' files alone.

- **Default:** `./mppi_sim walk` writes `analysis/data/mppi_sim/mppi_sim.*`.
- **`--name <run>`:** writes to the same folder under your name. You don't need the YAML or a path.
  ```bash
  ./mppi_sim walk_rough --name rough_test1       # analysis/data/mppi_sim/rough_test1.csv, rough_test1_qpos.csv, rough_test1.gif
  ./pd_mppi_sim walk_rough --name rough_test1    # analysis/data/pd_mppi_sim/rough_test1.*
  ./mppi_sim walk --name rough/test2             # a subfolder: analysis/data/mppi_sim/rough/test2.*
  ```
- **Full path:** a third positional argument still sets the output anywhere, e.g. `./mppi_sim walk ../muscle/utils/tasks.yaml /tmp/x.csv`. Give either a path or `--name`, not both.

Leaving off `.csv` is fine in both forms; it's added for you. `--name` can't contain `..` or be an absolute path. Use a full path for that.

## Rollout GIFs

After each run, both sims render the logged rollout to a GIF next to the CSV (`<output>.gif`, e.g. `mppi_sim.gif`), using the task the run was simulated with so the scene matches. Every run first deletes any existing GIF of that name, then renders a new one, so the GIF next to the CSVs is always from the same run. That holds even when no new GIF is made: with `--no-gif`, after a failed render, or when the robot falls before logging starts, no GIF is left there.

Rendering runs `analysis/render_gif.py` with `python3` and headless EGL (`MUJOCO_GL=egl` unless you've set `MUJOCO_GL`). That Python needs `mujoco`, `numpy`, `pyyaml` and `Pillow`. To use a different interpreter:

```bash
export MUSCLE_MPPI_PYTHON=/home/rml3/anaconda3/envs/mujoco/bin/python3
```

If rendering fails, the sim prints a warning; its CSVs are unaffected. To re-render a run by hand (e.g. with a different output name):

```bash
MUJOCO_GL=egl python3 ../../analysis/render_gif.py ../../analysis/data/mppi_sim/mppi_sim_qpos.csv ../../analysis/data/mppi_sim/walk_rough_test.gif walk_rough
```

# Saving Trials

By default `mppi_sim` and `pd_mppi_sim` overwrite their working CSVs every run. Pass `--save <name>` to also copy that run's logs into a numbered trial directory, so repeated runs accumulate for analysis:

```bash
cd muscle_mppi/controllers/build
./mppi_sim walk --save walk_baseline
./mppi_sim walk --save walk_baseline
./pd_mppi_sim walk --save walk_baseline_pd
```

Each run lands in its own directory under `analysis/log/trials/`:

```
analysis/log/trials/walk_baseline/
  trial_001/mppi_sim.csv, mppi_sim_qpos.csv, mppi_sim.gif
  trial_002/mppi_sim.csv, mppi_sim_qpos.csv, mppi_sim.gif
```

The flags work alongside the positional arguments in any order (`./mppi_sim walk --name baseline_run --save walk_baseline --no-gif`). `--save` picks the trial folder; the files inside keep the run's name (e.g. `baseline_run.csv`). With `--no-gif` the trial gets only the two CSVs. The working CSVs (and GIF, unless `--no-gif`) are still written to their usual location under `analysis/data/`, so `plot_walk_leg.py` keeps plotting the latest run by default.

To plot a specific saved trial, point the script at that trial directory instead (run from `muscle_mppi/controllers/build/`):

```bash
TRIAL=../../analysis/log/trials/walk_baseline/trial_001

python3 ../../analysis/plot/plot_walk_leg.py $TRIAL/mppi_sim.csv <name> --outdir $TRIAL

# only needed for a trial saved with --no-gif:
MUJOCO_GL=egl python3 ../../analysis/render_gif.py $TRIAL/mppi_sim_qpos.csv $TRIAL/mppi_sim.gif walk
```

`plot_walk_leg.py` writes to `analysis/plot/figures/` by default; `--outdir $TRIAL` keeps the plots with that trial's data instead. `render_gif.py` takes its output path explicitly (second argument), so give it a path inside the trial directory to keep the GIF there too. Its third argument is the task the trial was run with (`walk` above) — it must match, since the model path and dt are looked up from it.

# Batch Trials

`run_trials.sh` (repo root) runs both sims N times each and, unlike a bare `--save`, keeps each run's **console output** — the per-solve `[cost]` diagnostics, stand-up height, convergence solve time, and whether/when the robot fell — as `console.log` inside that run's trial directory.

```bash
./run_trials.sh                          # 100 × mppi_sim then 100 × pd_mppi_sim, task "guinea_fowl", under workshop/
./run_trials.sh -n 20                    # 20 of each
./run_trials.sh -t walk -N flat          # different task, saved under flat/
./run_trials.sh --muscle-only            # or --pd-only
./run_trials.sh --tee                    # also mirror each run's output to the terminal
./run_trials.sh --gif                    # also render each run's rollout GIF into its trial
```

Unlike a single sim run, batches skip GIFs by default (`run_trials.sh` passes `--no-gif`). Rendering would add to every run's `wall_s` and cost tens of MB per trial across a 100-run batch; `--gif` turns them back on. The CMA-ES objective (`analysis/optimize/objective.py`) likewise runs `mppi_sim --no-gif`, since it renders its own wandb rollouts.

Runs are strictly sequential — one sim process at a time, muscle batch first, then PD. Both binaries use every core for their rollout loop (`num_threads: 0`), so overlapping them would make the solve-time numbers meaningless. An `flock` on `analysis/log/trials/.run_trials.lock` also refuses to start a second batch while one is running.

Output:

```
analysis/log/trials/workshop/
  muscle/trial_NNN/{mppi_sim.csv, mppi_sim_qpos.csv, console.log}
  pd/trial_NNN/{pd_mppi_sim.csv, pd_mppi_sim_qpos.csv, console.log}
  batch_<timestamp>/
    batch.log      # batch header (task, git rev, binary hashes) + one line per run
    summary.csv    # one row per run, ready to load in pandas
    failed/        # console output of any run that died before saving a trial
```

`summary.csv` columns: `controller, run_index, trial, trial_dir, exit_code, wall_s, stand_height_m, converged_solve_ms, avg_solve_ms, solves, fell, fall_t, reached_goal, phase_advances, early_stop_t, logged_rows, logged_t_end`.

`reached_goal` is the success column to compare across controllers — it comes from the controller's own `[phase] task complete.`, which both variants emit. `early_stop_t` is the sim time at which the run stopped on that success; it is empty for muscle trials logged before `MPPILocomotion` gained a `task_success()` accessor, since `mppi_sim.cpp` had no early stop then and ran the full `sim_duration` instead.

Trial indices continue from whatever is already under that name, so re-running adds to the existing set instead of overwriting it. At the default `guinea_fowl` settings (25 s `sim_duration`) each run takes ~16 s, so 100 + 100 is about 55 minutes; `-t walk` is ~6 s per run, about 20 minutes.

#CMA-ES

```bash
cd muscle_mppi/analysis/optimize
python3 cmaes_walk.py --workers <x> --run-name <xxx>
```