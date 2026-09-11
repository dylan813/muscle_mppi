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

Build implementations
```bash
cd muscle_mppi/muscle_mppi/
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
cd muscle_mppi/muscle_mppi/build
./muscle_mppi_controller
```

<!-- record headless
```bash
sudo apt install xvfb
Xvfb :99 -screen 0 1280x720x24 &
DISPLAY=:99 ./unitree_mujoco -r go2w -s scene_terrain.xml -o ../../../analysis/videos/run.mp4
``` -->

# Unit Tests

```bash
cd muscle_mppi/analysis/unit_tests
python3 plot_force_length.py
python3 plot_force_velocity.py
```

```bash
cd muscle_mppi/muscle_mppi/build
./mppi_sim
MUJOCO_GL=egl /home/rml3/anaconda3/envs/mujoco/bin/python3 ../../analysis/render_gif.py ../mppi_sim/mppi_sim_qpos.csv ../mppi_sim/test.gif
python3 ../../analysis/log/plot_walk_leg.py <name>
```

```bash
./mppi_sim walk_rough
 MUJOCO_GL=egl /home/rml3/anaconda3/envs/mujoco/bin/python3 ../../analysis/render_gif.py ../mppi_sim/mppi_sim_qpos.csv ../mppi_sim/walk_rough_test.gif walk_rough
```

# Saving Trials

By default `mppi_sim` and `pd_mppi_sim` overwrite their working CSVs every run. Pass `--save <name>` to also copy that run's logs into a numbered trial directory, so repeated runs accumulate for analysis:

```bash
cd muscle_mppi/muscle_mppi/build
./mppi_sim walk --save walk_baseline
./mppi_sim walk --save walk_baseline
./pd_mppi_sim walk --save walk_baseline_pd
```

Each run lands in its own directory under `analysis/log/trials/`:

```
analysis/log/trials/walk_baseline/
  trial_001/mppi_sim.csv, mppi_sim_qpos.csv
  trial_002/mppi_sim.csv, mppi_sim_qpos.csv
```

The flag works alongside the positional arguments in any order (`./mppi_sim walk ../utils/tasks.yaml out.csv --save <name>`), and the working CSVs are still written to their usual location, so `render_gif.py` and `plot_walk_leg.py` keep operating on the latest run unchanged.

To plot or render a specific saved trial, point the scripts at that trial directory instead (run from `muscle_mppi/muscle_mppi/build/`):

```bash
TRIAL=../../analysis/log/trials/walk_baseline/trial_001

python3 ../../analysis/log/plot_walk_leg.py $TRIAL/mppi_sim.csv <name>

MUJOCO_GL=egl /home/rml3/anaconda3/envs/mujoco/bin/python3 ../../analysis/render_gif.py $TRIAL/mppi_sim_qpos.csv $TRIAL/trial_001.gif walk
```

`plot_walk_leg.py` writes its plots beside the CSV it is given, so they stay with that trial's data. `render_gif.py` takes its output path explicitly (second argument), so give it a path inside the trial directory to keep the GIF there too. Its third argument is the task the trial was run with (`walk` above) — it must match, since the model path and dt are looked up from it.

# Batch Trials

`run_trials.sh` (repo root) runs both sims N times each and, unlike a bare `--save`, keeps each run's **console output** — the per-solve `[cost]` diagnostics, stand-up height, convergence solve time, and whether/when the robot fell — as `console.log` inside that run's trial directory.

```bash
./run_trials.sh                          # 100 × mppi_sim then 100 × pd_mppi_sim, task "guinea_fowl", under workshop/
./run_trials.sh -n 20                    # 20 of each
./run_trials.sh -t walk -N flat          # different task, saved under flat/
./run_trials.sh --muscle-only            # or --pd-only
./run_trials.sh --tee                    # also mirror each run's output to the terminal
```

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