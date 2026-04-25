# Muscle-Inspired Torque-Level Control for Wheeled-Quadrupedal Locomotion with Sampling-Based MPC

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

```bash
cd muscle_mppi/ref_free_mppi/
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Terminal 1: Open the MuJoCo simulation with the Unitree Go2-W
```bash
cd muscle_mppi/unitree_mujoco/simulate/build
./unitree_mujoco -r go2w -s scene.xml
```

For recording rollout,
```bash
sudo apt install ffmpeg
cd muscle_mppi/unitree_mujoco/simulate/build
./unitree_mujoco -r go2w -s scene.xml -o ../../../analysis/videos/run.mp4
```

Terminal 2: Run muscle-inspired torque-level MPPI controller
```bash
cd muscle_mppi/muscle_mppi/build
./muscle_mppi_controller
```

Or, run reference-free MPPI controller
```bash
cd muscle_mppi/ref_free_mppi/build
./ref_free_mppi_controller
```

<!-- record headless
```bash
sudo apt install xvfb
Xvfb :99 -screen 0 1280x720x24 &
DISPLAY=:99 ./unitree_mujoco -r go2w -s scene_terrain.xml -o ../../../analysis/videos/run.mp4
``` -->