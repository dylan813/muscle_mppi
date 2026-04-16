# muscle_mppi

# install

```bash
git clone https://github.com/dylan813/muscle_mppi.git
git submodule update --init --recursive
```

test go2 stand
```bash
cd unitree_mujoco/simulate/build
./unitree_mujoco -r go2 -s scene_terrain.xml
```

```bash
cd unitree_mujoco/example/cpp/build
export CYCLONEDDS_URI='<CycloneDDS><Domain><SharedMemory><Enable>false</Enable></SharedMemory></Domain></CycloneDDS>'
LD_LIBRARY_PATH=/opt/unitree_robotics/lib:$LD_LIBRARY_PATH ./stand_go2
```

building muscle mppi
```bash
cd unitree_mujoco/simulate/build
export CYCLONEDDS_URI='<CycloneDDS><Domain><SharedMemory><Enable>false</Enable></SharedMemory></Domain></CycloneDDS>'
./unitree_mujoco -r go2w -s scene_terrain.xml
```

```bash
cd ~/Documents/dylan/muscle_mppi/mppi/build
cmake ..
make -j$(nproc)
```