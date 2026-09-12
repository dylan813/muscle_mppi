cmake_minimum_required(VERSION 3.16)
project(muscle_mppi_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fPIC")

list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")
find_package(unitree_sdk2 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)

set(MUJOCO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../unitree_mujoco/simulate/mujoco")
include_directories(${MUJOCO_DIR}/include)
link_directories(${MUJOCO_DIR}/lib)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags,-rpath,/opt/unitree_robotics/lib")

set(COMMON_LIBS unitree_sdk2 mujoco yaml-cpp pthread OpenMP::OpenMP_CXX)

# Absolute repo root, compiled into every binary so default config, gait and
# output paths resolve the same from any working directory (common/paths.h).
get_filename_component(MUSCLE_MPPI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
add_compile_definitions(MUSCLE_MPPI_ROOT="${MUSCLE_MPPI_ROOT}")

# Layout: muscle/ (muscle-actuated controller), pd/ (PD-actuated baseline),
# common/ (code shared by both).

# --- Quadruped locomotion controller ---
add_executable(muscle_mppi_controller
    muscle/mppi_controller.cpp
    muscle/control/base_mppi.cpp
    muscle/control/mppi_locomotion.cpp
    muscle/utils/tasks.cpp
    common/task_loader.cpp
)
target_link_libraries(muscle_mppi_controller ${COMMON_LIBS})

# --- Standalone MuJoCo simulation test (no DDS, no real-time constraint) ---
add_executable(mppi_sim
    muscle/mppi_sim.cpp
    muscle/control/base_mppi.cpp
    muscle/control/mppi_locomotion.cpp
    muscle/utils/tasks.cpp
    common/task_loader.cpp
)
target_link_libraries(mppi_sim ${COMMON_LIBS})

# --- PD-actuated standalone MuJoCo simulation test (mirrors mppi_sim, but ---
# --- with direct joint-space PD control instead of the muscle model)     ---
add_executable(pd_mppi_sim
    pd/pd_mppi_sim.cpp
    pd/control/base_mppi_pd.cpp
    pd/control/mppi_locomotion_pd.cpp
    pd/utils/tasks_pd.cpp
    common/task_loader.cpp
)
target_link_libraries(pd_mppi_sim ${COMMON_LIBS})
