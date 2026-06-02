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

# --- Quadruped locomotion controller (NUM_JOINTS=12, JOINT_OFFSET=0) ---
add_executable(muscle_mppi_controller
    mppi_controller.cpp
    control/base_mppi.cpp
    control/mppi_locomotion.cpp
    utils/tasks.cpp
)
target_link_libraries(muscle_mppi_controller ${COMMON_LIBS})
# No compile definitions needed — tasks.h defaults to NUM_JOINTS=12, JOINT_OFFSET=0.

# --- Stage-1: torque-space MPPI (NUM_JOINTS=3, JOINT_OFFSET=3, NUM_MUSCLES=3) ---
# Headless validation runner — no Hill model, no Unitree SDK.
set(TORQUE_LIBS unitree_sdk2 mujoco yaml-cpp pthread OpenMP::OpenMP_CXX)
add_executable(single_leg_torque_controller
    single_leg_torque_controller.cpp
    control/base_mppi.cpp
    control/single_leg_torque.cpp
    utils/tasks.cpp
)
target_compile_definitions(single_leg_torque_controller PRIVATE
    NUM_JOINTS=3 JOINT_OFFSET=3 NUM_MUSCLES=3)
target_link_libraries(single_leg_torque_controller ${TORQUE_LIBS})

# --- Single-leg reach controller (NUM_JOINTS=3, JOINT_OFFSET=3) ---
# Sim-only: no unitree_sdk2 DDS dependency.
set(SINGLE_LEG_LIBS unitree_sdk2 mujoco yaml-cpp pthread OpenMP::OpenMP_CXX)
add_executable(single_leg_controller
    single_leg_controller.cpp
    control/base_mppi.cpp
    control/single_leg_reach.cpp
    utils/tasks.cpp
)
target_compile_definitions(single_leg_controller PRIVATE NUM_JOINTS=3 JOINT_OFFSET=3)
target_link_libraries(single_leg_controller ${SINGLE_LEG_LIBS})
