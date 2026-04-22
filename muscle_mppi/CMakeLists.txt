cmake_minimum_required(VERSION 3.16)
project(muscle_mppi_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fPIC")

list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")
find_package(unitree_sdk2 REQUIRED)
find_package(OpenMP REQUIRED)

set(MUJOCO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../unitree_mujoco/simulate/mujoco")
include_directories(${MUJOCO_DIR}/include)
link_directories(${MUJOCO_DIR}/lib)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags,-rpath,/opt/unitree_robotics/lib")

add_executable(muscle_mppi_controller
    mppi_controller.cpp
    control/base_mppi.cpp
    control/mppi_locomotion.cpp
    utils/tasks.cpp
)

target_link_libraries(muscle_mppi_controller
    unitree_sdk2
    mujoco
    pthread
    OpenMP::OpenMP_CXX
)
