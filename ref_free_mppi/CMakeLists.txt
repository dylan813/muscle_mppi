cmake_minimum_required(VERSION 3.16)
project(ref_free_mppi_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fPIC")
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Unitree SDK2
list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")
find_package(unitree_sdk2 REQUIRED)

# MuJoCo (bundled with unitree_mujoco)
set(MUJOCO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../unitree_mujoco/simulate/mujoco")
include_directories(${MUJOCO_DIR}/include)
link_directories(${MUJOCO_DIR}/lib)

set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags,-rpath,/opt/unitree_robotics/lib")

add_executable(ref_free_mppi_controller
    ref_free_mppi_controller.cpp
    control/ref_free_mppi.cpp
    utils/tasks.cpp
)

target_link_libraries(ref_free_mppi_controller
    unitree_sdk2
    mujoco
    pthread
    OpenMP::OpenMP_CXX
)

add_executable(generate_reference
    generate_reference.cpp
    control/ref_free_mppi.cpp
    utils/tasks.cpp
)

target_link_libraries(generate_reference
    mujoco
    pthread
    OpenMP::OpenMP_CXX
)
