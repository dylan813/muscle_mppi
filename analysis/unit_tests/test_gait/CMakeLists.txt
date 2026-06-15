cmake_minimum_required(VERSION 3.16)
project(test_gait)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fPIC")

set(MUSCLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../muscle_mppi")

list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")
find_package(unitree_sdk2 REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(
    ${MUSCLE_ROOT}/control
    ${MUSCLE_ROOT}/utils
)

set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags,-rpath,/opt/unitree_robotics/lib")

add_executable(test_gait
    test_gait.cpp
    ${MUSCLE_ROOT}/utils/tasks.cpp
)
target_link_libraries(test_gait
    unitree_sdk2
    yaml-cpp
    pthread
)
