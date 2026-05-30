cmake_minimum_required(VERSION 3.16)
project(muscle_unit_tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

add_executable(test_muscle test_muscle.cpp)
target_include_directories(test_muscle PRIVATE ..)
