cmake_minimum_required(VERSION 3.24)
project(matrix)

set(CMAKE_CXX_STANDARD 23)
add_compile_options(-O3)
set(CMAKE_CXX_FLAGS "-mavx2")
set(CMAKE_CXX_FLAGS "-mfma")

add_executable(matrix main.cpp)
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
