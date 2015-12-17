cmake_minimum_required (VERSION 2.8.11)
SET(CMAKE_C_COMPILER mpiicc)
SET(CMAKE_CXX_COMPILER mpiicpc)
project (LatteComm)
FIND_PACKAGE( OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_library(LatteComm SHARED comm.cpp comm.h)
