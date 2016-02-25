cmake_minimum_required (VERSION 2.8.11)
project (LatteComm)

FIND_PACKAGE(OpenMP REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_library(LatteComm SHARED comm.cpp comm.h)
target_link_libraries(LatteComm ${MPI_C_LIBRARIES})
