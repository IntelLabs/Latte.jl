cmake_minimum_required (VERSION 2.8.11)
project (LatteComm)

FIND_PACKAGE(OpenMP REQUIRED)
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -v OUTPUT_VARIABLE CXX_VERSION)
if (NOT (CXX_VERSION MATCHES "MPI"))
    message(FATAL_ERROR "Latte's communication library requires CXX be set as an MPI compiler.")
else()
    message("${PROJECT_NAME} building with MPI")
endif()

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_library(LatteComm SHARED comm.cpp comm.h)
