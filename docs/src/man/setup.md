# Setup

## Prerequisites

To build Latte, you will need HDF5 and cmake.  What to install will vary by
platform and your needs.  On Ubuntu, try:

```shell
$ sudo apt-get install hdf5-tools libhdf5-dev cmake
```

Latte currently depends on [Intel
MKL](https://software.intel.com/en-us/intel-mkl) and the [Intel C++ Compiler
(icpc)](https://software.intel.com/en-us/c-compilers).

## Quick Install
```julia
# Latte currently depends on the master branch of these packages
julia> Pkg.clone("https://github.com/IntelLabs/Latte.jl")
julia> Pkg.checkout("CompilerTools")
julia> Pkg.checkout("ParallelAccelerator")
julia> # To build with MPI enabled, uncomment these lines
julia> # ENV["LATTE_BUILD_MPI"] = 1
julia> # ENV["CXX"] = "mpiicpc"  # Replace with your mpi compiler wrapper
julia> Pkg.build("Latte")
```
