# Latte.jl

[![Build Status](https://travis-ci.org/IntelLabs/Latte.jl.svg?branch=master)](https://travis-ci.org/IntelLabs/Latte.jl)
[![Join the chat at https://gitter.im/IntelLabs/Latte.jl](https://badges.gitter.im/IntelLabs/Latte.jl.svg)](https://gitter.im/IntelLabs/Latte.jl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Coverage Status](https://coveralls.io/repos/github/IntelLabs/Latte.jl/badge.svg?branch=master)](https://coveralls.io/github/IntelLabs/Latte.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](http://intellabs.github.io/Latte.jl/latest/)

**IMPORTANT:** Latte is prelease (alpha) software, please report bugs using the [Github issue tracker](https://github.com/IntelLabs/Latte.jl/issues). For usage questions, visit the [Gitter room](https://gitter.im/IntelLabs/Latte.jl).

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

# Examples
## MNIST
```shell
$ cd ~/.julia/v0.4/Latte/examples/mnist/data
$ ./get-data.sh
$ cd ..
$ julia mnist.jl
```

## cifar10
```shell
$ cd ~/.julia/v0.4/Latte/examples/cifar10/data
$ ./get-data.sh
$ cd ..
$ julia vgg-mini.jl
```
