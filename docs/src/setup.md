# Setup

## Prerequisites

Latte depends on HDF5 and cmake.

To build Latte, you will need HDF5, cmake, and either OpenBLAS or MKL.  What to
install will vary by platform and your needs.

### Ubuntu

The easiest setup with serial HDF5, cmake, and OpenBLAS.
```shell
$ sudo apt-get install hdf5-tools libhdf5-dev cmake libopenblas-dev
```

For MPI support you can use the mpich provided packages or compile HDF5 yourself.
```shell
$ sudo apt-get install libmpich2-dev mpich2 hdf5-tools libhdf5-mpich2-dev
```

## Installing the Julia package
Until Latte is added to the Julia Package Repository, it must be directly
cloned off Github as follows:

```julia
julia> Pkg.clone("https://github.com/IntelLabs/Latte.jl")
```

Latte currently depends on the master branch of some dependencies, they
can be checked out as follows:

```
julia> Pkg.checkout("CompilerTools")
julia> Pkg.checkout("ParallelAccelerator")
```

Next, build the supporting libraries for Latte with
```
julia> Pkg.build("Latte")
```

To ensure everything was installed properly, run the test suite:
```
julia> Pkg.test("Latte")
...
...
...
INFO: Latte tests passed
```

