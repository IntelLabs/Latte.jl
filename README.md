# Latte.jl

[![Build Status](https://travis-ci.org/IntelLabs/Latte.jl.svg?branch=master)](https://travis-ci.org/IntelLabs/Latte.jl)
[![Join the chat at https://gitter.im/IntelLabs/Latte.jl](https://badges.gitter.im/IntelLabs/Latte.jl.svg)](https://gitter.im/IntelLabs/Latte.jl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Coverage Status](https://coveralls.io/repos/github/IntelLabs/Latte.jl/badge.svg?branch=master)](https://coveralls.io/github/IntelLabs/Latte.jl?branch=master)

*Note:* Latte is prelease (alpha) software, please report bugs to our [Github issue tracker](https://github.com/IntelLabs/Latte.jl/issues).

Check out the [wiki](https://github.com/IntelLabs/Latte.jl/wiki) for tutorials and documentation.

# Setup

## Prerequisites

To build Latte, you will need MPI >= 3 and HDF5.  What to install will
vary by platform and your needs.  On Ubuntu, try:

```shell
$ sudo apt-get install libmpich2-dev mpich2 hdf5-tools libhdf5-dev 
```

## Quick Install
```shell
# Latte currently depends on the master branch of these packages
$ julia -e 'Pkg.clone("https://github.com/IntelLabs/Latte.jl")'
$ julia -e 'Pkg.checkout("CompilerTools")'
$ julia -e 'Pkg.checkout("ParallelAccelerator")'
$ julia -e 'Pkg.build("Latte")'
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
