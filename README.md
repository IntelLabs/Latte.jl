# Latte.jl

[![Build Status](https://travis-ci.org/IntelLabs/Latte.jl.svg?branch=master)](https://travis-ci.org/IntelLabs/Latte.jl)
[![Join the chat at https://gitter.im/IntelLabs/Latte.jl](https://badges.gitter.im/IntelLabs/Latte.jl.svg)](https://gitter.im/IntelLabs/Latte.jl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Check out the [wiki](https://github.com/IntelLabs/Latte.jl/wiki) for tutorials and documentation.

# Setup
## Quick Install
```shell
# Latte currently depends on the master branch of these packages
$ julia -e 'Pkg.clone("https://github.com/IntelLabs/Latte.jl")'
$ julia -e 'Pkg.checkout("CompilerTools")'
$ julia -e 'Pkg.checkout("ParallelAccelerator")'
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
