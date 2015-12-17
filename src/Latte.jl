# Copyright (c) 2015 Intel Corporation. All rights reserved.
module Latte

using Base.Cartesian
using Iterators
using ArrayViews
using CompilerTools.AstWalker

export init_mpi

pkg_root = joinpath(dirname(@__FILE__), "..")
latte_library_path = "$pkg_root/deps"
latte_include_path = "$pkg_root/deps/include"
libsuffix = @osx ? ".dylib" : ".so"
libIO = "$latte_library_path/libLatteIO$libsuffix"
libComm = "$latte_library_path/libLatteComm$libsuffix"

typealias LatteFloat Float32
type LatteException <: Exception end

LATTE_MPI = false

if haskey(ENV, "LATTE_MPI")
    LATTE_MPI = true
end

@eval ccall((:init, $libIO), Void, (Cuchar,), LATTE_MPI)
atexit(() -> @eval ccall((:clean_up, $libIO), Void, ()))

@enum Phase TrainTest Train Test
export TrainTest, Train, Test

include("neuron.jl")

include("util.jl")

if LATTE_MPI
    @eval ccall((:init, $libComm), Void, ())
    info("Finished initializing comm library")
end
include("types.jl")
include("param.jl")
include("net.jl")
include("ensembles.jl")
include("comm.jl")

include("stdlib/solvers.jl")
include("stdlib/DL.jl")


end # module Latte

