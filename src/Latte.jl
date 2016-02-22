#=
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=#

module Latte

using Base.Cartesian
using Iterators
using ArrayViews
using CompilerTools.AstWalker
using JLD

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

    @eval ccall((:init, $libComm), Void, ())
    # info("Finished initializing comm library")
end
@eval ccall((:init, $libIO), Void, (Cuchar,), LATTE_MPI)
atexit(() -> @eval ccall((:clean_up, $libIO), Void, ()))

@enum Phase TrainTest Train Test
export TrainTest, Train, Test

include("neuron.jl")

include("util.jl")
include("types.jl")
include("param.jl")
include("net.jl")
include("ensembles.jl")
include("comm.jl")

include("stdlib/solvers.jl")
include("stdlib/DL.jl")


end # module Latte

