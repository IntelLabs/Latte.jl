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

export MemoryDataLayer

type MemoryDataEnsemble{N,M} <: DataEnsemble
    name         :: Symbol
    neurons      :: Array{DataNeuron,N}
    value        :: Array{Float32, M}  # M != N because of batch dimension
    connections  :: Vector{Connection}
    phase        :: Phase
    net_subgroup :: Cint
end

function MemoryDataEnsemble{N, M}(name::Symbol, neurons::Array{DataNeuron,N}, value::Array{Float32, M}, phase::Phase)
    MemoryDataEnsemble{N,M}(name, neurons, value, Connection[], phase, convert(Cint, 1))
end

function forward{N}(ens::MemoryDataEnsemble, data::Array{Float32,N}, net::Net, phase::Phase)
    if net.time_steps > 1
        data[:] = ens.value[[Colon() for _ in ndims(ens)]..., :, net.curr_time_step]
    else
        data[:] = ens.value[:]
    end
end

function backward{N}(ens::MemoryDataEnsemble, data::Array{Float32,N}, net::Net, phase::Phase)
end

function MemoryDataLayer(net::Net, name::Symbol, shape::Tuple; phase=TrainTest)
    data_neurons = Array(DataNeuron, shape...)
    for i in 1:length(data_neurons)
        data_neurons[i] = DataNeuron(0.0)
    end
    shape = [shape...]
    push!(shape, batch_size(net))
    if net.time_steps > 1
        push!(shape, net.time_steps)
    end
    value = Array(Float32, shape...)
    ens = MemoryDataEnsemble(name, data_neurons, value, phase)
    add_ensemble(net, ens)
    ens, value
end
