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

export AddLayer

@neuron type AddNeuron
end

@neuron forward(neuron::AddNeuron) do
    neuron.value = neuron.inputs[1][1] + neuron.inputs[2][1]
end

@neuron backward(neuron::AddNeuron) do
    neuron.∇inputs[1][1] = neuron.∇
    neuron.∇inputs[2][1] = neuron.∇
end

function AddEnsemble(name::Symbol, net::Net, shape)
    neurons = Array(AddNeuron, shape...)
    for i in 1:length(neurons)
        neurons[i] = AddNeuron()
    end
    Ensemble(net, name, neurons)
end

function AddLayer(name::Symbol, net::Net, input_1::AbstractEnsemble, input_2::AbstractEnsemble)
    @assert size(input_1) == size(input_2)
    ens = AddEnsemble(name, net, size(input_1))
    mapping = one_to_one(ndims(input_1))
    add_connections(net, input_1, ens, mapping)
    add_connections(net, input_2, ens, mapping)
    ens
end

import Base: +, *

function +(net::Net, ens1::AbstractEnsemble, ens2::AbstractEnsemble)
    AddLayer(gensym("ensemble"), net, ens1, ens2)
end

function +(net::Net, ens1::AbstractEnsemble, ens...)
    AddLayer(gensym("ensemble"), net, ens1, +(net, ens...))
end


export MulLayer

@neuron type MulNeuron
end

@neuron forward(neuron::MulNeuron) do
    neuron.value = neuron.inputs[1][1] * neuron.inputs[2][1]
end

@neuron backward(neuron::MulNeuron) do
    neuron.∇inputs[1][1] = neuron.inputs[2][1] * neuron.∇
    neuron.∇inputs[2][1] = neuron.inputs[1][1] * neuron.∇
end

function MulEnsemble(name::Symbol, net::Net, shape)
    neurons = Array(MulNeuron, shape...)
    for i in 1:length(neurons)
        neurons[i] = MulNeuron()
    end
    Ensemble(net, name, neurons)
end

function MulEnsemble(net::Net, shape)
    MulEnsemble(gensym("ensemble"), net, shape)
end

function MulLayer(name::Symbol, net::Net, input_1::AbstractEnsemble, input_2::AbstractEnsemble)
    @assert size(input_1) == size(input_2)
    ens = MulEnsemble(name, net, size(input_1))
    mapping = one_to_one(ndims(input_1))
    add_connections(net, input_1, ens, mapping)
    add_connections(net, input_2, ens, mapping)
    ens
end

function *(net::Net, ens1::AbstractEnsemble, ens2::AbstractEnsemble)
    MulLayer(gensym("ensemble"), net, ens1, ens2)
end
