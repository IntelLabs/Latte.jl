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

export AccuracyLayer
type AccuracyNeuron <: Neuron
    value    :: Float32
    inputs   :: Vector{Float32}
    AccuracyNeuron() = new(0.0, [])
end

type AccuracyEnsemble <: NormalizationEnsemble
    name         :: Symbol
    neurons      :: Vector{AccuracyNeuron}
    connections  :: Vector{Connection}
    phase        :: Phase
    net_subgroup :: Cint
end

function AccuracyEnsemble(net::Net, name::Symbol)
    ens = AccuracyEnsemble(name, [], [], Test, 1)
    add_ensemble(net, ens)
    ens
end

function get_forward_args(ens::AccuracyEnsemble)
    return [symbol(ens.name,:value)]
end

function get_backward_args(ens::AccuracyEnsemble)
    return []
end

function init(ensemble::AccuracyEnsemble, net::Net)
    set_buffer(net, symbol(ensemble.name,:value), Array(Float32, 1))
end

function AccuracyLayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble,
                       label_ensemble::AbstractEnsemble)
    accuracy = AccuracyEnsemble(net, name)
    add_connections(net, input_ensemble, accuracy, (i) -> (1:length(input_ensemble),))
    add_connections(net, label_ensemble, accuracy, (i) -> (1:1,))
    accuracy
end

@output AccuracyEnsemble function forward(acc::Array, input::Array, label::Array)
    accuracy = 0.0
    for n in 1:size(input, 2)
        max_idx = 1
        max_val = input[1, n]
        for i in 2:size(input, 1)
            if input[i, n] > max_val
                max_val = input[i, n]
                max_idx = i
            end
        end
        if max_idx == round(Int, label[1, n]) + 1
            accuracy += 1.0
        end
    end
    acc[1] = accuracy / size(input, 2)
    return 0
end
