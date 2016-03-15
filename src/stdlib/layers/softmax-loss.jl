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

export SoftmaxLossLayer

type SoftmaxLossNeuron <: Neuron
    value    :: Float32
    ∇        :: Float32
end

type SoftmaxLossEnsemble <: NormalizationEnsemble
    name         :: Symbol
    neurons      :: Vector{SoftmaxLossNeuron}
    connections  :: Vector{Connection}
    num_inputs   :: Int
    phase        :: Phase
    net_subgroup :: Cint
end

function SoftmaxLossEnsemble(name::Symbol, num_inputs::Int)
    SoftmaxLossEnsemble(name, [], [], num_inputs, Train, 1)
end

function init(ensemble::SoftmaxLossEnsemble, net::Net)
    set_buffer(net, symbol(ensemble.name,:prob),
               Array(Float32, ensemble.num_inputs, net.batch_size))
    set_buffer(net, symbol(ensemble.name,:value),
               Array(Float32, 1))
end

function get_forward_args(ens::SoftmaxLossEnsemble)
    return [symbol(ens.name,:value), symbol(ens.name,:prob)]
end

function get_backward_args(ens::SoftmaxLossEnsemble)
    return [symbol(ens.name,:prob)]
end

function SoftmaxLossLayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble,
                          label_ensemble::AbstractEnsemble)
    num_inputs = length(input_ensemble)
    softmax = SoftmaxLossEnsemble(name, num_inputs)
    add_ensemble(net, softmax)
    add_connections(net, input_ensemble, softmax, (i) -> (1:num_inputs,))
    add_connections(net, label_ensemble, softmax, (i) -> (1:1,))
    softmax
end

@output SoftmaxLossEnsemble function forward(loss::Array, prob::Array,
                                             input::Array, label::Array)
    loss[1] = 0.0
    # parallel_for(1:size(input, 2)) do n
    for n = 1:size(input, 2)
        maxval = -Inf32
        for i in 1:size(input, 1)
            maxval = max(maxval, input[i, n])
        end
        for i in 1:size(input, 1)
            prob[i, n] = exp(input[i, n] - maxval)
        end
        the_sum = 0.0f0
        for i in 1:size(input, 1)
            the_sum += prob[i, n]
        end
        for i in 1:size(input, 1)
            prob[i, n] /= the_sum
        end
    end
    for n in 1:size(input, 2)
        label_value = round(Int, label[1, n]) + 1
        loss[1] -= log(max(prob[label_value, n], eps(Float32)))
    end
    loss[1] /= size(input, 2)
    return 0
end

@output SoftmaxLossEnsemble function backward(prob::Array, diff::Array,
                  label::Array)
    # parallel_for(1:length(diff)) do i
    for i = 1:length(diff)
        diff[i] = prob[i]
    end
    # parallel_for(1:size(diff, 2)) do n
    for n = 1:size(diff, 2)
        label_value = round(Int, label[1, n]) + 1
        diff[label_value, n] -= 1
    end
    # parallel_for(1:length(diff)) do i
    for i = 1:length(diff)
        diff[i] /= size(diff, 2)
    end
    return 0
end
