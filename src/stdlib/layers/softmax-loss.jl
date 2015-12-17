# Copyright (c) 2015 Intel Corporation. All rights reserved.
export SoftmaxLossLayer

type SoftmaxLossNeuron <: Neuron
    value    :: Float32
    ∇        :: Float32
end

type SoftmaxLossEnsemble <: NormalizationEnsemble
    name        :: Symbol
    neurons     :: Vector{SoftmaxLossNeuron}
    connections :: Vector{Connection}
    num_inputs  :: Int
    phase       :: Phase
end

function SoftmaxLossEnsemble(name::Symbol, num_inputs::Int)
    SoftmaxLossEnsemble(name, [], [], num_inputs, Train)
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
