# Copyright (c) 2015 Intel Corporation. All rights reserved.
export AccuracyLayer
type AccuracyNeuron <: Neuron
    value    :: Float32
    inputs   :: Vector{Float32}
    AccuracyNeuron() = new(0.0, [])
end

type AccuracyEnsemble <: NormalizationEnsemble
    name        :: Symbol
    neurons     :: Vector{AccuracyNeuron}
    connections :: Vector{Connection}
    phase       :: Phase
end

function AccuracyEnsemble(net::Net, name::Symbol)
    ens = AccuracyEnsemble(name, [], [], Test)
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
