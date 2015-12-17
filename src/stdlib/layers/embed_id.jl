# Copyright (c) 2015 Intel Corporation. All rights reserved.
export EmbedNeuron, EmbedIDLayer

@neuron type EmbedNeuron
    weights  :: DenseArray{LatteFloat}
    ∇weights :: DenseArray{LatteFloat}
end

@neuron forward(neuron::EmbedNeuron) do
    idx = round(Int, neuron.inputs[1][1])
    neuron.value = neuron.weights[idx]
end

@neuron backward(neuron::EmbedNeuron) do
    _idx = round(Int, neuron.inputs[1][1])
    neuron.∇weights[_idx] += neuron.∇
end

function EmbedIDLayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble, in_size::Int, out_size::Int)
    @assert(length(input_ensemble) == 1)
    weights = xavier(Float32, in_size, out_size)
    ∇weights = zeros(Float32, in_size, out_size)

    neurons = [EmbedNeuron(view(weights, :, i), view(∇weights, :, i)) for i in 1:out_size]
    ens = Ensemble(net, name, neurons, [Param(net, name, :weights, 1.0)])
    add_connections(net, input_ensemble, ens, function (i)
        (1:1, )
    end)
    ens
end
