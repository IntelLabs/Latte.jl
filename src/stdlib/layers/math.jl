# Copyright (c) 2015 Intel Corporation. All rights reserved.
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
