# Copyright (c) 2015 Intel Corporation. All rights reserved.
export ReLULayer

@neuron type ReLUNeuron
end

@neuron forward(neuron::ReLUNeuron) do
    neuron.value = max(neuron.inputs[1], 0.0f0)
end

@neuron backward(neuron::ReLUNeuron) do
    neuron.∇inputs[1] = neuron.inputs[1] > 0.0f0 ? neuron.∇ : 0.0f0
end

function ReLULayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble)
    neurons = Array(ReLUNeuron, size(input_ensemble)...)
    for i in 1:length(neurons)
        neurons[i] = ReLUNeuron()
    end
    ActivationEnsemble(net, name, neurons, input_ensemble)
end
