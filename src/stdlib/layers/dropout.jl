# Copyright (c) 2015 Intel Corporation. All rights reserved.
export DropoutLayer

@neuron type DropoutNeuron
    ratio   :: Shared{Float32}
    randval :: Batch{LatteFloat}
end

DropoutNeuron(ratio::Float32) = DropoutNeuron(Shared(ratio), Batch(0.0f0))

@neuron forward(neuron::DropoutNeuron) do
    if neuron.randval > .5f0
        neuron.value = neuron.inputs[1] * 2.0f0
    else
        neuron.value = 0.0
    end
end

@neuron backward(neuron::DropoutNeuron) do
    if neuron.randval > .5f0
        neuron.∇inputs[1] = neuron.∇ * 2.0f0
    else
        neuron.∇inputs[1] = 0.0
    end
end

function DropoutLayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble, ratio=0.5f0)
    neurons = Array(DropoutNeuron, size(input_ensemble)...)
    for i in 1:length(neurons)
        neurons[i] = DropoutNeuron(ratio)
    end
    ActivationEnsemble(net, name, neurons, input_ensemble; phase=Train)
end
