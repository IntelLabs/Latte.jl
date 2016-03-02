export TanhLayer

@neuron type TanhNeuron
end

@neuron forward(neuron::TanhNeuron) do
    neuron.value = tanh(neuron.inputs[1])
end

∇tanh(x) = x * (1.0f0 - x)

@neuron backward(neuron::TanhNeuron) do
    neuron.∇inputs[1] = ∇tanh(neuron.∇)
end

function TanhLayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble)
    neurons = Array(TanhNeuron, size(input_ensemble)...)
    for i in 1:length(neurons)
        neurons[i] = TanhNeuron()
    end
    ActivationEnsemble(net, name, neurons, input_ensemble)
end
