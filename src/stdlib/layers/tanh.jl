export TanhLayer

@neuron type TanhNeuron
end

@neuron forward(neuron::TanhNeuron) do
    neuron.value = tanh(neuron.inputs[1])
end

∇tanh(x) = x * (1.0f0 - x)

@neuron backward(neuron::TanhNeuron) do
    neuron.∇inputs[1] = ∇tanh(neuron.value) * neuron.∇
end

function TanhLayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble; copy=false)
    neurons = Array(TanhNeuron, size(input_ensemble)...)
    for i in 1:length(neurons)
        neurons[i] = TanhNeuron()
    end
    if copy
        ens = Ensemble(net, name, neurons)
        mapping = one_to_one(ndims(input_ensemble))
        add_connections(net, input_ensemble, ens, mapping)
        ens
    else
        ActivationEnsemble(net, name, neurons, input_ensemble)
    end
end

function tanh(net::Net, ens::AbstractEnsemble; copy=false)
    TanhLayer(gensym("ensemble"), net, ens; copy=copy)
end

export SigmoidLayer

sigmoid(x) = 1 / (1 + exp(-x))
∇sigmoid(x) = 1.0f0 - x * x

@neuron type SigmoidNeuron
end

@neuron forward(neuron::SigmoidNeuron) do
    neuron.value = sigmoid(neuron.inputs[1])
end

@neuron backward(neuron::SigmoidNeuron) do
    neuron.∇inputs[1] = ∇sigmoid(neuron.value) * neuron.∇
end

function SigmoidLayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble)
    neurons = Array(SigmoidNeuron, size(input_ensemble)...)
    for i in 1:length(neurons)
        neurons[i] = SigmoidNeuron()
    end
    ActivationEnsemble(net, name, neurons, input_ensemble)
end

function σ(net, ens::AbstractEnsemble)
    SigmoidLayer(gensym("ensemble"), net, ens)
end
