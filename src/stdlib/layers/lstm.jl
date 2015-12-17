# Copyright (c) 2015 Intel Corporation. All rights reserved.
export LSTMLayer

@neuron type LSTMNeuron
    state  :: Batch{LatteFloat}
    ∇state :: Batch{LatteFloat}
end

function LSTMNeuron()
    LSTMNeuron(Batch(0.0f0), Batch(0.0f0))
end

@noinline function tanh(x)
    Base.tanh(x)
end

sigmoid(x)  = 1.0f0 / (1.0f0 + exp(-x))
∇sigmoid(x) = 1.0f0 - x * x
∇tanh(x)    = x * (1.0f0 - x)

@neuron forward(neuron::LSTMNeuron) do 
    _a = tanh(neuron.inputs[1][1])
    _i = sigmoid(neuron.inputs[1][2])
    _f = sigmoid(neuron.inputs[1][3])
    _o = sigmoid(neuron.inputs[1][4]) 
    neuron.state = _a * _i + _f * neuron.state 
    neuron.value = _o * tanh(neuron.state)
end

@neuron backward(neuron::LSTMNeuron) do
    _a = tanh(neuron.inputs[1][1])
    _i = sigmoid(neuron.inputs[1][2])
    _f = sigmoid(neuron.inputs[1][3])
    _o = sigmoid(neuron.inputs[1][4])
    co = tanh(neuron.state)
    neuron.∇state = neuron.∇ * _o * ∇tanh(co) + neuron.∇state
    neuron.∇inputs[1][1] = neuron.∇state * _i * ∇tanh(_a)
    neuron.∇inputs[1][2] = neuron.∇state * _a * ∇sigmoid(_i)
    neuron.∇inputs[1][3] = neuron.∇state * neuron.state * ∇sigmoid(_f)
    neuron.∇inputs[1][4] = neuron.∇ * co * ∇sigmoid(_o)
    neuron.∇state *= _f
end

function FCEnsemble(name::Symbol, net::Net, n_inputs::Int, n_outputs::Int)
    neurons = Array(WeightedNeuron, n_outputs)
    weights = xavier(Float32, n_inputs, n_outputs)
    ∇weights = zeros(Float32, n_inputs, n_outputs)

    bias = zeros(Float32, 1, n_outputs)
    ∇bias = zeros(Float32, 1, n_outputs)
    for i in 1:n_outputs
        neurons[i] = WeightedNeuron(view(weights, :, i), view(∇weights, :, i), 
            view(bias, :, i), view(∇bias, :, i))
    end
    Ensemble(net, name, neurons, [Param(net, name, :weights, 1.0), 
                                  Param(net, name, :bias, 2.0)])
end

function LSTMLayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble, output_dim::Int)
    @assert ndims(input_ensemble) == 1
    neurons = Array(LSTMNeuron, size(input_ensemble)...)
    n_inputs = length(neurons)
    for i = 1:n_inputs
        neurons[i] = LSTMNeuron()
    end

    lstm = Ensemble(net, name, neurons, [Param(net, name, :state, 1.0)])
    add_connections(net, _sum, lstm, function (i)
        return ((i-1)*4+1:i*4,)
    end)
    add_connections(net, lstm, h, function (i)
        tuple([Colon() for d in size(input_ensemble)]...)
    end; recurrent=true)
    lstm
end
