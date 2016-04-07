using Latte, ArrayViews

@neuron type MLPNeuron
    weights  :: DenseArray{Float32}
    ∇weights :: DenseArray{Float32}

    bias     :: DenseArray{Float32}
    ∇bias    :: DenseArray{Float32}
end

@neuron forward(neuron::MLPNeuron) do
    # dot product of weights and inputs
    for i in 1:length(neuron.inputs[1])
        neuron.value += neuron.weights[i] * neuron.inputs[1][i]
    end
    # shift by the bias value
    neuron.value += neuron.bias[1]
end

@neuron backward(neuron::MLPNeuron) do
    for i in 1:length(neuron.inputs[1])
        neuron.∇inputs[1][i] += neuron.weights[i] * neuron.∇
    end
    for i in 1:length(neuron.inputs[1])
        neuron.∇weights[i] += neuron.inputs[1][i] * neuron.∇
    end
    neuron.∇bias[1] += neuron.∇
end

function FCLayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble, num_neurons::Int)
    neurons = Array(MLPNeuron, num_neurons)
    num_inputs = length(input_ensemble)
    weights = xavier(Float32, num_inputs, num_neurons)
    ∇weights = zeros(Float32, num_inputs, num_neurons)

    bias = zeros(Float32, 1, num_neurons)
    ∇bias = zeros(Float32, 1, num_neurons)
    for i in 1:num_neurons
        neurons[i] = MLPNeuron(view(weights, :, i), view(∇weights, :, i),
                               view(bias, :, i), view(∇bias, :, i))
    end
    ens = Ensemble(net, name, neurons, [Param(name,:weights, 1.0f0, 1.0f0), 
                                        Param(name,:bias, 2.0f0, 0.0f0)])
    add_connections(net, input_ensemble, ens,
                    (i) -> (tuple([Colon() for d in size(input_ensemble)]... )))
    return ens
end

net = Net(100)
data, label = HDF5DataLayer(net, "data/train.txt", "data/test.txt")

fc1 = FCLayer(:fc1, net, data, 100)
fc2 = FCLayer(:fc2, net, fc1, 10)

loss     = SoftmaxLossLayer(:loss, net, fc2, label)
accuracy = AccuracyLayer(:accuracy, net, fc2, label)

params = SolverParameters(
    lr_policy    = LRPolicy.Inv(0.01, 0.0001, 0.75),
    mom_policy   = MomPolicy.Fixed(0.9),
    max_epoch    = 50,
    regu_coef    = .0005,
    snapshot_dir = "mlp")
sgd = SGD(params)
solve(sgd, net)
