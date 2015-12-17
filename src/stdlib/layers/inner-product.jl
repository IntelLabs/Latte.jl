# Copyright (c) 2015 Intel Corporation. All rights reserved.
export InnerProductLayer
function InnerProductLayer(name::Symbol, net::Net,
                           input_ensemble::AbstractEnsemble,
                           num_outputs::Int; weight_init=xavier, bias_init::Real=0)
    # Create a 1-D array of `num_outputs` WeightedNeurons
    neurons = Array(WeightedNeuron, num_outputs)

    weights = weight_init(Float32, length(input_ensemble), num_outputs)
    ∇weights = zeros(Float32, length(input_ensemble), num_outputs)

    bias = Array(Float32, 1, num_outputs)
    fill!(bias, bias_init)
    ∇bias = zeros(Float32, 1, num_outputs)

    # Instantiate each neuron
    for i in 1:num_outputs
        # One weight for each input neuron
        neurons[i] = WeightedNeuron(view(weights, :, i), view(∇weights, :, i),
                                    view(bias, :, i), view(∇bias, :, i))
    end
    # Construct the ensemble
    ip = Ensemble(net, name, neurons, [Param(net, name,:weights, 1.0f0, 1.0f0), 
                                       Param(net, name,:bias, 2.0f0, 0.0f0)])

    # Connect each neuron in input_ensemble to each neuron in ip
    add_connections(net, input_ensemble, ip,
                    (i) -> (tuple([Colon() for d in size(input_ensemble)]... )))
    ip
end
