# Copyright (c) 2015 Intel Corporation. All rights reserved.
export MaxPoolingLayer

@neuron type MaxNeuron
    maxidx :: Batch{Int}
end

MaxNeuron() = MaxNeuron(Batch(0))

@neuron forward(neuron::MaxNeuron) do
    begin
        maxval = -Inf32
        max_idx = 1
        for i in 1:length(neuron.inputs[1])
            curr = neuron.inputs[1][i]
            if curr > maxval
                maxval = curr
                max_idx = i
            end
        end
        neuron.value = maxval
        neuron.maxidx = max_idx
    end
end

@neuron backward(neuron::MaxNeuron) do
    begin
        idx = neuron.maxidx
        neuron.∇inputs[1][idx] += neuron.∇
    end
end

function MaxPoolingLayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble,
                         kernel::Int, stride::Int, pad::Int)
    width, height, channels = size(input_ensemble)
    width_out = div((width - kernel + 2 * pad), stride) + 1
    height_out = div((height - kernel + 2 * pad), stride) + 1
    neurons =
        [MaxNeuron() for _ = 1:width_out, _ = 1:height_out, _ = 1:channels]

    pool = Ensemble(net, name, neurons)
    add_connections(net, input_ensemble, pool, function (i, j, c)
        in_x = (i-1)*stride - pad
        in_y = (j-1)*stride - pad
        return (in_x+1:in_x+kernel, in_y+1:in_y+kernel, c:c)
    end; padding=pad)
    pool
end



export MeanPoolingLayer

@neuron type MeanNeuron
end

@neuron forward(neuron::MeanNeuron) do
    begin
        num_inputs = length(neuron.inputs[1])
        the_sum = 0.0f0
        for i in 1:num_inputs
            the_sum += neuron.inputs[1][i]
        end
        neuron.value = the_sum / num_inputs
    end
end

@neuron backward(neuron::MeanNeuron) do
    begin
        num_inputs = length(neuron.∇inputs[1])
        val = neuron.∇ / num_inputs
        for i in 1:num_inputs
            neuron.∇inputs[1][i] += val
        end
    end
end

function MeanPoolingLayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble,
                          kernel::Int, stride::Int, pad::Int)
    width, height, channels = size(input_ensemble)
    width_out = div((width - kernel + 2 * pad), stride) + 1
    height_out = div((height - kernel + 2 * pad), stride) + 1
    neurons =
        [MeanNeuron() for _ = 1:width_out, _ = 1:height_out, _ = 1:channels]

    pool = Ensemble(net, name, neurons)
    add_connections(net, input_ensemble, pool, function (i, j, c)
        in_x = (i-1)*stride - pad
        in_y = (j-1)*stride - pad
        return (in_x+1:in_x+kernel, in_y+1:in_y+kernel, c:c)
    end;padding=pad)
    pool
end

