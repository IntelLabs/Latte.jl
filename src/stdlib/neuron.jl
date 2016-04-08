export DataNeuron
type DataNeuron <: Neuron
    value :: Float32
    ∇     :: Float32
end

function DataNeuron(value::AbstractFloat)
    DataNeuron(value, 0.0)
end

export WeightedNeuron
@neuron type WeightedNeuron
    weights  :: DenseArray{Float32}
    ∇weights :: DenseArray{Float32}

    bias     :: DenseArray{Float32}
    ∇bias    :: DenseArray{Float32}
end

@neuron forward(neuron::WeightedNeuron) do
    for i in 1:length(neuron.inputs[1])
        neuron.value += neuron.weights[i] * neuron.inputs[1][i]
    end
    neuron.value += neuron.bias[1]
end

@neuron backward(neuron::WeightedNeuron) do
    # Only backpropogate if input layer requests it
    # Length 0 indicates no buffer was created to receive backpropogated ∇s
    # if length(neuron.∇inputs[1]) > 0
        for i in 1:length(neuron.inputs[1])
            neuron.∇inputs[1][i] += neuron.weights[i] * neuron.∇
        end
    # end
    for i in 1:length(neuron.inputs[1])
        neuron.∇weights[i] += neuron.inputs[1][i] * neuron.∇
    end
    neuron.∇bias[1] += neuron.∇
end
