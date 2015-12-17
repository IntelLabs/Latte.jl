# Copyright (c) 2015 Intel Corporation. All rights reserved.
export DataNeuron, WeightedNeuron, gaussian, xavier

type DataNeuron <: Neuron
    value :: Float32
    ∇     :: Float32
end

function DataNeuron(value::AbstractFloat)
    DataNeuron(value, 0.0)
end

@neuron type WeightedNeuron
    weights  :: DenseArray{LatteFloat}
    ∇weights :: DenseArray{LatteFloat}

    bias     :: DenseArray{LatteFloat}
    ∇bias    :: DenseArray{LatteFloat}
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

function gaussian(;mean=0.0, std=1.0)
    function gaussian_init(eltype::Type, dims...)
        rand(eltype, dims...) * std + mean
    end
end

function xavier_init(eltype::Type, dims...)
    fan_in = prod(dims[1:end-1])
    scale = sqrt(3.0 / fan_in)
    rand(eltype, dims...) * 2scale - scale
end

xavier = xavier_init

include("layers/convolution.jl")
include("layers/pooling.jl")
include("layers/relu.jl")
include("layers/inner-product.jl")
include("layers/softmax-loss.jl")
include("layers/accuracy.jl")
include("layers/hdf5-data.jl")
include("layers/memory-data.jl")
include("layers/dropout.jl")
include("layers/math.jl")
include("layers/lstm.jl")
include("layers/embed_id.jl")
include("layers/GRU.jl")
include("layers/transform.jl")

# type WeightedEnsemble <: AbstractEnsemble
#     name         :: Symbol
#     neurons      :: Array{WeightedNeuron}
#     connections  :: Vector{Connection}
#     batch_fields :: Vector{Symbol}
#     arg_dim_info :: Dict{Symbol, Vector{Bool}}
#
#     weights      :: Array{Float32}
#     ∇weights     :: Array{Float32}
#
#     bias         :: Array{Float32}
#     ∇bias        :: Array{Float32}
# end
#
# function WeightedEnsemble(name::Symbol, neurons::Array{WeightedNeuron},
#                           weights::Array{Float32},
#                           ∇weights::Array{Float32},
#                           bias::Array{Float32},
#                           ∇bias::Array{Float32})
#     WeightedEnsemble(name, neurons, [], [:value, :∇, :inputs, :∇inputs],
#         Dict(), weights, ∇weights, bias, ∇bias)
# end
