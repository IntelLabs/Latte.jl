# Copyright (c) 2015 Intel Corporation. All rights reserved.
export MemoryDataLayer

type MemoryDataEnsemble{N,M} <: DataEnsemble
    name       :: Symbol
    neurons    :: Array{DataNeuron,N}
    value      :: Array{Float32, M}  # M != N because of batch dimension
end

function forward{N}(ens::MemoryDataEnsemble, data::Array{Float32,N}, phase::Phase)
    data[:] = ens.value[:]
end

function backward{N}(ens::MemoryDataEnsemble, data::Array{Float32,N}, phase::Phase)
end

function MemoryDataLayer(net::Net, name::Symbol, shape::Tuple; time_steps=1)
    data_neurons = Array(DataNeuron, shape...)
    for i in 1:length(data_neurons)
        data_neurons[i] = DataNeuron(0.0)
    end
    value = Array(Float32, shape..., batch_size(net))
    ens = MemoryDataEnsemble(name, data_neurons, value)
    add_ensemble(net, ens)
    ens, value
end
