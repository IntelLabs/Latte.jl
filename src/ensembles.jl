#=
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=#

export Ensemble, DataEnsemble, AbstractEnsemble, length, output,
    ActivationEnsemble

Base.length(ens::AbstractEnsemble)         = Base.length(ens.neurons)
Base.start(ens::AbstractEnsemble)          = Base.start(ens.neurons)
Base.done(ens::AbstractEnsemble, i::Int64) = Base.done(ens.neurons, i)
Base.next(ens::AbstractEnsemble, i::Int64) = Base.next(ens.neurons, i)
Base.ndims(ens::AbstractEnsemble)          = Base.ndims(ens.neurons)
Base.size(ens::AbstractEnsemble)           = Base.size(ens.neurons)
Base.size(ens::AbstractEnsemble, i::Int64) = Base.size(ens.neurons, i)
Base.getindex(ens::AbstractEnsemble, a...) = Base.getindex(ens.neurons, a...)

function Base.length(conn::Connection)
    conn.size
end

function Ensemble{T,N}(net::Net, name::Symbol, neurons::Array{T,N},
                       params::Vector=Param[]; phase::Phase=TrainTest)
    ens = Ensemble{T,N}(name, neurons, [], [:value, :∇, :inputs, :∇inputs],
                        Dict(), params, phase)
    add_ensemble(net, ens)
    ens
end

function AbstractString(ensemble::AbstractEnsemble)
    return "$(ensemble.name) $(size(ensemble.neurons))"
end

function init_inputs(ensemble::AbstractEnsemble, net::Net)
    batch_size = net.batch_size
    neuron = ensemble.neurons[1]
    shape = size(ensemble)
    for name in filter((x) -> x in [:inputs, :∇inputs], fieldnames(neuron))
        field = getfield(neuron, name)
        typ = typeof(field)
        key = symbol(ensemble.name,name)
        for t = 1:net.time_steps
            target = name == :inputs ? :value : :∇
            for (source_index, connection) in enumerate(ensemble.connections)
                key = symbol(ensemble.name,name,source_index)
                source_target = symbol(connection.source.name, target)
                source = get(net.buffers[t],
                             source_target,
                             nothing)
                if source == nothing
                    @assert !connection.recurrent
                    sink = Array(eltype(typ), 0, shape..., batch_size)
                    set_buffer(net, key, sink, t)
                elseif all(connection.is_dim_fixed)
                    connection.copy = false
                    if connection.recurrent && t == 1
                        set_buffer(net, key, zeros(eltype(typ), connection.size, batch_size), t)
                    elseif connection.recurrent && t > 1
                        source = get_buffer(net, source_target, t - 1)
                        set_buffer(net, key, reshape(source, (connection.size, batch_size)), t)
                    else
                        set_buffer(net, key, reshape(source, (connection.size, batch_size)), t)
                    end
                elseif connection.is_one_to_one
                    connection.copy = false
                    if connection.recurrent && t == 1
                        set_buffer(net, key, zeros(eltype(typ), size(ensemble)..., batch_size), t)
                    elseif connection.recurrent && t > 1
                        source = get_buffer(net, source_target, t - 1)
                        set_buffer(net, key, source, t)
                    else
                        set_buffer(net, key, source, t)
                    end
                else
                    _shape = shape
                    if LATTE_DISABLE_TILING
                        _shape[2] = TILE_SIZE * 2
                        num_tiles = batch_size
                    else
                        num_tiles = num_threads
                    end
                    _shape = _shape[!connection.is_dim_fixed]
                    sink = zeros(eltype(typ), connection.size, _shape..., num_tiles)
                    set_buffer(net, key, sink, t)
                end
            end
        end
    end
end

@doc """
Initialize `ensemble` of neuron type `T`
Allocate a buffer for each field in T

If the neurons are connected to a contigious region in another ensemble, map the
inputs array using ArrayViews and remove the need for a CopyTask
"""
@inbounds function init(ensemble::AbstractEnsemble, net::Net)
    batch_size = net.batch_size
    neuron = ensemble.neurons[1]
    shape = size(ensemble)
    for name in fieldnames(neuron)
        # log_info("Initializing field $(name) for ensemble $(ensemble.name)")
        # @time begin
        field = getfield(neuron, name)
        typ = typeof(field)
        key = symbol(ensemble.name,name)
        if name in [:inputs, :∇inputs]
            # skip
        elseif name in [:value, :∇]
            # :value and :∇ should allocate space for every item in a batch
            init_buffer(net, key, (shape..., batch_size))
        elseif isa(field, Batch)
            arr = Array(typeof(field.init), shape..., batch_size)
            fill!(arr, field.init)
            set_buffer(net, key, arr)
            push!(ensemble.batch_fields, name)
        elseif isa(field, Shared)
            # Skip
        elseif !isbits(typ)
            uniform_across_dim = [true for _ in 1:length(shape)]
            first = getfield(ensemble.neurons[ones(Int, length(shape))...], name)
            for d in 1:length(shape)
                for i in 1:shape[d]
                    idx = [x == d ? i : 1 for x in 1:length(shape)]
                    if first !== getfield(ensemble.neurons[idx...], name)
                        uniform_across_dim[d] = false
                        break
                    end
                end
            end
            ensemble.arg_dim_info[key] = uniform_across_dim
            _shape = []
            iter = []
            for i in 1:length(shape)
                if !uniform_across_dim[i]
                    push!(_shape, shape[i])
                    push!(iter, shape[i])
                else
                    push!(iter, 1)
                end
            end
            elem = getfield(ensemble.neurons[ones(Int, length(shape))...], name)
            buf = Array(eltype(typ), size(elem)..., _shape...)
            for index in CartesianRange(CartesianIndex(tuple(ones(Int, length(iter))...)),
                                        CartesianIndex(tuple(iter...)))
                _index = []
                for i in 1:length(shape)
                    if !uniform_across_dim[i]
                        push!(_index, index.I[i])
                    end
                end
                @inbounds buf[:, _index...] = getfield(ensemble.neurons[index], name)
            end
            set_buffer(net, key, buf)
        else
            # Neuron specific fields, only one instance per neuron instead of
            # per batch
            arr = map((elem) -> getfield(elem, name), ensemble.neurons)
            for t = 1:net.time_steps
                set_buffer(net, key, arr, t)
            end
        end
        # end  # @time begin
    end
end


"""
Get the forward function for a neuron in `ens`
"""
function get_forward(ens::AbstractEnsemble)
    neuron = ens.neurons[1]
    return forward(neuron)
end


"""
Get the backward function for a neuron in `ens`
"""
function get_backward(ens::AbstractEnsemble)
    neuron = ens.neurons[1]
    return backward(neuron)
end

function one_to_one(ndims)
    idx = [symbol(:i_, d) for d in 1:ndims]
    fn = Expr(:(->), Expr(:tuple, idx...), 
                     Expr(:block, Expr(:tuple, idx...)))
    return eval(fn)
end

type ActivationEnsemble{T <: Neuron, N} <: AbstractEnsemble
    name         :: Symbol
    neurons      :: Array{T,N}
    connections  :: Vector{Connection}
    batch_fields :: Vector{Symbol}
    arg_dim_info :: Dict{Symbol, Vector{Bool}}
    params       :: Vector{Param}
    phase        :: Phase
end

function ActivationEnsemble{T,N}(net::Net, name::Symbol, neurons::Array{T,N},
                                 input_ensemble::AbstractEnsemble; phase::Phase=TrainTest)
    params = Param[]
    ens = ActivationEnsemble{T,N}(name, neurons, [], 
        [:value, :∇, :inputs, :∇inputs], Dict(), params, phase)
    add_ensemble(net, ens)
    mapping = one_to_one(ndims(input_ensemble))
    add_connections(net, input_ensemble, ens, mapping)
    ens
end


function init_inputs(ensemble::ActivationEnsemble, net::Net)
    # Skip
end

@inbounds function init(ensemble::ActivationEnsemble, net::Net)
    batch_size = net.batch_size
    neuron = ensemble.neurons[1]
    shape = size(ensemble)
    for name in fieldnames(neuron)
        field = getfield(neuron, name)
        typ = typeof(field)
        key = symbol(ensemble.name,name)
        @assert length(ensemble.connections) == 1
        if name in [:inputs, :∇inputs]
            target = name == :inputs ? :value : :∇
            for (source_index, connection) in enumerate(ensemble.connections)
                connection.copy = false
                key = symbol(ensemble.name,name,source_index)
                for t = 1:net.time_steps
                    source = get_buffer(net, symbol(connection.source.name, target), t)
                    set_buffer(net, key, reshape(source, (1, size(source)...)), t)
                    key = symbol(ensemble.name,target)
                    set_buffer(net, key, source, t)
                end
            end
        elseif name in [:value, :∇]
            # Skip
        elseif isa(field, Batch)
            arr = Array(typeof(field.init), shape..., batch_size)
            fill!(arr, field.init)
            set_buffer(net, key, arr)
            push!(ensemble.batch_fields, name)
        elseif isa(field, Shared)
            # Skip
        else
            throw("Not implemented error")
        end
    end
end

macro output(ensemble_type, fn)
    return Expr(:escape, quote 
        function $(fn.args[1].args[1])(ens::$ensemble_type)
            return $(Expr(:quote, fn))
        end 
    end)
end

function init_inputs(ensemble::NormalizationEnsemble, net::Net)
end
