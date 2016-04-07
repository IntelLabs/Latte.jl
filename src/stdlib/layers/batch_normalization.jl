# Copyright (c) 2015, Intel Corporation
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

export BatchNormalizationLayer

type BatchNormalizationNeuron <: Neuron
    value :: Float32
    ∇     :: Float32
end

type BatchNormalizationEnsemble <: NormalizationEnsemble
    name         :: Symbol
    neurons      :: Array{BatchNormalizationNeuron}
    connections  :: Vector{Connection}
    num_out      :: Int
    eps          :: Float32
    momentum     :: Float32
    phase        :: Phase
    net_subgroup :: Cint
end

function BatchNormalizationEnsemble(name::Symbol, shape, num_out, eps::Float32, momentum::Float32)
    neurons = Array(BatchNormalizationNeuron, shape...)
    for i = 1:length(neurons)
        neurons[i] = BatchNormalizationNeuron(0.0f0, 0.0f0)
    end

    BatchNormalizationEnsemble(name, neurons, [], num_out, eps, momentum, Train, 1)
end

function init(ensemble::BatchNormalizationEnsemble, net::Net)
    shape = size(ensemble)
    set_buffer(net, symbol(ensemble.name,:value), zeros(Float32, shape..., net.batch_size))
    set_buffer(net, symbol(ensemble.name,:∇), zeros(Float32, shape..., net.batch_size))
    set_buffer(net, symbol(ensemble.name,:running_mean), zeros(Float32, net.batch_size))
    set_buffer(net, symbol(ensemble.name,:running_var), ones(Float32, net.batch_size))
    set_buffer(net, symbol(ensemble.name,:weight), rand(Float32, net.batch_size))
    set_buffer(net, symbol(ensemble.name,:∇weight), zeros(Float32, net.batch_size))
    set_buffer(net, symbol(ensemble.name,:bias), zeros(Float32, net.batch_size))
    set_buffer(net, symbol(ensemble.name,:∇bias), zeros(Float32, net.batch_size))
    set_buffer(net, symbol(ensemble.name,:momentum), Float32[ensemble.momentum])
    set_buffer(net, symbol(ensemble.name,:eps), Float32[ensemble.eps])
end

function get_forward_args(ens::BatchNormalizationEnsemble)
    return [symbol(ens.name,:value), 
            symbol(ens.name,:weight),
            symbol(ens.name,:bias),
            symbol(ens.name,:running_mean),
            symbol(ens.name,:running_var),
            symbol(ens.name,:momentum),
            symbol(ens.name,:eps)]
end

function get_backward_args(ens::BatchNormalizationEnsemble)
    return [
        symbol(ens.name, :∇),
        symbol(ens.name, :weight),
        symbol(ens.name, :∇weight),
        symbol(ens.name, :∇bias),
        symbol(ens.name, :momentum),
        symbol(ens.name, :eps)]
end

function BatchNormalizationLayer(name::Symbol, net::Net, input_ens::AbstractEnsemble, eps::Float32, momentum::Float32)
    @assert(ndims(input_ens) == 3)
    ens = BatchNormalizationEnsemble(name, size(input_ens), 3, eps, momentum)
    add_ensemble(net, ens)
    add_connections(net, input_ens, ens, (i, j, k) -> (i:i, j:j, k:k))
    ens
end

@output BatchNormalizationEnsemble function forward(output::Array,
    weight::Array, bias::Array, running_mean::Array, running_var::Array,
    momentum::Array, epsilon::Array, x::Array)

    _momentum = momentum[1]
    _eps = epsilon[1]

    N = size(x, 4)
    inner_size = size(x, 1) * size(x, 2) * size(x, 3)
    for n in 1:N
        x_sum = 0.0f0
        for i_3 in 1:size(x, 3)
            for i_2 in 1:size(x, 2)
                for i_1 in 1:size(x, 1)
                    x_sum += x[i_1, i_2, i_3, n]
                end
            end
        end
        x_mean = x_sum / inner_size
        x_var = 0.0f0
        for i_3 in 1:size(x, 3)
            for i_2 in 1:size(x, 2)
                for i_1 in 1:size(x, 1)
                    dev = x[i_1, i_2, i_3, n] - x_mean
                    x_var += dev * dev
                end
            end
        end
        x_var /= inner_size

        invstd = 1.0f0 / sqrt(x_var + _eps)
        running_mean[n] = _momentum * x_mean + (1 - _momentum) * running_mean[n]
        running_var[n] = _momentum * x_var + (1 - _momentum) * running_var[n]
        for i_3 in 1:size(x, 3)
            for i_2 in 1:size(x, 2)
                for i_1 in 1:size(x, 1)
                    output[i_1, i_2, i_3, n] = ((x[i_1, i_2, i_3, n] - x_mean) * invstd) * weight[n] + bias[n]
                end
            end
        end
    end
end

@output BatchNormalizationEnsemble function backward(output_diff::Array,
    weight::Array, weight_diff::Array, bias_diff::Array,
    momentum::Array, eps::Array, x::Array, x_diff::Array)

    _momentum = momentum[1]
    _eps = eps[1]

    #= x_sum = sum(x, 4) =#
    #= x_var = var(x, 4) =#
    #= xhat = (x - x_sum) ./ sqrt(x_var .+ eps) =#
    #= grad_weight[:] = sum(xhat .* grad_output, 4) =#
    #= grad_bias[:] = sum(grad_output, ndims(x)) =#
    #= xtmp = (xhat .* grad_weight .+ grad_bias) / size(x, 4) =#
    #= grad_input[:] = weight .* (grad_output - xtmp) / sqrt(x_var + eps) =#
end
