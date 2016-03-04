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

export EmbedNeuron, EmbedIDLayer

@neuron type EmbedNeuron
    weights  :: DenseArray{Float32}
    ∇weights :: DenseArray{Float32}
end

@neuron forward(neuron::EmbedNeuron) do
    idx = round(Int, neuron.inputs[1][1])
    neuron.value = neuron.weights[idx]
end

@neuron backward(neuron::EmbedNeuron) do
    _idx = round(Int, neuron.inputs[1][1])
    neuron.∇weights[_idx] += neuron.∇
end

function EmbedIDLayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble, in_size::Int, out_size::Int)
    @assert(length(input_ensemble) == 1)
    weights = xavier(Float32, in_size, out_size)
    ∇weights = zeros(Float32, in_size, out_size)

    neurons = [EmbedNeuron(view(weights, :, i), view(∇weights, :, i)) for i in 1:out_size]
    ens = Ensemble(net, name, neurons, [Param(name, :weights, 1.0f0, 1.0f0)])
    add_connections(net, input_ensemble, ens, function (i)
        (1:1, )
    end)
    ens
end
