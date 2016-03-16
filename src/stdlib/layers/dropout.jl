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

export DropoutLayer

@neuron type DropoutNeuron
    ratio   :: Shared{Float32}
    randval :: Batch{Float32}
end

DropoutNeuron(ratio::Float32) = DropoutNeuron(Shared(ratio), Batch(0.0f0))

# FIXME: CGen does not support rand()
function float_rand()
    val = ccall((:drand48, "stdlib"), Cdouble, ())
end

@neuron forward(neuron::DropoutNeuron) do
    neuron.randval = float_rand()
    ratio = .5f0
    if neuron.randval > ratio
        neuron.value = neuron.inputs[1] * (1.0f0 / ratio)
    else
        neuron.value = 0.0
    end
end

@neuron backward(neuron::DropoutNeuron) do
    ratio = .5f0
    if neuron.randval > ratio
        neuron.∇inputs[1] = neuron.∇ * (1.0f0 / ratio)
    else
        neuron.∇inputs[1] = 0.0
    end
end

function DropoutLayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble, ratio=0.5f0)
    neurons = Array(DropoutNeuron, size(input_ensemble)...)
    for i in 1:length(neurons)
        neurons[i] = DropoutNeuron(ratio)
    end
    ActivationEnsemble(net, name, neurons, input_ensemble; phase=Train)
end
