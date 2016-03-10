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

export InnerProductLayer

function FullyConnectedEnsemble(name::Symbol, net::Net, num_inputs::Int, num_outputs::Int; weight_init=xavier, bias_init=0)
    # Create a 1-D array of `num_outputs` WeightedNeurons
    neurons = Array(WeightedNeuron, num_outputs)

    weights = weight_init(Float32, num_inputs, num_outputs)
    ∇weights = zeros(Float32, num_inputs, num_outputs)

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
    Ensemble(net, name, neurons, [Param(name,:weights, 1.0f0, 1.0f0), 
                                  Param(name,:bias, 2.0f0, 0.0f0)])
end

function FullyConnectedEnsemble(net::Net, num_inputs::Int, num_outputs::Int; weight_init=xavier, bias_init=0)
    FullyConnectedEnsemble(gensym("ensemble"), net, num_inputs, num_outputs; weight_init=weight_init, bias_init=bias_init)
end

function InnerProductLayer(name::Symbol, net::Net,
                           input_ensemble::AbstractEnsemble,
                           num_outputs::Int; weight_init=xavier, bias_init::Real=0)
    ip = FullyConnectedEnsemble(name, net, length(input_ensemble),
                                num_outputs; weight_init=weight_init,
                                bias_init=bias_init)

    # Connect each neuron in input_ensemble to each neuron in ip
    add_connections(net, input_ensemble, ip,
                    (i) -> (tuple([Colon() for d in size(input_ensemble)]... )))
    ip
end

function InnerProductLayer(net::Net,
                           input_ensemble::AbstractEnsemble,
                           num_outputs::Int; weight_init=xavier, bias_init::Real=0)
    InnerProductLayer(gensym("ensemble"), net, input_ensemble, num_outputs; weight_init=weight_init, bias_init=bias_init)
end
