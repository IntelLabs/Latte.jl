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
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# FIXME: This should be moved into the Latte core as it uses special internal
# handling.
export ConcatLayer

function ConcatEnsemble(name::Symbol, inputs::Vector)
    # shape = (size(inputs[1])..., length(inputs))
    shape = [size(inputs[1])...]
    inner_size = prod(shape)
    shape[end] *= length(inputs)
    neurons = Array(ConcatNeuron, shape...)
    for i in 1:length(neurons)
        neurons[i] = ConcatNeuron(0.0, 0.0)
    end
    ConcatEnsemble(name, neurons, [], TrainTest, inputs, inner_size, 1)
end

function init(ensemble::ConcatEnsemble, net::Net)
    set_buffer(net, symbol(ensemble.name, :value), Array(Float32, size(ensemble)..., net.batch_size))
    set_buffer(net, symbol(ensemble.name, :∇), Array(Float32, size(ensemble)..., net.batch_size))
end

function ConcatLayer(name::Symbol, net::Net, input_ensembles...)
    @assert length(input_ensembles) > 1 "Must concatenate at least two ensembles"
    for ens in input_ensembles[2:end]
        @assert(size(input_ensembles[1]) == size(input_ensembles[2]), 
                "All inputs to ConcatLayer must have the same size")
    end

    ens = ConcatEnsemble(name, [input_ensembles...])
    add_ensemble(net, ens)
    ens
end
