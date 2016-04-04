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
                         kernel::Int, stride::Int, pad::Int; ceil_mode=false)
    width, height, channels = size(input_ensemble)
    if ceil_mode
        # Round up
        div_op = cld
    else
        div_op = fld
    end
    width_out = div_op((width - kernel + 2 * pad), stride) + 1
    height_out = div_op((height - kernel + 2 * pad), stride) + 1
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
                          kernel::Int, stride::Int, pad::Int; ceil_mode=false)
    if ceil_mode
        # Round up
        div_op = cld
    else
        div_op = fld
    end
    width, height, channels = size(input_ensemble)
    width_out = div_op((width - kernel + 2 * pad), stride) + 1
    height_out = div_op((height - kernel + 2 * pad), stride) + 1
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

