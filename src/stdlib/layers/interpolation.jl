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

export InterpolationLayer

@neuron type InterpolatedNeuron
    ih :: Int
    iw :: Int
    resize_factor :: Float32
end


@neuron forward(neuron::InterpolatedNeuron) do
    begin
        r_f = neuron.ih * neuron.resize_factor
        c_f = neuron.iw * neuron.resize_factor
        delta_r = r_f - floor(Int, r_f)
        delta_c = c_f - floor(Int, c_f)
        
        x = 1
        y = 1

        y_plus_1 = length(neuron.inputs[1]) 
        x_plus_1 = length(neuron.inputs[2]) 

        neuron.value =  neuron.inputs[1][y] * (1-delta_r) * (1-delta_c) 
                    + neuron.inputs[2][x] * delta_r * (1-delta_c)  
                    + neuron.inputs[1][x_plus_1] * (1-delta_r) * delta_c
                    + neuron.inputs[2][y_plus_1] * delta_r * delta_c 

    end
end

@neuron backward(neuron::InterpolatedNeuron) do
    begin
        r_f = neuron.ih * neuron.resize_factor
        c_f = neuron.iw * neuron.resize_factor
        delta_r = r_f - floor(r_f)
        delta_c = c_f - floor(c_f)

        x = 1
        y = 1

        y_plus_1 = length(neuron.inputs[1]) 
        x_plus_1 = length(neuron.inputs[2]) 

        neuron.∇inputs[1][y] += (1-delta_r) * (1-delta_c) * neuron.∇
        neuron.∇inputs[2][x] += delta_r * (1-delta_c) * neuron.∇
        neuron.∇inputs[1][x_plus_1] += (1-delta_r) * delta_c * neuron.∇
        neuron.∇inputs[2][y_plus_1] += delta_r * delta_c * neuron.∇

    end
end

function InterpolationLayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble,
                          kernel::Int, pad::Int, resize_factor::Float32)
    width, height, channels = size(input_ensemble)
    width_out = floor(Int, width * resize_factor)
    height_out = floor(Int, height * resize_factor)
    neurons =
        [InterpolatedNeuron(ih, iw, resize_factor) for iw = 1:width_out, ih = 1:height_out, _ = 1:channels]

    interpolate = Ensemble(net, name, neurons)
    add_connections(net, input_ensemble, interpolate, function (i, j, c)
        in_x = floor(Int, (i-1) / resize_factor) - pad 
        in_y = floor(Int, (j-1) / resize_factor) - pad 
        return (max(in_x+1, 1):min(in_x+kernel, width), max(in_y+1, 1):max(in_y+1, 1), c:c)
    end;padding=pad)
    add_connections(net, input_ensemble, interpolate, function (i, j, c)
        in_x = floor(Int, (i-1) / resize_factor) - pad 
        in_y = floor(Int, (j-1) / resize_factor) - pad 
        return (max(in_x+1, 1):min(in_x+kernel, width), min(in_y+kernel, height):min(in_y+kernel, height), c:c)
    end;padding=pad)
    interpolate
end

