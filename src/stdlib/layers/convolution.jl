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

export ConvolutionLayer
function compute_sliding_window_output_shape(input_shape, kernel, pad, stride)
    width, height, channels = input_shape
    width_out = div((width - kernel + 2 * pad), stride) + 1
    height_out = div((height - kernel + 2 * pad), stride) + 1
    width_out, height_out
end

function ConvolutionLayer(name::Symbol, net::Net,
                          input_ensemble::AbstractEnsemble, num_filters::Int,
                          kernel::Int, stride::Int, pad::Int; weight_init=xavier, bias_init::Real=0)
    @assert(ndims(input_ensemble) == 3,
            "Input ensemble to convolution layer must be 3-dimensional")
    width, height, channels = size(input_ensemble)
    # compute the output shape as a function of the input
    width_out, height_out = compute_sliding_window_output_shape(
        size(input_ensemble), kernel, pad, stride)

    # instantiate weights
    weights = weight_init(Float32, kernel*kernel*channels, num_filters)
    ∇weights = zeros(Float32, kernel*kernel*channels, num_filters)

    # instantiate shared bias
    bias = Array(Float32, 1, num_filters)
    fill!(bias, bias_init)
    ∇bias = zeros(Float32, 1, num_filters)

    # allocate neurons
    neurons = Array(WeightedNeuron, width_out, height_out, num_filters)

    # Neurons in the same output channel `o` share filters
    for o = 1:num_filters, h = 1:height_out, w = 1:width_out
        neurons[w, h, o] = WeightedNeuron(view(weights, :, o), view(∇weights, :, o),
                                          view(bias, :, o), view(∇bias, :, o))
    end

    conv = Ensemble(net, name, neurons, [Param(name, :weights, 1.0f0, 1.0f0), 
                                         Param(name, :bias, 2.0f0, 0.0f0)])
    add_connections(net, input_ensemble, conv, function (x, y, _)
        in_x = (x-1)*stride - pad
        in_y = (y-1)*stride - pad
        return (in_x+1:in_x+kernel, in_y+1:in_y+kernel, 1:channels)
    end; padding=pad)
    conv
end
