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

using FactCheck
using Latte

function convolution_forward(filter::Array, bias::Array, input::Array,
                             output::Array, stride, pad, kernel)
    width, height, channel, num = size(input)
    width_out, height_out, n_filter, _ = size(output)

    for n = 1:num, o = 1:n_filter, y = 1:height_out, x = 1:width_out,
        k = 1:channel, p = 1:kernel, q = 1:kernel

        in_y = (y-1) * stride - pad + p
        in_x = (x-1) * stride - pad + q
        if (in_y >= 1 && in_y <= height && in_x >= 1 && in_x <= width)
            filter_idx = ((k - 1) * kernel + (p - 1)) * kernel + q
            output[x, y, o, n] += input[in_x, in_y, k, n] *
                                  filter[filter_idx, o]
        end
    end

    # add bias
    for n = 1:num, o = 1:n_filter, y = 1:height_out, x = 1:width_out
        output[x, y, o, n] += bias[o]
    end

    return output
end

function convolution_backward(filter::Array, bias::Array, input::Array,
                              top_diff::Array, stride, pad, kernel,
                              ∇filter::Array, ∇bias::Array, ∇input::Array)
    width, height, channels, num = size(input)
    width_out, height_out, n_filter, _ = size(top_diff)

    # ∇bias
    for n = 1:num, o = 1:n_filter, y = 1:height_out, x = 1:width_out
        ∇bias[o] += top_diff[x, y, o, n]
    end

    # ∇filter and ∇input
    for n = 1:num, o = 1:n_filter, k = 1:channels, y = 1:height_out,
        x = 1:width_out, p = 1:kernel, q = 1:kernel

        in_y = (y-1) * stride - pad + p
        in_x = (x-1) * stride - pad + q
        if (in_y >= 1 && in_y <= height && in_x >= 1 && in_x <= width)
            filter_idx = ((k - 1) * kernel + (p - 1)) * kernel + q
            ∇filter[filter_idx, o] += top_diff[x,y,o,n] * input[in_x,in_y,k,n]
            ∇input[in_x,in_y,k,n] += top_diff[x,y,o,n] * filter[filter_idx,o]
        end
    end

    return (∇filter, ∇bias, ∇input)
end

pad = 1

srand(1234)
net = Net(8)
data,  data_value   = MemoryDataLayer(net, :data, (227, 227, 3))
label, label_value = MemoryDataLayer(net, :label, (1,))
data_value[:]  = rand(Float32, size(data_value)...) * 256
label_value[:] = map(floor, rand(Float32, size(label_value)...) * 10)
conv1        = ConvolutionLayer(:conv1, net, data, 10, 3, 1, pad)
conv2        = ConvolutionLayer(:conv2, net, conv1, 10, 3, 1, pad)
fc1          = InnerProductLayer(:fc1, net, conv2, 10)
loss         = SoftmaxLossLayer(:loss, net, fc1, label)

init(net)
ϵ = 1e-3

input    = get_buffer(net, :conv1value)
∇input   = get_buffer(net, :conv1∇)
filters  = get_buffer(net, :conv2weights)
∇filters = get_buffer(net, :conv2∇weights)
bias     = get_buffer(net, :conv2bias)
rand!(bias)
∇bias    = get_buffer(net, :conv2∇bias)
top_diff = get_buffer(net, :conv2∇)
# rand!(top_diff)
expected = zeros(get_buffer(net, :conv2value))
∇input_expected   = zeros(∇input)
∇filters_expected = zeros(size(∇filters)[1:end-1])
∇bias_expected    = zeros(size(∇bias)[1:end-1])

params = SolverParameters(
    lr_policy    = LRPolicy.Decay(.01f0, 5.0f-7),
    mom_policy   = MomPolicy.Fixed(0.9),
    max_epoch    = 300,
    regu_coef    = .0005)
sgd = SGD(params)

facts("Testing Convolution Layer") do
    context("Forward") do
        forward(net; solver=sgd)
        convolution_forward(filters, bias, input, expected, 1, pad, 3)
        # @fact expected --> roughly(net.buffers[:conv2value])
        @fact all(-ϵ .< expected - get_buffer(net, :conv2value) .< ϵ) --> true
    end
    context("Backward") do
        clear_∇(net)
        backward(net)
        convolution_backward(filters, bias, input, top_diff, 1, pad, 3,
                             ∇filters_expected, ∇bias_expected,
                             ∇input_expected)
        @fact all(-ϵ .< ∇input - ∇input_expected .< ϵ) --> true
        @fact all(-ϵ .< sum(∇filters, 3)[:,:,1] - ∇filters_expected .< ϵ) --> true
        @fact all(-ϵ .< sum(∇bias, 3)[:,:,1] - ∇bias_expected .< ϵ) --> true
    end
end

FactCheck.exitstatus()
