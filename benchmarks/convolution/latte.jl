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


using Latte
using HDF5

batch_size = 64

net = Net(batch_size)
data, data_value = MemoryDataLayer(net, :data, (112, 112, 64))
rand!(data_value)

conv1_1 = ConvolutionLayer( :conv1_1, net, data,    128,  3, 1, 1)

init(net)

forward(net; phase=Latte.Test)
backward(net)

forward_task = net.forward_tasks[Latte.Test][end]
forward_args = []
for arg in forward_task.args
    if isa(arg, Symbol)
        push!(forward_args, get_buffer(net,arg))
    else
        push!(forward_args, arg)
    end
end
function forward_bench()
    forward_task.func(forward_args...)
end

backward_task = net.backward_tasks[Latte.Train][end]
backward_args = []
for arg in backward_task.args
    if isa(arg, Symbol)
        push!(backward_args, get_buffer(net,arg))
    else
        push!(backward_args, arg)
    end
end
function backward_bench()
    backward_task.func(backward_args...)
end

for i = 1:3
    forward_bench()
    backward_bench()
end

num_trials = 10

forward_time = 0.0
backward_time = 0.0
for i = 1:num_trials
    tic()
    forward_bench()
    forward_time += toq()
    tic()
    backward_bench()
    backward_time += toq()
end
println("Avg forward time for $num_trials runs: $(forward_time / num_trials * 1000.0)ms")
println("Avg backward time for $num_trials runs: $(backward_time / num_trials * 1000.0)ms")

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

input   = data_value
filters = get_buffer(net, :conv1_1weights)
bias    = get_buffer(net, :conv1_1bias)
actual  = get_buffer(net, :conv1_1value)
fill!(actual, 0.0f0)
expected = zeros(actual)
forward_bench()
convolution_forward(filters, bias, input, expected, 1, 1, 3)
ϵ = 1e-4
println("Checking correctness")
if !all(-ϵ .< expected - actual .< ϵ)
    throw("Error: ")
end
println("Done")
