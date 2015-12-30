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
using Latte
using HDF5

batch_size = 128

net = Net(batch_size)
data, data_value = MemoryDataLayer(net, :data, (224, 224, 3))

conv1 = ConvolutionLayer( :conv1, net, data,  96,  11, 4, 0)
relu1 = ReLULayer(        :relu1, net, conv1)
pool1 = MaxPoolingLayer(  :pool1, net, relu1, 2,   2,  0)

conv2 = ConvolutionLayer( :conv2, net, pool1, 256, 5,  1, 2)
relu2 = ReLULayer(        :relu2, net, conv2)
pool2 = MaxPoolingLayer(  :pool2, net, relu2, 2,   2,  0)

conv3 = ConvolutionLayer( :conv3, net, pool2, 512, 3,  1, 1)
relu3 = ReLULayer(        :relu3, net, conv3)
conv4 = ConvolutionLayer( :conv4, net, relu3, 1024, 3,  1, 1)
relu4 = ReLULayer(        :relu4, net, conv4)
conv5 = ConvolutionLayer( :conv5, net, relu4, 1024, 3,  1, 1)
relu5 = ReLULayer(        :relu5, net, conv5)
pool5 = MaxPoolingLayer(  :pool5, net, relu5, 2,   2,  0)

fc6     = InnerProductLayer(:fc6,     net, pool5,   3072)
fc7     = InnerProductLayer(:fc7,     net, fc6,   4096)
fc8     = InnerProductLayer(:fc8,     net, fc7,   1000)

# loss     = SoftmaxLossLayer(:loss, net, fc8, label)
# accuracy = AccuracyLayer(:accuracy, net, fc8, label)

init(net)

forward(net)
backward(net)

forward_task = net.forward_tasks[Latte.Train][end]
function forward_bench()
    args = []
    for arg in forward_task.args
        if isa(arg, Symbol)
            push!(args, net.buffers[arg])
        else
            push!(args, arg)
        end
    end
    forward_task.func(args...)
end

backward_task = net.backward_tasks[Latte.Train][end]
function backward_bench()
    args = []
    for arg in backward_task.args
        if isa(arg, Symbol)
            push!(args, net.buffers[arg])
        else
            push!(args, arg)
        end
    end
    backward_task.func(args...)
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
