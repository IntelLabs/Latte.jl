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

batch_size = 128
# batch_size = 128

net = Net(batch_size)
data, data_value = MemoryDataLayer(net, :data, (224, 224, 3))

conv1_1 = ConvolutionLayer( :conv1_1, net, data,    64,  3, 1, 1)
relu1_1 = ReLULayer(        :relu1_1, net, conv1_1)
pool1   = MaxPoolingLayer(  :pool1,   net, relu1_1, 2,   2, 0)

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
