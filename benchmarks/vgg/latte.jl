# Copyright (c) 2015 Intel Corporation. All rights reserved.
using Latte
using HDF5

batch_size = 64

net = Net(batch_size)
data, data_value = MemoryDataLayer(net, :data, (224, 224, 3))

conv1_1 = ConvolutionLayer( :conv1_1, net, data,    64,  3, 1, 1)
relu1_1 = ReLULayer(        :relu1_1, net, conv1_1)
pool1   = MaxPoolingLayer(  :pool1,   net, relu1_1, 2,   2, 0)

conv2_1 = ConvolutionLayer( :conv2_1, net, pool1,   128, 3, 1, 1)
relu2_1 = ReLULayer(        :relu2_1, net, conv2_1)
pool2   = MaxPoolingLayer(  :pool2,   net, relu2_1, 2,   2, 0)

conv3_1 = ConvolutionLayer( :conv3_1, net, pool2,   256, 3, 1, 1)
relu3_1 = ReLULayer(        :relu3_1, net, conv3_1)
conv3_2 = ConvolutionLayer( :conv3_2, net, relu3_1, 256, 3, 1, 1)
relu3_2 = ReLULayer(        :relu3_2, net, conv3_2)
pool3   = MaxPoolingLayer(  :pool3,   net, relu3_2, 2,   2, 0)

conv4_1 = ConvolutionLayer( :conv4_1, net, pool3,   512, 3, 1, 1)
relu4_1 = ReLULayer(        :relu4_1, net, conv4_1)
conv4_2 = ConvolutionLayer( :conv4_2, net, relu4_1, 512, 3, 1, 1)
relu4_2 = ReLULayer(        :relu4_2, net, conv4_2)
pool4   = MaxPoolingLayer(  :pool4,   net, relu4_2, 2,   2, 0)

conv5_1 = ConvolutionLayer( :conv5_1, net, pool4,   512, 3, 1, 1)
relu5_1 = ReLULayer(        :relu5_1, net, conv5_1)
conv5_2 = ConvolutionLayer( :conv5_2, net, relu5_1, 512, 3, 1, 1)
relu5_2 = ReLULayer(        :relu5_2, net, conv5_2)
pool5   = MaxPoolingLayer(  :pool5,   net, relu5_2, 2,   2, 0)

fc6     = InnerProductLayer(:fc6,     net, pool5,   4096)
fc7     = InnerProductLayer(:fc7,     net, fc6,   4096)
fc8     = InnerProductLayer(:fc8,     net, fc7,   1000)

init(net)

forward(net)
backward(net)

forward_task = net.forward_tasks[Latte.Train][end]
forward_args = []
for arg in forward_task.args
    if isa(arg, Symbol)
        push!(forward_args, net.buffers[arg])
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
        push!(backward_args, net.buffers[arg])
    else
        push!(backward_args, arg)
    end
end
function backward_bench()
    backward_task.func(backward_args...)
end

# for i = 1:3
for i = 1:1
    forward_bench()
    backward_bench()
end

num_trials = 2

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
