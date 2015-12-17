# Copyright (c) 2015 Intel Corporation. All rights reserved.
using FactCheck
using Latte

net = Net(8)
data,  data_value   = MemoryDataLayer(net, :data, (227, 227, 3))
label, label_value = MemoryDataLayer(net, :label, (1,))
data_value[:]  = rand(Float32, size(data_value)...) * 256
label_value[:] = map(floor, rand(Float32, size(label_value)...) * 10)
conv1       = ConvolutionLayer(:conv1, net, data, 10, 3, 1, 1)
relu1       = ReLULayer(:relu1, net, conv1)
fc1         = InnerProductLayer(:fc1, net, relu1, 10)
loss        = SoftmaxLossLayer(:loss, net, fc1, label)

init(net)


ϵ = 1e-5
input   = net.buffers[:conv1value]
facts("Testing ReLU Layer") do
    context("Forward") do
        forward(net)

        expected = map((x) -> x > 0.0f0 ? x : 0.0f0, input)
        @fact expected --> roughly(net.buffers[:relu1value])
    end
    context("Backward") do
        backward(net)
        top_diff = net.buffers[:relu1∇]
        expected = map((x, y) -> x > 0.0f0 ? y : 0.0f0, input, top_diff)
        @fact expected --> roughly(net.buffers[:conv1∇])
    end
end

FactCheck.exitstatus()
