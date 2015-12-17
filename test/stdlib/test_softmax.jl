# Copyright (c) 2015 Intel Corporation. All rights reserved.
using FactCheck
using Latte

net = Net(8)
data,  data_value   = MemoryDataLayer(net, :data, (227, 227, 3))
label, label_value = MemoryDataLayer(net, :label, (1,))
data_value[:]  = rand(Float32, size(data_value)...) * 256
label_value[:] = map(floor, rand(Float32, size(label_value)...) * 10)
fc1         = InnerProductLayer(:fc1, net, data, 10)
softmax     = SoftmaxLossLayer(:softmax, net, fc1, label)

init(net)

input = net.buffers[:fc1value]
label = net.buffers[:labelvalue]

expected_grad = zeros(Float32, size(input))

epsilon = 1e-5
forward(net)
batch_size = size(input)[end]
pred = zeros(input)
expected_loss = 0
for n in 1:batch_size
    pred[:,n] = exp(input[:,n] .- maximum(input[:,n]))
    pred[:,n] /= sum(pred[:,n])
    label_value = round(Int, label[1, n]) + 1
    expected_loss -= log(max(pred[label_value, n], eps(Float32)))
end
expected_loss /= batch_size

facts("Testing SoftmaxLoss layer") do
    context("Forward") do
        @fact net.buffers[:softmaxvalue][1] --> roughly(expected_loss)
    end
    context("Backward") do
        backward(net)
        expected_grad = copy(pred)
        for n in 1:batch_size
            label_value = round(Int, label[1, n]) + 1
            expected_grad[label_value, n] -= 1
        end
        expected_grad /= batch_size
        @fact net.buffers[:fc1∇] --> roughly(expected_grad)
    end
end

FactCheck.exitstatus()
