# Copyright (c) 2015 Intel Corporation. All rights reserved.
using FactCheck
using Latte

net = Net(8)
data,  data_value   = MemoryDataLayer(net, :data, (227, 227, 3))
label, label_value = MemoryDataLayer(net, :label, (1,))
data_value[:]  = rand(Float32, size(data_value)...) * 256
label_value[:] = map(floor, rand(Float32, size(label_value)...) * 10)
fc1         = InnerProductLayer(:fc1, net, data, 10)
acc         = AccuracyLayer(:accuracy, net, fc1, label)

init(net)

input = get_buffer(net, :fc1value)
label = get_buffer(net, :labelvalue)

expected_accuracy = 0.0

forward(net, phase=Latte.Test)
batch_size = size(input)[end]
for n in 1:batch_size
    if indmax(input[:, n]) == round(Int64, label[1, n]) + 1
        expected_accuracy += 1
    end
end
expected_accuracy /= batch_size

facts("Testing AccuracyLayer layer") do
    context("Forward") do
        @fact get_buffer(net, :accuracyvalue)[1] --> roughly(expected_accuracy)
    end
end

FactCheck.exitstatus()
