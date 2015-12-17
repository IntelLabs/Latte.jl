# Copyright (c) 2015 Intel Corporation. All rights reserved.
using FactCheck
using Latte

net = Net(8)
data,  data_value   = MemoryDataLayer(net, :data, (227, 227, 3))
label, label_value = MemoryDataLayer(net, :label, (1,))
data_value[:]  = rand(Float32, size(data_value)...) * 256
label_value[:] = map(floor, rand(Float32, size(label_value)...) * 10)
fc1         = InnerProductLayer(:fc1, net, data, 20)
ratio = 0.5f0
drop        = DropoutLayer(:drop, net, fc1, ratio)
# loss        = SoftmaxLossLayer(:loss, net, drop, label)

init(net)


input   = net.buffers[:fc1value]
facts("Testing Dropout Layer") do
    randvals = net.buffers[:droprandval]
    rand!(randvals)
    context("Forward") do
        forward(net)
        to_check = []
        for i in 1:length(randvals)
            if randvals[i] <= ratio
                push!(to_check, net.buffers[:dropvalue][i])
            end
        end
        @fact all(to_check .== 0.0f0) --> true
    end
    context("Backward") do
        top_diff = net.buffers[:drop∇]
        rand!(top_diff)
        expected = (top_diff .* (randvals .> ratio)) .* (1.0 / ratio)
        backward(net)

        @fact expected --> roughly(net.buffers[:fc1∇])
    end
end

FactCheck.exitstatus()
