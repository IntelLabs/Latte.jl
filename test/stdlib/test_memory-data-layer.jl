# Copyright (c) 2015 Intel Corporation. All rights reserved.
using Latte
using FactCheck

net = Net(8)
data_layer, data_value = MemoryDataLayer(net, :data, (227, 227, 3))

init(net)

facts("Testing Memory Data Layer") do
    rand!(data_value)
    forward(net)
    @fact data_value --> get_buffer(net, :datavalue)

    rand!(data_value)
    forward(net)
    @fact data_value --> get_buffer(net, :datavalue)
end

FactCheck.exitstatus()
