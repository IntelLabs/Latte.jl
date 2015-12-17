# Copyright (c) 2015 Intel Corporation. All rights reserved.
using FactCheck
using Latte

facts("Testing Math Layers") do
    context("Sum") do
        net = Net(8)
        data1, data_value1 = MemoryDataLayer(net, :data1, (227, 227, 3))
        data2, data_value2 = MemoryDataLayer(net, :data2, (227, 227, 3))
        data_value1[:]    = rand(Float32, size(data_value1)...) * 256
        data_value2[:]    = rand(Float32, size(data_value2)...) * 256
        add = AddLayer(:add, net, data1, data2)
        init(net)
        forward(net)
        actual = get_buffer(net, :addvalue)
        expected = data_value1 + data_value2
        @fact actual --> roughly(expected)
    end
end
