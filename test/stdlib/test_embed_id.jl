# Copyright (c) 2015 Intel Corporation. All rights reserved.
using Latte
using FactCheck

facts("Testing embed_id layer") do
    net = Net(8)
    data, data_value = MemoryDataLayer(net, :data, (1,))
    embed = EmbedIDLayer(:embed, net, data, 100, 128)
    for i in 1:8
        data_value[1,i] = i
    end
    init(net)
    forward(net)
    expected = get_buffer(net, :embedweights)
    actual = get_buffer(net, :embedvalue)
    for i in 1:8
        @fact actual[:, i][:] --> expected[i, :][:]
    end
    rand!(net.buffers[:embed∇])
    backward(net)
    expected = get_buffer(net, :embed∇)
    actual = get_buffer(net, :embed∇weights)
    for i in 1:8
        @fact actual[i, :][:] --> expected[:, i][:]
    end
end
