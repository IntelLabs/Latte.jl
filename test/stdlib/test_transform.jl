# Copyright (c) 2015 Intel Corporation. All rights reserved.
using Latte
net = Net(2)
data,  data_value   = MemoryDataLayer(net, :data, (4, 4, 2))
transform = TransformLayer(net, :transform, data; crop=(2, 2))
rand!(data_value)
init(net)
forward(net)
println(data_value)
println(get_buffer(net, :transformvalue))
