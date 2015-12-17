# Copyright (c) 2015 Intel Corporation. All rights reserved.
using Latte
using FactCheck

facts("Testing LSTM layer") do
    net = RNN(8, 5)
    data, data_value   = MemoryDataLayer(net, :data, (2,))
    rand!(data_value)
    lstm1 = LSTMLayer(:lstm1, net, data)

    init(net)
    forward(net)
    for t in 2:2
        c_prev = get_buffer(net, :lstm1state, t-1)[:,:]
        input = data_value
        y_weights = get_buffer(net, :lstm1yweights, t)
        y = input' * y_weights
        h_input = get_buffer(net, :lstm1value, t-1)
        # Ensure recurrent connection is correct
        @fact h_input --> get_buffer(net, :lstm1hinputs1, t)
        h_weights = get_buffer(net, :lstm1hweights, t)
        h = h_input' * h_weights 
        _sum = y .+ h
        a = _sum[[1, 5], :]
        i = _sum[[2, 6], :]
        f = _sum[[3, 7], :]
        o = _sum[[4, 8], :]

        sigmoid(x)  = 1.0f0 ./ (1.0f0 .+ exp(-x))
        ∇sigmoid(x) = 1.0f0 .- x .* x
        ∇tanh(x)    = x .* (1.0f0 .- x)
        c_expected = sigmoid(i) .* tanh(a) .+ sigmoid(f) .* c_prev
        h_expected = sigmoid(o) .* tanh(c_expected)

        @fact c_expected --> roughly(get_buffer(net, :lstm1state, t))
        @fact h_expected --> roughly(get_buffer(net, :lstm1value, t))
    end
end
