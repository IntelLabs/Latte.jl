# Copyright (c) 2015 Intel Corporation. All rights reserved.
using Latte
using FactCheck

facts("Testing GRU layer") do
    net = Net(8; time_steps=5)
    data, data_value   = MemoryDataLayer(net, :data, (2,))
    inputs = rand(Float32, size(data_value)..., 8, 5)
    gru1 = GRULayer(:gru1, net, data, 2)
    init(net)
    for t = 1:5
        data_value[:,:] = inputs[:,:,t]
        forward(net; t=t)
    end
    for t in 2:5
        W_z = get_buffer(net, :gru1W_z, 1)
        U_z = get_buffer(net, :gru1U_z, 1)
        b_z = get_buffer(net, :gru1b_z, 1)

        W_r = get_buffer(net, :gru1W_r, 1)
        U_r = get_buffer(net, :gru1U_r, 1)
        b_r = get_buffer(net, :gru1b_r, 1)

        W_h = get_buffer(net, :gru1W_h, 1)
        U_h = get_buffer(net, :gru1U_h, 1)
        b_h = get_buffer(net, :gru1b_h, 1)

        x_z = W_z' * inputs[:,:,t] .+ b_z[:]
        x_r = W_r' * inputs[:,:,t] .+ b_r[:]
        x_h = W_h' * inputs[:,:,t] .+ b_h[:]

        h_tm1 = get_buffer(net, :gru1value, t-1)
        @fact h_tm1 --> get_buffer(net, :gru1inputs2, t)
        sigmoid(x) = 1.0f0 ./ (1.0f0 .+ exp(-x))
        z = sigmoid(x_z .+ (U_z' * h_tm1))
        r = sigmoid(x_r .+ (U_r' * h_tm1))
        hh = tanh(x_h .+ (U_h' * (r .* h_tm1)))
        h = z .* h_tm1 + (1-z) .* hh
        eps = 1e-1
        # @fact all(-eps .< get_buffer(net, :gru1value, t) - h .< eps) --> true
        @fact get_buffer(net, :gru1value, t) --> roughly(h, atol=eps)
    end
    backward(net)
end
