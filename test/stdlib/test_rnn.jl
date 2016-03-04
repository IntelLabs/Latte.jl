using Latte
using FactCheck

facts("Testing simple RNN layer") do
    n_time_steps = 5
    net = Net(2; time_steps=n_time_steps)
    data, data_value = MemoryDataLayer(net, :data, (8,))
    rand!(data_value)

    rnn = RNNBlock(:rnn, net, data)
    init(net)

    context("Forward") do
        h_weights = get_buffer(net, :rnnstateweights)
        h_bias = get_buffer(net, :rnnstatebias)

        x_weights = get_buffer(net, :rnnxweights)
        x_bias = get_buffer(net, :rnnxbias)

        weights = get_buffer(net, :rnnweights)
        bias = get_buffer(net, :rnnbias)

        forward(net; phase=Latte.Test)
        h_prev = zeros(Float32, size(get_buffer(net, :rnntanhvalue))...)
        for t in 1:n_time_steps
            @fact get_buffer(net, :datavalue, t) --> data_value[:,:,t]

            x_expected = x_weights' * data_value[:, :, t] .+ reshape(x_bias, prod(size(x_bias)))
            @fact get_buffer(net, :rnnxvalue, t) --> roughly(x_expected)

            state_expected = h_weights' * h_prev .+ reshape(h_bias, prod(size(h_bias)))
            @fact get_buffer(net, :rnnstatevalue, t) --> roughly(state_expected)

            # h_expected = state_expected .+ x_expected
            h_expected = tanh(state_expected .+ x_expected)
            @fact get_buffer(net, :rnntanhvalue, t) --> roughly(h_expected)
            h_prev = h_expected

            output_expected = weights' * h_expected .+ reshape(bias, prod(size(bias)))
            @fact get_buffer(net, :rnnvalue, t) --> roughly(output_expected)
        end
    end
end

FactCheck.exitstatus()
