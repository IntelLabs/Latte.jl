using Latte, FactCheck

function batch_normalization_forward(x, output, weight, bias, running_mean,
    running_var, momentum, eps; train=true)
    
    x_mean = [mean(x[:,:,:,n]) for n in 1:size(x, 4)]
    x_var = [var(x[:,:,:,n]) for n in 1:size(x, 4)]
    xhat = Array(Float32, size(x)...)
    for n in 1:size(x, 4)
        xhat[:,:,:,n] = x[:,:,:,n] .- x_mean[n] ./ sqrt(x_var[n] + eps)
    end
    running_mean[:] = running_mean .* momentum .+ (1.0f0 - momentum) .* x_mean
    running_var[:] = running_var .* momentum .+ (1.0f0 - momentum) .* x_var
    for n in 1:size(x, 4)
        output[:,:,:,n] = xhat[:,:,:,n] .* weight[n] .+ bias[n]
    end
end

function batch_normalization_backward(x, grad_output, grad_input,
    grad_weight, grad_bias, weight, save_mean, save_std, eps)

    x_sum = sum(x, ndims(x))
    x_var = var(x, ndims(x))
    xhat = (x - x_sum) ./ sqrt(x_var .+ eps)
    grad_weight[:] = sum(xhat .* grad_output, ndims(x))
    grad_bias[:] = sum(grad_output, ndims(x))
    xtmp = (xhat .* grad_weight .+ grad_bias) / float(size(x)[end])
    grad_input[:] = weight .* (grad_output - xtmp) / sqrt(x_var + eps)
end

net = Net(8)

data, data_value = MemoryDataLayer(net, :data, (27, 27, 3))
rand!(data_value)
conv1 = ConvolutionLayer(:conv1, net, data, 10, 3, 1, 1)
eps = 1f-5
momentum = .1f0
bn1 = BatchNormalizationLayer(:bn1, net, conv1, eps, momentum)

init(net)

params = SolverParameters(
    LRPolicy.Inv(0.01, 0.0001, 0.75),
    MomPolicy.Fixed(0.9),
    100000,
    .0005,
    100)
sgd = SGD(params)

facts("Testing batch normalization") do
    context("Forward") do
        forward(net; solver=sgd)
        input = get_buffer(net, :conv1value)
        weight = get_buffer(net, :bn1weight)
        bias = get_buffer(net, :bn1bias)
        running_mean = get_buffer(net, :bn1running_mean)
        running_var = get_buffer(net, :bn1running_var)
        expected_running_mean = zeros(running_mean)
        expected_running_var = zeros(running_var)
        output = get_buffer(net, :bn1value)
        expected_output = zeros(output)
        batch_normalization_forward(input, expected_output, weight, bias,
                                    expected_running_mean,
                                    expected_running_var,
                                    momentum, eps; train=true)
        @fact expected_running_mean --> running_mean
        @fact expected_running_var --> running_var
        #= @fact expected_output --> output =#
    end
end
