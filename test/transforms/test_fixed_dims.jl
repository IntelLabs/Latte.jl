# Copyright (c) 2015 Intel Corporation. All rights reserved.
using Latte
using FactCheck

facts("Testing drop_fixed_dims") do
    ast = :(arr[_neuron_index_3, _neuron_index_2, _neuron_index_1])
    arg_dim_info = Dict(:arr => [true, false, false])
    @fact drop_fixed_dims(ast, arg_dim_info) --> :(arr[_neuron_index_2, _neuron_index_1])

    ast = :(arr[_neuron_index_3, _neuron_index_2, _neuron_index_1])
    arg_dim_info = Dict(:arr => [false, true, true])
    @fact drop_fixed_dims(ast, arg_dim_info) --> :(arr[_neuron_index_3])

    ast = :(arr[_neuron_index_3, _neuron_index_2, _neuron_index_1])
    arg_dim_info = Dict(:arr => [false, false, false])
    @fact drop_fixed_dims(ast, arg_dim_info) --> :(arr[_neuron_index_3, _neuron_index_2, _neuron_index_1])
end

