# Copyright (c) 2015 Intel Corporation. All rights reserved.
using Latte
using FactCheck

facts("Testing neuron trasnform") do
    context("Testing simple") do
        net = Net(8)
        neurons = [DataNeuron(0.0) for _ = 1:10, _ = 1:10]
        ens = Ensemble(net, :test_ens, neurons, [])
        fn = :(function forward(neuron::DataNeuron)
            for i = 1:length(neuron.inputs[1])
                neuron.value += neuron.inputs[1][i]
            end
        end)
        actual, args = transform_neuron_fn(fn, ens)
        @fact args --> Set([:test_ensvalue, :test_ensinputs1])
        @fact remove_line_nodes(actual) --> remove_line_nodes(:(
            function forward(neuron::DataNeuron)
                for i = 1:size(test_ensinputs1,1)
                    test_ensvalue[_neuron_index_1, _neuron_index_2, _neuron_index_3] += 
                        test_ensinputs1[i, _neuron_index_1, _neuron_index_2, _neuron_index_3]
                end
            end
        ))
    end
    context("Testing ActivationEnsemble") do
        net = Net(8)
        neurons = [DataNeuron(0.0) for _ = 1:10, _ = 1:10]
        ens = Ensemble(net, :test_ens, neurons, [])
        neurons = [DataNeuron(0.0) for _ = 1:10, _ = 1:10]
        act_ens = ActivationEnsemble(net, :act_ens, neurons, ens)
        fn = :(function forward(neuron::DataNeuron)
            neuron.value = neuron.inputs[1] + 1
        end)
        actual, args = transform_neuron_fn(fn, act_ens)
        @fact args --> Set([:act_ensvalue])
        @fact remove_line_nodes(actual) --> remove_line_nodes(:(
            function forward(neuron::DataNeuron)
                act_ensvalue[_neuron_index_1, _neuron_index_2, _neuron_index_3] = 
                    act_ensvalue[_neuron_index_1, _neuron_index_2, _neuron_index_3] + 1
            end
        ))
    end
end
