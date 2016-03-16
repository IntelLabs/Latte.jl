#=
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=#

using Latte
using FactCheck

facts("Testing neuron trasnform") do
    context("Testing simple") do
        net = Net(8)
        neurons = [DataNeuron(0.0) for _ = 1:10, _ = 1:10]
        ens = Ensemble(net, :test_ens, neurons, [])
        ens2 = Ensemble(net, :test_ens2, neurons, [])
        add_connections(net, ens2, ens, (i, j) -> (i:i+1, j:j+1); padding=1)
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

FactCheck.exitstatus()
