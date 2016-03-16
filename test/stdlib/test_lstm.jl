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

facts("Testing LSTM layer") do
    n_time_steps = 5
    net = Net(2; time_steps=n_time_steps)
    data, data_value = MemoryDataLayer(net, :data, (8,))
    rand!(data_value)

    lstm = LSTMLayer(:lstm, net, data, length(data))
    init(net)
    forward(net; phase=Latte.Test)

    W_ih = net.params[1].value
    W_ix = net.params[3].value

    W_C_simh = net.params[5].value
    W_C_simx = net.params[7].value

    W_fh = net.params[9].value
    W_fx = net.params[11].value

    W_oh = net.params[13].value
    W_ox = net.params[15].value

    V = net.params[17].value

    h = zeros(Float32, 8, 2)
    C = zeros(Float32, 8, 2)
    sigmoid(x)  = 1.0f0 ./ (1.0f0 .+ exp(-x))
    for t = 1:n_time_steps
        x = get_buffer(net, :datavalue, t)
        ih = W_ih' * h
        ix = W_ix' * x
        @fact ih --> roughly(get_buffer(net, symbol(net.ensembles[2].name, :value), t))
        @fact ix --> roughly(get_buffer(net, symbol(net.ensembles[3].name, :value), t))

        C_simh = W_C_simh' * h
        C_simx = W_C_simx' * x
        @fact C_simh --> roughly(get_buffer(net, symbol(net.ensembles[4].name, :value), t))
        @fact C_simx --> roughly(get_buffer(net, symbol(net.ensembles[5].name, :value), t))

        fh = W_fh' * h
        fx = W_fx' * x
        @fact fh --> roughly(get_buffer(net, symbol(net.ensembles[6].name, :value), t))
        @fact fx --> roughly(get_buffer(net, symbol(net.ensembles[7].name, :value), t))

        oh = W_oh' * h
        ox = W_ox' * x
        @fact oh --> roughly(get_buffer(net, symbol(net.ensembles[8].name, :value), t))
        @fact ox --> roughly(get_buffer(net, symbol(net.ensembles[9].name, :value), t))

        i = sigmoid(ih .+ ix)
        @fact i --> roughly(get_buffer(net, symbol(net.ensembles[11].name, :value), t))

        C_sim = tanh(C_simh .+ C_simx)
        @fact C_sim --> roughly(get_buffer(net, symbol(net.ensembles[13].name, :value), t))

        f = sigmoid(fh .+ fx)
        @fact f --> roughly(get_buffer(net, symbol(net.ensembles[15].name, :value), t))

        f_C = f .* C
        @fact f_C --> roughly(get_buffer(net, symbol(net.ensembles[16].name, :value), t))

        i_C = i .* C_sim
        @fact i_C --> roughly(get_buffer(net, symbol(net.ensembles[17].name, :value), t))

        C = i_C .+ f_C
        @fact C --> roughly(get_buffer(net, symbol(net.ensembles[18].name, :value), t))
        o = sigmoid(V' * C .+ oh .+ ox)
        h = o .* tanh(C)
        h_actual = get_buffer(net, symbol(net.ensembles[end].name, :value), t)
        @fact h_actual --> roughly(h)
    end
end

FactCheck.exitstatus()
