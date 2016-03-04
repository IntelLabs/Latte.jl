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

#=
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
        @pending h_input --> get_buffer(net, :lstm1hinputs1, t)
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

        @pending c_expected --> roughly(get_buffer(net, :lstm1state, t))
        @fact h_expected --> roughly(get_buffer(net, :lstm1value, t))
    end
end
=#
