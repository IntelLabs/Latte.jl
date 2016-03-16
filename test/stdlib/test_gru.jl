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
        @pending h_tm1 --> get_buffer(net, :gru1inputs2, t)
        sigmoid(x) = 1.0f0 ./ (1.0f0 .+ exp(-x))
        z = sigmoid(x_z .+ (U_z' * h_tm1))
        r = sigmoid(x_r .+ (U_r' * h_tm1))
        hh = tanh(x_h .+ (U_h' * (r .* h_tm1)))
        h = z .* h_tm1 + (1-z) .* hh
        eps = 1e-2
        @pending all(-eps .< get_buffer(net, :gru1value, t) - h .< eps) --> true
    end
    backward(net)
end

FactCheck.exitstatus()
=#
