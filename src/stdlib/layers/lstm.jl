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

export LSTMLayer

function LSTMLayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble, n_outputs::Int)
    @assert ndims(input_ensemble) == 1

    for ens in [:i, :C_sim, :f, :o]
        @eval $(symbol(ens, :h)) = FullyConnectedEnsemble($net, length($input_ensemble), $n_outputs)
        @eval $(symbol(ens, :x)) = InnerProductLayer($net, $input_ensemble, $n_outputs)
    end

    # i = σ(ih + ix)
    i = σ(net, +(net, ih, ix))
    # C_sim = tanh(C_simh + C_simx)
    C_sim = tanh(net, +(net, C_simh, C_simx))
    # f = σ(fh + fx)
    f = σ(net, +(net, fh, fx))

    # C_t = i * C_sim + f * C_{t-1}
    f_C = MulEnsemble(net, (n_outputs, ))
    add_connections(net, f, f_C, (i) -> (i,))
    C = +(net, *(net, i, C_sim), f_C)
    add_connections(net, C, f_C, (i) -> (i,); recurrent=true)

    # oC = V * C
    oC = InnerProductLayer(net, C, n_outputs)

    # o = sigmoid(oh + ox + oC)
    o = σ(net, +(net, oC, oh, ox))

    # h = o * tanh(C)
    h = *(net, o, tanh(net, C; copy=true))

    # Connect h back to each gate
    for ens in [ih, C_simh, fh, oh]
        add_connections(net, h, ens, (i) -> (1:n_outputs, ); recurrent=true)
    end
end
