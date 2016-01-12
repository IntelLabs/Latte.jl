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

export GRULayer
sigmoid(x)  = 1.0f0 / (1.0f0 + exp(-x))
∇sigmoid(x) = 1.0f0 - x * x
∇tanh(x)    = x * (1.0f0 - x)

@neuron type GRUNeuron
    W_z  :: DenseArray{Float32}
    ∇W_z :: DenseArray{Float32}
    U_z  :: DenseArray{Float32}
    ∇U_z :: DenseArray{Float32}
    b_z  :: DenseArray{Float32}
    ∇b_z :: DenseArray{Float32}

    W_r  :: DenseArray{Float32}
    ∇W_r :: DenseArray{Float32}
    U_r  :: DenseArray{Float32}
    ∇U_r :: DenseArray{Float32}
    b_r  :: DenseArray{Float32}
    ∇b_r :: DenseArray{Float32}

    W_h  :: DenseArray{Float32}
    ∇W_h :: DenseArray{Float32}
    U_h  :: DenseArray{Float32}
    ∇U_h :: DenseArray{Float32}
    b_h  :: DenseArray{Float32}
    ∇b_h :: DenseArray{Float32}

    r    :: Batch{Float32}
    z    :: Batch{Float32}
    hh   :: Batch{Float32}
end
function GRUNeuron(
        W_z  :: DenseArray{Float32},
        ∇W_z :: DenseArray{Float32},
        U_z  :: DenseArray{Float32},
        ∇U_z :: DenseArray{Float32},
        b_z  :: DenseArray{Float32},
        ∇b_z :: DenseArray{Float32},

        W_r  :: DenseArray{Float32},
        ∇W_r :: DenseArray{Float32},
        U_r  :: DenseArray{Float32},
        ∇U_r :: DenseArray{Float32},
        b_r  :: DenseArray{Float32},
        ∇b_r :: DenseArray{Float32},

        W_h  :: DenseArray{Float32},
        ∇W_h :: DenseArray{Float32},
        U_h  :: DenseArray{Float32},
        ∇U_h :: DenseArray{Float32},
        b_h  :: DenseArray{Float32},
        ∇b_h :: DenseArray{Float32}
    )
    GRUNeuron(
        W_z, ∇W_z, U_z, ∇U_z, b_z, ∇b_z,
        W_r, ∇W_r, U_r, ∇U_r, b_r, ∇b_r,
        W_h, ∇W_h, U_h, ∇U_h, b_h, ∇b_h,
        Batch(0.0f0),
        Batch(0.0f0),
        Batch(0.0f0))
end

@neuron forward(neuron::GRUNeuron) do
    x_z = neuron.b_z[1]
    x_r = neuron.b_r[1]
    x_h = neuron.b_h[1]
    for i in 1:length(neuron.inputs[1])
        x_z += neuron.W_z[i] * neuron.inputs[1][i]
        x_r += neuron.W_r[i] * neuron.inputs[1][i]
        x_h += neuron.W_h[i] * neuron.inputs[1][i]
    end

    u_z = 0.0f0
    u_r = 0.0f0
    for i = 1:length(neuron.inputs[2])
        u_z += neuron.U_z[i] * neuron.inputs[2][i]
        u_r += neuron.U_r[i] * neuron.inputs[2][i]
    end

    neuron.z = sigmoid(x_z + u_z)
    neuron.r = sigmoid(x_r + u_r)
    u_h = 0.0f0
    for i = 1:length(neuron.inputs[2])
        u_h += neuron.U_h[i] * neuron.r * neuron.inputs[2][i]
    end
    neuron.hh = tanh(x_h + u_h)
    neuron.value = neuron.z * neuron.inputs[2][neuron.index] + (1 - neuron.z) * neuron.hh
end

@neuron backward(neuron::GRUNeuron) do
    ∇_z = neuron.∇ * neuron.hh
    ∇hh = neuron.∇ * (1 - neuron.z)
    ∇z  = ∇sigmoid(neuron.∇ * neuron.inputs[2][neuron.index])
    neuron.∇inputs[2][neuron.index] = neuron.∇ * neuron.z
    ∇h = ∇tanh(∇hh)
    ∇r = 0.0f0
    for i = 1:length(neuron.inputs[2])
        neuron.∇U_h[i] += ∇h * neuron.r * neuron.inputs[2][i]
        neuron.∇inputs[2][i] += ∇h * neuron.r * neuron.U_h[i]
        ∇r += ∇h * neuron.U_h[i] * neuron.inputs[2][i]
    end
    ∇r = ∇sigmoid(∇r)
    for i = 1:length(neuron.inputs[2])
        neuron.∇U_z[i] += ∇z * neuron.inputs[2][i]
        neuron.∇inputs[2][i] += ∇z * neuron.U_z[i]

        neuron.∇U_r[i] += ∇r * neuron.inputs[2][i]
        neuron.∇inputs[2][i] += ∇r * neuron.U_r[i]
    end
    for i = 1:length(neuron.inputs[2])
        neuron.∇W_z[i] += ∇z * neuron.inputs[1][i]
        neuron.∇inputs[1][i] += ∇z * neuron.W_z[i]

        neuron.∇W_r[i] += ∇r * neuron.inputs[1][i]
        neuron.∇inputs[2][i] += ∇r * neuron.W_r[i]

        neuron.∇W_h[i] += ∇h * neuron.inputs[1][i]
        neuron.∇inputs[2][i] += ∇h * neuron.W_h[i]
    end
    neuron.∇b_h[1] = ∇h
    neuron.∇b_r[1] = ∇r
    neuron.∇b_z[1] = ∇z
end

function GRULayer(name::Symbol, net::Net, input_ensemble::AbstractEnsemble, num_outputs::Int)
    weight_init = xavier
    in_out_shape = (length(input_ensemble), num_outputs)
    out_out_shape = (num_outputs, num_outputs)

    W_z = weight_init(Float32, in_out_shape...)
    ∇W_z = zeros(Float32, in_out_shape...)

    U_z = weight_init(Float32, out_out_shape...)
    ∇U_z = zeros(Float32, out_out_shape...)

    b_z = zeros(Float32, 1, num_outputs)
    ∇b_z = zeros(Float32, 1, num_outputs)

    W_r = weight_init(Float32, in_out_shape...)
    ∇W_r = zeros(Float32, in_out_shape...)

    U_r = weight_init(Float32, out_out_shape...)
    ∇U_r = zeros(Float32, out_out_shape...)

    b_r = zeros(Float32, 1, num_outputs)
    ∇b_r = zeros(Float32, 1, num_outputs)

    W_h = weight_init(Float32, in_out_shape...)
    ∇W_h = zeros(Float32, in_out_shape...)

    U_h = weight_init(Float32, out_out_shape...)
    ∇U_h = zeros(Float32, out_out_shape...)

    b_h = zeros(Float32, 1, num_outputs)
    ∇b_h = zeros(Float32, 1, num_outputs)

    neurons = Array(GRUNeuron, num_outputs)
    for i in 1:num_outputs
        neurons[i] = GRUNeuron(
            view(W_z, :, i), view(∇W_z, :, i),
            view(U_z, :, i), view(∇U_z, :, i),
            view(b_z, :, i), view(∇b_z, :, i),

            view(W_r, :, i), view(∇W_r, :, i),
            view(U_r, :, i), view(∇U_r, :, i),
            view(b_r, :, i), view(∇b_r, :, i),

            view(W_h, :, i), view(∇W_h, :, i),
            view(U_h, :, i), view(∇U_h, :, i),
            view(b_h, :, i), view(∇b_h, :, i),
        )
    end

    ens = Ensemble(net, name, neurons, [Param(net, name,:W_z, 1.0), 
                                        Param(net, name,:U_z, 1.0), 
                                        Param(net, name,:b_z, 2.0), 
                                        Param(net, name,:W_r, 1.0), 
                                        Param(net, name,:U_r, 1.0), 
                                        Param(net, name,:b_r, 2.0), 
                                        Param(net, name,:W_h, 1.0), 
                                        Param(net, name,:U_h, 1.0), 
                                        Param(net, name,:b_h, 2.0)])
    add_connections(net, input_ensemble, ens,
                    (i) -> (tuple([Colon() for d in size(input_ensemble)]... )))
    add_connections(net, ens, ens,
                    (i) -> (tuple([Colon() for d in size(input_ensemble)]... ));
                    recurrent=true)
    ens

end
