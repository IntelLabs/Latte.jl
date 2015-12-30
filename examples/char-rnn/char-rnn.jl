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

function load_data()
    vocab = Dict()
    open("input.txt", "r") do text
        words = collect(readall(text))
        dataset = Array(Int32, length(words))
        for (i, word) in enumerate(words)
            if !haskey(vocab, word)
                vocab[word] = length(keys(vocab)) + 1
            end
            dataset[i] = vocab[word]
        end
        println("corpus length : ", length(words))
        println("vocab size    : ", length(keys(vocab)))
        return dataset, words, vocab
    end
end

dataset, words, vocab = load_data()

n_units = 128
n_vocab = length(keys(vocab))
batch_size = 50
bprop_len = 50

net = Net(batch_size; time_steps=50)
data, data_value = MemoryDataLayer(net, :data, (1,))
label, label_value = MemoryDataLayer(net, :label, (1,))
embed = EmbedIDLayer(:embed, net, data, n_vocab, n_units)
gru1 = GRULayer(:gru1, net, embed, n_units)
gru2 = GRULayer(:gru2, net, gru1, n_units)
fc1 = InnerProductLayer(:fc1, net, gru2, n_vocab)
loss  = SoftmaxLossLayer(:loss, net, fc1, label)

init(net)

whole_len = length(dataset)
jump = div(whole_len, batch_size)
n_epochs = 100

params = SolverParameters(
    LRPolicy.Inv(0.01, 0.0001, 0.75),
    MomPolicy.Fixed(0.9),
    100000,
    .0005,
    1000)
sgd = SGD(params)
solver = sgd
accum_loss = 0.0
for i in 1:n_epochs
    loss = 0.0f0
    for t = 1:50
        for j in 1:batch_size
            data_value[j] = dataset[(jump*(j-1) + (i-1) * 50 + t - 1) % whole_len + 1]
            label_value[j] = dataset[(jump*(j-1) + (i-1) * 50 + t) % whole_len + 1]
        end
        forward(net; t=t)
        loss += get_buffer(net, :lossvalue, t)[1]
    end
    info("Iter $i - Loss: $(loss / 50.0f0)")
    backward(net)
    solver.state.learning_rate = get_learning_rate(solver.params.lr_policy, solver.state)
    solver.state.momentum = get_momentum(solver.params.mom_policy, solver.state)
    update(solver, net)
    clear_values(net)
    clear_∇(net)
end
