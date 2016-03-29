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

using FactCheck
using Latte

net = Net(8)
data,  data_value   = MemoryDataLayer(net, :data, (227, 227, 3))
label, label_value = MemoryDataLayer(net, :label, (1,))
data_value[:]  = rand(Float32, size(data_value)...) * 256
label_value[:] = map(floor, rand(Float32, size(label_value)...) * 10)
fc1         = InnerProductLayer(:fc1, net, data, 10)
softmax     = SoftmaxLossLayer(:softmax, net, fc1, label)

init(net)

params = SolverParameters(
    lr_policy    = LRPolicy.Decay(.01f0, 5.0f-7),
    mom_policy   = MomPolicy.Fixed(0.9),
    max_epoch    = 300,
    regu_coef    = .0005)
sgd = SGD(params)

input = get_buffer(net, :fc1value)
label = get_buffer(net, :labelvalue)

expected_grad = zeros(Float32, size(input))

epsilon = 1e-5
forward(net; solver=sgd)
batch_size = size(input)[end]
pred = zeros(input)
expected_loss = 0
for n in 1:batch_size
    pred[:,n] = exp(input[:,n] .- maximum(input[:,n]))
    pred[:,n] /= sum(pred[:,n])
    label_value = round(Int, label[1, n]) + 1
    expected_loss -= log(max(pred[label_value, n], eps(Float32)))
end
expected_loss /= batch_size

facts("Testing SoftmaxLoss layer") do
    context("Forward") do
        @fact get_buffer(net, :softmaxvalue)[1] --> roughly(expected_loss)
    end
    context("Backward") do
        backward(net)
        expected_grad = copy(pred)
        for n in 1:batch_size
            label_value = round(Int, label[1, n]) + 1
            expected_grad[label_value, n] -= 1
        end
        expected_grad /= batch_size
        @fact get_buffer(net, :fc1∇) --> roughly(expected_grad)
    end
end

FactCheck.exitstatus()
