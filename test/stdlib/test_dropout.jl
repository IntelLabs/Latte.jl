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
fc1 = InnerProductLayer(:fc1, net, data, 20)
ratio = 0.5f0
drop = DropoutLayer(:drop, net, fc1, ratio)

init(net)
params = SolverParameters(
    LRPolicy.Inv(0.01, 0.0001, 0.75),
    MomPolicy.Fixed(0.9),
    100000,
    .0005,
    100)
sgd = SGD(params)

input = get_buffer(net, :fc1value)

facts("Testing Dropout Layer") do
    context("Forward") do
        forward(net; solver=sgd)
        randvals = get_buffer(net, :droprandval)
        value = get_buffer(net, :dropvalue)
        to_check = []
        for i in 1:length(randvals)
            if randvals[i] < ratio
                push!(to_check, value[i])
            end
        end
        @fact all(to_check .== 0.0f0) --> true
    end
    context("Backward") do
        top_diff = get_buffer(net, :drop∇)
        rand!(top_diff)
        randvals = get_buffer(net, :droprandval)
        expected = (top_diff .* (randvals .> ratio)) .* (1.0 / ratio)
        backward(net)

        @fact expected --> roughly(get_buffer(net, :fc1∇))
    end
end

FactCheck.exitstatus()
