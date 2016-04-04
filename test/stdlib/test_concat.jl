# Copyright (c) 2015, Intel Corporation
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

using FactCheck
using Latte

net = Net(8)
data, data_value = MemoryDataLayer(net, :data, (10,))
fc1 = InnerProductLayer(:fc1, net, data, 20)
fc2 = InnerProductLayer(:fc2, net, fc1, 20)
concat1 = ConcatLayer(:concat1, net, fc1, fc2)

init(net)
facts("Testing Concat Layer") do
    context("Forward") do
        forward(net; phase=Latte.Test)
        fc1value = get_buffer(net, :fc1value)
        fc2value = get_buffer(net, :fc2value)
        inner_size = [size(fc1value)[1:end-1]...]
        inner_size[end] *= 2
        expected = zeros(Float32, inner_size..., size(fc1value)[end])
        for n = 1:8
            expected[1:20, n] = fc1value[:, n]
            expected[21:40, n] = fc2value[:, n]
        end
        @fact get_buffer(net, :concat1value) --> expected
    end
    context("Backward") do
        concat∇ = get_buffer(net, :concat1∇)
        rand!(concat∇)
        fc1∇ = get_buffer(net, :fc1∇)
        fc2∇ = get_buffer(net, :fc2∇)
        backward(net)
        for n = 1:8
            @fact fc1∇[:, n] --> concat∇[1:20, n]
            @fact fc2∇[:, n] --> concat∇[21:40, n]
        end
    end
end

FactCheck.exitstatus()
