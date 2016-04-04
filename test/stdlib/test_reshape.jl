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
rand!(data_value)
fc1 = InnerProductLayer(:fc1, net, data, 20)
reshape = ReshapeLayer(:reshape, net, fc1, (10, 2))

init(net)
facts("Testing Reshape Layer") do
    @fact size(get_buffer(net, :reshapevalue)) --> (10, 2, 8)
    @fact size(get_buffer(net, :reshape∇)) --> (10, 2, 8)

    forward(net; phase=Latte.Test)
    @fact get_buffer(net, :reshapevalue)[:] --> get_buffer(net, :fc1value)[:]

    rand!(get_buffer(net, :reshape∇))
    backward(net)
    @fact get_buffer(net, :reshape∇)[:] --> get_buffer(net, :fc1∇)[:]
end

FactCheck.exitstatus()
