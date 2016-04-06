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

include("fusion.jl")
include("test_snapshot.jl")
include("transforms/test_fixed_dims.jl")
include("transforms/test_neuron.jl")
include("transforms/test_tree_cleaners.jl")
include("stdlib/test_accuracy.jl")
include("stdlib/test_concat.jl")
include("stdlib/test_convolution.jl")
include("stdlib/test_dropout.jl")
include("stdlib/test_embed_id.jl")
include("stdlib/test_gru.jl")
include("stdlib/test_hdf5-data.jl")
include("stdlib/test_inner-product.jl")
include("stdlib/test_lstm.jl")
include("stdlib/test_math.jl")
include("stdlib/test_memory-data-layer.jl")
include("stdlib/test_pooling.jl")
include("stdlib/test_relu.jl")
include("stdlib/test_reshape.jl")
include("stdlib/test_rnn.jl")
include("stdlib/test_softmax.jl")
include("stdlib/test_tanh.jl")
include("stdlib/test_transform.jl")

FactCheck.exitstatus()
