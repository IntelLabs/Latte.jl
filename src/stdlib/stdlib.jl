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

include("neuron.jl")

# Common initialization functions
export gaussian
function gaussian(;mean=0.0, std=1.0)
    function gaussian_init(eltype::Type, dims...)
        rand(eltype, dims...) * std + mean
    end
end

export xavier, xavier_init
function xavier_init(eltype::Type, dims...)
    fan_in = prod(dims[1:end-1])
    scale = sqrt(3.0 / fan_in)
    rand(eltype, dims...) * 2scale - scale
end

xavier = xavier_init

include("layers/convolution.jl")
include("layers/pooling.jl")
include("layers/relu.jl")
include("layers/inner-product.jl")
include("layers/softmax-loss.jl")
include("layers/accuracy.jl")
include("layers/hdf5-data.jl")
include("layers/memory-data.jl")
include("layers/dropout.jl")
include("layers/math.jl")
include("layers/lstm.jl")
include("layers/embed_id.jl")
include("layers/GRU.jl")
include("layers/transform.jl")
include("layers/concat.jl")
include("layers/rnn.jl")
include("layers/tanh.jl")
include("layers/reshape.jl")

