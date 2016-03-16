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

export TransformLayer
using HDF5

type TransformEnsemble <: JuliaEnsemble
    net           :: Net
    name          :: Symbol
    neurons       :: Array{DataNeuron}
    mean          :: Array{Float32}
    random_mirror :: Bool
    scale         :: Float32
    connections   :: Vector{Connection}
    phase         :: Phase
    net_subgroup  :: Cint
end

function TransformLayer(net::Net, name::Symbol,
                        input_ensemble::AbstractEnsemble; crop=false,
                        random_mirror=false, scale=1.0f0, mean_file=nothing)
    @assert ndims(input_ensemble) == 3
    if crop != false
        neurons = Array(DataNeuron, crop..., size(input_ensemble, 3))
    else
        neurons = Array(DataNeuron, size(input_ensemble)...)
    end
    mean = zeros(Float32, size(input_ensemble)...)
    if mean_file != nothing
        h5open(mean_file, "r") do h5
            mean = h5["mean"]
            if ndims(mean) == 4
                mean = mean[:,:,:,1] .* scale
                mean = reshape(mean, size(mean)[1:3])
            else
                mean = mean[:,:,:] .* scale
            end
            @assert size(mean) == size(input_ensemble)
        end
    end
    for i = 1:length(neurons)
        neurons[i] = DataNeuron(0.0f0)
    end
    ens = TransformEnsemble(net, name, neurons, mean, random_mirror, scale, [], TrainTest, convert(Cint, 1))
    add_ensemble(net, ens)
    add_connections(net, input_ensemble, ens, (i, j, k) -> (i:i, j:j, k:k))
    ens
end

function forward(ens::TransformEnsemble, input::Array, value::Array, phase::Phase)
    if phase == Train
        w_off = abs(rand(Int)) % (size(input,1)-size(value, 1)+1)
        h_off = abs(rand(Int)) % (size(input,2)-size(value, 2)+1)
    else
        w_off = div(size(input,1)-size(input, 1), 2)
        h_off = div(size(input,2)-size(input, 2), 2)
    end
    for i in 1:size(input, 4)
        input[:,:,:,i] -= ens.mean
    end
    crop_w, crop_h, channels, num = size(value)
    if rand(UInt) % 2 == 0 || phase == Test
        for n = 1:num
            for c = 1:channels
                for h = 1:crop_h
                    @simd for w = 1:crop_w
                        @inbounds value[w,h,c,n] = input[w+w_off,h+h_off,c,n] * ens.scale
                    end
                end
            end
        end
    else
        for n = 1:num
            for c = 1:channels
                for h = 1:crop_h
                    @simd for w = 1:crop_w
                        @inbounds value[crop_w-w+1,h,c,n] = input[w+w_off,h+h_off,c,n] * ens.scale
                    end
                end
            end
        end
    end
end

function backward(ens::TransformEnsemble, args...)
end
