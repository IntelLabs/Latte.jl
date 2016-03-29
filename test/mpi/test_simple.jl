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
using FactCheck

function pooling_forward(input, mask, output, kernel, stride, pad, _type)
  width, height, channels, num = size(input)
  pooled_width, pooled_height, _, _ = size(output)
  kernel_size = kernel * kernel

  for n = 1:num
    for c = 1:channels
      for ph = 1:pooled_height
        for pw = 1:pooled_width
          hstart = (ph-1)*stride - pad + 1
          wstart = (pw-1)*stride - pad + 1
          hend = min(hstart + kernel - 1, height)
          wend = min(wstart + kernel - 1, width)
          hstart = max(1, hstart)
          wstart = max(1, wstart)

          region = sub(input, wstart:wend, hstart:hend, c, n)
          if _type == :max
              index = indmax(region)
              mask[pw, ph, c, n] = index # note this is local index in region
              output[pw, ph, c, n] = region[index]
          elseif _type == :mean
              output[pw, ph, c, n] = sum(region) / kernel_size
          end
          # if isa(state.layer.pooling, Pooling.Max)
          #   index = indmax(region)
          #   mask[pw, ph, c, n] = index # note this is local index in region
          #   output[pw, ph, c, n] = region[index]
          # elseif isa(state.layer.pooling, Pooling.Mean)
          #   output[pw, ph, c, n] = sum(region) / kernel_size
          # else
          #   error("Unknown pooling $(state.layer.pooling)")
          # end
        end
      end
    end
  end
end

function convolution_forward(filter::Array, bias::Array, input::Array,
                             output::Array, stride, pad, kernel)
    width, height, channel, num = size(input)
    width_out, height_out, n_filter, _ = size(output)

    for n = 1:num, o = 1:n_filter, y = 1:height_out, x = 1:width_out,
        k = 1:channel, p = 1:kernel, q = 1:kernel

        in_y = (y-1) * stride - pad + p
        in_x = (x-1) * stride - pad + q
        if (in_y >= 1 && in_y <= height && in_x >= 1 && in_x <= width)
            filter_idx = ((k - 1) * kernel + (p - 1)) * kernel + q
            output[x, y, o, n] += input[in_x, in_y, k, n] *
                                  filter[filter_idx, o]
        end
    end

    # add bias
    for n = 1:num, o = 1:n_filter, y = 1:height_out, x = 1:width_out
        output[x, y, o, n] += bias[o]
    end

    return output
end

net = Net(8)
data,  data_value   = MemoryDataLayer(net, :data, (224, 224, 3))
label, label_value = MemoryDataLayer(net, :label, (1,))
rand!(data_value)
rand!(label_value)
conv1        = ConvolutionLayer(:conv1, net, data, 10, 3, 1, 1)
relu1        = ReLULayer(:relu1, net, conv1)
pool1        = MaxPoolingLayer(:pool1, net, relu1, 2, 2, 0)

conv2_1      = ConvolutionLayer(:conv2_1, net, pool1, 10, 3, 1, 1)
relu2_1      = ReLULayer(:relu2_1, net, conv2_1)
conv2_2      = ConvolutionLayer(:conv2_2, net, relu2_1, 10, 3, 1, 1)
relu2_2      = ReLULayer(:relu2_2, net, conv2_2)
pool2        = MaxPoolingLayer(:pool2, net, relu2_2, 2, 2, 0)

net.num_subgroups = 2
for ens in net.ensembles[6:end]
    ens.net_subgroup = 2
end

init(net)

params = SolverParameters(
    lr_policy    = LRPolicy.Decay(.01f0, 5.0f-7),
    mom_policy   = MomPolicy.Fixed(0.9),
    max_epoch    = 1,
    regu_coef    = .0005)
sgd = SGD(params)

facts("Testing Model Parallelism") do
    forward(net; solver=sgd)
    first_pool1val = copy(get_buffer(net, :pool1value))
    second_pool1val = copy(get_buffer(net, :pool1value))
    @eval ccall((:broadcast_intra, $(Latte.libComm)), Void, (Ptr{Float32}, Cint, Cint), $first_pool1val, length($first_pool1val), 0)
    @eval ccall((:broadcast_intra, $(Latte.libComm)), Void, (Ptr{Float32}, Cint, Cint), $second_pool1val, length($second_pool1val), 1)
    @fact first_pool1val --> second_pool1val
    println(first_pool1val[1:10])
end
