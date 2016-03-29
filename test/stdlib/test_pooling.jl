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

function pooling_backward(gradient, mask, diff, kernel, stride, pad, _type)
  width, height, channels, num = size(gradient)
  pooled_width, pooled_height, _ = size(diff)
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

          region = sub(gradient, wstart:wend, hstart:hend, c, n)
          if _type == :max
              index = mask[pw, ph, c, n]
              region[index] += diff[pw, ph, c, n]
          elseif _type == :mean
              region[:] += diff[pw, ph, c, n] / kernel_size
          end
          # if isa(state.layer.pooling, Pooling.Max)
          #   index = payload[pw, ph, c, n]
          #   region[index] += diff[pw, ph, c, n]
          # elseif isa(state.layer.pooling, Pooling.Mean)
          #   region[:] += diff[pw, ph, c, n] / kernel_size
          # else
          #   error("Unknown pooling $(state.layer.pooling)")
          # end
        end
      end
    end
  end
end

facts("Testing MaxPooling Layer") do
    for (h, w) in [(24, 24), (2, 2)]
        net = Net(8)
        data,  data_value   = MemoryDataLayer(net, :data, (w, h, 3))
        label, label_value = MemoryDataLayer(net, :label, (1,))
        data_value[:]  = rand(Float32, size(data_value)...) * 256
        label_value[:] = map(floor, rand(Float32, size(label_value)...) * 10)
        conv1        = ConvolutionLayer(:conv1, net, data, 2, 3, 1, 1)
        pool1        = MaxPoolingLayer(:pool1, net, conv1, 2, 2, 0)
        fc1          = InnerProductLayer(:fc1, net, pool1, 10)
        loss         = SoftmaxLossLayer(:loss, net, fc1, label)

        init(net)

        params = SolverParameters(
            lr_policy    = LRPolicy.Decay(.01f0, 5.0f-7),
            mom_policy   = MomPolicy.Fixed(0.9),
            max_epoch    = 300,
            regu_coef    = .0005)
        sgd = SGD(params)

        input    = get_buffer(net, :conv1value)
        mask     = get_buffer(net, :pool1maxidx)

        context("Forward input shape=($h x $w)") do
            forward(net; solver=sgd)

            mask_expected = zeros(mask)
            expected = zeros(get_buffer(net, :pool1value))
            pooling_forward(input, mask_expected, expected, 2, 2, 0, :max)
            @fact expected --> roughly(get_buffer(net, :pool1value))
            @fact mask_expected --> mask
        end

        context("Backward input shape=($h x $w)") do
            top_diff = get_buffer(net, :pool1∇)

            ∇input = get_buffer(net, :conv1∇)
            ∇input_expected = deepcopy(∇input)

            backward(net)
            pooling_backward(∇input_expected, mask, top_diff, 2, 2, 0, :max)
            @fact ∇input   --> roughly(∇input_expected)
        end
    end
end

facts("Testing MeanPooling Layer") do
    net = Net(8)
    data,  data_value   = MemoryDataLayer(net, :data, (227, 227, 3))
    label, label_value = MemoryDataLayer(net, :label, (1,))
    data_value[:]  = rand(Float32, size(data_value)...) * 256
    label_value[:] = map(floor, rand(Float32, size(label_value)...) * 10)
    conv1        = ConvolutionLayer(:conv1, net, data, 2, 3, 1, 1)
    pool1        = MeanPoolingLayer(:pool1, net, conv1, 2, 2, 0)
    fc1          = InnerProductLayer(:fc1, net, pool1, 10)
    loss         = SoftmaxLossLayer(:loss, net, fc1, label)

    init(net)

    params = SolverParameters(
        lr_policy    = LRPolicy.Decay(.01f0, 5.0f-7),
        mom_policy   = MomPolicy.Fixed(0.9),
        max_epoch    = 300,
        regu_coef    = .0005)
    sgd = SGD(params)

    input    = get_buffer(net, :conv1value)

    context("Forward") do
        forward(net; solver=sgd)

        expected = zeros(get_buffer(net, :pool1value))
        pooling_forward(input, [], expected, 2, 2, 0, :mean)
        @fact expected --> roughly(get_buffer(net, :pool1value))
    end

    context("Backward") do
        top_diff = get_buffer(net, :pool1∇)

        ∇input = get_buffer(net, :conv1∇)
        ∇input_expected = deepcopy(∇input)

        backward(net)
        pooling_backward(∇input_expected, [], top_diff, 2, 2, 0, :mean)
        @fact ∇input   --> roughly(∇input_expected)
    end
end

FactCheck.exitstatus()
