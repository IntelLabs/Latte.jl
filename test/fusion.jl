# Copyright (c) 2015 Intel Corporation. All rights reserved.
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
data_value[:]  = rand(Float32, size(data_value)...) * 256
label_value[:] = map(floor, rand(Float32, size(label_value)...) * 10)
conv1        = ConvolutionLayer(:conv1, net, data, 10, 3, 1, 1)
relu1        = ReLULayer(:relu1, net, conv1)
pool1        = MaxPoolingLayer(:pool1, net, relu1, 2, 2, 0)

conv2_1      = ConvolutionLayer(:conv2_1, net, pool1, 10, 3, 1, 1)
relu2_1      = ReLULayer(:relu2_1, net, conv2_1)
conv2_2      = ConvolutionLayer(:conv2_2, net, relu2_1, 10, 3, 1, 1)
relu2_2      = ReLULayer(:relu2_2, net, conv2_2)
pool2        = MaxPoolingLayer(:pool2, net, relu2_2, 2, 2, 0)

init(net)

input     = get_buffer(net, :datavalue)
filters1  = get_buffer(net, :conv1weights)
bias1     = get_buffer(net, :conv1bias)
rand!(bias1)
conv1_out = zeros(get_buffer(net, :conv1value))

mask1     = get_buffer(net, :pool1maxidx)
mask1_out = zeros(mask1)
pool1_out = zeros(get_buffer(net, :pool1value))

filters2_1  = get_buffer(net, :conv2_1weights)
bias2_1     = get_buffer(net, :conv2_1bias)
rand!(bias2_1)
conv2_1_out = zeros(get_buffer(net, :conv2_1value))

filters2_2  = get_buffer(net, :conv2_2weights)
bias2_2     = get_buffer(net, :conv2_2bias)
rand!(bias2_2)
conv2_2_out = zeros(get_buffer(net, :conv2_2value))

mask2     = get_buffer(net, :pool2maxidx)
mask2_out = zeros(mask2)
pool2_out = zeros(get_buffer(net, :pool2value))

facts("Testing Conv-Relu-Pool fusion") do
    forward(net)
    convolution_forward(filters1, bias1, input, conv1_out, 1, 1, 3)
    map!((x) -> x > 0.0f0 ? x : 0.0f0, conv1_out)
    @fact conv1_out --> roughly(get_buffer(net, :conv1value))
    pooling_forward(conv1_out, mask1_out, pool1_out, 2, 2, 0, :max)
    @fact pool1_out --> roughly(get_buffer(net, :pool1value))
    @fact mask1_out --> mask1
    convolution_forward(filters2_1, bias2_1, pool1_out, conv2_1_out, 1, 1, 3)
    map!((x) -> x > 0.0f0 ? x : 0.0f0, conv2_1_out)
    @fact conv2_1_out --> roughly(get_buffer(net, :conv2_1value))
    convolution_forward(filters2_2, bias2_2, conv2_1_out, conv2_2_out, 1, 1, 3)
    map!((x) -> x > 0.0f0 ? x : 0.0f0, conv2_2_out)
    @fact conv2_2_out --> roughly(get_buffer(net, :conv2_2value))
    pooling_forward(conv2_2_out, mask2_out, pool2_out, 2, 2, 0, :max)
    @fact pool2_out --> roughly(get_buffer(net, :pool2value))
    @fact mask2_out --> mask2
end
