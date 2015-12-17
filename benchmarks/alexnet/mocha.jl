# Copyright (c) 2015 Intel Corporation. All rights reserved.
ENV["MOCHA_USE_NATIVE_EXT"] = "true"

using Mocha

backend = CPUBackend()
init(backend)
img_width, img_height, img_channels = (224, 224, 3)
batch_size = 128  # could be larger if you want to classify a bunch of images at a time

layers = [
  MemoryDataLayer(name="data", tops=[:data], batch_size=batch_size,
      data = Array[zeros(img_width, img_height, img_channels, batch_size)])
  ConvolutionLayer(name="conv1", tops=[:conv1], bottoms=[:data],
      kernel=(11,11), stride=(4,4), n_filter=96, neuron=Neurons.ReLU())
  PoolingLayer(name="pool1", tops=[:pool1], bottoms=[:conv1],
      kernel=(2,2), stride=(2,2), pooling=Pooling.Max())
  ConvolutionLayer(name="conv2", tops=[:conv2], bottoms=[:pool1],
      kernel=(5,5), pad=(2,2), n_filter=256, neuron=Neurons.ReLU())
  PoolingLayer(name="pool2", tops=[:pool2], bottoms=[:conv2],
      kernel=(2,2), stride=(2,2), pooling=Pooling.Max())
  ConvolutionLayer(name="conv3", tops=[:conv3], bottoms=[:pool2],
      kernel=(3,3), pad=(1,1), n_filter=384, neuron=Neurons.ReLU())
  ConvolutionLayer(name="conv4", tops=[:conv4], bottoms=[:conv3],
      kernel=(3,3), pad=(1,1), n_filter=384, neuron=Neurons.ReLU())
  ConvolutionLayer(name="conv5", tops=[:conv5], bottoms=[:conv4],
      kernel=(3,3), pad=(1,1), n_filter=256, neuron=Neurons.ReLU())
  PoolingLayer(name="pool5", tops=[:pool5], bottoms=[:conv5],
      kernel=(2,2), stride=(2,2), pooling=Pooling.Max())
  InnerProductLayer(name="fc6", tops=[:fc6], bottoms=[:pool5],
      output_dim=4096)
  InnerProductLayer(name="fc7", tops=[:fc7], bottoms=[:fc6],
      output_dim=4096)
  InnerProductLayer(name="fc8", tops=[:fc8], bottoms=[:fc7],
      output_dim=1000)
]

net = Net("imagenet", backend, layers)

init(net)

for i = 1:3
    forward(net)
    backward(net)
end

num_trials = 10

forward_time = 0.0
backward_time = 0.0
for i = 1:num_trials
    tic()
    forward(net)
    forward_time += toq()
    tic()
    backward(net)
    backward_time += toq()
end
println("Avg forward time for $num_trials runs: $(forward_time / num_trials * 1000.0)ms")
println("Avg backward time for $num_trials runs: $(backward_time / num_trials * 1000.0)ms")
