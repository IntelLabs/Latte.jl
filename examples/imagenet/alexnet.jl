# Copyright (c) 2015 Intel Corporation. All rights reserved.
using Latte

batch_size = 64

net = Net(batch_size)
data, label  = HDF5DataLayer(net, "data/train.txt", "data/test.txt")
trans = TransformLayer(net, :trans, data; crop=(224, 224), random_mirror=true, mean_file="data/mean.hdf5")

conv1 = ConvolutionLayer(:conv1, net, trans, 96, 11, 4, 1;
                         weight_init=gaussian(std=0.01))
relu1 = ReLULayer(:relu1, net, conv1)
pool1 = MaxPoolingLayer(:pool1, net, relu1, 2, 2, 0)

conv2 = ConvolutionLayer(:conv2, net, pool1, 256, 5, 1, 2;
                         weight_init=gaussian(std=0.01), bias_init=0.1f0)
relu2 = ReLULayer(:relu2, net, conv2)
pool2 = MaxPoolingLayer(:pool2, net, relu2, 2, 2, 0)

conv3 = ConvolutionLayer(:conv3, net, pool2, 384, 3, 1, 1;
                         weight_init=gaussian(std=0.01))
relu3 = ReLULayer(:relu3, net, conv3)
conv4 = ConvolutionLayer(:conv4, net, relu3, 256, 3, 1, 1;
                         weight_init=gaussian(std=0.01), bias_init=0.1f0)
relu4 = ReLULayer(:relu4, net, conv4)
conv5 = ConvolutionLayer(:conv5, net, relu4, 256, 3, 1, 1;
                         weight_init=gaussian(std=0.01), bias_init=0.1f0)
relu5 = ReLULayer(:relu5, net, conv5)
pool5 = MaxPoolingLayer(:pool5, net, relu5, 2, 2, 0)

fc6     = InnerProductLayer(:fc6, net, pool5, 4096;
                            weight_init=gaussian(std=0.005), bias_init=0.1f0)
relu6   = ReLULayer(:relu6, net, fc6)
fc7     = InnerProductLayer(:fc7, net, relu6, 4096;
                            weight_init=gaussian(std=0.005), bias_init=0.1f0)
relu7   = ReLULayer(:relu7, net, fc7)
fc8     = InnerProductLayer(:fc8, net, relu7, 1000)

loss     = SoftmaxLossLayer(:loss, net, fc8, label)
accuracy = AccuracyLayer(:accuracy, net, fc8, label)

params = SolverParameters(
    LRPolicy.Step(.01, .1, 100000),
    MomPolicy.Fixed(0.9),
    450000,
    .0005,
    1000)
sgd = SGD(params)
solve(sgd, net)
