# Copyright (c) 2015 Intel Corporation. All rights reserved.
using Latte

net = Net(100)
data, label = HDF5DataLayer(net, "data/train.txt", "data/test.txt")
conv1    = ConvolutionLayer(:conv1, net, data, 20, 5, 1, 1)
relu1    = ReLULayer(:relu1, net, conv1)
pool1    = MaxPoolingLayer(:pool1, net, relu1, 2, 2, 0)
conv2    = ConvolutionLayer(:conv2, net, pool1, 50, 5, 1, 1)
relu2    = ReLULayer(:relu2, net, conv2)
pool2    = MaxPoolingLayer(:pool2, net, relu2, 2, 2, 0)
conv3    = ConvolutionLayer(:conv3, net, pool1, 50, 3, 1, 1)
relu3    = ReLULayer(:relu3, net, conv3)
pool3    = MaxPoolingLayer(:pool3, net, relu3, 2, 2, 0)
fc4      = InnerProductLayer(:fc4, net, pool3, 512)
relu4    = ReLULayer(:relu4, net, fc4)
fc5      = InnerProductLayer(:fc5, net, relu4, 512)
relu5    = ReLULayer(:relu5, net, fc5)
fc6      = InnerProductLayer(:fc6, net, relu5, 10)
loss     = SoftmaxLossLayer(:loss, net, fc6, label)
accuracy = AccuracyLayer(:accuracy, net, fc6, label)

params = SolverParameters(
    LRPolicy.Inv(0.01, 0.0001, 0.75),
    MomPolicy.Fixed(0.9),
    100000,
    .0005,
    1000)
sgd = SGD(params)
solve(sgd, net)
