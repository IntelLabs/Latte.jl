# Copyright (c) 2015 Intel Corporation. All rights reserved.
using Latte

net = Net(64)
data, label  = HDF5DataLayer(net, "data/train.txt", "data/test.txt")

conv1_1 = ConvolutionLayer( :conv1_1, net, data,    64,  3, 1, 1)
relu1_1 = ReLULayer(        :relu1_1, net, conv1_1)
conv1_2 = ConvolutionLayer( :conv1_2, net, relu1_1, 64,  3, 1, 1)
relu1_2 = ReLULayer(        :relu1_2, net, conv1_2)
pool1   = MaxPoolingLayer(  :pool1,   net, relu1_2, 2,   2, 0)

conv2_1 = ConvolutionLayer( :conv2_1, net, pool1,   128, 3, 1, 1)
relu2_1 = ReLULayer(        :relu2_1, net, conv2_1)
conv2_2 = ConvolutionLayer( :conv2_2, net, relu2_1, 128, 3, 1, 1)
relu2_2 = ReLULayer(        :relu2_2, net, conv2_2)
pool2   = MaxPoolingLayer(  :pool2,   net, relu2_2, 2,   2, 0)

conv3_1 = ConvolutionLayer( :conv3_1, net, pool2,   256, 3, 1, 1)
relu3_1 = ReLULayer(        :relu3_1, net, conv3_1)
conv3_2 = ConvolutionLayer( :conv3_2, net, relu3_1, 256, 3, 1, 1)
relu3_2 = ReLULayer(        :relu3_2, net, conv3_2)
conv3_3 = ConvolutionLayer( :conv3_3, net, relu3_2, 256, 3, 1, 1)
relu3_3 = ReLULayer(        :relu3_3, net, conv3_3)
conv3_4 = ConvolutionLayer( :conv3_4, net, relu3_3, 256, 3, 1, 1)
relu3_4 = ReLULayer(        :relu3_4, net, conv3_4)
pool3   = MaxPoolingLayer(  :pool3,   net, relu3_4, 2,   2, 0)

fc4     = InnerProductLayer(:fc4,     net, pool3,   1024)
relu4   = ReLULayer(        :relu4,   net, fc4)
# drop4   = DropoutLayer(     :drop4,   net, relu4, .5f0)
# fc5     = InnerProductLayer(:fc5,     net, drop4,   1024)
fc5     = InnerProductLayer(:fc5,     net, relu4,   1024)
relu5   = ReLULayer(        :relu5,   net, fc5)
# drop5   = DropoutLayer(     :drop5,   net, relu5, .5f0)
# fc6     = InnerProductLayer(:fc6,     net, drop5,   10)
fc6     = InnerProductLayer(:fc6,     net, relu5,   10)

loss     = SoftmaxLossLayer(:loss, net, fc6, label)
accuracy = AccuracyLayer(:accuracy, net, fc6, label)

params = SolverParameters(
    LRPolicy.Step(.01, .1, 100000),
    MomPolicy.Fixed(0.9),
    100000,
    .0005,
    1000)
sgd = SGD(params)
solve(sgd, net)
