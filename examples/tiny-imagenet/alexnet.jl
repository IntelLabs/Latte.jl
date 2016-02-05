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

batch_size = 32

net = Net(batch_size)
data, label  = HDF5DataLayer(net, "data/train.txt", "data/val.txt")
trans = TransformLayer(net, :trans, data; crop=(56, 56), scale=1/256.0f0, random_mirror=true, mean_file="data/mean.hdf5")

conv1 = ConvolutionLayer(:conv1, net, trans, 64, 3, 1, 1)
relu1 = ReLULayer(:relu1, net, conv1)
pool1 = MaxPoolingLayer(:pool1, net, relu1, 2, 2, 0)

conv2 = ConvolutionLayer(:conv2, net, pool1, 256, 3, 1, 2; bias_init=0.1f0)
relu2 = ReLULayer(:relu2, net, conv2)
pool2 = MaxPoolingLayer(:pool2, net, relu2, 2, 2, 0)

conv3 = ConvolutionLayer(:conv3, net, pool2, 384, 3, 1, 1)
relu3 = ReLULayer(:relu3, net, conv3)
conv4 = ConvolutionLayer(:conv4, net, relu3, 256, 3, 1, 1; bias_init=0.1f0)
relu4 = ReLULayer(:relu4, net, conv4)
conv5 = ConvolutionLayer(:conv5, net, relu4, 256, 3, 1, 1; bias_init=0.1f0)
relu5 = ReLULayer(:relu5, net, conv5)
pool5 = MaxPoolingLayer(:pool5, net, relu5, 2, 2, 0)

fc6     = InnerProductLayer(:fc6, net, pool5, 256;
                            weight_init=gaussian(std=0.005), bias_init=0.1f0)
relu6   = ReLULayer(:relu6, net, fc6)
fc7     = InnerProductLayer(:fc7, net, relu6, 256;
                            weight_init=gaussian(std=0.005), bias_init=0.1f0)
relu7   = ReLULayer(:relu7, net, fc7)
fc8     = InnerProductLayer(:fc8, net, relu7, 200)

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
