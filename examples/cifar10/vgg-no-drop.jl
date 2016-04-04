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

using Latte

net = Net(128)
data, label  = HDF5DataLayer(net, "data/train.txt", "data/test.txt")

conv1_1 = ConvolutionLayer( :conv1_1, net, data,    64,  3, 1, 1)
relu1_1 = ReLULayer(        :relu1_1, net, conv1_1)
conv1_2 = ConvolutionLayer( :conv1_2, net, relu1_1, 64,  3, 1, 1)
relu1_2 = ReLULayer(        :relu1_2, net, conv1_2)
pool1   = MaxPoolingLayer(  :pool1,   net, relu1_2, 2,   2, 0; ceil_mode=true)

conv2_1 = ConvolutionLayer( :conv2_1, net, pool1,   128, 3, 1, 1)
relu2_1 = ReLULayer(        :relu2_1, net, conv2_1)
conv2_2 = ConvolutionLayer( :conv2_2, net, relu2_1, 128, 3, 1, 1)
relu2_2 = ReLULayer(        :relu2_2, net, conv2_2)
pool2   = MaxPoolingLayer(  :pool2,   net, relu2_2, 2,   2, 0; ceil_mode=true)

conv3_1 = ConvolutionLayer( :conv3_1, net, pool2,   256, 3, 1, 1)
relu3_1 = ReLULayer(        :relu3_1, net, conv3_1)
conv3_2 = ConvolutionLayer( :conv3_2, net, relu3_1, 256, 3, 1, 1)
relu3_2 = ReLULayer(        :relu3_2, net, conv3_2)
conv3_3 = ConvolutionLayer( :conv3_3, net, relu3_2, 256, 3, 1, 1)
relu3_3 = ReLULayer(        :relu3_3, net, conv3_3)
pool3   = MaxPoolingLayer(  :pool3,   net, relu3_3, 2,   2, 0; ceil_mode=true)

conv4_1 = ConvolutionLayer( :conv4_1, net, pool3,   512, 3, 1, 1)
relu4_1 = ReLULayer(        :relu4_1, net, conv4_1)
conv4_2 = ConvolutionLayer( :conv4_2, net, relu4_1, 512, 3, 1, 1)
relu4_2 = ReLULayer(        :relu4_2, net, conv4_2)
conv4_3 = ConvolutionLayer( :conv4_3, net, relu4_2, 512, 3, 1, 1)
relu4_3 = ReLULayer(        :relu4_3, net, conv4_3)
pool4   = MaxPoolingLayer(  :pool4,   net, relu4_3, 2,   2, 0; ceil_mode=true)

conv5_1 = ConvolutionLayer( :conv5_1, net, pool4,   512, 3, 1, 1)
relu5_1 = ReLULayer(        :relu5_1, net, conv5_1)
conv5_2 = ConvolutionLayer( :conv5_2, net, relu5_1, 512, 3, 1, 1)
relu5_2 = ReLULayer(        :relu5_2, net, conv5_2)
conv5_3 = ConvolutionLayer( :conv5_3, net, relu5_2, 512, 3, 1, 1)
relu5_3 = ReLULayer(        :relu5_3, net, conv5_3)
pool5   = MaxPoolingLayer(  :pool5,   net, relu5_3, 2,   2, 0; ceil_mode=true)

fc6     = InnerProductLayer(:fc6,     net, pool5, 512)
relu6   = ReLULayer(        :relu6,   net, fc6)
fc7     = InnerProductLayer(:fc7,     net, relu6,   10)

loss     = SoftmaxLossLayer(:loss, net, fc7, label)
accuracy = AccuracyLayer(:accuracy, net, fc7, label)

params = SolverParameters(
    lr_policy    = LRPolicy.Decay(.01f0, 5.0f-7),
    mom_policy   = MomPolicy.Fixed(0.9),
    max_epoch    = 300,
    regu_coef    = .0005,
    snapshot_dir = "no_dropout")
sgd = SGD(params)
solve(sgd, net)
