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

function create_net()
	net = Net(100)

    data, label = HDF5DataLayer(net, "train.txt", "test.txt")

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

	net
end
facts("Testing saving and loading snapshots") do
    mkdir("snapshot_test")
    cp("../examples/mnist/data/get-data.sh", joinpath("snapshot_test", "get-data.sh"))
    cp("../examples/mnist/data/convert.jl", joinpath("snapshot_test", "convert.jl"))
    cd("snapshot_test")
    run(`bash get-data.sh`)
	net = create_net()

    params = SolverParameters(
        lr_policy    = LRPolicy.Decay(.01f0, 5.0f-7),
        mom_policy   = MomPolicy.Fixed(0.9),
        max_epoch    = 1,
        regu_coef    = .0005)
	sgd = SGD(params)
	solve(sgd, net)

	save_snapshot(net, "snapshot-1.jld")

	net2 = create_net()

	init(net2)

	load_snapshot(net2, "snapshot-1.jld")

	for (param1, param2) in zip(net.params, net2.params)
		@fact param1.value --> param2.value
	end

end
