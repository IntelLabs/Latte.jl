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
using HDF5
facts("Testing HDF5 Layer") do
    _file = "temp"

    w = 224
    h = 224
    c = 3
    n = 16

    data_value = rand(Float32, w, h, c, n) * 256
    label_value = map(floor, rand(Float32, 1, n) * 10)

    h5open("$_file.hdf5", "w") do h5
        dset_data = d_create(h5, "data", datatype(Float32), dataspace(w, h, c, n))
        dset_label = d_create(h5, "label", datatype(Float32), dataspace(1, n))
        dset_data[:,:,:,:] = data_value
        dset_label[:,:] = label_value
    end

    open("$_file.txt", "w") do f
        write(f, "$_file.hdf5")
    end

    net = Net(8)
    data, label  = HDF5DataLayer(net, "$_file.txt", "$_file.txt"; shuffle=false)

    init(net)
    forward(net)
    @fact get_buffer(net, :datavalue) --> data_value[:,:,:,1:8]
    @fact get_buffer(net, :labelvalue) --> label_value[:,1:8]
    forward(net)
    @fact get_buffer(net, :datavalue) --> data_value[:,:,:,9:16]
    @fact get_buffer(net, :labelvalue) --> label_value[:,9:16]
    # Test wrap around
    forward(net)
    @fact get_buffer(net, :datavalue) --> data_value[:,:,:,1:8]
    @fact get_buffer(net, :labelvalue) --> label_value[:,1:8]
    rm("$_file.txt")
    rm("$_file.hdf5")
end

FactCheck.exitstatus()
