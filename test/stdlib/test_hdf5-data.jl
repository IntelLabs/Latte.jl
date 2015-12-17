# Copyright (c) 2015 Intel Corporation. All rights reserved.
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
