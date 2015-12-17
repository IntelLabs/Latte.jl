# Copyright (c) 2015 Intel Corporation. All rights reserved.
export HDF5DataLayer

type HDF5DataEnsemble <: DataEnsemble
    net         :: Net
    name        :: Symbol
    neurons     :: Array{DataNeuron}
    train_id    :: Cint
    test_id     :: Cint
    train_epoch :: Int
    test_epoch  :: Int
end

@eval function HDF5DataEnsemble(net::Net, train_id::Cint, test_id::Cint, target::Symbol)
    if target == :data
        ndim = ccall((:get_data_ndim, $libIO), Cint, (Cint,), train_id)
        _shape = ccall((:get_data_shape, $libIO), Ptr{Cint}, (Cint,), train_id)
    else
        ndim = ccall((:get_label_ndim, $libIO), Cint, (Cint,), train_id)
        _shape = ccall((:get_label_shape, $libIO), Ptr{Cint}, (Cint,), train_id)
    end
    # first index is batch so skip, reverse c order
    shape = pointer_to_array(_shape, ndim)[end:-1:2]
    neurons = Array(DataNeuron, shape...)
    for i = 1:length(neurons)
        neurons[i] = DataNeuron(0.0f0)
    end
    ens = HDF5DataEnsemble(net, target, neurons, train_id, test_id, 1, 1)
    add_ensemble(net, ens)
    ens
end

@eval function init(ens::HDF5DataEnsemble, net::Net)
    arr = Array(Float32, size(ens)..., net.batch_size)
    set_buffer(net, symbol(ens.name,:value), arr; _copy=false)
    set_buffer(net, symbol(ens.name,:∇), Array(Float32, 0); _copy=false)
    if ens.name == :data
        ccall((:set_data_pointer, $libIO), Void, (Cint, Ptr{Float32},), ens.train_id, arr)
        ccall((:set_data_pointer, $libIO), Void, (Cint, Ptr{Float32},), ens.test_id, arr)
    else
        ccall((:set_label_pointer, $libIO), Void, (Cint, Ptr{Float32},), ens.train_id, arr)
        ccall((:set_label_pointer, $libIO), Void, (Cint, Ptr{Float32},), ens.test_id, arr)
    end
end

@eval function forward{N}(ens::HDF5DataEnsemble, data::Array{Float32,N}, phase::Phase)
    if phase == Train
        id = ens.train_id
    else
        id = ens.test_id
    end
    if ens.name == :data
        _shape = ccall((:get_next_batch, $libIO), Void, (Cint,), id)
    end
    epoch = ccall((:get_epoch, $libIO), Cint, (Cint,), id)
    if phase == Train
        ens.net.train_epoch = epoch
    else
        ens.net.test_epoch = epoch
    end
end

function parse_hdf5_source(source::AbstractString)
    open(source, "r") do s
        strip(readlines(s)[1])
    end
end

@eval function HDF5DataLayer(net::Net, train_data_source::AbstractString,
                       test_data_source::AbstractString;
                       shuffle=true, scale=1.0f0)
    batch_size = net.batch_size
    @assert(batch_size > 0, "Data Layer batch_size must be greater than 0")
    train_data_source = parse_hdf5_source(train_data_source)
    test_data_source = parse_hdf5_source(test_data_source)
    train_id = ccall((:init_dataset, $libIO), Cint, (Cint, Ptr{UInt8}, Cuchar, Cuchar, Cuchar), batch_size, train_data_source, shuffle, LATTE_MPI, false)
    test_id = ccall((:init_dataset, $libIO), Cint, (Cint, Ptr{UInt8}, Cuchar, Cuchar, Cuchar), batch_size, test_data_source, false, LATTE_MPI, true)
    HDF5DataEnsemble(net, train_id, test_id, :data), HDF5DataEnsemble(net, train_id, test_id, :label)
end
