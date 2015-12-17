# Copyright (c) 2015 Intel Corporation. All rights reserved.
abstract AbstractEnsemble
abstract DataEnsemble <: AbstractEnsemble
abstract NormalizationEnsemble <: AbstractEnsemble


type JuliaTask
    func :: Function
    args :: Tuple
end

type Batch{T}
    init :: T
end

type Shared{T}
    value :: T
end

abstract Net

type Param
    net            :: Net
    name           :: Symbol
    gradient_name  :: Symbol
    learning_rate  :: Float32
    regu_coef      :: Float32
    clip_gradients :: Float32

    value    :: Array
    gradient :: Array
    hist     :: Array
    request  :: Cint
    Param(net::Net, ensemble_name::Symbol, name::Symbol,
          learning_rate::Float32, regu_coef::Float32) =
              new(net, symbol(ensemble_name, name),
                  symbol(ensemble_name,:∇,name), learning_rate, regu_coef, -1.0f0)
end

type TaskSet
    tasks :: Dict{Phase, Vector{JuliaTask}}
    TaskSet() = new(Dict{Phase, Vector{JuliaTask}}(Train => [], Test => []))
end

function TaskSet(tasks::Dict{Phase, Set})
    @assert haskey(tasks, Train) && haskey(tasks, Test)
end

function Base.getindex(task_set::TaskSet, phase::Phase)
    return task_set.tasks[phase]
end

function Base.append!(first::TaskSet, second::TaskSet)
    append!(first[Train], second[Train])
    append!(first[Test], second[Test])
end

type ArgSet
    args :: Dict{Phase, Set}
    ArgSet() = new(Dict{Phase, Set}(Train => Set(), Test => Set()))
end

function ArgSet(args::Dict{Phase, Set})
    @assert haskey(args, Train) && haskey(args, Test)
end

function Base.getindex(arg_set::ArgSet, phase::Phase)
    arg_set.args[phase]
end

function Base.union!(first::ArgSet, second::ArgSet)
    union!(first[Train], second[Train])
    union!(first[Test], second[Test])
end

type SingleNet <: Net
    ensembles      :: Vector{AbstractEnsemble}
    ensembles_map  :: Dict{Symbol, AbstractEnsemble}
    buffers        :: Vector{Dict{Symbol, Array}}
    forward_tasks  :: TaskSet
    backward_tasks :: TaskSet
    update_tasks   :: Vector{JuliaTask}
    params         :: Vector{Param}
    run_where      :: Int
    signal         :: Array{Cint, 1}
    batch_size     :: Int
    train_epoch    :: Int
    test_epoch     :: Int
    time_steps     :: Int
    SingleNet(batch_size, time_steps=1) = new([],
                Dict{Symbol, AbstractEnsemble}(),
                [Dict{Symbol,Array}() for _ in 1:time_steps],
                TaskSet(),
                TaskSet(),
                [], [], -1, Array(Cint, 1), batch_size, 1, 1, time_steps)
end

Net(batch_size::Int; time_steps=1) = SingleNet(batch_size, time_steps)

batch_size(net::SingleNet) = net.batch_size

type Connection
    source        :: AbstractEnsemble
    mapping       :: Function
    shape         :: Tuple
    size          :: Int
    copy          :: Bool
    is_dim_fixed  :: Vector{Bool}
    padding       :: Int
    recurrent     :: Bool
end

type Ensemble{T <: Neuron, N} <: AbstractEnsemble
    name         :: Symbol
    neurons      :: Array{T,N}
    connections  :: Vector{Connection}
    batch_fields :: Vector{Symbol}
    arg_dim_info :: Dict{Symbol, Vector{Bool}}
    params       :: Vector
    phase        :: Phase
end

abstract JuliaEnsemble <: AbstractEnsemble
