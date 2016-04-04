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

abstract AbstractEnsemble
abstract DataEnsemble <: AbstractEnsemble
abstract NormalizationEnsemble <: AbstractEnsemble

abstract LatteTask

type JuliaTask <: LatteTask
    func :: Function
    args :: Tuple
end

type UpdateTask <: LatteTask
    param_id   :: UInt64
end

type Batch{T}
    init :: T
end

type Shared{T}
    value :: T
end

abstract Net

type Param
    name           :: Symbol
    gradient_name  :: Symbol
    hist_name      :: Symbol
    learning_rate  :: Float32
    regu_coef      :: Float32
    clip_gradients :: Float32

    value    :: Array
    gradient :: Array
    hist     :: Array
    request  :: Cint
    Param(ensemble_name::Symbol, name::Symbol,
          learning_rate::Float32, regu_coef::Float32) =
              new(symbol(ensemble_name, name),
                  symbol(ensemble_name,:∇,name), 
                  symbol(ensemble_name, name, :hist), learning_rate, regu_coef, -1.0f0)
end

type TaskSet
    tasks :: Dict{Phase, Vector{LatteTask}}
    TaskSet() = new(Dict{Phase, Vector{LatteTask}}(Train => [], Test => []))
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
    ensembles          :: Vector{AbstractEnsemble}
    ensembles_map      :: Dict{Symbol, AbstractEnsemble}
    buffers            :: Tuple{Vector{Dict{Symbol, Array}},Vector{Dict{Symbol, Array}}}
    curr_buffer_set    :: Int
    forward_tasks      :: TaskSet
    backward_tasks     :: TaskSet
    update_tasks       :: Vector{LatteTask}
    params             :: Vector{Param}
    run_where          :: Int
    signal             :: Array{Cint, 1}
    batch_size         :: Int
    train_epoch        :: Int
    test_epoch         :: Int
    curr_time_step     :: Int
    time_steps         :: Int
    num_subgroups      :: Int
    ensemble_send_list :: Dict{Symbol, Vector{Tuple{Int, Int}}}
    SingleNet(batch_size, time_steps=1, num_subgroups=1) = new([],
                Dict{Symbol, AbstractEnsemble}(),
                tuple([Dict{Symbol,Array}() for _ in 1:time_steps],
                      [Dict{Symbol,Array}() for _ in 1:time_steps]),
                1,
                TaskSet(),
                TaskSet(),
                [], [], -1, Array(Cint, 1), batch_size, 1, 1, 1, time_steps, num_subgroups,
                Dict{Symbol, Vector{Int}}())
end

function Net(batch_size::Int; time_steps=1, num_subgroups=1)
    net = SingleNet(batch_size, time_steps, num_subgroups)
    @latte_mpi initialize_communicators(net)
    net
end

batch_size(net::SingleNet) = net.batch_size

type Connection
    source        :: AbstractEnsemble
    mapping       :: Function
    shape         :: Tuple
    size          :: Int
    copy          :: Bool
    is_dim_fixed  :: Vector{Bool}
    is_one_to_one :: Bool
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
    net_subgroup :: Cint
end

abstract JuliaEnsemble <: AbstractEnsemble

type ConcatNeuron <: Neuron
    value :: Float32
    ∇     :: Float32
end

type ConcatEnsemble <: AbstractEnsemble
    name :: Symbol
    neurons :: Array{ConcatNeuron}
    connections :: Vector{Connection}
    phase :: Phase
    inputs :: Vector{AbstractEnsemble}
    inner_size :: Int
end

type ReshapeNeuron <: Neuron
    value :: Float32
    ∇     :: Float32
end

type ReshapeEnsemble <: AbstractEnsemble
    name :: Symbol
    neurons :: Array{ReshapeNeuron}
    connections :: Vector{Connection}
    phase :: Phase
    net_subgroup :: Cint
    input :: AbstractEnsemble
    shape :: Tuple
end
