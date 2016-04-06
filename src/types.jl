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

"""
A task that calls `func` with `args...`
"""
type JuliaTask <: LatteTask
    func :: Function
    args :: Tuple
end

"""
A task that updates parameter `param_id`
"""
type UpdateTask <: LatteTask
    param_id   :: UInt64
end

"""
Used in neuron definitions to mark a field to be unique per batch item
"""
type Batch{T}
    init :: T
end

"""
A parameter in a `Net` (learned during training)

** Fields **

- `name`          -- the name of the parameter
- `gradient_name` -- the name of the gradient (should be ∇`name`)
- `hist_name`     -- the name of the history buffer (should be `name`hist)
- learning_rate   -- local learning rate for the parameter
- regu_coef       -- local regularization coefficient
- clip_gradients  -- NOT IMPLEMENTED, gradient clipping parameter
- value           -- buffer containing the value of the parameter
- gradient        -- buffer containing the gradient of the parameter
- hist            -- buffer containing the history of the parameter
- request         -- request id, used for MPI data parallelism
"""
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

"""
A container for tasks for multiple `Phase`s

** Fields **

- tasks -- a dictionary containing a `Vector` of tasks for each `Phase`
"""
type TaskSet
    tasks :: Dict{Phase, Vector{LatteTask}}
    TaskSet() = new(Dict{Phase, Vector{LatteTask}}(Train => [], Test => []))
end

#= function TaskSet(tasks::Dict{Phase, Set}) =#
#=     @assert haskey(tasks, Train) && haskey(tasks, Test) =#
#= end =#

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

"""
** Fields **

- `ensembles`          -- a vector containing each `Ensemble` in the network.
- `ensembles_map`      -- a mapping between ensemble names in the network to the
                          instance stored in `ensembles`
- `buffers`            -- the internal buffers used by the network
- `curr_buffer_set`    -- the current buffer set (used for double buffering)
- `forward_tasks`      -- a set of tasks for performing forward propogation of the network
- `backward_tasks`     -- a set of tasks for performing back propogation of the network
- `update_tasks`       -- a set of tasks for performing parameter updates
- `params`             -- a vector of `Param` instances corresponding to network parameters
- `run_where`          -- DEPRECATED
- `signal`             -- DEPRECATED
- `batch_size`         -- the batch size of the network 
                          TODO: support different batch sizes for train/test
- `train_epoch`        -- number of epochs completed for training
- `test_epoch`         -- number of epochs completed for testing
- `curr_time_step`     -- the current time step (for RNNs)
- `time_steps`         -- total number of time steps to unroll (for RNNs)
- `num_subgroups`      -- number of partitions in network (for model parallelism)
- `ensemble_send_list` -- a mapping between ensembles and a list of subgroups to send
                          values to, used internally when synthesizing code for
                          model parallelism
"""
type Net
    ensembles          :: Vector{AbstractEnsemble}
    ensembles_map      :: Dict{Symbol, AbstractEnsemble}

    buffers            :: Tuple{Vector{Dict{Symbol, Array}},
                                Vector{Dict{Symbol, Array}}}
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

    Net(batch_size, time_steps=1, num_subgroups=1) = new(
                [],                                                  # ensembles
                Dict{Symbol, AbstractEnsemble}(),                    # ensembles_map
                tuple([Dict{Symbol,Array}() for _ in 1:time_steps],
                      [Dict{Symbol,Array}() for _ in 1:time_steps]), # buffers
                1,                                                   # curr_buffer_set 
                TaskSet(),                                           # forward_tasks
                TaskSet(),                                           # backward_tasks
                [],                                                  # update_tasks
                Param[],                                             # params
                -1,                                                  # run_where 
                Array(Cint, 1),                                      # signal
                batch_size,                                          # batch_size
                1,                                                   # train_epoch
                1,                                                   # test_epoch
                1,                                                   # curr_time_step
                time_steps,                                          # time_steps
                num_subgroups,                                       # num_subgroups
                Dict{Symbol, Vector{Int}}())                         # ensemble_send_list
end

"""
Main `Net` constructor that should be used.
** Params **
- `batch_size`    -- batch size of the network
- `time_steps`    -- number of time steps to unroll the network (for RNNs)
- `num_subgroups` -- number of subgroups in the network (for model parallelism)
"""
function Net(batch_size::Int; time_steps=1, num_subgroups=1)
    net = Net(batch_size, time_steps, num_subgroups)
    @latte_mpi initialize_communicators(net)
    net
end

"""
Returns the batch size of the network
"""
batch_size(net::Net) = net.batch_size

"""
A connection between two ensembles.

** Fields **

- source        -- the source `Ensemble`
- mapping       -- a mapping function to a range of neurons in `source`
- shape         -- shape of the connected neurons returned by `mapping`
- size          -- length of the connected neurons returned by `mapping`
- copy          -- whether the connection requires input values to be copied
- is_dim_fixed  -- vector of booleans that are true if the connection is fixed along a dimension
- is_one_to_one -- whether the connection is one to one
- padding       -- amount of padding used for the connection
- recurrent     -- whether the connection is recurrent
"""
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

"""
An ensemble

** Fields **

- name         -- name of the ensemble
- neurons      -- an array of neurons of type `T`
- connections  -- a list of `Ensemble`s connected to this ensemble
- batch_fields -- a vector of `Batch` fields for `T` (used internally)
- arg_dim_info -- 
- params       -- a vector of `Param`s associated with the ensemble
- phase        -- phase(s) in which this ensemble is active
- net_subgroup -- the net subgroup the ensemble is a member of (use for model parallelism)
"""
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
