# Internals

    {meta}
    CurrentModule = Latte

## Contents

    {contents}
    Pages = ["internals.md"]

## Index

    {index}
    Pages = ["internals.md"]

## Net
The `Net` datastructure is the main container used by the Latte compiler and
runtime.  When constructing a `Net` a user adds ensembles and applies
connections between them.  This implicitly constructs a task graph with
connections as data dependencies and ensembles as groups of comptue tasks.  The
`init(net::Net)` routine is the main entry point for the Latte compiler.
Inside this function, the Latte compiler consumes the implicit task graph,
synthesizes functions to compute various tasks and optimizes these functions.

    {docs}
    init(net::Net)

    init_buffer(net::Net, name::Symbol, shape; func=zeros)
    set_buffer(net::Net, name::Symbol, arr::Array; _copy=true)
    set_buffer(net::Net, name::Symbol, arr::Array, t::Int)
    get_buffer(net::Net, ens::AbstractEnsemble, name::Symbol)

    rand_values(net::Net)
    clear_values(net::Net)
    clear_∇(net::Net)

## Connections

    {docs}
    check_one_to_one(mapping::Function, shape::Tuple)
    check_dimensions_fixed(mapping::Function, sink_shape::Tuple)

## Synthesis and Optimization

    {docs}
    add_send_exprs(net::Net, ensemble::AbstractEnsemble,
                         compute_body::Dict{Phase, Vector},
                         compute_args::ArgSet)
    add_recv_expr(net::Net, source::AbstractEnsemble,
                        ensemble::AbstractEnsemble, 
                        compute_body::Dict{Phase, Vector}, compute_args::ArgSet)
    init_forward(ensemble::ReshapeEnsemble, net::Net, compute_args::ArgSet, compute_body)
    init_forward(ensemble::ConcatEnsemble, net::Net, compute_args::ArgSet,
                 compute_body::Dict{Phase, Vector})
    init_forward(ensemble::NormalizationEnsemble, net::Net,
                 compute_args::ArgSet, compute_body::Dict{Phase, Vector})
    init_backward(ensemble::ReshapeEnsemble, net::Net, compute_args::ArgSet, compute_body)
    init_backward(ensemble::ConcatEnsemble, net::Net, compute_args::ArgSet,
                  compute_body::Dict{Phase, Vector})
    init_backward(ensemble::NormalizationEnsemble, net::Net,
                  compute_args::ArgSet, compute_body::Dict{Phase, Vector})
    add_forward_data_tasks(ensemble::DataEnsemble, tasks::TaskSet, net::Net)
    add_forward_julia_tasks(ensemble::JuliaEnsemble, tasks::TaskSet, net::Net)
    push_compute_tasks!(tasks::TaskSet, buffers::Dict,
                        compute_body::Dict, compute_args::ArgSet,
                        run_where::Int, signal::Array{Cint, 1};
                        distribute::Bool=false)
    generate_c_function(func::Function, signature::Tuple,
                        run_where::Int, signal::Array{Cint,1},
                        buffers::Dict)
    gen_neuron_backward(ensemble::AbstractEnsemble, net::Net,
                        compute_body::Vector,
                        compute_args::Set)
    gen_neuron_forward(ensemble::AbstractEnsemble, net::Net,
                       compute_body::Dict{Phase, Vector},
                       compute_args::Set)
    gen_copy_block(net::Net, ensemble::AbstractEnsemble,
                   connection::Connection, index::Int; ∇::Bool=false)
    get_src_idx(mapping::Expr)
    optimize(args::Vector, tile_fusion_factors::Vector, fn::Expr)

    unpack_tiled_loop
    is_tiled_loop
    get_tile_fusion_factor_forward
    get_tile_fusion_factor_backward
    update_tile_var
    inner_loop_tiler
    get_inner_loop_tiler
    tile_size_inliner
    get_tile_loops

## Utility Datastructures

    {docs}
    TaskSet
    JuliaTask
    UpdateTask
    Batch
