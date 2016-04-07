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

export Net, init, forward, backward, clear_∇, clear_values, add_ensemble,
       add_connections, copy_from_mic, copy_to_mic, get_buffer,
       get_value, get_gradient
importall ParallelAccelerator
import ParallelAccelerator.CGen
import ParallelAccelerator.CGen.mk_parallel_loophead
import CompilerTools
CGen.set_include_blas()
# using ParallelAccelerator.J2CArray
ENV["KMP_AFFINITY"] = "granularity=fine,compact"

num_threads = get(ENV, "OMP_NUM_THREADS", nothing)
if num_threads == nothing
    num_threads = Base.CPU_CORES
    ENV["OMP_NUM_THREADS"] = "$num_threads"
else
    num_threads = parse(Int, num_threads)
end
const TILE_SIZE = 2
const MICRO_BATCH_SIZE = num_threads
NOFUSE = 0
LATTE_DISABLE_TILING = false
LATTE_DISABLE_TILE_FUSION = false

include("transforms/util.jl")
include("transforms/fixed_dims.jl")
include("transforms/distribute.jl")
include("transforms/tree_cleaners.jl")
include("transforms/neuron.jl")

include("optimizers/tiling.jl")
include("optimizers/fusion.jl")
include("optimizers/gemm_pattern_match.jl")
include("optimizers/array_expr_inline.jl")
include("optimizers/wrap_for_loops.jl")
include("optimizers/parallelize.jl")

"""
Optimize a function `fn`

** Params **

- `args`                -- an ordered vector of arguments (buffers)
- `tile_fusion_factors` -- factors for tile fusions determined by connection
                           structure
- `fn`                  -- the ast of the function to be optimized
"""
function optimize(args::Vector, tile_fusion_factors::Vector, fn::Expr)
    ast = macroexpand(fn)
    ast = remove_line_nodes(ast)
    arg_name_map = build_arg_name_map(args, ast)
    debugp(2, "----- Begin : Function Before Optimization -----")
    debugp(2, ast)
    debugp(2, "----- End   : Function Before Optimization -----")
    for (var1, var2, var3) in simple_ijk_orders
        fn = symbol(:pattern_match_gemm, var1, var2, var3)
        ast = AstWalk(ast, eval(fn), arg_name_map)
    end
    if !LATTE_DISABLE_TILING
        map!(get_inner_loop_tiler(tile_fusion_factors), ast.args[2].args)
    end
    ast = clean_tree(ast)
    ast = AstWalk(ast, pattern_match_gemm6, arg_name_map)
    ast = AstWalk(ast, pattern_match_gemm5, arg_name_map)
    ast = AstWalk(ast, pattern_match_gemm4, arg_name_map)
    # TODO: Above pattern matches should be cleaned up to ingest the "clean"
    # ast like fuse_loops below
    ast = clean_tree(ast)
    ast.args[2].args = fuse_loops(ast.args[2].args)
    debugp(1, "----- Begin : Function After Optimization  -----")
    debugp(1, ast)
    debugp(1, "----- End   : Function After Optimization  -----")
    return ast
end

mk_clamp(idx, _max) = :($idx > 0 && $idx <= $_max)

"""
Synthesize a loopnest to copy value or ∇ to the appropriate buffer
"""
function gen_copy_block(net::Net, ensemble::AbstractEnsemble,
                        connection::Connection, index::Int; ∇::Bool=false)
    if ∇
        shape = size(get_buffer(net, ensemble, :∇))
        sink_name  = symbol(ensemble.name, :∇inputs, index)
        source_name = symbol(connection.source.name, :∇)
    else
        shape = size(get_buffer(net, ensemble, :value))
        sink_name   = symbol(ensemble.name,:inputs,index)
        source_name = symbol(connection.source.name,:value)
    end
    N = length(shape)
    source = get_buffer(net, source_name)

    # connection.is_dim_fixed is a Bool[], we use the inverse to select non
    # fixed indices
    non_fixed_indices = [collect(1:N-1)[!connection.is_dim_fixed]..., N]

    sink_idx = [symbol(:_neuron_index_,d) for d in non_fixed_indices]

    mapping_args = [symbol(:_neuron_index_,d) for d in 1:N-1]
    mapping_inlined = inline(connection.mapping, mapping_args)
    src_idx = get_src_idx(mapping_inlined)

    body = Expr(:block)
    idx = [symbol(:_u__, i) for i = 1:length(src_idx)]
    for_block = gen_loop_nest(body, idx, src_idx)

    if ∇
        assign = :($source_name[$(idx...), $(symbol(:_neuron_index_,N))] +=
                       $sink_name[count, $(sink_idx...)])
        if connection.padding > 0
            cond = mk_clamp(idx[1], size(source, 1))
            for (i, d) in zip(idx[2:end], size(source)[2:end-1])
                cond = :($cond && $(mk_clamp(i, d)))
            end
            assign = :(if $cond
                $assign
            end)
        end
        statements = [
            assign,
            :($sink_name[count, $(sink_idx...)] = 0.0f0)
        ]
    else
        rhs = :($source_name[$(idx...), $(symbol(:_neuron_index_,N))])
        if connection.padding > 0
            cond = mk_clamp(idx[1], size(source, 1))
            for (i, d) in zip(idx[2:end], size(source)[2:end-1])
                cond = :($cond && $(mk_clamp(i, d)))
            end
            rhs = :(ifelse($cond, $rhs, 0.0f0))
        end
        statements = [:($sink_name[count, $(sink_idx...)] = $rhs)]
    end
    append!(body.args, (quote
        $(statements...)
        count += 1
    end).args)
    copy_block = quote
        $(mapping_inlined.args[1:end-1]...)
        count = 1
        $for_block
    end
    vars   = [symbol(:_neuron_index_,i) for i in 1:N]
    ranges = [connection.is_dim_fixed[i] ? :(1:1) : :(1:$(shape[i])) for i in 1:N-1]
    push!(ranges, :(1:$(shape[N])))
    gen_loop_nest(copy_block, vars, ranges)
end

"""
Extract the final expression of the mapping function to be used as an indexing
expression.  Handles cases where the final expression can be a Tuple or a
single value.

** Params **

- `mapping` -- an ast for a mapping function
"""
function get_src_idx(mapping::Expr)
    if isa(mapping.args[end], Expr) &&
            mapping.args[end].head == :call &&
            isa(mapping.args[end].args[1], TopNode)
        return mapping.args[end].args[2:end]
    else
        return [mapping.args[end]]
    end
end

"""
Synthesize the code to execute forward for an ensemble

** Params **

- `ensemble`     -- the `AbstractEnsemble` that is having its forward synthesized
- `net`          -- the `Net` containing `ensemble`
- `compute_body` -- the body that the synthesized code will be added to
- `compute_args` -- a set of arguments reference in `compute_body`.  Any values
                    referenced in the synthesized code should be added to
                    `compute_args`
"""
function gen_neuron_forward(ensemble::AbstractEnsemble, net::Net,
                            compute_body::Dict{Phase, Vector},
                            compute_args::ArgSet)
    forward_tasks = net.forward_tasks

    neuron_forward = get_forward(ensemble)
    value = get_buffer(net, ensemble, :value)
    N = ndims(value)
    body = []
    args = []
    arg_dim_info = ensemble.arg_dim_info
    for (index, connection) in enumerate(ensemble.connections)
        # Store argument info
        sink_name = symbol(ensemble.name,:inputs,index)
        arg_dim_info[sink_name] = connection.is_dim_fixed

        # If copying, add source buffer to args and generate the copy block
        if connection.copy
            source_name = symbol(connection.source.name,:value)
            push!(args, source_name)
            push!(body, gen_copy_block(net, ensemble, connection, index))
        end
    end
    # Transform body of neuron function
    ast, _args = transform_neuron_fn(neuron_forward, ensemble)
    map((a) -> push!(args, a), _args)  # FIXME: append! doesn't support sets

    ast = remove_line_nodes(ast)

    # Drop indices for shared value dimensions
    ast = drop_fixed_dims(ast, arg_dim_info)

    # Synthesize loop nest and append to body 
    append_neuron_loop!(body, ast.args, value)

    # Collect arguments
    args = collect(args)

    fn = symbol(ensemble.name,:forward)
    arg_bufs = map((arg) -> get_buffer(net, arg), args)
    params = [:($arg::$(typeof(buf))) for (arg, buf) in zip(args, arg_bufs)]
    tile_fusion_factors = []
    if length(ensemble.connections) == 1 && ndims(ensemble) > 2
        for d in ensemble.connections[1].shape[2:end]
            push!(tile_fusion_factors, (d, 0))
        end
    else
        for d in ensemble.connections[1].shape[2:end]
            push!(tile_fusion_factors, (0, 0))
        end
    end
    defn = optimize(arg_bufs, tile_fusion_factors, :(function $fn($(params...))
        $(body...)
    end))
    body = defn.args[2].args
    # Disable fusion across ensemble code if there exists a dependency
    is_dependence(conn) = length(conn.shape) > 2 && any(collect(conn.shape[3:end]) .> 1)
    if any(map(is_dependence, ensemble.connections))
        unshift!(body, :NOFUSE)
    end
    if ensemble.phase in [Train, TrainTest]
        for param in ensemble.params
            push!(compute_body[Train], :(update_param($(object_id(param)))))
        end
        append!(compute_body[Train], deepcopy(body))
        union!(compute_args[Train], args)
    end
    if ensemble.phase in [Test, TrainTest]
        append!(compute_body[Test], deepcopy(body))
        union!(compute_args[Test], args)
    end
end

"""
TODO: doc
"""
function gen_neuron_backward(ensemble::AbstractEnsemble, net::Net,
                             compute_body::Vector,
                             compute_args::Set)
    if !(ensemble.phase in [Train, TrainTest])
        return
    end
    backward_tasks = net.backward_tasks

    neuron_backward = get_backward(ensemble)
    ∇ = get_buffer(net, ensemble, :∇)
    N = ndims(∇)

    # Transform neuron backward definition and collect arguments
    ast, args = transform_neuron_fn(neuron_backward, ensemble)
    ast = remove_line_nodes(ast)

    body = []
    arg_dim_info = ensemble.arg_dim_info

    # Collect source buffer info
    for (index, connection) in enumerate(ensemble.connections)
        arg_dim_info[symbol(ensemble.name, :inputs, index)] =
            connection.is_dim_fixed
        sink_name = symbol(ensemble.name, :∇inputs, index)
        arg_dim_info[sink_name] = connection.is_dim_fixed
    end

    # Replace indexing expressions for shared values
    ast = drop_fixed_dims(ast, arg_dim_info)

    # Synthesize loop nest and append to body
    append_neuron_loop!(body, ast.args, ∇)
    fn = symbol(ensemble.name,:backward)
    arg_bufs = map((arg) -> get_buffer(net, arg), args)
    params = [:($arg::$(typeof(buf))) for (arg, buf) in zip(args, arg_bufs)]
    tile_fusion_factors = []
    if length(ensemble.connections) == 1 && body[1] != :NOFUSE && ndims(ensemble) > 2
        for d in ensemble.connections[1].shape[2:end]
            push!(tile_fusion_factors, (0, d))
        end
    else
        for d in ensemble.connections[1].shape[2:end]
            push!(tile_fusion_factors, (0, 0))
        end
    end
    for (index, connection) in enumerate(ensemble.connections)
        if !isa(connection.source, DataEnsemble) && connection.copy
            source_name = symbol(connection.source.name, :∇)
            push!(args, source_name)
            push!(body, gen_copy_block(net, ensemble, connection, index; ∇=true))
        end
    end
    defn = optimize(arg_bufs, tile_fusion_factors, :(function $fn($(params...))
        $(body...)
    end))
    body = defn.args[2].args
    for (index, connection) in enumerate(ensemble.connections)
        if length(connection.shape) > 2 && any(collect(connection.shape[3:end]) .> 1)
            push!(body, :NOFUSE)
            break
        end
    end
    prepend!(compute_body, body)
    union!(compute_args, args)
end

"""
TODO: doc
"""
function generate_c_function(func::Function, signature::Tuple,
                             run_where::Int, signal::Array{Cint,1},
                             buffers::Dict)
    m = methods(func, signature)
    def = m[1].func.code
    func_ref = GlobalRef(def.module, symbol(string(func)))
    ast = code_typed(func, signature)[1]

    ast = remove_temp_nodes(ast)
    ast = transform_to_raw_array(ast)
    ast = collect_parallel_loop_private_vars(ast)

    # dir_ast = Driver.toDomainIR(func_ref, ast, signature)
    # pir_ast = Driver.toParallelIR(func_ref, dir_ast, signature)

    function_name_string = CGen.canonicalize(string(func_ref.name))
    for varinfo in ast.args[2][1]
        if varinfo[2] <: Array
            varinfo[2] = CGen.RawArray{eltype(varinfo[2]), ndims(varinfo[2])}
        end
    end
    s = ""
    @latte_mpi(s *= "#include \"comm.h\"\n")
    s *= CGen.from_root_entry(ast, function_name_string)
    proxy_name = string("_", function_name_string, "_j2c_proxy")
    proxy_sym = symbol(proxy_name)
    j2c_name = string("_", function_name_string, "_unaliased_")
    args = [:($(symbol(:__arg_,d))) for d in 1:length(signature)]
    if run_where >= 0
        @assert false
    end
    outfile_name = CGen.writec(s)
    cflags = []
    @latte_mpi push!(cflags, "-I$latte_library_path/communication")
    @latte_mpi ENV["CGEN_COMPILER"] = "mpiicpc"
    CGen.compile(outfile_name; flags=cflags)

    lflags = []
    @latte_mpi push!(lflags, "-L$latte_library_path")
    @latte_mpi push!(lflags, "-lLatteComm")
    dyn_lib = CGen.link(outfile_name; flags=lflags)

    proxy_params = [:($arg::$typ) for (arg, typ) in zip(args, signature)]
    sig_types = [:(Ptr{$(eltype(typ))}) for typ in signature]
    if run_where < 0
        j2c_typs = Expr(:tuple, Cint, sig_types..., Ptr{Cint})
        return @eval function $proxy_sym($(proxy_params...))
            ccall(($j2c_name, $dyn_lib), Void, $j2c_typs, $(run_where), $(args...), &(0))
        end
    else
        @assert false
    end
end

"""
TODO: doc
FIXME: My, my this function is ugly, clean this up some day...
"""
function push_compute_tasks!(tasks::TaskSet, buffers::Dict,
                             compute_body::Dict, compute_args::ArgSet,
                             run_where::Int, signal::Array{Cint, 1};
                             distribute::Bool=false)
    for phase in [Train, Test]
        args = collect(compute_args[phase])
        if length(args) == 0
            continue
        end
        # BEGIN REPLACE ALIASED ARRAYS
        # TODO: Abstract this into a function
        # When two arrays x and y are aliased such that x === y, we replace all
        # references to y with x to assist the compiler with vectorization
        filtered_args = []
        arg_map = Dict{Symbol, Any}()
        for arg in args
            index = find((x) -> buffers[x] === buffers[arg], filtered_args)
            if length(index) == 0
                push!(filtered_args, arg)
            else
                @assert length(index) == 1
                arg_map[arg] = filtered_args[index[1]]
            end
        end
        compute_body[phase] = replace_symbols(compute_body[phase], arg_map)
        # END REPLACE ALIASED ARRAYS

        arg_bufs = [buffers[arg] for arg in filtered_args]
        signature = tuple([typeof(buf) for buf in arg_bufs]...)
        params = [:($arg::$typ) for (arg, typ) in zip(filtered_args, signature)]
        compute_body[phase] = clean_for_loops(compute_body[phase])
        # println(:(function $compute_task($(params...))
        #     $(compute_body[phase]...)
        # end))
        compute_body[phase] = fuse_loops(compute_body[phase])
        # println(:(function $compute_task($(params...))
        #     $(compute_body[phase]...)
        # end))
        # throw("Err")
        compute_body[phase] = inline_tile_size(compute_body[phase])
        compute_body[phase] = distribute_batch_loops(compute_body[phase])
        compute_body[phase] = parallelize_batch_tile_loops(compute_body[phase])
        compute_body[phase] = fuse_loops(compute_body[phase])

        is_update_param_call = (node) -> isa(node, Expr) && node.head == :call && node.args[1] == :update_param
        to_push = []
        for statement in compute_body[phase]
            if is_update_param_call(statement)
                if length(to_push) > 0
                    compute_task = gensym("compute_task")
                    func = :(function $compute_task($(params...))
                        $(to_push...)
                        return 0
                    end)
                    func = wrap_for_loops(func)
                    func = inline_array_expressions(func, buffers)
                    func = @eval $func
                    # println(code_typed(func, signature))
                    # throw("Err")

                    proxy_func = generate_c_function(func, signature, run_where,
                                                     signal, buffers)
                    proxy_args = arg_bufs

                    push!(tasks[phase], JuliaTask(proxy_func, tuple(filtered_args...)))
                end
                to_push = []
                push!(tasks[phase], UpdateTask(eval(statement.args[2])))
            else
                push!(to_push, statement)
            end
        end
        if length(to_push) > 0
            compute_task = gensym("compute_task")
            func = :(function $compute_task($(params...))
                $(to_push...)
                return 0
            end)
            func = wrap_for_loops(func)
            func = inline_array_expressions(func, buffers)
            func = @eval $func
            # println(code_typed(func, signature))
            # throw("Err")

            proxy_func = generate_c_function(func, signature, run_where,
                                             signal, buffers)
            proxy_args = arg_bufs

            push!(tasks[phase], JuliaTask(proxy_func, tuple(filtered_args...)))
        end
    end
end

"""
TODO: doc
"""
function add_forward_data_tasks(ensemble::DataEnsemble, tasks::TaskSet, net::Net)
    test_args = (ensemble, symbol(ensemble.name, :value), net, Test)
    train_args = (ensemble, symbol(ensemble.name, :value), net, Train)
    push!(tasks[Train], JuliaTask(forward, train_args))
    push!(tasks[Test], JuliaTask(forward, test_args))
end

"""
TODO: doc
"""
function add_forward_julia_tasks(ensemble::JuliaEnsemble, tasks::TaskSet, net::Net)
    inputs = []
    for connection in ensemble.connections
        push!(inputs, symbol(connection.source.name, :value))
    end
    test_args = (ensemble, inputs..., symbol(ensemble.name, :value), Test)
    train_args = (ensemble, inputs..., symbol(ensemble.name, :value), Train)
    push!(tasks[Train], JuliaTask(forward, train_args))
    push!(tasks[Test], JuliaTask(forward, test_args))
end

"""
TODO: doc
"""
function init_forward(ensemble::ReshapeEnsemble, net::Net,
                      compute_args::ArgSet, compute_body::Dict{Phase, Vector})
end

"""
TODO: doc
"""
function init_backward(ensemble::ReshapeEnsemble, net::Net,
                       compute_args::ArgSet, compute_body::Dict{Phase, Vector})
end

"""
TODO: doc
"""
function init_forward(ensemble::ConcatEnsemble, net::Net, compute_args::ArgSet,
                      compute_body::Dict{Phase, Vector})
    asts = []
    output_name = symbol(ensemble.name, :value)
    push!(compute_args[Train], output_name)
    push!(compute_args[Test], output_name)
    for (index, input) in enumerate(ensemble.inputs)
        input_name = symbol(input.name, :value)
        push!(compute_args[Train], input_name)
        push!(compute_args[Test], input_name)
        loopvars = []
        loopranges = []
        for i in size(input)
            push!(loopvars, gensym("loopvar"))
            push!(loopranges, :(1:$i))
        end
        # Add batch loop
        push!(loopvars, gensym("loopvar"))
        push!(loopranges, :(1:$(net.batch_size)))
        index_vars = Any[loopvars[1:end-1]...]
        if index > 1
            index_vars[end] = :($(index_vars[end]) + $((index - 1) * ensemble.inner_size))
        end

        push!(asts, gen_loop_nest(
            :($output_name[$(index_vars...), $(loopvars[end])] = 
                $input_name[$(loopvars...)]), 
            loopvars, loopranges))
    end
    for phase in [Train, Test]
        append!(compute_body[phase], asts)
    end
end

"""
TODO: doc
"""
function init_backward(ensemble::ConcatEnsemble, net::Net,
                       compute_args::ArgSet, compute_body::Dict{Phase, Vector})
    asts = []
    output_name = symbol(ensemble.name, :∇)
    push!(compute_args[Train], output_name)
    for (index, input) in enumerate(ensemble.inputs)
        input_name = symbol(input.name, :∇)
        push!(compute_args[Train], input_name)
        loopvars = []
        loopranges = []
        for i in size(input)
            push!(loopvars, gensym("loopvar"))
            push!(loopranges, :(1:$i))
        end
        # Add batch loop
        push!(loopvars, gensym("loopvar"))
        push!(loopranges, :(1:$(net.batch_size)))
        index_vars = Any[loopvars[1:end-1]...]
        if index > 1
            index_vars[end] = :($(index_vars[end]) + $((index - 1) * ensemble.inner_size))
        end

        push!(asts, gen_loop_nest(
            :($input_name[$(loopvars...)] = 
              $output_name[$(index_vars...), $(loopvars[end])]),
            loopvars, loopranges))
    end
    append!(compute_body[Train], asts)
end

"""
TODO: doc
"""
function init_forward(ensemble::NormalizationEnsemble, net::Net,
                      compute_args::ArgSet, compute_body::Dict{Phase, Vector})
    args = get_forward_args(ensemble)
    union!(compute_args[ensemble.phase], args)

    for connection in ensemble.connections
        arg = symbol(connection.source.name,:value)
        push!(args, arg)
        push!(compute_args[ensemble.phase], arg)
    end

    ast = forward(ensemble)
    symbol_map = Dict{Symbol, Symbol}()
    for (param, arg) in zip(ast.args[1].args[2:end], args)
        if isa(param, Expr) && param.head == :(::)
            param = param.args[1]
        end
        symbol_map[param] = arg
    end
    body = AstWalk(ast.args[2], symbol_replacer, symbol_map)
    push!(compute_body[ensemble.phase], body)
end

"""
TODO: doc
"""
function init_backward(ensemble::NormalizationEnsemble, net::Net,
                       compute_args::ArgSet, compute_body::Dict{Phase, Vector})
    args = get_backward_args(ensemble)
    union!(compute_args[ensemble.phase], args)
    for connection in ensemble.connections
        # Don't backprop to label's ∇, but pass it's value
        if :label == connection.source.name
            arg = symbol(connection.source.name,:value)
        else
            arg = symbol(connection.source.name,:∇)
        end
        push!(args, arg)
        push!(compute_args[ensemble.phase], arg)
    end
    ast = backward(ensemble)
    symbol_map = Dict{Symbol, Symbol}()
    for (param, arg) in zip(ast.args[1].args[2:end], args)
        if isa(param, Expr) && param.head == :(::)
            param = param.args[1]
        end
        symbol_map[param] = arg
    end
    body = AstWalk(ast.args[2], symbol_replacer, symbol_map)
    unshift!(compute_body[ensemble.phase], body)
end

"""
TODO: doc
"""
function add_recv_expr(net::Net, source::AbstractEnsemble,
                       ensemble::AbstractEnsemble, 
                       compute_body::Dict{Phase, Vector}, compute_args::ArgSet)
    key = symbol(source.name, :value)
    source_subgroup = source.net_subgroup - 1  # -1 for zero based indexing with MPI ranks
    tag = find(x -> x == ensemble, net.ensembles)[1]
    expr = :(ccall((:recv_intra, $libComm), Void, 
                   (Ptr{Float32}, Cint, Cint, Cint), 
                   pointer($key), length($key), $tag, $source_subgroup))
    if ensemble.phase in [Train, TrainTest] &&
            connection.source.phase in [Train, TrainTest]
        push!(compute_body[Train], expr)
        push!(compute_args[Train], key)
    end
    if ensemble.phase in [Test, TrainTest] &&
            connection.source.phase in [Train, TrainTest]
        push!(compute_body[Test], expr)
        push!(compute_args[Test], key)
    end
end

"""
TODO: doc
"""
function add_send_exprs(net::Net, ensemble::AbstractEnsemble,
                        compute_body::Dict{Phase, Vector},
                        compute_args::ArgSet)
    for (target, tag) in net.ensemble_send_list[ensemble.name]
        target = target - 1 # 0-based indexing for MPI
        target_phase = net.ensembles[tag].phase
        target_buf = symbol(ensemble.name, :value)
        expr = :(ccall((:send_intra, $libComm), Void, 
                       (Ptr{Float32}, Cint, Cint, Cint), 
                       pointer($target_buf), length($target_buf), $tag, $target))
        if ensemble.phase in [Train, TrainTest] &&
                target_phase in [Train, TrainTest]
            push!(compute_body[Train], expr)
            push!(compute_args[Train], target_buf)
        end
        if ensemble.phase in [Test, TrainTest] &&
                target_phase in [Test, TrainTest]
            push!(compute_body[Test], expr)
            push!(compute_args[Test], target_buf)
        end
    end
end

"""
Initialize the network `net`.

This function begins by initializing each `Ensemble` in the Network.  This is
done by calling `init(ens)` which will be dispatched to the appropriate
initialization routine.  These initialization routines are responsible for
adding buffers to the network to contain neuron fields and output values.  See
the specific initialization functions for different Ensemble types for more
information (TODO: Reference these).  

After initializaing the local fields and output values for each ensemble, we
then initialize the input buffers for each ensemble.  This is done after all
output values have been initialized so that we can analyze recurrent
connections.  This is done by calling the `init_inputs(ens)` routine.
"""
function init(net::Net)
    log_info("Initializing net...")
    forward_tasks = net.forward_tasks
    backward_tasks = net.backward_tasks
    forward_compute_body = Dict{Phase, Vector}(Train => [], Test => [])
    forward_compute_args = ArgSet()
    forward_data_tasks   = TaskSet()
    backward_compute_body = Dict{Phase, Vector}(Train => [], Test => [])
    backward_compute_args = ArgSet()
    seen_names = Set()
    log_info("  Initializing ensembles.")
    for ensemble in net.ensembles
        # Check for duplicate ensemble names
        if ensemble.name in seen_names
            throw("Error: Found duplicate ensemble name: $(ensemble.name)")
        end
        push!(seen_names, ensemble.name)

        # If in MPI mode skip ensembles not assigned to this subrank
        @latte_mpi(if ensemble.net_subgroup != get_net_subrank(net) + 1
            continue  # skip
        end)
        net.ensemble_send_list[ensemble.name] = Tuple{Int, Int}[]

        log_info("    $(ensemble.name) size=$(size(ensemble))")

        # Initialize the ensemble
        init(ensemble, net)
        init_params(ensemble, net)
    end

    for (index, ensemble) in enumerate(net.ensembles)
        # If in MPI mode, populate the send_list for connected ensembles not in
        # this subrank and skip initializations
        @latte_mpi if ensemble.net_subgroup != get_net_subrank(net) + 1
            for connection in ensemble.connections
                if connection.source.net_subgroup == get_net_subrank(net) + 1
                    push!(net.ensemble_send_list[connection.source.name], 
                          (ensemble.net_subgroup, index))
                end
            end
            continue  # skip
        end

        # Initialize the inputs for the ensemble, this is done after all
        # ensembles have initialized their outputs (necessary for rnns)
        init_inputs(ensemble, net)
    end
    log_info("  Finished initializing ensembles.")

    log_info("  Synthesizing forward functions.")
    # Generate forward tasks
    for ensemble in net.ensembles
        @latte_mpi if ensemble.net_subgroup != get_net_subrank(net) + 1
            continue  # skip
        end

        @latte_mpi for connection in ensemble.connections
            if connection.source.net_subgroup != ensemble.net_subgroup
                add_recv_expr(net, connection.source, ensemble,
                              forward_compute_body, forward_compute_args)

            end
        end
        # Skip code generation for data and loss ensembles
        if typeof(ensemble) <: DataEnsemble
            add_forward_data_tasks(ensemble, forward_data_tasks, net)
        elseif typeof(ensemble) <: JuliaEnsemble
            add_forward_julia_tasks(ensemble, forward_data_tasks, net)
        elseif isa(ensemble, Union{NormalizationEnsemble, ConcatEnsemble, ReshapeEnsemble})
            init_forward(ensemble, net, forward_compute_args, forward_compute_body)
        elseif isa(ensemble, Union{Ensemble, ActivationEnsemble})
            gen_neuron_forward(ensemble, net, forward_compute_body, forward_compute_args)
        else
            throw("Latte Error: Encountered unsupported ensemble type $(typeof(ensemble)).")
        end

        @latte_mpi add_send_exprs(net, ensemble, forward_compute_body,
                                  forward_compute_args)
    end
    append!(net.forward_tasks, forward_data_tasks)

    push_compute_tasks!(forward_tasks, net.buffers[1][1],
                        forward_compute_body, forward_compute_args,
                        net.run_where, net.signal; distribute=true)

    log_info("  Synthesizing backward functions.")
    # Backward tasks
    for ensemble in net.ensembles
        @latte_mpi (if ensemble.net_subgroup != get_net_subrank(net) + 1
            continue
        end)
        if typeof(ensemble) <: DataEnsemble || ensemble.phase == Test
            continue  # skip
        end
        for (target, tag) in net.ensemble_send_list[ensemble.name]
            target_phase = net.ensembles[tag].phase
            if !(target_phase in [Train, TrainTest])
                continue
            end
            target = target - 1 # 0-based indexing for MPI
            target_buf = symbol(ensemble.name, :∇)
            tag += length(net.ensembles)
            expr = :(ccall((:recv_intra, $libComm), Void, 
                           (Ptr{Float32}, Cint, Cint, Cint), 
                           pointer($target_buf), length($target_buf), $tag, $target))
            push!(backward_compute_body[Train], expr)
            push!(backward_compute_args[Train], target_buf)
        end
        if isa(ensemble, Union{NormalizationEnsemble, ConcatEnsemble, ReshapeEnsemble})
            init_backward(ensemble, net, backward_compute_args, backward_compute_body)
        elseif typeof(ensemble) <: JuliaEnsemble
            # Skip
            # throw("NotImplementedError")
        elseif isa(ensemble, Union{Ensemble, ActivationEnsemble})
            for param in ensemble.params
                @latte_mpi(
                    unshift!(backward_compute_body[Train], quote
                        ccall((:sync_gradients, $libComm), Void, 
                              (Ptr{Float32}, Cint, Cint), 
                              pointer($(param.gradient_name)),
                              $(length(param.gradient)), 
                              $(param.request))
                    end)
                )
                push!(net.params, param)
            end
            gen_neuron_backward(ensemble, net, backward_compute_body[Train], backward_compute_args[Train])
        else
            throw("Latte Error: Encountered unsupported ensemble type $(typeof(ensemble)).")
        end
        for connection in ensemble.connections
            if :label == connection.source.name
                # Skip
                continue
            end
            if connection.source.net_subgroup != ensemble.net_subgroup
                key = symbol(connection.source.name, :∇)
                source_subgroup = connection.source.net_subgroup - 1  # 0-based indexing for MPI
                tag = find(x -> x == ensemble, net.ensembles)[1] + length(net.ensembles)
                push!(backward_compute_args[Train], key)
                push!(backward_compute_body[Train],
                    :(ccall((:send_intra, $libComm), Void, 
                            (Ptr{Float32}, Cint, Cint, Cint), 
                            pointer($key), length($key), $tag, $source_subgroup)))
            end
        end
    end

    push_compute_tasks!(backward_tasks, net.buffers[1][1], backward_compute_body,
                        backward_compute_args, net.run_where, net.signal)
    if net.run_where >= 0
        @assert false
    end
    log_info("Initialization finished.")
end

function get_task_args(net, task_args, t)
    args = []
    for arg in task_args
        if isa(arg, Symbol)
            push!(args, get_buffer(net, arg, t))
        else
            push!(args, arg)
        end
    end
    args
end

# Use metaprogramming to generate single and multi versions of forward
# and backward.
for direction in [:forward, :backward]
    tasks = symbol(direction,:_tasks)
    @eval function $direction(net::Net; phase=Train, solver=nothing)
        for t = 1:net.time_steps
            net.curr_time_step = t
            for task in net.$tasks[phase]
                if isa(task, JuliaTask)
                    task.func(get_task_args(net, task.args, t)...)
                elseif isa(task, UpdateTask)
                    @assert phase == Train && solver != nothing
                    update(solver, net, task.param_id)
                else
                    throw("Unsupported task type $(typeof(task))")
                end
            end
        end
    end
end

"""
Add an ensemble to the network `net`
"""
function add_ensemble(net::Net, ens::AbstractEnsemble)
    push!(net.ensembles, ens)
    net.ensembles_map[ens.name] = ens
end

"""
TODO: doc
"""
function check_dimensions_fixed(mapping::Function, sink_shape::Tuple)
    n = length(sink_shape)
    is_dim_fixed = [true for _ in 1:n]
    first = mapping(ones(Int, n)...)
    if !all(map((x) -> isa(x, UnitRange) || isa(x, Colon), first))
        is_dim_fixed = [false for _ in 1:n]
    else
        for d in 1:n
            for i in 1:sink_shape[d]
                idx = ones(Int, n)
                idx[d] = i
                if first != mapping(idx...)
                    is_dim_fixed[d] = false
                    break
                end
            end
        end
    end
    is_dim_fixed
end

"""
TODO: doc
"""
function check_one_to_one(mapping::Function, shape::Tuple)
    is_one_to_one = true
    for i in CartesianRange(shape)
        if mapping(i.I...) != i.I
            is_one_to_one = false
            break
        end
    end
    is_one_to_one
end

"""
Connect neurons in `source` to neurons in `sink` using the function `mapping`.
`mapping` should be a function with a parameter for the index in each dimension
of sink.
For example, if sink is a 3-d ensemble, mapping = f(i, j, k) -> ...

`mapping` should return a tuple of continuous ranges corresponding to the
indices of neurons in source that should be connected to the neuron at the
current index
"""
function add_connections(net::Net, source::AbstractEnsemble,
                         sink::AbstractEnsemble, mapping::Function; padding=0,
                         recurrent=false)
    n = ndims(sink)

    # Compute the size and shape of the connection
    range_size = 1
    range_shape = []
    for (index, d) in enumerate(mapping(ones(Int, n)...))
        if isa(d, Colon)
            range_size *= size(source)[index]
            push!(range_shape, size(source)[index])
        else
            range_size *= length(d)
            push!(range_shape, length(d))
        end
    end

    # Determine if any dimensions are fixed
    is_dim_fixed = check_dimensions_fixed(mapping, size(sink))
    is_one_to_one = false
    if !all(is_dim_fixed)
        is_one_to_one = check_one_to_one(mapping, size(sink))
    end
    push!(sink.connections, Connection(source, mapping, tuple(range_shape...),
                                       range_size, true, is_dim_fixed,
                                       is_one_to_one, padding, recurrent))
end

"""
Test `net` for one epoch
"""
function test(net::Net)
    curr_epoch = net.test_epoch
    accuracy = 0.0f0
    num_batches = 0.0f0
    while net.test_epoch == curr_epoch
        forward(net, phase=Test)
        num_batches += 1.0f0
        if haskey(net.buffers[net.curr_buffer_set][1], :accuracyvalue)
            accuracy += get_buffer(net, :accuracyvalue)[1]
        end
        clear_values(net)
        @latte_mpi sync_intra_test_epoch(net)
    end
    accuracy / num_batches * 100.0f0
end

export save_snapshot, load_snapshot

"""
Save a snapshot of `net` to `file`
"""
function save_snapshot(net::Net, file::AbstractString)
    param_dict = Dict{Symbol, Vector{Param}}()
    for ens in net.ensembles
        if :params in fieldnames(ens) && length(ens.params) > 0
            param_dict[ens.name] = ens.params
        end
    end
    save(file, "param_dict", param_dict)
end

"""
Load a network snapshot from `file`.

TODO: Can we save the structure of `net` in the snapshot?
"""
function load_snapshot(net::Net, file::AbstractString)
    param_dict = load(file, "param_dict")
    for ens in net.ensembles
        if :params in fieldnames(ens) && length(ens.params) > 0
            for (param, saved) in zip(ens.params, param_dict[ens.name])
                param.value[:] = saved.value[:]
            end
        end
    end
end

"""
Get the current loss for `net`

TODO: This is not general, assumes one :loss ensemble
"""
function get_loss(net::Net)
    @latte_mpi(if haskey(net.buffers[net.curr_buffer_set][1], :lossvalue)
        loss = get_buffer(net, :lossvalue)[1]
        sync_intra_loss(net, loss)
        return loss
    else
        return sync_intra_loss(net, 0.0f0)
    end, 
    return get_buffer(net, :lossvalue)[1])
end

"""
Fill buffers with names containing `∇` with zeros.
"""
function clear_∇(net::Net)
    i = net.curr_buffer_set
    for t in 1:net.time_steps
        for name in keys(net.buffers[i][t])
            if contains(string(name), "∇")
                fill!(net.buffers[i][t][name], 0.0)
            end
        end
    end
end

"""
Fill buffers with names containing `value` with zeros.
"""
function clear_values(net::Net)
    i = net.curr_buffer_set
    for t in 1:net.time_steps
        for name in keys(net.buffers[i][t])
            if contains(string(name), "value")
                fill!(net.buffers[i][t][name], 0.0)
            end
        end
    end
end

"""
Fill buffers with names containing `randval` with random values
"""
function rand_values(net::Net)
    i = net.curr_buffer_set
    for t in 1:net.time_steps
        for name in keys(net.buffers[i][t])
            if contains(string(name), "randval")
                rand!(net.buffers[i][t][name])
            end
        end
    end
end

"""
Get a buffer associated with an ensemble

** Params **

- `net`   -- network to get buffer
- `ens`   -- the ensemble
- `name`  -- name of buffer associated with `ens`
"""
function get_buffer(net::Net, ens::AbstractEnsemble, name::Symbol)
    return net.buffers[net.curr_buffer_set][1][symbol(ens.name, name)]
end

"""
Get a buffer at the time_step `t`

** Params **

- `net`   -- network to get buffer
- `name`  -- name of the buffer
- `t`     -- time step
"""
function get_buffer(net::Net, name::Symbol, t::Int=1)
    return net.buffers[net.curr_buffer_set][t][name]
end

"""
Add or update a buffer at a particular time step `t`

** Params **

- `net`   -- network to add/update buffer
- `name`  -- name of the buffer
- `arr`   -- buffer
- `t`     -- time step to add buffer
"""
function set_buffer(net::Net, name::Symbol, arr::Array, t::Int)
    net.buffers[1][t][name] = arr
    net.buffers[2][t][name] = arr
end

"""
Add or update a buffer

** Params **

- `net`   -- network to add/update buffer
- `name`  -- name of the buffer
- `arr`   -- buffer
- `_copy` -- whether to copy the buffer
"""
function set_buffer(net::Net, name::Symbol, arr::Array; _copy=true)
    for t in 1:net.time_steps
        if _copy
            net.buffers[1][t][name] = copy(arr)
            net.buffers[2][t][name] = copy(arr)
        else
            net.buffers[1][t][name] = arr
            net.buffers[2][t][name] = arr
        end
    end
end

"""
Initialize a buffer in `net`

** Params **

- `net`   -- network to receive initialized buffer
- `name`  -- name of the buffer
- `shape` -- shape of the buffer
- `func`  -- function used to initialize the buffer, should return an Array and
             have a signature (Float32, dims...)
"""
function init_buffer(net::Net, name::Symbol, shape; func=zeros)
    for t in 1:net.time_steps
        net.buffers[1][t][name] = func(Float32, shape...)
        net.buffers[2][t][name] = func(Float32, shape...)
    end
end
