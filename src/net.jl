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

export Net, init, forward, backward, clear_∇, clear_values, add_ensemble,
       add_connections, copy_from_mic, copy_to_mic, get_buffer,
       get_param, get_value, get_gradient
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

"""
Construct a Dict{Symbol, Any} that maps the arguments of the function
`ast` to the correspondnig value in Vector `args`.
"""
function build_arg_name_map(args::Vector, ast::Expr)
    @assert is_function(ast)
    map = Dict()
    for (index, arg) in enumerate(ast.args[1].args[2:end])
        map[arg.args[1]] = args[index]
    end
    map
end

include("transforms/fixed_dims.jl")

function distribute_batch_loops(statements)
    new_body = []
    for statement in statements
        if isa(statement, Expr) && statement.head == :for
            for expr in statement.args[2].args
                loop = deepcopy(statement)
                if isa(expr, LineNumberNode) || isa(expr, LabelNode) || expr.head == :line
                    continue
                end
                loop.args[2].args = [expr]
                push!(new_body, loop)
            end
        else
            push!(new_body, statement)
        end
    end
    new_body
end

include("optimizers/tiling.jl")
include("optimizers/fusion.jl")
include("optimizers/gemm_pattern_match.jl")
include("optimizers/array_expr_inline.jl")
include("optimizers/wrap_for_loops.jl")
include("optimizers/parallelize.jl")

function optimize(args::Vector, tile_fusion_factors::Vector, fn)
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
    # FIXME: Above pattern matches should be cleaned up to ingest the "clean"
    # ast like fuse_loops below
    ast = clean_tree(ast)
    ast.args[2].args = fuse_loops(ast.args[2].args)
    debugp(1, "----- Begin : Function After Optimization  -----")
    debugp(1, ast)
    debugp(1, "----- End   : Function After Optimization  -----")
    return ast
end

include("transforms/neuron.jl")

function clear_∇(net::Net)
    for t in 1:net.time_steps
        for name in keys(net.buffers[t])
            if contains(string(name), "∇")
                fill!(net.buffers[t][name], 0.0)
            end
        end
    end
end

function clear_values(net::Net)
    for t in 1:net.time_steps
        for name in keys(net.buffers[t])
            if contains(string(name), "value")
                fill!(net.buffers[t][name], 0.0)
            end
        end
    end
end

function update_rand(net::Net)
    for t in 1:net.time_steps
        for name in keys(net.buffers[t])
            if contains(string(name), "rand")
                rand!(net.buffer[t]s[name])
            end
        end
    end
end

function get_param(net::SingleNet, name::Symbol)
    for param in net.params
        if param.name == name
            return param
        end
    end
    throw("Param $name not found")
end

function get_buffer(net::Net, ens::AbstractEnsemble, name::Symbol)
    return net.buffers[1][symbol(ens.name, name)]
end

function get_buffer(net::SingleNet, name::Symbol, t::Int=1)
    return net.buffers[t][name]
end

function set_buffer(net::SingleNet, name::Symbol, arr::Array, t::Int)
    net.buffers[t][name] = arr
end

function set_buffer(net::SingleNet, name::Symbol, arr::Array; _copy=true)
    for t in 1:net.time_steps
        if _copy
            net.buffers[t][name] = copy(arr)
        else
            net.buffers[t][name] = arr
        end
    end
end

"""
Generates a loopnest with `body` as the body of the inner most loop using
`vars` as a list of loop variables and `ranges` as a list of ranges for each
loop nest.

Example:
    julia> gen_loop_nest(:(println((a, b))), [:a, :b], [1:2, 1:3])
    :(for b = 1:3
          for a = 1:2
              println((a, b))
          end
      end)
"""
function gen_loop_nest(body, vars, ranges)
    nest = body
    for (var, range) in zip(vars, ranges)
        nest = :(for $var = $range
            $nest
        end)
    end
    nest
end

"""
Synthesizes a loop nest around body based on the dimensionality of `buffer`.
We split each statement in `statements` into a separate synthesized loop nest
to facilitate pattern matching of statements.  If no pattern matching occurs,
the identical loop nests will be fused later.
"""
function append_neuron_loop!(body::Vector, statements::Vector, buffer::Array)
    N = ndims(buffer)
    for statement in statements
        vars   = [symbol(:_neuron_index_,i) for i in 1:N]
        ranges = [:(1:$(size(buffer,i)))    for i in 1:N]
        push!(body, gen_loop_nest(statement, vars, ranges))
    end
end

mk_clamp(idx, _max) = :($idx > 0 && $idx <= $_max)

function get_src_idx(mapping)
    if isa(mapping.args[end], Expr) &&
            mapping.args[end].head == :call &&
            isa(mapping.args[end].args[1], TopNode)
        return mapping.args[end].args[2:end]
    else
        return [mapping.args[end]]
    end
end

"""
Generate a loop nest to perform the copy from the source of `connection`
to an input buffer for `ensemble`
"""
function gen_copy_block_inputs(net::Net, ensemble::AbstractEnsemble,
                               connection::Connection, index::Int)
    value = get_buffer(net, ensemble, :value)
    N = ndims(value)
    sink_name   = symbol(ensemble.name,:inputs,index)
    source_name = symbol(connection.source.name,:value)
    if connection.recurrent
        source_name = symbol(source_name, :prev)
    end
    source      = get_buffer(net, source_name)

    # connection.is_dim_fixed is a Bool[], we use the inverse to select non
    # fixed indices
    non_fixed_indices = [collect(1:N-1)[!connection.is_dim_fixed]..., N]

    sink_idx = [symbol(:_neuron_index_,d) for d in non_fixed_indices]

    mapping_inlined = inline(connection.mapping, [symbol(:_neuron_index_,d) for d in 1:N-1])
    src_idx = get_src_idx(mapping_inlined)
    body = Expr(:block)
    idx = [symbol(:_u__, i) for i = 1:length(src_idx)]

    for_block = gen_loop_nest(body, idx, src_idx)

    rhs = :($source_name[$(idx...), $(symbol(:_neuron_index_,N))])
    if connection.padding > 0
        cond = mk_clamp(idx[1], size(source, 1))
        for (i, d) in zip(idx[2:end], size(source)[2:end-1])
            cond = :($cond && $(mk_clamp(i, d)))
        end
        rhs = :(ifelse($cond, $rhs, 0.0f0))
    end
    append!(body.args, (quote
        $sink_name[count, $(sink_idx...)] = $rhs
        count += 1
    end).args)
    copy_block = quote
        $(mapping_inlined.args[1:end-1]...)
        count = 1
        $for_block
    end
    vars   = [symbol(:_neuron_index_,i) for i in non_fixed_indices]
    ranges = [:(1:$(size(value,i)))     for i in non_fixed_indices]
    gen_loop_nest(copy_block, vars, ranges)
end

function gen_neuron_forward(ensemble::AbstractEnsemble, net::Net, compute_body,
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
            push!(body, gen_copy_block_inputs(net, ensemble, connection,
                index))
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
        append!(compute_body[Train], deepcopy(body))
        union!(compute_args[Train], args)
    end
    if ensemble.phase in [Test, TrainTest]
        append!(compute_body[Test], deepcopy(body))
        union!(compute_args[Test], args)
    end
end

function gen_copy_block_∇inputs(net::Net, ensemble::AbstractEnsemble,
                                connection::Connection, index::Int)
    ∇ = get_buffer(net, ensemble, :∇)
    N = ndims(∇)
    sink_name   = symbol(ensemble.name, :∇inputs, index)
    source_name = symbol(connection.source.name, :∇)
    source      = get_buffer(net, connection.source, :∇)
    non_fixed_indices = [collect(1:N-1)[!connection.is_dim_fixed]..., N]
    sink_idx = [symbol(:_neuron_index_,d) for d in non_fixed_indices]
    mapping_args = [connection.is_dim_fixed[d] ? 1 : symbol(:_neuron_index_,d) for d in 1:N-1]

    mapping_inlined = inline(connection.mapping, mapping_args)
    src_idx = get_src_idx(mapping_inlined)

    for_block = Expr(:for, Expr(:block), Expr(:block))
    block = Expr(:block)
    idx = [symbol(:_u__, i) for i = 1:length(src_idx)]
    for_block = gen_loop_nest(block, idx, src_idx)

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
    append!(block.args, (quote
        $assign
        $sink_name[count, $(sink_idx...)] = 0.0f0
        count += 1
    end).args)
    copy_block = quote
        $(mapping_inlined.args[1:end-1]...)
        count = 1
        $for_block
    end
    vars   = [symbol(:_neuron_index_,i) for i in non_fixed_indices]
    ranges = [:(1:$(size(∇,i)))     for i in non_fixed_indices]
    return gen_loop_nest(copy_block, vars, ranges)
end

function gen_neuron_backward(ensemble::AbstractEnsemble, net::Net,
                             compute_body, compute_args::Set)
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
        if isa(connection.source, DataEnsemble) || !connection.copy
            continue
        end
        source_name = symbol(connection.source.name, :∇)
        push!(args, source_name)
        push!(body, gen_copy_block_∇inputs(net, ensemble, connection, index))
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
    if ensemble.phase in [Train, TrainTest]
        prepend!(compute_body, body)
        union!(compute_args, args)
    end
end

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
    s = CGen.from_root_entry(ast, function_name_string)
    proxy_name = string("_", function_name_string, "_j2c_proxy")
    proxy_sym = symbol(proxy_name)
    j2c_name = string("_", function_name_string, "_unaliased_")
    args = [:($(symbol(:__arg_,d))) for d in 1:length(signature)]
    if run_where >= 0
        @assert false
    end
    if LATTE_MPI
        s = "#include \"comm.h\"\n" * s
        outfile_name = CGen.writec(s)
        CGen.compile(outfile_name; flags=["-I$latte_library_path/communication"])
        dyn_lib = CGen.link(outfile_name; flags=["-L$latte_library_path", "-lLatteComm"])
    else
        outfile_name = CGen.writec(s)
        CGen.compile(outfile_name)
        dyn_lib = CGen.link(outfile_name)
    end

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

include("transforms/tree_cleaners.jl")

function push_compute_tasks!(tasks::TaskSet, buffers::Dict,
                             compute_body::Dict, compute_args::ArgSet,
                             run_where::Int, signal::Array{Cint, 1};
                             distribute=false)
    for phase in [Train, Test]
        args = collect(compute_args[phase])
        if length(args) == 0
            continue
        end
        compute_task = gensym("compute_task")
        arg_bufs = [buffers[arg] for arg in args]
        signature = tuple([typeof(buf) for buf in arg_bufs]...)
        params = [:($arg::$typ) for (arg, typ) in zip(args, signature)]
        push!(compute_body[phase], :(return 0))
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
        func = :(function $compute_task($(params...))
            $(compute_body[phase]...)
        end)
        # println(func)
        func = wrap_for_loops(func)
        func = inline_array_expressions(func, buffers)
        func = @eval $func
        # println(code_typed(func, signature))
        # throw("Err")

        proxy_func = generate_c_function(func, signature, run_where,
                                         signal, buffers)
        proxy_args = arg_bufs

        push!(tasks[phase], JuliaTask(proxy_func, tuple(args...)))
    end
end

function add_forward_data_tasks(ensemble::DataEnsemble, tasks::TaskSet, net::Net)
    test_args = (ensemble, symbol(ensemble.name, :value), net, Test)
    train_args = (ensemble, symbol(ensemble.name, :value), net, Train)
    push!(tasks[Train], JuliaTask(forward, train_args))
    push!(tasks[Test], JuliaTask(forward, test_args))
end

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

function init_forward(ensemble::NormalizationEnsemble, compute_args::ArgSet,
                      compute_body)
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

function init_backward(ensemble::NormalizationEnsemble, compute_args::ArgSet, compute_body)
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


function init(net::SingleNet)
    log_info("Initializing net...")
    forward_tasks = net.forward_tasks
    backward_tasks = net.backward_tasks
    forward_compute_body = Dict{Phase, Vector}(Train => [], Test => [])
    forward_compute_args = ArgSet()
    forward_data_tasks   = TaskSet()
    backward_compute_body = Dict{Phase, Vector}(Train => [], Test => [])
    backward_compute_args = ArgSet()
    seen_names = Set()
    # Initialize ensembles
    for ensemble in net.ensembles
        if ensemble.name in seen_names
            throw("Error: Found duplicate ensemble name: $(ensemble.name)")
        end
        push!(seen_names, ensemble.name)
        map(init, ensemble)
        init(ensemble, net)
    end
    for ensemble in net.ensembles
        init_inputs(ensemble, net)
    end
    # Generate forward tasks
    for ensemble in net.ensembles
        # Skip code generation for data and loss ensembles
        if typeof(ensemble) <: DataEnsemble
            add_forward_data_tasks(ensemble, forward_data_tasks, net)
        elseif typeof(ensemble) <: JuliaEnsemble
            add_forward_julia_tasks(ensemble, forward_data_tasks, net)
        elseif typeof(ensemble) <: NormalizationEnsemble
            init_forward(ensemble, forward_compute_args, forward_compute_body)
        else
            gen_neuron_forward(ensemble, net, forward_compute_body, forward_compute_args)
        end
    end
    append!(net.forward_tasks, forward_data_tasks)

    push_compute_tasks!(forward_tasks, net.buffers[1],
                        forward_compute_body, forward_compute_args,
                        net.run_where, net.signal; distribute=true)

    # Backward tasks
    for ensemble in net.ensembles
        if typeof(ensemble) <: DataEnsemble || ensemble.phase == Test
            # Don't backprop on data ensembles
            continue
        elseif typeof(ensemble) <: NormalizationEnsemble
            init_backward(ensemble, backward_compute_args, backward_compute_body)
        elseif typeof(ensemble) <: JuliaEnsemble
            # Not implemented yet
        else
            for param in ensemble.params
                param.value = get_buffer(net, param.name)
                gradient = get_buffer(net, param.gradient_name)
                param.gradient = gradient
                param.hist = zeros(param.value)
                println(param.name)
                if LATTE_MPI
                    param.request = @eval ccall((:init_request, $libComm), Cint, ())
                    unshift!(backward_compute_body[Train],quote
                        ccall((:sync_gradients, $libComm), Void, (Ptr{Float32}, Ptr{Float32}, Cint, Cint), pointer($(param.gradient_name)), pointer($(param.name)), $(length(param.gradient)), $(param.request));
                        #ccall((:sync_gradients, $libComm), Void, (Ptr{Float32}, Cint, Cint), pointer($(param.gradient_name)), $(length(param.gradient)), $(param.request));
                    end)
                end
                push!(net.params, param)
            end
            aa = @eval ccall((:print_stat, $libComm), Cint, ())
            gen_neuron_backward(ensemble, net, backward_compute_body[Train], backward_compute_args[Train])
        end
    end

    push_compute_tasks!(backward_tasks, net.buffers[1], backward_compute_body,
                        backward_compute_args, net.run_where, net.signal)
    if net.run_where >= 0
        @assert false
    end
end

# function init(net::RNN)
#     init(net.subnet)
#     push!(net.buffers, net.subnet.buffers)
#     # for t = 2:net.time_steps
#     #     push!(net.buffers, deepcopy(net.subnet.buffers))
#     #     predicate = (x) -> :connections in fieldnames(x)
#     #     for ensemble in filter(predicate, net.subnet.ensembles)
#     #         conn_pred = (x) -> x[2].recurrent
#     #         for (index, connection) in filter(conn_pred, enumerate(ensemble.connections))
#     #             target = symbol(connection.source.name,:value)
#     #             if connection.copy
#     #                 net.buffers[t][symbol(target, :prev)] = net.buffers[t-1][target]
#     #             else
#     #                 inputs = symbol(ensemble.name,:inputs, index)
#     #                 net.buffers[t][inputs] = reshape(net.buffers[t-1][target],
#     #                     (connection.size, net.batch_size))
#     #             end
#     #         end
#     #     end
#     # end
# end

# Use metaprogramming to generate single and multi versions of forward
# and backward.
for direction in [:forward, :backward]
    tasks = symbol(direction,:_tasks)
    @eval function $direction(net::SingleNet; phase=Train, t=1)
        for (index, task) in enumerate(net.$tasks[phase])
            args = []
            for arg in task.args
                if isa(arg, Symbol)
                    push!(args, get_buffer(net, arg, t))
                else
                    push!(args, arg)
                end
            end
            task.func(args...)
        end
    end
end

# function forward(net::RNN)
#     for t = 1:net.time_steps
#         net.subnet.buffers = net.buffers[t]
#         forward(net.subnet)
#     end
# end

"""
Add an ensemble to the network `net`
"""
function add_ensemble(net::Net, ens::AbstractEnsemble)
    push!(net.ensembles, ens)
    net.ensembles_map[ens.name] = ens
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
                         sink::AbstractEnsemble, mapping::Function; padding=0, recurrent=false)
    n = ndims(sink)
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
    is_dim_fixed = [true for _ in 1:n]
    first = mapping(ones(Int, n)...)
    if !all(map((x) -> isa(x, UnitRange) || isa(x, Colon), first))
        is_dim_fixed = [false for _ in 1:n]
    else
        for d in 1:n
            for i in 1:size(sink, d)
                idx = ones(Int, n)
                idx[d] = i
                if first != mapping(idx...)
                    is_dim_fixed[d] = false
                    break
                end
            end
        end
    end
    push!(sink.connections, Connection(source, mapping, tuple(range_shape...),
                                       range_size, true, is_dim_fixed, padding, recurrent))
end

function test(net::SingleNet)
    curr_epoch = net.test_epoch
    accuracy = 0.0f0
    num_batches = 0.0f0
    while net.test_epoch == curr_epoch
        forward(net, phase=Test)
        num_batches += 1.0f0
        accuracy += get_buffer(net, :accuracyvalue)[1]
        clear_values(net)
    end
    accuracy / num_batches * 100.0f0
end

# import Base.show
# function Base.show(io::IO, net::Net)
#     println(io, "Net")
#     println(io, "Ensembles")
#     println(io, "---------")
#     for ensemble in net.ensembles
#         println(io, "    ", AbstractString(ensemble))
#     end
#     println(io, " ")
#     println(io, "Forward Tasks")
#     println(io, "-------------")
#     for task in net.forward_tasks
#         println(io, "    ", task.func)
#     end
# end

export save_snapshot, load_snapshot

function save_snapshot(net::Net, file::AbstractString)
    param_dict = Dict{Symbol, Vector{Param}}()
    for ens in net.ensembles
        if :params in fieldnames(ens) && length(ens.params) > 0
            param_dict[ens.name] = ens.params
        end
    end
    save(file, "param_dict", param_dict)
end

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
