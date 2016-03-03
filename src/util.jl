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

export set_debug_level, remove_line_nodes, gradient_check

debug_level = parse(Int, get(ENV, "LATTE_DEBUG_LEVEL", "0"))

function debugp(level::Int, args...)
    global debug_level
    if debug_level >= level
        println(args...)
    end
end

function set_debug_level(level::Int)
    global debug_level
    debug_level = level
end

macro recurse_unless(pred)
    quote
        if !$pred
            return ASTWALK_RECURSE
        end
    end
end

function isexpr(node)
    isa(node, Expr)
end

function line_node_remover(node, cbdata, index, top_level, read)
    if isa(node, LineNumberNode) || isa(node, Expr) && node.head == :line
        return ASTWALK_REMOVE
    end
    return ASTWALK_RECURSE
end

function remove_line_nodes(ast)
    AstWalk(ast, line_node_remover, nothing)
end

function Base.contains(symbol::Symbol, value::AbstractString)
    return contains(string(symbol), string(value))
end

"""
Replaces symbols in keys(cdata) with cbdata[symbol]
"""
function symbol_replacer(node, cbdata, index, top_level, read)
    if isa(node, Symbol)
        if haskey(cbdata, node)
            return cbdata[node]
        end
    elseif isa(node, Expr) && node.head == :return
        return AstWalk(node.args[1], symbol_replacer, cbdata)
    elseif isa(node, Expr) && node.head in [:loophead, :loopend, :parallel_loophead, :parallel_loopend]
        for i in 1:length(node.args)
            if isa(node.args[i], Array)
                node.args[i] = map!((x) ->AstWalk(x, symbol_replacer, cbdata), node.args[i])
            elseif isa(node.args[i], Set)
                # skip
            else
                node.args[i] = AstWalk(node.args[i], symbol_replacer, cbdata)
            end
        end
        return node
    end
    ASTWALK_RECURSE
end

"""
Inline a function with a set of arguments `args`.
"""
function inline(func, args)
    ast = Base.uncompressed_ast(func.code)
    env = Dict{Symbol, Any}()
    for (closure_var, value) in zip(ast.args[2][2], func.env)
        if isa(value, Box)
            # FIXME: Hack for Boxed vars that should be unboxed properly
            value = value.contents
        end
        env[closure_var[1]] = value
    end
    for (param, arg) in zip(ast.args[1], args)
        env[param.args[1]] = arg
    end
    # Replace arguments and escaped variables with their values
    AstWalk(ast.args[3], symbol_replacer, env)
end

intel_runtime = "/home/truongle/pse-hpc/intel-julia/intel-runtime/lib/libintel-runtime.so"

double_buffer = true
# @eval ccall((:pert_init,$intel_runtime), Cint, (Cint,), convert(Cint, double_buffer))
# @eval function pert_shutdown()
#   ccall((:pert_shutdown, $intel_runtime), Cint, ())
# end
# atexit(pert_shutdown)

using Base.LinAlg
import Base.LinAlg: BlasInt

@eval function new_sgemm(order, transA, transB,
    M::BlasInt, N::BlasInt, K::BlasInt, alpha::Float32,
    Ap::Ptr{Float32}, lda::BlasInt, Bp::Ptr{Float32}, ldb::BlasInt,
    beta::Float32, Cp::Ptr{Float32}, ldc::BlasInt)
  ccall((:new_sgemm_LD, $intel_runtime), Ptr{Void},
    (UInt8, UInt8, UInt8, BlasInt, BlasInt, BlasInt, Float32, Ptr{Float32}, BlasInt,
     Ptr{Float32}, BlasInt, Float32, Ptr{Float32}, BlasInt),
    order, transA, transB, M, N, K, alpha, Ap, lda, Bp, ldb, beta, Cp, ldc)
end

@eval function insert_task(obj, flag)
  ccall((:insert_task, $intel_runtime), BlasInt, (Ptr{Void},Cint), obj, flag)
end

elty = Float32
const blas = Base.libblas_name
function gemm!(transA::Char, transB::Char, M::Int, N::Int, K::Int,
               alpha::Float32, A::Ptr{Float32}, lda::Int,
               B::Ptr{Float32}, ldb::Int, beta::Float32,
               C::Ptr{Float32}, ldc::Int)
    # lda = (transA == 'N') ? M : K
    # ldb = (transB == 'N') ? K : N
    # ldc = M
    CblasNoTrans = 111
    CblasTrans = 112
    _transA = transA == 'N' ? CblasNoTrans : CblasTrans
    _transB = transB == 'N' ? CblasNoTrans : CblasTrans
    CblasColMajor = 102
    ccall((:cblas_sgemm, blas), Void,
        (Clonglong, Clonglong, Clonglong, Clonglong, Clonglong,
        Clonglong, Float32, Ptr{Float32}, Clonglong,
        Ptr{Float32}, Clonglong, Float32, Ptr{Float32},
        Clonglong),
        CblasColMajor, _transA, _transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
end

function tree_cleaner(node, cbdata, index, top_level, read)
    if isa(node, Expr)
        # Don't collapse :block nodes for nodes that need one as args[2]
        if node.head == :for || node.head == :function || node.head == :if
            new_args = []
            for arg in node.args[2].args
                result = clean_tree(arg)
                if !(isa(result, Symbol) && result == :nothing)
                    if isa(result, Array)
                        append!(new_args, result)
                    else
                        push!(new_args, result)
                    end
                end
            end
            node.args[2].args = new_args
            return node
        # Collapse all other block nodes
        # TODO: There are probably other blocks we shouldn't collapse
        elseif node.head == :block
            map!(clean_tree, node.args)
            return node
        end
    end
    return ASTWALK_RECURSE
end

function clean_tree(ast)
    ast = AstWalk(ast, tree_cleaner, nothing)
    remove_line_nodes(ast)
end

function log_info(args...)
    _time = string(Libc.strftime("%d-%b %H:%M:%S",time())," - ")
    if LATTE_MPI
        rank = @eval ccall((:get_rank, $libComm), Cint, ())
        Base.info(_time, "RANK $rank: ",  args...)
    else
        Base.info(_time, args...)
    end
end

function _omp_get_thread_num()
    return ccall((:omp_get_thread_num, "libiomp"), Cint, ())
end

for head in [:(:for), :(:call), :(:function), :(:ref)]
    fn_name = symbol(:is_,@eval $head)
    @eval function $fn_name(expr::Expr)
        return expr.head == $head
    end

    @eval function $fn_name(expr::Any)
        return false
    end
end

function is_call_target(call_node::Expr, name::Symbol)
    target = call_node.args[1]
    if isa(target, GlobalRef)
        return target.name == name
    else
        return target == name
    end
end

is_colon(expr::Expr) = expr.head == :(:)
is_colon(expr::Any)  = false

is_num(value::Any) = isa(value, Number)

# BEGIN - :for helpers
# These are utility methods for working with :for ast nodes.

function get_loopvar(for_node::Expr)
    @assert is_for(for_node) "Expected a :for expression"
    for_node.args[1].args[1]
end

function get_loop_length(for_node::Expr)
    @assert is_for(for_node) "Expected a :for expression"
    return :(length($(for_node.args[1].args[2])))
end

function get_loop_start(for_node::Expr)
    @assert is_for(for_node) "Expected a :for expression"
    for_node.args[1].args[2].args[1]
end

# END - :for helpers

@noinline function _REMOVE_THIS_LINE()
    return 0
end

function exp{T}(arg::T)
    ccall((:exp, "libm"), T, (T, ), arg)
end

function log{T}(arg::T)
    ccall((:log, "libm"), T, (T, ), arg)
end

function tanh{T}(arg::T)
    ccall((:tanh, "libm"), T, (T, ), arg)
end

function gradient_check(f::Function, inputs::Vector{Array}, grad_outputs::Vector{Array}, eps=1e-3)
    grads = [zeros(x) for x in inputs]
    for (x, gx) in zip(inputs, grads)
        flat_x = flatten_view(x)
        flat_gx = flatten_view(gx)
        for i in 1:length(flat_x)
            orig = flat_x[i]
            flat_x[i] = orig + eps
            ys1 = [copy(x) for x in f()]
            flat_x[i] = orig - eps
            ys2 = [copy(x) for x in f()]
            flat_x[i] = orig

            for (y1, y2, gy) in zip(ys1, ys2, grad_outputs)
                dot = float(sum(((y1 - y2) .* gy)[:]))
                flat_gx[i] += dot / (2 * eps)
            end
        end
    end
    return grads
end

macro NotImplemented()
    return :(throw("Not Implemented"))
end

function remove_temp_nodes(ast)
    function walker(node, cbdata, index, top_level, read)
        if isa(node, Expr) && node.head == :(=) &&
           isa(node.args[2], Expr) && node.args[2].head == :call &&
           isa(node.args[2].args[1], GlobalRef) &&
           node.args[2].args[1].mod == Latte &&
           node.args[2].args[1].name == :_REMOVE_THIS_LINE
            return ASTWALK_REMOVE
        elseif isa(node, Expr) && node.head in [:parallel_loophead,
                                                :parallel_loopend,
                                                :loophead, :loopend]
            return node
        end
        ASTWALK_RECURSE
    end
    AstWalk(ast, walker, nothing)
end

@noinline function pointer(args...)
    Base.pointer(args...)
end

function transform_to_raw_array(ast)
    function walker(node, cbdata, index, top_level, read)
        if isa(node, Expr) && node.head == :call
            if isa(node.args[1], GlobalRef)
                if node.args[1].name == :arrayref
                    node.args[1] = :raw_arrayref
                    return node
                elseif node.args[1].name == :arrayset
                    node.args[1] = :raw_arrayset
                    return node
                elseif node.args[1].name == :pointer
                    node.args[1] = :raw_pointer
                    return node
                end
            end
        elseif isa(node, Expr) && node.head in [:parallel_loophead,
                                                :parallel_loopend,
                                                :loophead, :loopend]
            return node
        end
        ASTWALK_RECURSE
    end
    AstWalk(ast, walker, nothing)
end

"""
Defines @expr only if LATTE_MPI is enabled
"""
macro latte_mpi(expr)
    LATTE_MPI ? esc(expr) : nothing
end

