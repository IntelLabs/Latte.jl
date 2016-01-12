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

function get_flattened_index(node)
    @assert isa(node, Expr) && node.head == :ref
    idx = []
    if isexpr(node.args[1]) && node.args[1].head == :ref
        append!(idx, get_flattened_index(node.args[1]))
    end
    append!(idx, node.args[2:end])
    idx
    # filter((i) -> i != 1, idx)
end

function search_for_symbol(node, cbdata, index, top_level, read)
    if isa(node, Symbol) && node == cbdata[:expected]
        cbdata[:found] = true
    end
    ASTWALK_RECURSE
end

function contains_index(actual_idxs, expected_idxs)
    function check_contains_index(actual, expected)
        cbdata = Dict(:expected => expected, :found => false)
        AstWalk(actual, search_for_symbol, cbdata)
        cbdata[:found]
    end
    all(map(check_contains_index, actual_idxs, expected_idxs))
end

function get_target(A, offsets)
    @assert isa(A, Expr) && A.head == :ref
    idx = A.args[2:end]
    offset = 0
    if length(idx) > length(offsets)
        for i in length(idx):-1:length(offsets)+1
            if offset == 0
                offset = :(($(idx[i]) - 1) * size($(A.args[1]), $(i - 1)))
            else
                offset = :(($offset + ($(idx[i]) - 1)) * size($(A.args[1]), $(i - 1)))
            end
        end
    end
    for i in length(offsets):-1:1
        if offsets[i] != 1
            inner_size = 1
            for d in 1:i - 1
                inner_size = :(size($(A.args[1]), $d) * $inner_size)
            end
            offset = :(($offset + ($(offsets[i]) - 1)))
        end
        if i > 1 && offset != 0
            offset = :($offset * size($(A.args[1]), $(i - 1)))
        end
    end
    return :(ParallelAccelerator.API.pointer($(A.args[1]), $offset))
end
function collapse_blocks(node)
    body = []
    for arg in node.args
        if isa(arg, Expr) && arg.head == :block
            append!(body, collapse_blocks(arg))
        elseif arg != :nothing
            push!(body, arg)
        end
    end
    body
end

function isa_gemm_statement(statement)
    return isa(statement, Expr) &&
           statement.head == :(+=) &&
           isa(statement.args[2], Expr) &&
           statement.args[2].head == :call &&
           statement.args[2].args[1] == :(*) &&
           length(statement.args[2].args) == 3
end

function replace_index_with_expr(expr, idx, replace_with)
    function walker(node, cbdata, index, top_level, read)
        if isa(node, Symbol) && haskey(cbdata, node)
            return cbdata[node]
        end
        ASTWALK_RECURSE
    end
    cbdata = Dict(idx => replace_with)
    AstWalk(expr, walker, cbdata)
end

simple_ijk_orders = [(:i, :j, :k), (:j, :k, :i), (:j, :i, :k), (:k, :j, :i),
                     (:i, :k, :j), (:k, :i, :j)]
for (var1, var2, var3) in simple_ijk_orders
    fn = symbol(:pattern_match_gemm, var1, var2, var3)
    @eval function $fn(node, cbdata, index, top_level, read)
        @recurse_unless isa(node, Expr)
        @recurse_unless node.head == :for
        $(symbol(var1,:_loop)) = node
        $(symbol(var1,:_loopvar)) = get_loopvar(node)

        $(symbol(var2,:_loop)) = collapse_blocks($(symbol(var1,:_loop)).args[2])[1]
        @recurse_unless $(symbol(var2,:_loop)).head == :for
        $(symbol(var2,:_loopvar)) = get_loopvar($(symbol(var2,:_loop)))

        $(symbol(var3,:_loop)) = collapse_blocks($(symbol(var2,:_loop)).args[2])[1]
        @recurse_unless $(symbol(var3,:_loop)).head == :for
        $(symbol(var3,:_loopvar)) = get_loopvar($(symbol(var3,:_loop)))
        N = get_loop_length(i_loop)
        M = get_loop_length(j_loop)
        K = get_loop_length(k_loop)

        @recurse_unless length($(symbol(var3,:_loop)).args[2].args) == 1
        accum_stmt = $(symbol(var3,:_loop)).args[2].args[1]

        @recurse_unless isa_gemm_statement(accum_stmt)

        accum_target = accum_stmt.args[1]
        C = accum_target
        @recurse_unless length(get_flattened_index(accum_target)) >= 2
        C_idx = get_flattened_index(accum_target)
        @recurse_unless contains_index(C_idx[1:2], [j_loopvar, i_loopvar])

        i_start = get_loop_start(i_loop)
        j_start = get_loop_start(j_loop)
        k_start = get_loop_start(k_loop)

        (A, B) = accum_stmt.args[2].args[2:3]
        @recurse_unless length(get_flattened_index(A)) >= 2
        @recurse_unless length(get_flattened_index(B)) >= 2
        A_idx = get_flattened_index(A)[1:2]
        B_idx = get_flattened_index(B)[1:2]
        if contains_index(B_idx, [k_loopvar, i_loopvar])
            transB = 'N'
            B_offset = map(replace_index_with_expr, B_idx,[k_loopvar, i_loopvar], [k_start, i_start])
        elseif contains_index(B_idx, [i_loopvar, k_loopvar])
            transB = 'T'
            B_offset = map(replace_index_with_expr, B_idx,[i_loopvar, k_loopvar], [i_start, k_start])
        else
            return ASTWALK_RECURSE
        end
        if contains_index(A_idx, [j_loopvar, k_loopvar])
            transA = 'N'
            A_offset = map(replace_index_with_expr, A_idx, [j_loopvar, k_loopvar], [j_start, k_start])
        elseif contains_index(A_idx, [k_loopvar, j_loopvar])
            transA = 'T'
            A_offset = map(replace_index_with_expr, A_idx, [k_loopvar, j_loopvar], [k_start, j_start])
        else
            return ASTWALK_RECURSE
        end
        C_offset = map(replace_index_with_expr, C_idx, [j_loopvar, i_loopvar], [j_start, i_start])
        # if length(C_idx) > 2 && C_idx[3] == k_loopvar &&
        #         contains(string(C.args[1]), "∇") && 
        #         length(split(string(C.args[1]), "∇")[2]) > 0
        #     push!(C_offset, 1)
        # end
        return :(gemm!($transA, $transB, $M, $N, $K, 1.0f0,
                       $(get_target(A, A_offset)), size($(A.args[1]), 1),
                       $(get_target(B, B_offset)), size($(B.args[1]), 1), 1.0f0,
                       $(get_target(C, C_offset)), size($(C.args[1]), 1)))
    end
end

"""
The case of the "flattened" j loop
"""
function pattern_match_gemm4(node, cbdata, index, top_level, read)
    @recurse_unless isa(node, Expr)
    @recurse_unless node.head == :for
    i_loopvar = get_loopvar(node)
    i_loop = node

    j1_loop = collapse_blocks(node.args[2])[1]
    @recurse_unless j1_loop.head == :for
    j1_loopvar = get_loopvar(j1_loop)

    j2_loop = collapse_blocks(j1_loop.args[2])[1]
    @recurse_unless j2_loop.head == :for
    j2_loopvar = get_loopvar(j2_loop)

    k_loop = collapse_blocks(j2_loop.args[2])[1]
    @recurse_unless k_loop.head == :for
    k_loopvar = get_loopvar(k_loop)

    N = get_loop_length(i_loop)
    M1 = get_loop_length(j1_loop)
    M2 = get_loop_length(j2_loop)
    M = :($M1 * $M2)
    K = get_loop_length(k_loop)

    @recurse_unless length(k_loop.args[2].args) == 1
    accum_stmt = k_loop.args[2].args[1]

    @recurse_unless isa_gemm_statement(accum_stmt)

    accum_target = accum_stmt.args[1]
    C = accum_target
    @recurse_unless length(get_flattened_index(accum_target)) >= 3
    (j2, j1, i) = get_flattened_index(accum_target)[1:3]
    C_idx = [j2, j1, i]
    @recurse_unless contains_index(C_idx, [j2_loopvar, j1_loopvar, i_loopvar])

    i_start = get_loop_start(i_loop)
    j1_start = get_loop_start(j1_loop)
    j2_start = get_loop_start(j2_loop)
    k_start = get_loop_start(k_loop)

    (B, A) = accum_stmt.args[2].args[2:3]
    @recurse_unless length(get_flattened_index(A)) >= 3
    A_idx = get_flattened_index(A)[1:3]
    B_idx = get_flattened_index(B)[1:2]
    if contains_index(B_idx, [k_loopvar, i_loopvar])
        transB = 'N'
        B_offset = map(replace_index_with_expr, B_idx,[k_loopvar, i_loopvar], [k_start, i_start])
    elseif contains_index(B_idx, [i_loopvar, k_loopvar])
        transB = 'T'
        B_offset = map(replace_index_with_expr, B_idx,[i_loopvar, k_loopvar], [i_start, k_start])
    else
        return ASTWALK_RECURSE
    end
    if contains_index(A_idx, [j2_loopvar, j1_loopvar, k_loopvar])
        transA = 'N'
        lda = :(size($(A.args[1]), 1) * size($(A.args[1]), 2))
        A_offset = map(replace_index_with_expr, A_idx, [j2_loopvar, j1_loopvar, k_loopvar], [j2_start, j2_start, k_start])
    elseif contains_index(A_idx, [k_loopvar, j2_loopvar, j1_loopvar])
        transA = 'T'
        lda = :(size($(A.args[1]), 1))
        A_offset = map(replace_index_with_expr, A_idx, [k_loopvar, j2_loopvar, j1_loopvar], [k_start, j2_start, j2_start])
    else
        return ASTWALK_RECURSE
    end
    C_offset = map(replace_index_with_expr, C_idx, [j2_loopvar, j1_loopvar, i_loopvar], [j2_start, j1_start, i_start])
    return :(gemm!($transA, $transB, $M, $N, $K, 1.0f0,
             $(get_target(A, A_offset)), $lda,
             $(get_target(B, B_offset)), size($(B.args[1]), 1), 1.0f0,
             $(get_target(C, C_offset)), size($(C.args[1]), 1) * size($(C.args[1]), 2)))
end

"""
The case of the "flattened" k loop
"""
function pattern_match_gemm5(node, cbdata, index, top_level, read)
    @recurse_unless isa(node, Expr)
    @recurse_unless node.head == :for
    i_loopvar = get_loopvar(node)
    i_loop = node

    k1_loop = collapse_blocks(node.args[2])[1]
    @recurse_unless k1_loop.head == :for
    k1_loopvar = get_loopvar(k1_loop)

    k2_loop = collapse_blocks(k1_loop.args[2])[1]
    @recurse_unless k2_loop.head == :for
    k2_loopvar = get_loopvar(k2_loop)

    j_loop = collapse_blocks(k2_loop.args[2])[1]
    @recurse_unless j_loop.head == :for
    j_loopvar = get_loopvar(j_loop)

    N = get_loop_length(i_loop)
    M = get_loop_length(j_loop)
    K1 = get_loop_length(k1_loop)
    K2 = get_loop_length(k2_loop)
    K = :($K1 * $K2)

    @recurse_unless length(j_loop.args[2].args) == 1
    accum_stmt = j_loop.args[2].args[1]

    @recurse_unless isa_gemm_statement(accum_stmt)

    accum_target = accum_stmt.args[1]
    C = accum_target
    @recurse_unless length(get_flattened_index(accum_target)) >= 2
    C_idx = get_flattened_index(accum_target)[1:2]
    @recurse_unless contains_index(C_idx, [j_loopvar, i_loopvar])
    (j, i) = C_idx

    i_start = get_loop_start(i_loop)
    j_start = get_loop_start(j_loop)
    k1_start = get_loop_start(k1_loop)
    k2_start = get_loop_start(k2_loop)

    (A, B) = accum_stmt.args[2].args[2:3]
    @recurse_unless length(get_flattened_index(A)) >= 3
    @recurse_unless length(get_flattened_index(B)) >= 3
    A_idx = get_flattened_index(A)[1:3]
    B_idx = get_flattened_index(B)[1:3]
    if contains_index(B_idx, [k2_loopvar, k1_loopvar, i_loopvar])
        transB = 'N'
        ldb = :(size($(B.args[1]), 1) * size($(B.args[1]), 2))
        B_offset = map(replace_index_with_expr, B_idx, [k2_loopvar, k1_loopvar, i_loopvar], [k2_start, k1_start, i_start])
    elseif contains_index(B_idx, [i_loopvar, k2_loopvar, k1_loopvar])
        transB = 'T'
        ldb = :(size($(B.args[1]), 1))
        B_offset = map(replace_index_with_expr, B_idx, [i_loopvar, k2_loopvar, k1_loopvar], [i_start, k2_start, k1_start])
    else
        return ASTWALK_RECURSE
    end
    if contains_index(A_idx, [j_loopvar, k2_loopvar, k1_loopvar])
        transA = 'N'
        lda = :(size($(A.args[1]), 1))
        A_offset = map(replace_index_with_expr, A_idx, [j_loopvar, k2_loopvar, k1_loopvar], [j_start, k2_start, k2_start])
    elseif contains_index(A_idx, [k2_loopvar, k1_loopvar, j_loopvar])
        transA = 'T'
        lda = :(size($(A.args[1]), 1) * size($(A.args[1]), 2))
        A_offset = map(replace_index_with_expr, A_idx, [k2_loopvar, k1_loopvar, j_loopvar], [k2_start, k2_start, j_start])
    else
        return ASTWALK_RECURSE
    end
    C_offset = map(replace_index_with_expr, C_idx, [j_loopvar, i_loopvar], [j_start, i_start])
    return :(gemm!($transA, $transB, $M, $N, $K, 1.0f0,
             $(get_target(A, A_offset)), $lda,
             $(get_target(B, B_offset)), $ldb, 1.0f0,
             $(get_target(C, C_offset)), size($(C.args[1]), 1)))
end

function pattern_match_gemm6(node, cbdata, index, top_level, read)
    @recurse_unless isa(node, Expr)
    @recurse_unless node.head == :for
    k_loopvar = get_loopvar(node)
    k_loop = node

    i1_loop = collapse_blocks(node.args[2])[1]
    @recurse_unless i1_loop.head == :for
    i1_loopvar = get_loopvar(i1_loop)

    i2_loop = collapse_blocks(i1_loop.args[2])[1]
    @recurse_unless i2_loop.head == :for
    i2_loopvar = get_loopvar(i2_loop)

    j_loop = collapse_blocks(i2_loop.args[2])[1]
    @recurse_unless j_loop.head == :for
    j_loopvar = get_loopvar(j_loop)

    N1 = get_loop_length(i1_loop)
    N2 = get_loop_length(i2_loop)
    N = :($N1 * $N2)
    M = get_loop_length(j_loop)
    K = get_loop_length(k_loop)

    @recurse_unless length(j_loop.args[2].args) == 1
    accum_stmt = j_loop.args[2].args[1]

    @recurse_unless isa_gemm_statement(accum_stmt)

    accum_target = accum_stmt.args[1]
    C = accum_target
    @recurse_unless length(get_flattened_index(accum_target)) >= 3
    C_idx = get_flattened_index(accum_target)[1:3]
    @recurse_unless contains_index(C_idx, [j_loopvar, i2_loopvar, i1_loopvar])

    i1_start = get_loop_start(i1_loop)
    i2_start = get_loop_start(i2_loop)
    j_start = get_loop_start(j_loop)
    k_start = get_loop_start(k_loop)

    (A, B) = accum_stmt.args[2].args[2:3]
    @recurse_unless length(get_flattened_index(B)) >= 3
    A_idx = get_flattened_index(A)[1:2]
    B_idx = get_flattened_index(B)[1:3]
    if contains_index(B_idx, [k_loopvar, i2_loopvar, i1_loopvar])
        transB = 'N'
        ldb = :(size($(B.args[1]), 1))
        B_offset = map(replace_index_with_expr, B_idx, [k_loopvar, i2_loopvar, i1_loopvar], [k_start, i2_start, i1_start])
    elseif contains_index(B_idx, [i2_loopvar, i1_loopvar, k_loopvar])
        transB = 'T'
        ldb = :(size($(B.args[1]), 1) * size($(B.args[1]), 2))
        B_offset = map(replace_index_with_expr, B_idx, [i2_loopvar, i1_loopvar, k_loopvar], [i2_start, i1_start, k_start])
    else
        return ASTWALK_RECURSE
    end
    if contains_index(A_idx, [j_loopvar, k_loopvar])
        transA = 'N'
        A_offset = map(replace_index_with_expr, A_idx, [j_loopvar, k_loopvar], [j_start, k_start])
    elseif contains_index(A_idx, [k_loopvar, j_loopvar])
        transA = 'T'
        A_offset = [k_start, j_start]
        A_offset = map(replace_index_with_expr, A_idx, [k_loopvar, j_loopvar], [k_start, j_start])
    else
        return ASTWALK_RECURSE
    end
    C_offset = map(replace_index_with_expr, C_idx, [j_loopvar, i2_loopvar, i1_loopvar], [j_start, i2_start, i1_start])
    return :(gemm!($transA, $transB, $M, $N, $K, 1.0f0,
             $(get_target(A, A_offset)), size($(A.args[1]), 1),
             $(get_target(B, B_offset)), $ldb, 1.0f0,
             $(get_target(C, C_offset)), size($(C.args[1]), 1)))
end
