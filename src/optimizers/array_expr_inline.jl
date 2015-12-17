# Copyright (c) 2015 Intel Corporation. All rights reserved.
function array_expr_inliner(node, cbdata, index, top_level, read)
    if isa(node, Expr)
        if node.head == :ref
            arr = cbdata[node.args[1]]
            idxs = node.args[2:end]
            map!((i) -> AstWalk(i, array_expr_inliner, cbdata), idxs)
            if length(idxs) == ndims(arr)
                flattened_idx = :($(idxs[end]) - 1)
                for i in length(idxs)-1:-1:1
                    flattened_idx = :(($(idxs[i]) - 1) + $(size(arr, i)) * ($flattened_idx))
                end
                node.args = [node.args[1], :(1 + $flattened_idx)]
            else
                # 1-d indexing
                @assert length(idxs) == 1
            end
            return node
        elseif node.head == :call && node.args[1] == :size
            arr = cbdata[node.args[2]]
            dim = node.args[3]
            return size(arr, dim)
        elseif node.head == :call && node.args[1] == :length
            if haskey(cbdata, node.args[2])
                arr = cbdata[node.args[2]]
                return length(arr)
            end
        elseif node.head in [:loophead, :parallel_loophead, :loopend, :parallel_loopend]
            return node
        end
    end
    ASTWALK_RECURSE
end

function inline_array_expressions(expr, env)
    AstWalk(expr, array_expr_inliner, env)
end
