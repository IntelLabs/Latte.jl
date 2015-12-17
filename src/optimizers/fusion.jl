# Copyright (c) 2015 Intel Corporation. All rights reserved.
function fuse_loops(block::Vector{Any})
    first = block[1]
    new_block = Any[first]
    if isa(first, Expr)
        if first.head == :for
            first.args[2].args = fuse_loops(first.args[2].args)
        elseif first.head == :block
            first.args = fuse_loops(first.args)
        end
    end
    for statement in block[2:end]
        if isa(statement, Expr) && statement.head == :for
            if isa(new_block[end], Expr) && new_block[end].head == :for
                curr_loop = statement
                prev_loop = new_block[end]
                curr_len = curr_loop.args[1].args[2]
                prev_len = prev_loop.args[1].args[2]
                if curr_len == prev_len
                    prev_loopvar = get_loopvar(prev_loop)
                    curr_loopvar = get_loopvar(curr_loop)
                    curr_body = AstWalk(curr_loop.args[2], symbol_replacer, Dict(curr_loopvar => prev_loopvar)).args
                    append!(prev_loop.args[2].args, curr_body)
                    prev_loop.args[2].args = fuse_loops(prev_loop.args[2].args)
                elseif is_tiled_loop(prev_loop) && is_tiled_loop(curr_loop) && !LATTE_DISABLE_TILE_FUSION
                    forward_factor = get_tile_fusion_factor_forward(curr_len)
                    backward_factor = get_tile_fusion_factor_backward(prev_len)
                    prev_end = prev_len.args[2].args[2]
                    curr_end = curr_len.args[2].args[2]
                    if curr_end * forward_factor == prev_end
                        prev_len.args[2].args[2] = div(prev_end, forward_factor)
                        if forward_factor > 1
                            unshift!(prev_loop.args[2].args, :(TILE_SIZE *= $forward_factor))
                            push!(prev_loop.args[2].args, :(TILE_SIZE /= $forward_factor))
                        end
                        append!(prev_loop.args[2].args, curr_loop.args[2].args)
                        prev_loop.args[2].args = fuse_loops(prev_loop.args[2].args)
                    elseif curr_end == prev_end * backward_factor
                        if backward_factor > 1
                            unshift!(curr_loop.args[2].args, :(TILE_SIZE *= $backward_factor))
                            push!(curr_loop.args[2].args, :(TILE_SIZE /= $backward_factor))
                        end
                        prev_len.args[4] *= get_tile_fusion_factor_backward(curr_len)
                        append!(prev_loop.args[2].args, curr_loop.args[2].args)
                        prev_loop.args[2].args = fuse_loops(prev_loop.args[2].args)
                    else
                        statement.args[2].args = fuse_loops(statement.args[2].args)
                        push!(new_block, statement)
                    end
                else
                    statement.args[2].args = fuse_loops(statement.args[2].args)
                    push!(new_block, statement)
                end
            else
                statement.args[2].args = fuse_loops(statement.args[2].args)
                push!(new_block, statement)
            end
        else
            push!(new_block, statement)
        end
    end
    filter((x) -> !(isa(x, Symbol) && x == :NOFUSE), new_block)
end
