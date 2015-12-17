# Copyright (c) 2015 Intel Corporation. All rights reserved.
function unpack_tiled_loop(node)
    range = node.args[1].args[2]
    node.args[1].args[2] = range.args[2]
    node
end

function is_tiled_loop(node)
    if !(isa(node, Expr) && node.head == :for)
        return false
    end
    range = node.args[1].args[2]
    isa(range, Expr) && range.head == :call && range.args[1] == :tiled_loop
end

function get_tile_fusion_factor_forward(expr)
    return expr.args[3]
end

function get_tile_fusion_factor_backward(expr)
    return expr.args[4]
end

function update_tile_var(node, cbdata, index, top_level, read)
    if isa(node, Expr) && node.head == :ref && contains(string(node.args[1]), "inputs")
        for i in 2:length(node.args)-1
            if node.args[i] == cbdata
                node.args[i] = :(($(node.args[i]) - 1) % TILE_SIZE + 1)
            end
        end
        node.args[end] = :(_omp_get_thread_num() + 1)
        # println(node)
        # throw("err")
        return node
    end
    ASTWALK_RECURSE
end

function inner_loop_tiler(node, cbdata, index, top_level, read)
    if isa(node, Expr) && node.head == :for && isa(get_loopvar(node), Symbol)
        loopvar = string(get_loopvar(node))
        if contains(loopvar, "_neuron_index_") && parse(Int, loopvar[end]) == 2 # || parse(Int, loopvar[end]) == 3
            loop_idx = parse(Int, loopvar[end])
            loop_range = node.args[1].args[2]
            loop_end = loop_range.args[2]
            unshift!(cbdata[:loop_lengths], loop_end)
            unshift!(cbdata[:loop_vars], symbol(loopvar, "_tile_idx"))
            loop_range.args[1] = :((($(symbol(loopvar, "_tile_idx")) - 1) * TILE_SIZE + 1))
            loop_range.args[2] = :(min($(symbol(loopvar, "_tile_idx")) * TILE_SIZE, $loop_end))
            node.args[2] = AstWalk(node.args[2], update_tile_var, get_loopvar(node))
            return node
        end
    end
    ASTWALK_RECURSE
end

global unique_id = 0
function get_inner_loop_tiler(tile_fusion_factors)
    global unique_id += 1
    function tile_inner_loops(statement)
        if isa(statement, Expr) && statement.head == :for
            loopvar = string(get_loopvar(statement))
            if contains(loopvar, "_neuron_index_") && parse(Int, loopvar[end]) >= 4
                loop_lengths = []
                loop_vars = []
                cbdata = Dict(:loop_lengths => loop_lengths, 
                              :TILE_SIZE => TILE_SIZE,
                              :loop_vars => loop_vars)
                statement = AstWalk(statement, inner_loop_tiler, cbdata)
                body = statement.args[2]
                num_tiles = []
                for len in loop_lengths
                    n_tiles = div(len, TILE_SIZE)
                    if len % TILE_SIZE != 0
                        n_tiles += 1
                    end
                    push!(num_tiles, n_tiles)
                end
                tile_loop = quote
                    $(body.args...)
                end
                for (loopvar, n_tiles, factor) in zip(loop_vars, num_tiles, tile_fusion_factors)
                    tile_loop = quote
                        for $loopvar = tiled_loop(1:$n_tiles, $(factor...), $unique_id)
                            $(tile_loop.args...)
                        end
                    end
                end
                statement.args[2] = tile_loop
            end
        end
        statement
    end
end

function tile_size_inliner(node, cbdata, index, top_level, read)
    if isa(node, Expr)
        if node.head == :(*=) && node.args[1] == :TILE_SIZE
            cbdata[1] *= node.args[2]
            return ASTWALK_REMOVE
        elseif node.head == :(/=) && node.args[1] == :TILE_SIZE
            cbdata[1] /= node.args[2]
            return ASTWALK_REMOVE
        elseif is_tiled_loop(node)
            node = unpack_tiled_loop(node)
            node.args[2] = AstWalk(node.args[2], tile_size_inliner, cbdata)
            node.args[1] = AstWalk(node.args[1], tile_size_inliner, cbdata)
            return node
        end
    elseif isa(node, Symbol) && node == :TILE_SIZE
        return cbdata[1]
    end
    ASTWALK_RECURSE
end

function inline_tile_size(statements)
    map((x) -> AstWalk(x, tile_size_inliner, [TILE_SIZE]), statements)
end

"""
Tile loop variables use `_tile_idx` loopvars.
"""
function get_tile_loops(node, cbdata, index, top_level, read)
    if isa(node, Expr) && node.head == :for && isa(get_loopvar(node), Symbol)
        loopvar = string(get_loopvar(node))
        if contains(loopvar, "_tile_idx")
            push!(cbdata, node)
        end
    end
    ASTWALK_RECURSE
end
