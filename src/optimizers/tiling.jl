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

"""
Convert from an internal Latte tiled_loop to a normal Julia :for expression
"""
function unpack_tiled_loop(node::Expr)
    @assert node.head == :for
    range = node.args[1].args[2]
    node.args[1].args[2] = range.args[2]
    node
end

"""
Check if `node` is an internal Latte tiled_loop
"""
function is_tiled_loop(node)
    if !(isa(node, Expr) && node.head == :for)
        return false
    end
    range = node.args[1].args[2]
    isa(range, Expr) && range.head == :call && range.args[1] == :tiled_loop
end

"""
TODO: doc
"""
function get_tile_fusion_factor_forward(expr)
    return expr.args[3]
end

"""
TODO: doc
"""
function get_tile_fusion_factor_backward(expr)
    return expr.args[4]
end

"""
Replace indexing expressions with `tile_var` with the proper tiled expression
When `inputs` are not copied, the neuron transformer appends :NOTILE to the
index expression to force Latte not to tile the expression.
"""
function update_tile_var(node, tile_var, index, top_level, read)
    if isa(node, Expr) && node.head == :ref && contains(string(node.args[1]), "inputs")
        if node.args[end] != :NOTILE
            for i in 2:length(node.args)-1
                if node.args[i] == tile_var
                    node.args[i] = :(($(node.args[i]) - 1) % TILE_SIZE + 1)
                end
            end
            node.args[end] = :(_omp_get_thread_num() + 1)
        end
        return node
    end
    ASTWALK_RECURSE
end

"""
TODO: doc
"""
function inner_loop_tiler(node, cbdata, index, top_level, read)
    if isa(node, Expr) && node.head == :for && isa(get_loopvar(node), Symbol)
        loopvar = string(get_loopvar(node))
        if contains(loopvar, "_neuron_index_") && parse(Int, loopvar[end]) == 2
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

"""
TODO: doc
"""
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

"""
TODO: doc
"""
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
        elseif node.head == :ref && node.args[end] == :NOTILE
            pop!(node.args)
            return node
        elseif node.head in [:loophead, :parallel_loophead, :loopend, :parallel_loopend]
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
    elseif isa(node, Expr) && node.head in [:loophead, :parallel_loophead, :loopend, :parallel_loopend]
        return node
    end
    ASTWALK_RECURSE
end
