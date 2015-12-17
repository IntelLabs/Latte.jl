# Copyright (c) 2015 Intel Corporation. All rights reserved.
export drop_fixed_dims
"""
Drop indexing expressions along dimension for an array if the compiler
has determined it to be uniform.  `cbdata` should be a Dict{Symbol,
Vector{Bool}} that maps array names to their uniformity.
"""
function drop_fixed_dims(ast, arg_dim_info)
    function walker(node, cbdata::Dict{Symbol, Vector{Bool}}, index, top_level, read)
        if is_ref(node) && haskey(cbdata, node.args[1])
            info = cbdata[node.args[1]]
            count = 1
            # First argument is reference name so we skip it
            idx = Any[node.args[1]]
            for arg in node.args[2:end]
                if isa(arg, Symbol) && contains(string(arg), "_neuron_index")
                    if count > length(info) || !info[count]
                        push!(idx, arg)
                    end
                    count += 1
                else
                    push!(idx, arg)
                end
            end
            node.args = idx
            return node
        end
        ASTWALK_RECURSE
    end
    AstWalk(ast, walker, arg_dim_info)
end
