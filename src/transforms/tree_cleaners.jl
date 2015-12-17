# Copyright (c) 2015 Intel Corporation. All rights reserved.
export clean_for_loops

function clean_for_loops(statements)
    function walker(node, cbdata, index, top_level, read)
        if is_for(node) && length(node.args[2].args) == 1 &&
                isa(node.args[2].args[1], Expr) &&
                node.args[2].args[1].head == :block
            node.args[2].args = clean_for_loops(node.args[2].args[1].args)
            return node
        end
        ASTWALK_RECURSE
    end
    map((s) -> AstWalk(s, walker, nothing), statements)
end
