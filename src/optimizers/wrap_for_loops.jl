# Copyright (c) 2015 Intel Corporation. All rights reserved.
function for_loop_wrapper(node, cbdata, index, top_level, read)
    if is_for(node)
        loopvar = node.args[1].args[1]
        range = node.args[1].args[2]
        if is_call(range) && is_call_target(range, :colon)
            (start, stop) = range.args[2:3]
        elseif !is_colon(range)
            return ASTWALK_RECURSE
        else
            (start, stop) = range.args[1:2]
        end
        body = node.args[2]
        # Wrap nested loops
        map!(wrap_for_loops, body.args)
        # Inline constant loop bounds
        if is_num(start) && is_num(stop)
            return quote
                $loopvar = _REMOVE_THIS_LINE()
                $(Expr(:loophead, loopvar, start, stop))
                $(body.args...)
                $(Expr(:loopend, loopvar))
            end
        end
        start_var = gensym("latte_loop_start")
        stop_var = gensym("latte_loop_stop")
        return quote
            $loopvar = _REMOVE_THIS_LINE()
            $start_var = $start
            $stop_var = $stop
            $(Expr(:loophead, loopvar, start_var, stop_var))
            $(body.args...)
            $(Expr(:loopend, loopvar))
        end
    elseif isa(node, Expr) && node.head in [:parallel_loophead, :loopend, :parallel_loopend]
        return node
    end
    ASTWALK_RECURSE
end

function wrap_for_loops(expr)
    AstWalk(expr, for_loop_wrapper, nothing)
end
