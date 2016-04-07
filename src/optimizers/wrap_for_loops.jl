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
    elseif isa(node, Expr) && node.head in [:parallel_loophead, :loophead, :loopend, :parallel_loopend]
        return node
    end
    ASTWALK_RECURSE
end

function wrap_for_loops(expr)
    AstWalk(expr, for_loop_wrapper, nothing)
end
