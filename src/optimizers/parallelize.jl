g=
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

function parallelize_batch_tile_loops(statements)
    new_body = []
    for batch_loop in statements
        if isa(batch_loop, Expr) && batch_loop.head == :for
            batch_body = batch_loop.args[2]
            batch_body.args = fuse_loops(batch_body.args)
            batch_range = batch_loop.args[1].args[2]
            num_micro_batches = cld(batch_range.args[2], MICRO_BATCH_SIZE)
            batch_loopvar = batch_loop.args[1].args[1]
            statement = batch_body.args[1]
            tile_loops = []
            AstWalk(batch_loop, get_tile_loops, tile_loops)
            loopvars = [batch_loopvar]
            body = batch_body
            # batch_range = :((micro_batch - 1) * $MICRO_BATCH_SIZE + 1:
            #                 min(micro_batch * $MICRO_BATCH_SIZE, $(batch_range.args[2])))
            ranges = [batch_range]
            if length(tile_loops) > 0 && length(batch_body.args) == 1
                for loop in tile_loops
                    push!(ranges, loop.args[1].args[2])
                    push!(loopvars, loop.args[1].args[1])
                end
                body = tile_loops[end].args[2]
            end
            pre = []
            for var in loopvars
                push!(pre, :($var = _REMOVE_THIS_LINE()))
            end
            unique = gensym("loop_ranges")
            start_vars = Union{Symbol,Int}[]
            stop_vars = Union{Symbol,Int}[]
            for (i, range) in enumerate(ranges)
                # range_var = symbol(unique,:range,i)
                # push!(pre, :($range_var = $range))
                start_var = symbol(unique,:start,i)
                push!(pre, :($start_var = $(range.args[1])))
                push!(start_vars, start_var)
                stop_var = symbol(unique,:stop,i)
                push!(pre, :($stop_var = $(range.args[2])))
                push!(stop_vars, stop_var)
            end
            push!(new_body, quote# for micro_batch = 1:$num_micro_batches
                $(pre...)
                $(mk_parallel_loophead(loopvars, start_vars, stop_vars;schedule="schedule(static,1)"))
                $(body.args...)
                $(Expr(:parallel_loopend, length(loopvars)))
                # parallel_for($(ranges...)) do $(loopvars...)
                #     $body
                # end
            # end))
            end)
        else
            push!(new_body, batch_loop)
        end
    end
    new_body
end

function collect_parallel_loop_private_vars(expr)
    function private_var_collector(node, cbdata, index, top_level, read)
        if cbdata[1] != nothing
            if isa(node, Expr) && node.head == :(=)
                lhs = node.args[1]
                if isa(lhs, Symbol)
                    push!(cbdata[1], lhs)
                elseif isa(lhs, SymbolNode)
                    push!(cbdata[1], lhs.name)
                elseif isa(lhs, GenSym)
                    push!(cbdata[1], lhs)
                end
                return node
            elseif isa(node, Expr) && node.head == :loophead
                private_var_collector(node.args[1], cbdata, index, top_level, read)
                return node
            end
        end
        if isa(node, Expr) && node.head == :parallel_loophead
            cbdata[1] = node.args[4]
            return node
        elseif isa(node, Expr) && node.head == :parallel_loopend
            cbdata[1] = nothing
            return node
        elseif isa(node, Expr) && node.head in [:loopend, :loophead]
            return node
        end
        ASTWALK_RECURSE
    end
    AstWalk(expr, private_var_collector, Any[nothing])
end
