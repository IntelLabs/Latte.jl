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
