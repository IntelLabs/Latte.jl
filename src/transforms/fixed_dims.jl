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
