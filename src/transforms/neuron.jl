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

export transform_neuron_fn
"""
Data used by `neuron_fn_transformer` during the ASTWalk process

    `args`  : Stores the neuron fields encountered in the body of the function
              which should be passed as arguments to the generated function
"""
type NeuronTransformerData
    ensemble     :: AbstractEnsemble
    args         :: Set
    NeuronTransformerData(ensemble::AbstractEnsemble) =
        new(ensemble, Set())
end

@doc """
Process a neuron function replacing field references with array reference
expressions
"""
function transform_neuron_fn(fn, ensemble)
    function walker(node, cbdata, index, top_level, read)
        if !isa(node, Expr)
            return ASTWALK_RECURSE
        end
        if node.head == :(.) && node.args[1] == :neuron
            name = node.args[2].value
            N = ndims(cbdata.ensemble)
            if name == :index
                # N += 1  # batch_dim
                idx = :($(symbol(:_neuron_index_,N)) - 1)
                buffer = symbol(cbdata.ensemble.name, :value)
                for i in N-1:-1:1
                    size = :(size($buffer, $i))
                    idx = :($idx * $size + $(symbol(:_neuron_index_,i)) - 1)
                end
                return :($idx + 1)
            end
            if name in cbdata.ensemble.batch_fields
                N += 1
            end
            name = symbol(cbdata.ensemble.name,name)
            idx = Any[symbol(:_neuron_index_,i) for i in 1:N]
            str_name = string(name)
            # if contains(str_name, "∇")
            #     result = split(str_name, "∇")
            #     if !(result[2] == "" || contains(result[2], "inputs"))
            #         push!(idx, :(_omp_get_thread_num() + 1))
            #     end
            # end
            if !contains(str_name, "inputs")
                push!(cbdata.args, name)
            end
            return :($name[$(idx...)])
        elseif node.head == :ref
            for i in 2:length(node.args)
                node.args[i] = AstWalk(node.args[i], walker, cbdata)
            end
            result = AstWalk(node.args[1], walker, cbdata)
            str_target = string(result.args[1])
            if endswith(str_target, "inputs")
                if isa(cbdata.ensemble, ActivationEnsemble)
                    if contains(str_target, "∇")
                        result.args[1] = symbol(cbdata.ensemble.name,:∇)
                    else
                        result.args[1] = symbol(cbdata.ensemble.name,:value)
                    end
                else
                    result.args[1] = symbol(result.args[1], node.args[2])
                end
                push!(cbdata.args, result.args[1])
                node = Expr(:ref, result.args[1], node.args[3:end]..., result.args[2:end]...)
            else
                node = Expr(:ref, result.args[1], node.args[2:end]..., result.args[2:end]...)
            end
            return node
        elseif node.head == :call && node.args[1] == :length
            node.args[2] = AstWalk(node.args[2], walker, cbdata)
            if isa(node.args[2], Expr) && node.args[2].head == :ref
                if contains(node.args[2].args[1], "inputs")
                    return Expr(:call, :size, node.args[2].args[1], 1)
                end
            end
        end
        ASTWALK_RECURSE
    end
    cbdata = NeuronTransformerData(ensemble)
    ast = AstWalk(fn, walker, cbdata)
    ast, cbdata.args
end
