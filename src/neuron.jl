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

export Neuron, neuron, @neuron
abstract Neuron

function init(neuron::Neuron)
end

"""
Specify a neuron function using the syntax
```
@neuron forward(neuron::NeuronSubType) do
    ...
end
```
"""
macro neuron(expr)
    @assert isa(expr, Expr) "@neuron macro expects an expression"
    if expr.head == :call
        @eval function $(expr.args[1])($(expr.args[3:end]...))
            return $(Expr(:quote, expr.args[2].args[2]))
        end
    elseif expr.head == :type
        type_defn = expr.args[2]
        expr.args[2] = Expr(:(<:), type_defn, :Neuron)
        original = remove_line_nodes(copy(expr.args[3])).args
        T = Float32
        block = expr.args[3].args
        unshift!(block, :(∇inputs :: DenseArray{$T}))
        unshift!(block, :(inputs  :: DenseArray{$T}))
        unshift!(block, :(∇       :: $T))
        unshift!(block, :(value   :: $T))
        args = [a.args[1] for a in original]
        defn = quote
            function $type_defn($(original...))
                $type_defn(zero($T), zero($T), Array($T, 1), Array($T, 1),
                           $(args...))
            end
        end

        result = quote
            $expr
            $defn
        end
        @eval $result
        return Expr(:escape, quote
            typealias $type_defn Latte.$type_defn
        end)
    else
        throw("Not Implemented")
    end
end
