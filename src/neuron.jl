# Copyright (c) 2015 Intel Corporation. All rights reserved.
export Neuron, neuron
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
        return Expr(:escape, quote
            function $(expr.args[1])($(expr.args[3:end]...))
                return $(Expr(:quote, expr.args[2].args[2]))
            end
        end)
    elseif expr.head == :type
        type_defn = expr.args[2]
        expr.args[2] = Expr(:(<:), type_defn, :Neuron)
        original = remove_line_nodes(copy(expr.args[3])).args
        T = LatteFloat
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

        result = Expr(:escape, quote
            $expr
            $defn
        end)
        return result
    else
        throw("Not Implemented")
    end
end
