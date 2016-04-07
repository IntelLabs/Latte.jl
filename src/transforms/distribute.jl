"""
Performs loop distribution for any top-level for loops in `statements`
"""
function distribute_batch_loops(statements)
    new_body = []
    for statement in statements
        if isa(statement, Expr) && statement.head == :for
            for expr in statement.args[2].args
                loop = deepcopy(statement)
                if isa(expr, LineNumberNode) || isa(expr, LabelNode) || expr.head == :line
                    continue
                end
                loop.args[2].args = [expr]
                push!(new_body, loop)
            end
        else
            push!(new_body, statement)
        end
    end
    new_body
end
