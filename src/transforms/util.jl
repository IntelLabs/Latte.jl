"""
Construct a Dict{Symbol, Any} that maps the arguments of the function
`ast` to the corresponding value in Vector `args`.
"""
function build_arg_name_map(args::Vector, ast::Expr)
    @assert is_function(ast)
    map = Dict()
    for (index, arg) in enumerate(ast.args[1].args[2:end])
        map[arg.args[1]] = args[index]
    end
    map
end

"""
Generates a loopnest with `body` as the body of the inner most loop using
`vars` as a list of loop variables and `ranges` as a list of ranges for each
loop nest.

Example:
    julia> gen_loop_nest(:(println((a, b))), [:a, :b], [1:2, 1:3])
    :(for b = 1:3
          for a = 1:2
              println((a, b))
          end
      end)
"""
function gen_loop_nest(body, vars, ranges)
    nest = body
    for (var, range) in zip(vars, ranges)
        nest = :(for $var = $range
            $nest
        end)
    end
    nest
end

"""
Synthesizes a loop nest around body based on the dimensionality of `buffer`.
We split each statement in `statements` into a separate synthesized loop nest
to facilitate pattern matching of statements.  If no pattern matching occurs,
the identical loop nests will be fused later.
"""
function append_neuron_loop!(body::Vector, statements::Vector, buffer::Array)
    N = ndims(buffer)
    for statement in statements
        vars   = [symbol(:_neuron_index_,i) for i in 1:N]
        ranges = [:(1:$(size(buffer,i)))    for i in 1:N]
        push!(body, gen_loop_nest(statement, vars, ranges))
    end
end
