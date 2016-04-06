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
