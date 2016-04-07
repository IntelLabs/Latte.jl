using Documenter, Latte

makedocs(
    modules = [Latte],
    doctest  = false
)

custom_deps() = run(`pip install --user pygments mkdocs mkdocs-material`)

deploydocs(
    deps = custom_deps,
    repo = "github.com/IntelLabs/Latte.jl.git",
    julia = "0.4"
)
