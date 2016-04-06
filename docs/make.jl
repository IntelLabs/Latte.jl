using Documenter, Latte

makedocs()
deploydocs(
    repo = "github.com/IntelLabs/Latte.jl.git",
    julia = "0.4.3"
)
