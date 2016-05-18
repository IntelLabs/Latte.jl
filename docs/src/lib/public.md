# Public

```@meta
CurrentModule = Latte
```

## Contents

```@contents
Pages = ["public.md"]
```

## Index

```@index
Pages = ["public.md"]
```

## API

```@docs
Net
Net(batch_size::Int; time_steps=1, num_subgroups=1)
Ensemble
Connection
Param
get_buffer(net::Net, name::Symbol, t::Int)

add_connections(net::Net, source::AbstractEnsemble,
                sink::AbstractEnsemble, mapping::Function; padding=0,
                recurrent=false)
add_ensemble(net::Net, ens::AbstractEnsemble)
init(net::Net)

load_snapshot(net::Net, file::AbstractString)
save_snapshot(net::Net, file::AbstractString)

test(net::Net)
```
