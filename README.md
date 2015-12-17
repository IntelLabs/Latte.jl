# Latte.jl

## Developer

### Julia Dependencies
```shell
julia -e 'Pkg.add("ArrayViews")'
julia -e 'Pkg.add("Iterators")'
julia -e 'Pkg.add("HDF5")'
julia -e 'Pkg.add("FactCheck")'
```

### Testing
Run the entire test suite with
```shell
cd test
julia runtests.jl
```

or run an individual test file with
```shell
julia <test_file.jl>
```
