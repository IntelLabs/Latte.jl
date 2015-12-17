# Copyright (c) 2015 Intel Corporation. All rights reserved.
summary = []
function runtests(testdir)
    istest(f) = endswith(f, ".jl") && f != "runtests.jl"
    testfiles = sort(filter(istest, readdir(testdir)))
    nfail = 0
    exename = joinpath(JULIA_HOME, Base.julia_exename())
    for f in testfiles
        try
            run(`$exename $(joinpath(testdir, f))`)
            push!(summary, ((:green, STDOUT),  "SUCCESS: $f"))
        catch ex
            push!(summary, ((:red, STDERR), "Error: $(joinpath(testdir, f))"))
            nfail += 1
        end
    end
    return nfail
end
nfail = 0
print_with_color(:white, "Running Latte.jl tests\n")
testdir = dirname(@__FILE__)
nfail += runtests(testdir)
nfail += runtests(joinpath(testdir, "stdlib"))
nfail += runtests(joinpath(testdir, "transforms"))
for message in summary
    Base.with_output_color(message[1]...) do io
        println(io, message[2])
    end
end
exit(nfail)
