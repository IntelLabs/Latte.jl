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
