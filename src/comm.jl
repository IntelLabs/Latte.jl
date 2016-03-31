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

@eval function broadcast_initial_params(net::Net)
    log_info("Broadcasting initial parameters")
    for param in net.params
        ccall((:broadcast_inter, $libComm), Void, (Ptr{Float32}, Cint, Cint), param.value, length(param.value), 0)
    end
    log_info("Done")
end

@eval function get_net_subrank(net::Net)
    rank = ccall((:get_rank, $libComm), Cint, ())
    rank % net.num_subgroups
end

@eval function get_inter_rank(net::Net)
    rank = ccall((:get_rank, $libComm), Cint, ())
    div(rank, net.num_subgroups)
end

@eval function get_rank()
    ccall((:get_rank, $libComm), Cint, ())
end

@eval function initialize_communicators(net::Net)
    ccall((:initialize_communicators, $libComm), Void, (Cint,), net.num_subgroups)
end

@eval function sync_intra_loss(net::Net, loss::Float32)
    loss_val = Array(Float32, 1)
    loss_val[1] = loss
    ccall((:broadcast_intra, $libComm), Void, (Ptr{Float32}, Cint, Cint), loss_val, 1, net.num_subgroups - 1)
    loss_val[1]
end

@eval function sync_intra_train_epoch(net::Net)
    epoch = Array(Float32, 1)
    epoch[1] = net.train_epoch
    ccall((:broadcast_intra, $libComm), Void, (Ptr{Float32}, Cint, Cint), epoch, 1, 0)
    net.train_epoch = epoch[1]
end

@eval function sync_intra_test_epoch(net::Net)
    epoch = Array(Float32, 1)
    epoch[1] = net.test_epoch
    ccall((:broadcast_intra, $libComm), Void, (Ptr{Float32}, Cint, Cint), epoch, 1, 0)
    net.test_epoch = epoch[1]
end
