# Copyright (c) 2015 Intel Corporation. All rights reserved.
@eval function broadcast_initial_params(net::Net)
    for param in net.params
        ccall((:broadcast, $libComm), Void, (Ptr{Float32}, Cint), param.value, length(param.value))
    end
end
