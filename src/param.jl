# Copyright (c) 2015 Intel Corporation. All rights reserved.
function get_value(param::Param)
    return param.value
end

function get_gradient(param::Param)
    return param.gradient
end
