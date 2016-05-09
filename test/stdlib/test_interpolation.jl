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

using Latte
using FactCheck

function interpolation_forward(input, resize_factor, kernel, pad, output)
    width, height, channels, num = size(input)
    interpolated_height, interpolated_width, _, _ = size(output)

    for n = 1:num
        for c = 1:channels
            for ih = 1:interpolated_height
                r_f = ih * resize_factor
                delta_r = r_f - floor(r_f)
                for iw = 1:interpolated_width
                    c_f = iw * resize_factor         
                    delta_c = c_f - floor(c_f)

                    hstart = floor(Int, (ih-1) / resize_factor) - pad + 1
                    wstart = floor(Int, (iw-1) / resize_factor) - pad + 1
                    hend = min(hstart + kernel - 1, height)
                    wend = min(wstart + kernel - 1, width)
                    hstart = max(1, hstart)
                    wstart = max(1, wstart)

                    region =  sub(input, wstart:wend, hstart:hend, c, n) 
                   
                    r = 1
                    c = 1
                    r_plus_1 = size(region, 1) > 1 ? 2 : 1
                    c_plus_1 = size(region, 2) > 1 ? 2 : 1
                    output[iw, ih, c, n] = region[r, c] * (1-delta_r) * (1-delta_c) 
                                        +  region[r_plus_1, c] * delta_r * (1-delta_c) 
                                        +  region[r, c_plus_1]  * (1-delta_r) * delta_c
                                        +  region[r_plus_1, c_plus_1] * delta_r * delta_c
                end
            end
        end
    end
end

function interpolation_backward(input, resize_factor, kernel, pad, output)
    width, height, channels, num = size(input)
    interpolated_height, interpolated_width, _, _ = size(output)

    for n = 1:num
        for c = 1:channels
            for ih = 1:interpolated_height
                r_f = ih * resize_factor
                delta_r = r_f - floor(r_f)
                for iw = 1:interpolated_width
                    c_f = iw * resize_factor
                    delta_c = c_f - floor(c_f)

                    hstart = floor(Int, (ih-1) / resize_factor) - pad + 1
                    wstart = floor(Int, (iw-1) / resize_factor) - pad + 1
                    hend = min(hstart + kernel - 1, height)
                    wend = min(wstart + kernel - 1, width)
                    hstart = max(1, hstart)
                    wstart = max(1, wstart)

                    region =  sub(input, wstart:wend, hstart:hend, c, n) 
                    r = 1
                    c = 1
                    r_plus_1 = size(region, 1) > 1 ? 2 : 1
                    c_plus_1 = size(region, 2) > 1 ? 2 : 1

                    region[r,c] += (1-delta_r) * (1-delta_c) * output[iw, ih, c, n]
                    region[r_plus_1, c] += delta_r * (1-delta_c) * output[iw, ih, c, n]
                    region[r, c_plus_1] += (1-delta_r) * delta_c * output[iw, ih, c, n]
                    region[r_plus_1, c_plus_1] += delta_r * delta_c * output[iw, ih, c, n]
                end
            end
        end
    end
end



facts("Testing Bilinear Interpolation Layer (Enlarge)") do

    net = Net(1)
    resize_factor = Float32(8.0)
    data, data_value = MemoryDataLayer(net, :data, (64, 64, 1))
    data_value[:] = rand(Float32, size(data_value)...)
    interp1 = InterpolationLayer(:interp1, net, data, 2, 0, resize_factor)
    
    init(net)

    params = SolverParameters(
        lr_policy    = LRPolicy.Decay(.01f0, 5.0f-7),
        mom_policy   = MomPolicy.Fixed(0.9),
        max_epoch    = 300,
        regu_coef    = .0005)
    sgd = SGD(params)

    
    context("Forward") do
        forward(net; solver=sgd)
        input = get_buffer(net, :datavalue)
        expected = zeros(get_buffer(net, :interp1value))
        interpolation_forward(input, resize_factor, 2, 0, expected)
        @fact expected --> roughly(get_buffer(net, :interp1value)) 
    end

    context("Backward") do
        top_diff = get_buffer(net, :interp1∇)            
            ∇input = get_buffer(net, :data∇)
            ∇input_expected = deepcopy(∇input)
        
        backward(net; solver=sgd)  
            
        interpolation_backward(∇input_expected, resize_factor, 2, 0, top_diff)
        @fact ∇input --> roughly(∇input_expected)
    end 
    
end

facts("Testing Bilinear Interpolation Layer (Shrink)") do

    net = Net(1)
    resize_factor = Float32(1/8.0)
    data, data_value = MemoryDataLayer(net, :data, (64, 64, 1))
    data_value[:] = rand(Float32, size(data_value)...)
    interp1 = InterpolationLayer(:interp1, net, data, 2, 0, resize_factor)
    
    init(net)

    params = SolverParameters(
        lr_policy    = LRPolicy.Decay(.01f0, 5.0f-7),
        mom_policy   = MomPolicy.Fixed(0.9),
        max_epoch    = 300,
        regu_coef    = .0005)
    sgd = SGD(params)

    
    context("Forward") do
        forward(net; solver=sgd)
        input = get_buffer(net, :datavalue)
        expected = zeros(get_buffer(net, :interp1value))
        interpolation_forward(input, resize_factor, 2, 0, expected)
        @fact expected --> roughly(get_buffer(net, :interp1value)) 
    end

    context("Backward") do
        top_diff = get_buffer(net, :interp1∇)            
            ∇input = get_buffer(net, :data∇)
            ∇input_expected = deepcopy(∇input)
        
        backward(net; solver=sgd)  
            
        interpolation_backward(∇input_expected, resize_factor, 2, 0, top_diff)
        @fact ∇input --> roughly(∇input_expected)
    end 
    
end

FactCheck.exitstatus()
