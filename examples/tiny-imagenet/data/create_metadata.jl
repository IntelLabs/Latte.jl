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

id_to_label = Dict{AbstractString, Int}()
open("tiny-imagenet-200/wnids.txt", "r") do ids
    for (index, line) in enumerate(eachline(ids))
        id_to_label[chomp(line)] = index - 1
    end
end

val_images = []
open("tiny-imagenet-200/val/val_annotations.txt", "r") do val
    for line in eachline(val)
        _line = split(line)
        push!(val_images, (_line[1], id_to_label[_line[2]]))
    end
end
open("val_metadata.txt", "w") do metadata
    for val in val_images
        write(metadata, "tiny-imagenet-200/val/images/$(val[1]) $(val[2])\n")
    end
end

train_classes = readdir("tiny-imagenet-200/train")
open("train_metadata.txt", "w") do metadata
    for class in train_classes
        image_path = "tiny-imagenet-200/train/$class/images"
        for image in readdir(image_path)
            write(metadata, "$image_path/$image $(id_to_label[class])\n")
        end
    end
end
