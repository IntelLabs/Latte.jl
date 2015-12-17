# Copyright (c) 2015 Intel Corporation. All rights reserved.
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
