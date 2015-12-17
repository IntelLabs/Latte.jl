# Copyright (c) 2015 Intel Corporation. All rights reserved.
using Latte
using FactCheck

facts("Testing clean for loops") do
    tree = :(for i = 1:2
        begin
            println("Hi")
        end
    end)
    tree = remove_line_nodes(tree)
    actual = clean_for_loops([tree])[1]
    @fact actual --> remove_line_nodes(:(
        for i = 1:2
            println("Hi")
        end
    ))
end
