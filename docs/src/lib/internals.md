# Internals

## Contents

    {contents}
    Pages = ["internals.md"]

## Index

    {index}
    Pages = ["internals.md"]

## Net

    {docs}
    Latte.init_buffer
    Latte.set_buffer

    Latte.rand_values
    Latte.clear_values
    Latte.clear_∇

## Connections

    {docs}
    Latte.check_one_to_one
    Latte.check_dimensions_fixed

## Synthesis and Optimization

    {docs}
    Latte.add_send_exprs
    Latte.add_recv_expr
    Latte.init_backward
    Latte.init_forward
    Latte.add_forward_julia_tasks
    Latte.add_forward_data_tasks
    Latte.push_compute_tasks!
    Latte.generate_c_function
    Latte.gen_neuron_backward
    Latte.gen_neuron_forward
    Latte.gen_copy_block
    Latte.get_src_idx
    Latte.optimize

    Latte.unpack_tiled_loop
    Latte.is_tiled_loop
    Latte.get_tile_fusion_factor_forward
    Latte.get_tile_fusion_factor_backward
    Latte.update_tile_var
    Latte.inner_loop_tiler
    Latte.get_inner_loop_tiler
    Latte.tile_size_inliner
    Latte.get_tile_loops

## Utility Datastructures

    {docs}
    Latte.TaskSet
    Latte.JuliaTask
    Latte.UpdateTask
    Latte.Batch
