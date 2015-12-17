// Copyright (c) 2015 Intel Corporation. All rights reserved.
#include "dataset.h"

Dataset::Dataset(char* data_file_name, int _batch_size, bool _shuffle, bool _use_mpi, bool divide_by_rank) {
    use_mpi = use_mpi;
    shuffle = _shuffle;
    batch_size = _batch_size;
    epoch = 0;
    curr_item = 0;

    // Set up file access property list with parallel I/O access
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    assert(plist_id != -1);

    herr_t ret;
    if (use_mpi) {
#ifdef LATTE_BUILD_MPI
        /* set Parallel access with communicator */
        ret = H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
        assert(ret != -1);
#endif
    }
   
    // open file
    file_id = H5Fopen(data_file_name, H5F_ACC_RDONLY, plist_id);
    assert(file_id != -1);
    ret = H5Pclose(plist_id);
    assert(ret != -1);

    // open dataset
    label_dataset_id = H5Dopen2(file_id, "/label", H5P_DEFAULT);
    assert(label_dataset_id != -1);
    
    data_dataset_id = H5Dopen2(file_id, "/data", H5P_DEFAULT);
    assert(data_dataset_id != -1);

    hid_t space_id = H5Dget_space(data_dataset_id);
    assert(space_id != -1);

    data_ndim = H5Sget_simple_extent_ndims(space_id);
    assert(data_ndim > 2);

    /* get data dimension info */
    hsize_t space_dims[data_ndim];
    hsize_t space_maxdims[data_ndim];
    H5Sget_simple_extent_dims(space_id, space_dims, space_maxdims);
    // printf("dataset dimensions: ");
    // for (int i = 0; i < data_ndim; i++) {
    //     printf("%lu ", space_dims[i]);
    // }
    // printf(" max dims: ");
    // for (int i = 0; i < data_ndim; i++) {
    //     printf("%lu ", space_maxdims[i]);
    // }
    // printf("\n");

    data_shape = new int[data_ndim];
    for (int i = 0; i < data_ndim; i++) {
        data_shape[i] = space_dims[i];
    }

    num_total_items = data_shape[0];
    if (use_mpi && divide_by_rank) {
#ifdef LATTE_BUILD_MPI
        int rank, size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int chunk_size = num_total_items / size + 1;
        chunk_start = rank * chunk_size;
        chunk_end = std::min(chunk_start+chunk_size, num_total_items);
        num_total_items = chunk_end - chunk_start;
#endif
    } else {
        chunk_start = 0;
        chunk_end = num_total_items;
    }
    chunk_idx = chunk_start;
    data_item_size = 1;
    for (int i = 1; i < data_ndim; i++) {
        data_item_size *= data_shape[i];
    }
    unsigned long total_size = (num_total_items*data_item_size*sizeof(float));
    // printf("data_item_size %d\n", data_item_size);
    // printf("total_size %ul\n", total_size);
    // printf("num_total_items %d\n", num_total_items);
// #define LOCAL_SIZE 40000000000ul
#define LOCAL_SIZE 2000000000ul
    if (total_size > LOCAL_SIZE) {
        num_local_items = LOCAL_SIZE/(data_item_size*sizeof(float));
        // make it a multiple of batch_size
        num_local_items = (num_local_items/batch_size)*batch_size;
    } else {
        num_local_items = num_total_items;
    }
    // printf("num_local_items %d\n", num_local_items);
    batch_idxs = new int[num_local_items];
    for (int i = 0; i < num_local_items; i++) batch_idxs[i] = i;
    data_buffer = new float[num_local_items*data_item_size];


    space_id = H5Dget_space(label_dataset_id);
    assert(space_id != -1);

    label_ndim = H5Sget_simple_extent_ndims(space_id);
    assert(label_ndim > 1);

    hsize_t label_space_dims[label_ndim];
    hsize_t label_space_maxdims[label_ndim];
    H5Sget_simple_extent_dims(space_id, label_space_dims, label_space_maxdims);
    label_shape = new int[label_ndim];
    for (int i = 0; i < label_ndim; i++) {
        label_shape[i] = label_space_dims[i];
    }
    label_item_size = 1;
    for (int i = 1; i < label_ndim; i++) {
        label_item_size *= label_shape[i];
    }
    label_buffer = new float[num_local_items*label_item_size];
    fetch_next_chunk(true);
}

void Dataset::fetch_next_chunk(bool force) {
    // always shuffle batch_idxs
    if (shuffle) std::random_shuffle(batch_idxs, batch_idxs + num_local_items);
    // If dataset fits in memory we don't need to reload it
    if (num_local_items != num_total_items || force) {
        // Load the next local chunk
        hsize_t count[data_ndim];
        hsize_t start[data_ndim];
        count[0] = num_local_items;
        start[0] = chunk_idx;
        // printf("chunk_idx: %d\n", chunk_idx);
        // printf("num_local_items: %d\n", num_local_items);
        // printf("count: ");
        for (int i = 1; i < data_ndim; i++) {
            count[i] = data_shape[i];
            // printf("%d ", data_shape[i]);
            start[i] = 0;
        }
        // printf("\n");
        /* create a file dataspace independently */
        hid_t my_dataspace = H5Dget_space(data_dataset_id);
        assert(my_dataspace != -1);
        herr_t ret;
        ret=H5Sselect_hyperslab(my_dataspace, H5S_SELECT_SET, start, NULL, count, NULL);
        assert(ret != -1);

        /* create a memory dataspace independently */
        hid_t mem_dataspace = H5Screate_simple (data_ndim, count, NULL);
        assert (mem_dataspace != -1);

        hid_t xfer_plist = H5Pcreate (H5P_DATASET_XFER);
        assert(xfer_plist != -1);
        if (use_mpi) {
#ifdef LATTE_BUILD_MPI
            ret = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
            assert(ret != -1);
#endif
        }

        /* read data collectively */
        ret = H5Dread(data_dataset_id, H5T_NATIVE_FLOAT, mem_dataspace, my_dataspace,
                xfer_plist, data_buffer);
        // printf("Error %d", ret);
        assert(ret != -1);

        count[0] = num_local_items;
        start[0] = chunk_idx;
        for (int i = 1; i < label_ndim; i++) {
            count[i] = label_shape[i];
            start[i] = 0;
        }

        /* create a file dataspace independently */
        my_dataspace = H5Dget_space(label_dataset_id);
        mem_dataspace = H5Screate_simple(label_ndim, count, NULL);
        assert(my_dataspace != -1);
        // stride and block are NULL for contiguous hyperslab
        ret = H5Sselect_hyperslab(my_dataspace, H5S_SELECT_SET, start, NULL, count, NULL);
        assert(ret != -1);

        ret = H5Dread(label_dataset_id, H5T_NATIVE_FLOAT, mem_dataspace, my_dataspace,
                xfer_plist, label_buffer);
        assert(ret != -1);
        if (chunk_idx + 2 * num_local_items > chunk_end) {
            chunk_idx = chunk_start;
            epoch += 1;
        } else {
            chunk_idx += num_local_items;
        }
        H5Sclose(my_dataspace);
        H5Sclose(mem_dataspace);
    } else {
        // Data fits in memory, don't need to reload it
        epoch += 1;
    }
}

void Dataset::get_next_batch() {
    int start = curr_item;
    int end = std::min(curr_item + batch_size, num_local_items);
#pragma omp parallel for
    for (int i = start; i < end; i++) {
        int n = batch_idxs[i];
        memcpy(data_out + (i-start)*data_item_size, data_buffer + n*data_item_size,
               data_item_size*sizeof(float));
        memcpy(label_out + (i-start)*label_item_size, label_buffer + n*label_item_size,
               label_item_size*sizeof(float));
    }
    if (end != curr_item + batch_size) {
        fetch_next_chunk(false);
        int leftover_start = 0;
        int leftover_end = curr_item+batch_size-end;
        curr_item = 0;
#pragma omp parallel for
        for (int i = leftover_start; i < leftover_end; i++) {
            memcpy(data_out + (i + end - start)*data_item_size, data_buffer + batch_idxs[i]*data_item_size,
                   data_item_size*sizeof(float));
            memcpy(label_out + (i + end - start)*label_item_size, label_buffer + batch_idxs[i]*label_item_size,
                   label_item_size*sizeof(float));
        }
        curr_item += leftover_end;
    } else if (curr_item == num_local_items) {
        curr_item = 0;
        fetch_next_chunk(false);
    } else {
        curr_item = end;
    }
}
