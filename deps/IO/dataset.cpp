/*
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
*/

#include "dataset.h"
#ifdef LATTE_BUILD_MPI
#include "../communication/comm.h"
#endif

Dataset::Dataset(char* data_file_name, int _batch_size, bool _shuffle, bool _use_mpi, bool divide_by_rank) {
#ifdef LATTE_BUILD_MPI
    int rank;
    if (_use_mpi) {
        MPI_Comm_rank(get_inter_net_comm(), &rank);
        debug("Rank %d : Initializing dataset %s (shuffle=%d, use_mpi=%d).", rank, data_file_name, _shuffle, _use_mpi);
    }
#else
    debug("Initializing dataset %s.", data_file_name);
#endif
    use_mpi = _use_mpi;
    shuffle = _shuffle;
    batch_size = _batch_size;
    epoch = 0;
    curr_item = 0;

    // Set up file access property list with parallel I/O access
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    assert(plist_id != -1);

    herr_t ret;
    if (use_mpi) {
        debug("Rank %d : Setting up parallel access to dataset %s.", rank, data_file_name);
#ifdef LATTE_BUILD_MPI
        /* set Parallel access with communicator */
        ret = H5Pset_fapl_mpio(plist_id, get_inter_net_comm(), MPI_INFO_NULL);
        assert(ret != -1);
#else
        std::cerr << "Error: To use Latte in MPI mode, please rebuild IO library with -DLATTE_MPI=ON" << std::endl;
        assert(false);
#endif
    } else {
        debug("use_mpi=%d, accesing dataset %s in sequential mode.", use_mpi, data_file_name);
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
    debug("dataset dimensions: ");
    for (int i = 0; i < data_ndim; i++) {
        debug("  %lu", space_dims[i]);
    }
    debug(" max dims: ");
    for (int i = 0; i < data_ndim; i++) {
        debug("  %lu", space_maxdims[i]);
    }

    data_shape = new int[data_ndim];
    for (int i = 0; i < data_ndim; i++) {
        data_shape[i] = space_dims[i];
    }

    num_total_items = data_shape[0];
    if (use_mpi) { // && divide_by_rank) {
#ifdef LATTE_BUILD_MPI
        int size;
        MPI_Comm_size(get_inter_net_comm(), &size);
        int chunk_size = num_total_items / size + 1;
        chunk_start = rank * chunk_size;
        chunk_end = std::min(chunk_start+chunk_size, num_total_items);
        num_total_items = chunk_end - chunk_start;
        debug("Rank %d : chunk_size=%d, chunk_start=%d, chunk_end=%d, num_total_items=%d", rank, chunk_size, chunk_start, chunk_end, num_total_items);
#endif
    } else {
        chunk_start = 0;
        chunk_end = num_total_items;
    }
    data_item_size = 1;
    for (int i = 1; i < data_ndim; i++) {
        data_item_size *= data_shape[i];
    }
    debug("data_item_size %d", data_item_size);
    debug("num_total_items %d", num_total_items);
// #define LOCAL_SIZE 40000000000ul
#define LOCAL_SIZE 2000000000ul
    if (num_total_items > LOCAL_SIZE / (data_item_size*sizeof(float))) {
        num_local_items = LOCAL_SIZE/(data_item_size*sizeof(float));
        // make it a multiple of batch_size
        num_local_items = (num_local_items/batch_size)*batch_size;
    } else {
        num_local_items = num_total_items;
    }

    // chunk_idx = chunk_start;
    chunk_idx = 0;
    n_chunks = num_total_items / num_local_items;
    chunks = new int[n_chunks];
    for (int i = 0; i < n_chunks; i++) {
        chunks[i] = chunk_start + num_local_items * i;
    }
    if (shuffle) std::random_shuffle(chunks, chunks + n_chunks);

    debug("num_local_items %d", num_local_items);
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
        // start[0] = chunk_idx;
        start[0] = chunks[chunk_idx];
        debug("Fetching chunk %d", start[0]);
        debug("chunk_idx: %d", chunk_idx);
        debug("num_local_items: %d", num_local_items);
        debug("count: ");
        for (int i = 1; i < data_ndim; i++) {
            count[i] = data_shape[i];
            debug("  %d", data_shape[i]);
            start[i] = 0;
        }
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
//         if (use_mpi) {
// #ifdef LATTE_BUILD_MPI
//             ret = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
//             assert(ret != -1);
// #endif
//         }

        /* read data collectively */
        ret = H5Dread(data_dataset_id, H5T_NATIVE_FLOAT, mem_dataspace, my_dataspace,
                xfer_plist, data_buffer);
        // printf("Error %d", ret);
        assert(ret != -1);

        count[0] = num_local_items;
        // start[0] = chunk_idx;
        start[0] = chunks[chunk_idx];
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
        H5Pclose(xfer_plist);
        assert(ret != -1);
        if (chunk_idx + 1 >= n_chunks) {
            chunk_idx = 0;
        // if (chunk_idx + 2 * num_local_items > chunk_end) {
            // chunk_idx = chunk_start;
            epoch += 1;
            if (shuffle) std::random_shuffle(chunks, chunks + n_chunks);
        } else {
            // chunk_idx += num_local_items;
            chunk_idx++;
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
