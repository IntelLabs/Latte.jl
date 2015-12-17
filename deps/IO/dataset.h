// Copyright (c) 2015 Intel Corporation. All rights reserved.
#ifndef LATTE_IO_DATASET_H
#define LATTE_IO_DATASET_H
#include <algorithm>
#include "hdf5.h"
#include "stdlib.h"
#include <assert.h>
#include <string.h>
#ifdef LATTE_BUILD_MPI
#include <mpi.h>
#endif
#include <omp.h>

class Dataset {
    int* batch_idxs;
    float* data_buffer;
    float* label_buffer;

    bool shuffle;
    int curr_item;
    int batch_size;
    int data_item_size;
    int label_item_size;
    int num_local_items;
    int num_total_items;
    hid_t data_dataset_id;
    hid_t label_dataset_id;
    hid_t file_id;
    int chunk_start;
    int chunk_idx;
    int chunk_end;
    bool use_mpi;
    public:
        int epoch;
        int  data_ndim;
        int* data_shape;
        int  label_ndim;
        int* label_shape;
        float* data_out;
        float* label_out;
        void fetch_next_chunk(bool force);
        void get_next_batch();

        Dataset(char* data_file_name, int _batch_size, bool _shuffle, bool _use_mpi, bool divide_by_rank);
	~Dataset() {
	    H5Dclose(data_dataset_id);
	    H5Dclose(label_dataset_id);
	    H5Fclose(file_id);
	};
};

#endif /* LATTE_IO_DATASET_H */
