// Copyright (c) 2015 Intel Corporation. All rights reserved.
#include <assert.h>
#include <string.h>
#include <vector>
#include <algorithm>

#include "dataset.h"

int mpi_size;
int mpi_rank;
#ifdef LATTE_BUILD_MPI
MPI_Comm comm  = MPI_COMM_WORLD;
MPI_Info info  = MPI_INFO_NULL;
#endif


std::vector<Dataset*> datasets;

// initialize parallel IO library
extern "C" {
    void init(bool use_mpi);
    void clean_up();

    int init_dataset(int _batch_size, char *data_file_name, bool _shuffle, bool use_mpi, bool divide_by_rank);
    void get_next_batch(int dset_id);
    int get_epoch(int dset_id);
    int* get_data_shape(int dset_id);
    int  get_data_ndim(int dset_id);
    int* get_label_shape(int dset_id);
    int  get_label_ndim(int dset_id);
    void set_data_pointer(int dset_id, float* pointer);
    void set_label_pointer(int dset_id, float* pointer);
}
