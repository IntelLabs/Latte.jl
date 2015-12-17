// Copyright (c) 2015 Intel Corporation. All rights reserved.
#include "io.h"

void init(bool use_mpi) {
    if (use_mpi) {
        // MPI_Init(NULL, NULL);
        // MPI_Comm comm  = MPI_COMM_WORLD;
        // MPI_Info info  = MPI_INFO_NULL;

        // MPI_Comm_size(comm, &mpi_size);
        // MPI_Comm_rank(comm, &mpi_rank);  
        // MPI_Barrier(comm);
    }
}

int init_dataset(int _batch_size, char *data_file_name, bool _shuffle, bool use_mpi, bool divide_by_rank)
{
    Dataset* dset = new Dataset(data_file_name, _batch_size, _shuffle, use_mpi, divide_by_rank);

    int id = datasets.size();
    datasets.push_back(dset);
    return id;
}

int get_data_ndim(int dset_id) {
    assert(dset_id < datasets.size());
    return datasets[dset_id]->data_ndim;
}

int* get_data_shape(int dset_id) {
    assert(dset_id < datasets.size());
    return datasets[dset_id]->data_shape;
}

int get_label_ndim(int dset_id) {
    assert(dset_id < datasets.size());
    return datasets[dset_id]->label_ndim;
}

int* get_label_shape(int dset_id) {
    assert(dset_id < datasets.size());
    return datasets[dset_id]->label_shape;
}

void set_data_pointer(int dset_id, float* pointer) {
    assert(dset_id < datasets.size());
    datasets[dset_id]->data_out = pointer;
}

void set_label_pointer(int dset_id, float* pointer) {
    assert(dset_id < datasets.size());
    datasets[dset_id]->label_out = pointer;
}

void next_epoch(int dset_id)
{
}

void get_next_batch(int dset_id) {
    assert(dset_id < datasets.size());
    datasets[dset_id]->get_next_batch();
}

int get_epoch(int dset_id) {
    assert(dset_id < datasets.size());
    return datasets[dset_id]->epoch;
}


void clean_up() {
  datasets.clear();
}
