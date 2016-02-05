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

#include "io.h"

void init(bool use_mpi) {
    MPI_Init(NULL, NULL);
    if (use_mpi) {
        // MPI_Comm comm  = MPI_COMM_WORLD;
        // MPI_Info info  = MPI_INFO_NULL;

        // MPI_Comm_size(comm, &mpi_size);
        // MPI_Comm_rank(comm, &mpi_rank);  
        // MPI_Barrier(comm);
    }
}

int init_dataset(int _batch_size, char *data_file_name, bool _shuffle, float scale, bool use_mpi, bool divide_by_rank)
{
    Dataset* dset = new Dataset(data_file_name, _batch_size, _shuffle, scale, use_mpi, divide_by_rank);

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
  MPI_Finalize();
}
