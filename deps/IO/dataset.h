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

#ifndef LATTE_IO_DATASET_H
#define LATTE_IO_DATASET_H
#include <algorithm>
#include "hdf5.h"
#include "stdlib.h"
#include <assert.h>
#include <string.h>
#include <mkl.h>
#ifdef LATTE_BUILD_MPI
#include <mpi.h>
#endif
#include <omp.h>

class Dataset {
    int* batch_idxs;
    float* data_buffer;
    float* label_buffer;

    bool shuffle;
    float scale;
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

        Dataset(char* data_file_name, int _batch_size, bool _shuffle, float scale, bool _use_mpi, bool divide_by_rank);
	~Dataset() {
	    H5Dclose(data_dataset_id);
	    H5Dclose(label_dataset_id);
	    H5Fclose(file_id);
	};
};

#endif /* LATTE_IO_DATASET_H */
