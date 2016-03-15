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

#include "comm.h"
#include <mpi.h>

std::vector<MPI_Request *> requests;
MPI_Comm *Inter_net_communicator;
MPI_Comm *Intra_net_communicator;
void init() {
    MPI_Init(NULL, NULL);
}

int init_request() {
    MPI_Request *request = (MPI_Request *) malloc(sizeof(MPI_Request));
    // We initialize this request because forward pass begins with a 0 update
    MPI_Ibarrier(*Inter_net_communicator, request);
    int id = requests.size();
    requests.push_back(request);
    return id;
}

void sync_gradients(float *data, int count, int request_id) {
    MPI_Request *request = requests[request_id];
    MPI_Iallreduce(MPI_IN_PLACE, data, count, MPI_FLOAT, MPI_SUM, *Inter_net_communicator, request);
    // int size;
    // MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Scale for gradient accumulation normalization
    // #pragma omp parallel for simd
    // for (int i=0; i < count; i++) {
    //     data[i] /= (float) size;
    // }
}

void wait(int request_id) {
    MPI_Request *request = requests[request_id];
    // clock_t start_time = clock();
    MPI_Wait(request, MPI_STATUS_IGNORE);
    // clock_t end_time = clock();
    // double total_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    // std::cout << "Parameter " << request_id << " takes " << total_time << " seconds." << std::endl;
}

float reduce_accuracy(float acc) {
    MPI_Comm comm = *Inter_net_communicator;
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);  
    float total_acc = 0.0f;
    MPI_Reduce(&acc, &total_acc, 1, MPI_FLOAT, MPI_SUM, 0, comm);
    if (rank == 0) {
        return total_acc / size;
    } else {
        return -1.0f;
    }
}

void broadcast_inter(float* value, int length, int root) {
    MPI_Bcast(value, length, MPI_FLOAT, root, *Inter_net_communicator);
}

void broadcast_intra(float* value, int length, int root) {
    MPI_Bcast(value, length, MPI_FLOAT, root, *Intra_net_communicator);
}

int get_rank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

void initialize_communicators(int num_subgroups) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(size % num_subgroups == 0);
    Intra_net_communicator = (MPI_Comm *) malloc(sizeof(MPI_Comm));
    Inter_net_communicator = (MPI_Comm *) malloc(sizeof(MPI_Comm));
    // Initialize for each net replica
    MPI_Comm_split(MPI_COMM_WORLD, rank % num_subgroups, 0, Inter_net_communicator);
    MPI_Comm_split(MPI_COMM_WORLD, rank / num_subgroups, 0, Intra_net_communicator);
}

void recv_intra(float* data, int length, int tag, int source) {
    // std::cout << "Receiving " << length << " floats with tag " << tag << " from " << source << std::endl;
    MPI_Recv(data, length, MPI_FLOAT, source, tag, *Intra_net_communicator, MPI_STATUS_IGNORE);
}

void send_intra(float* data, int length, int tag, int dest) {
    // std::cout << "Sending " << length << " floats with tag " << tag << " to " << dest << std::endl;
    MPI_Send(data, length, MPI_FLOAT, dest, tag, *Intra_net_communicator);
}
