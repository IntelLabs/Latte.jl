// Copyright (c) 2015 Intel Corporation. All rights reserved.
#include "comm.h"
#include <vector>
#include <mpi.h>

std::vector<MPI_Request *> requests;
void init() {
    MPI_Init(NULL, NULL);
}

int init_request() {
    MPI_Request *request = (MPI_Request *) malloc(sizeof(MPI_Request));
    int id = requests.size();
    requests.push_back(request);
    return id;
}

void sync_gradients(float *data, int count, int request_id) {
    MPI_Request *request = requests[request_id];
    MPI_Iallreduce(MPI_IN_PLACE, data, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, request);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Scale for gradient accumulation normalization
    #pragma omp parallel for simd
    for (int i=0; i < count; i++) {
        data[i] /= (float) size;
    }
}

void wait(int request_id) {
    MPI_Request *request = requests[request_id];
    MPI_Status stat;
    MPI_Wait(request, &stat);
}

float reduce_accuracy(float acc) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    float total_acc = 0.0f;
    MPI_Reduce(&acc, &total_acc, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        return total_acc / size;
    } else {
        return -1.0f;
    }
}

void broadcast(float* value, int length) {
    MPI_Bcast(value, length, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

int get_rank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}
