// Copyright (c) 2015 Intel Corporation. All rights reserved.
#include <stdlib.h>

extern "C" {
    void init();
    int init_request();
    void sync_gradients(float* data, int count, int request_id);
    void wait(int request_id);
    float reduce_accuracy(float acc);
    void broadcast(float* value, int length);
    int get_rank();
}
