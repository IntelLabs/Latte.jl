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
#include <vector>
#include <mpi.h>
#include <cmath>
#include <map>

std::vector<MPI_Request *> requests;
std::vector<bool> isCompressed;

void init() {
    MPI_Init(NULL, NULL);
}

int init_request() {
    MPI_Request *request = (MPI_Request *) malloc(sizeof(MPI_Request));
    int id = requests.size();
    requests.push_back(request);
    isCompressed.push_back(false);
    return id;
}

uint64_t zero_grads = 0;
uint64_t all_grads = 0;
uint64_t large_grads = 0;

void sync_gradients_old(float *data, float* values, int count, int request_id) {
//void sync_gradients(float *data, int count, int request_id) {
    int rank = get_rank();
    all_grads += count;
   // for(int i=0; i<count; i++)
    //    if(std::abs(data[i])<std::abs(values[i])/100.0) {
      //      zero_grads++;
      //      data[i] = 0.0;
     //   }

    MPI_Request *request = requests[request_id];
    MPI_Iallreduce(MPI_IN_PLACE, data, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, request);
    // int size;
    // MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Scale for gradient accumulation normalization
    // #pragma omp parallel for simd
    // for (int i=0; i < count; i++) {
    //     data[i] /= (float) size;
    // }
}

int print_stat() {
    // printf("rank %d zeros %lld all %lld rate:%lf \n", get_rank(), zero_grads, all_grads, zero_grads/(double)all_grads);
    return 1;
}

std::map<float*,char*> buffer_map;
// request maps
std::map<int,float*> data_map;
std::map<int,float*> val_map;
std::map<int,int> count_map;
uint64_t  curr_iter=0;

void sync_gradients(float *data, float* values, int count, int request_id) {

    if(curr_iter++ < 10000)
    {
        // printf("id %d count %d\n",request_id,count);
        sync_gradients_old(data, values, count, request_id);
        return;
    }

//    isCompressed[request_id] = true;

    char* data_char_buff;
    if(buffer_map.find(data)!=buffer_map.end())
        data_char_buff = buffer_map[data];
    else 
    {
        buffer_map[data] = data_char_buff = new char[count];
        data_map[request_id] = data;
        val_map[request_id] = values;
        count_map[request_id] = count;
    }

    all_grads += count;
    for(int i=0; i<count; i++)
    {

        if(data[i]==0) {
            zero_grads++;
            continue;
        }

        if(std::abs(data[i])<std::abs(values[i])/(1<<2))
        {
            large_grads++;

            //printf("data %f value %f\n",data[i], values[i]);
            //printf("data %X value %X\n",*(int*)&(data[i]), *(int*)&(values[i]));

            int val_exp;
            frexp(values[i],&val_exp);
            int dat_exp;
            frexp(data[i],&dat_exp);
            //printf("data_exp %d value_ex %d\n", dat_exp, val_exp);

            // set IEEE754 implicit bit
            int idata = (*(unsigned int*)&(data[i])) | 0x800000;
            //
            // normalize to 2^(val_exp-2)
            // 24 bits (with implicit) - 17 = 7 bits needed
            int ddata = (idata&0xFFFFFF)>>(17+val_exp-dat_exp-2);
            char sign = std::signbit(data[i]);
            //printf("sign %d\n", sign);
            // sign + 7 bits
            volatile char res = (ddata & 0x7F) | (sign<<7);
            //printf("res %X\n",res); 
            // if no bit could be represented
            if(!(res&0x7F)) {
                zero_grads++;
                int mask = ( (1<<(17+val_exp-dat_exp-2)) -1) &0xFFFFFF; 
                printf("remains mask %X\n", mask);
                int remains = idata & mask;
                printf("remains %X\n", remains);
                int draw = rand() & mask;
                printf("draw %X\n", draw);
                if(draw<remains)
                    res |= 1;
            }
            
            // move sign to last bit
            int sign_o = ((int)res & 0x80) << 24; 
            char res2 = res & 0x7F;
            float out = (float)res2;
            //printf("raw out %f\n",out); 
            out = ldexp(out,-6+val_exp-1-2);
            out = sign_o? -out : out;
            //printf("out %f\n",out);

            if(std::abs(data[i])-std::abs(out)>std::abs(values[i])/4)
            {
                printf("Error! data %f out %f\n", data[i], out);
                // assert(false);
            }

            data[i] = out;
            
        }
        /*

           float rel_val = (data[i]/(double)values[i])*((double)(1<<12));

           if(std::abs(rel_val)<1.0f)
           {
            int chance = std::abs(rel_val)*1000;
            int draw = rand()%1000;
            // printf("rel_val %f chance %d draw %d\n",rel_val, chance, draw);
            if(draw<chance)
                rel_val == 1.01f;
        }
        
        if(std::abs(rel_val)>126.0f)
        {
            large_grads++;
            //printf("LARGE VALUE %f %d\n", rel_val, request_id);
        }
        else
        {
            //if(get_rank()==2) printf("data %lf ",data[i]);
            volatile char rel_char = (char)rel_val;
            data[i] = ((double)rel_char*values[i])/((double)(1<<12));
            //if(get_rank()==2) printf("%lf \n",data[i]);

        }
        */


    }
    if(curr_iter%10000==0)
    {
        printf("large ration: %lf large: %lld all: %lld zeros %lld zero ratio: %lf\n", large_grads/(double)all_grads, large_grads, all_grads, zero_grads, zero_grads/(double)all_grads);
    }
    sync_gradients_old(data, values, count, request_id);
    return;

//    printf("sent %d %lf %lf %lf %lf %d %d\n", request_id, data[0], data[count-1], values[0], 
  //          values[count-1], data_char_buff[0], data_char_buff[count-1]);

    //MPI_Request *request = requests[request_id];
    //MPI_Iallreduce(MPI_IN_PLACE, data_char_buff, count, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD, request);
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
    MPI_Status stat;
    MPI_Wait(request, &stat);
    if (!isCompressed[request_id])
        return;
    float* data = data_map[request_id];
    float* values = val_map[request_id];
    char* data_char = buffer_map[data];
    int count = count_map[request_id];
    for(int i=0; i<count; i++)
        data[i] = ((float)data_char[i]*values[i])/((float)(1<<6));
//    printf("received %d %lf %lf\n", request_id, data[0], data[count-1]);
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
