#include <cuda_runtime.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <string.h>
#include "template.h"
#include <cuda.h>


__global__ void spmv_0(unsigned short * global_row_index_of_warp_level_block, unsigned int * global_warp_nz_begin_offset, float * val_arr, unsigned short * col_index_arr, float * device_x_arr, float * device_y_arr)
{
    int tid_in_warp = threadIdx.x % 32;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int wum = (gridDim.x * blockDim.x) / 32;
    int wid = (blockDim.x * blockIdx.x + threadIdx.x) / 32;


    printf("wum:%d, wid:%d, tid:%d\n", wum, wid, tid);

    if (tid == 249)
    {
        assert(false);
    }

    unsigned long kernal_first_row_index;
    unsigned long kernal_first_col_index;

    for (unsigned long warp_level_block_id = wid; warp_level_block_id < 120050; warp_level_block_id = warp_level_block_id + wum)
    {
        // 获取warp首个非零元索引
        unsigned long this_warp_block_first_nz;
        unsigned long next_warp_block_first_nz;

        if (tid_in_warp == 0)
        {
            this_warp_block_first_nz = global_warp_nz_begin_offset[warp_level_block_id];
            next_warp_block_first_nz = global_warp_nz_begin_offset[warp_level_block_id + 1];
        }

        this_warp_block_first_nz = __shfl_sync(0xFFFFFFFF, this_warp_block_first_nz, 0, 32);
        next_warp_block_first_nz = __shfl_sync(0xFFFFFFFF, next_warp_block_first_nz, 0, 32);

        float result_tmp_result = 0;

        for (unsigned int global_nz_index = this_warp_block_first_nz + tid_in_warp; global_nz_index < next_warp_block_first_nz; global_nz_index = global_nz_index + 32)
        {
            result_tmp_result = result_tmp_result +  val_arr[global_nz_index] * device_x_arr[kernal_first_col_index + col_index_arr[global_nz_index]];
        }

        for (int offset = 16; offset > 0; offset = offset / 2)
        {
            result_tmp_result = result_tmp_result + __shfl_down_sync(0xFFFFFFFF, result_tmp_result, offset);
        }

        if (tid_in_warp == 0)
        {
            unsigned long global_row_index = global_row_index_of_warp_level_block[warp_level_block_id];
            // 写数据
            atomicAdd(&(device_y_arr[global_row_index]), result_tmp_result);
        }
    }
}

int main()
{
compressed_dense_block_0_t *dense_block_0_template_data = read_dense_block_0_from_file("/home/duzhen/spmv_builder/data_source/661391894");

unsigned short* device_dense_0_global_row_index_of_warp_level_block;
unsigned int* device_dense_0_global_warp_nz_begin_offset;
float* device_dense_0_val_arr;
unsigned short* device_dense_0_col_index_arr;

cudaMalloc(&device_dense_0_global_row_index_of_warp_level_block, sizeof(unsigned short) * 120051);
cudaMemcpy(device_dense_0_global_row_index_of_warp_level_block, dense_block_0_template_data->dense_0_global_row_index_of_warp_level_block, sizeof(unsigned short) * 120051, cudaMemcpyHostToDevice);

cudaMalloc(&device_dense_0_global_row_index_of_warp_level_block, sizeof(unsigned short) * 120051);
cudaMemcpy(device_dense_0_global_row_index_of_warp_level_block, dense_block_0_template_data->dense_0_global_row_index_of_warp_level_block, sizeof(unsigned short) * 120051, cudaMemcpyHostToDevice);

cudaMalloc(&device_dense_0_val_arr, sizeof(float) * 7681696);
cudaMemcpy(device_dense_0_val_arr, dense_block_0_template_data->dense_0_val_arr, sizeof(float) * 7681696, cudaMemcpyHostToDevice);

cudaMalloc(&device_dense_0_col_index_arr, sizeof(unsigned short) * 7681696);
cudaMemcpy(device_dense_0_col_index_arr, dense_block_0_template_data->dense_0_col_index_arr, sizeof(unsigned short) * 7681696, cudaMemcpyHostToDevice);

float *host_y_arr = (float *)malloc(sizeof(float) * 60050);
float *host_x_arr = (float *)malloc(sizeof(float) * 60001);

float *device_y_arr = NULL;
float *device_x_arr = NULL;

cudaMalloc(&device_y_arr, sizeof(float) * 60050);
cudaMalloc(&device_x_arr, sizeof(float) * 60001);

for (unsigned long i = 0; i < 60050; i++)
{
host_y_arr[i] = 0;
}

for (unsigned long i = 0; i < 60001; i++)
{
host_x_arr[i] = 1;
}

cudaMemcpy(device_y_arr, host_y_arr, sizeof(float) * 60050, cudaMemcpyHostToDevice);
cudaMemcpy(device_x_arr, host_x_arr, sizeof(float) * 60001, cudaMemcpyHostToDevice);

cudaStream_t stream_arr[1];
for(unsigned long i = 0; i < 1; i++)
{
cudaStreamCreate(&(stream_arr[i]));
}

cudaDeviceSynchronize();
struct timeval start,end;
gettimeofday(&start, NULL);
for (int i = 0; i < 1200; i++)
{

spmv_0<<<1, 512, 0, stream_arr[0]>>>(device_dense_0_global_row_index_of_warp_level_block, device_dense_0_global_warp_nz_begin_offset, device_dense_0_val_arr, device_dense_0_col_index_arr, device_x_arr, device_y_arr);

cudaDeviceSynchronize();
}

gettimeofday(&end, NULL);

long timeuse = 1000000 * (end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
double gflops = ((double)2.0 * 7200120 * 1200 / ((double)timeuse / 1000000)) / 1000000000;

cudaError_t cuda_err = cudaGetLastError();

if(cudaSuccess != cuda_err)

{

// fprintf(stderr,"%s:(%s:%s:%d)\n",message,file,function,line);

fprintf(stderr,"%s\n", cudaGetErrorString(cuda_err));

//exit(1);

return -1;
}

printf("time=%fms, gflops=%f\n",timeuse /1000.0, gflops);
cudaMemcpy(host_y_arr, device_y_arr, sizeof(float) * 60050, cudaMemcpyDeviceToHost);
print_arr_to_file_with_data_type(host_y_arr, FLOAT, 60050, "/home/duzhen/spmv_builder/data_source/test_result_3");

return 0;
}

