#include <cuda_runtime.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <string.h>
#include "ell_template.h"
#include <cuda.h>


__global__ void spmv_0(float * val_arr, unsigned int * col_index_arr, float * device_x_arr, float * device_y_arr)
{
int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
int total_thread_num = blockDim.x * gridDim.x;
for(unsigned int thread_level_block_id = global_tid; thread_level_block_id < 929920; thread_level_block_id = thread_level_block_id + total_thread_num)
{
float thread_block_tmp_result = 0;
unsigned int global_nz_index = thread_level_block_id;
for(unsigned char nz_index_inner_thread_level_block = 0; nz_index_inner_thread_level_block < 5; nz_index_inner_thread_level_block++)
{
thread_block_tmp_result = thread_block_tmp_result + val_arr[global_nz_index] * device_x_arr[col_index_arr[global_nz_index]];
global_nz_index = global_nz_index + 929920;
}
unsigned int global_row_index;

global_row_index = thread_level_block_id;
device_y_arr[global_row_index] = thread_block_tmp_result;
}
}

int main()
{
compressed_dense_block_0_t *dense_block_0_template_data = read_dense_block_0_from_file("/home/duzhen/spmv_builder/data_source/1085220678");

float* device_dense_0_val_arr;
unsigned int* device_dense_0_col_index_arr;

cudaMalloc(&device_dense_0_val_arr, sizeof(float) * 4649600);
cudaMemcpy(device_dense_0_val_arr, dense_block_0_template_data->dense_0_val_arr, sizeof(float) * 4649600, cudaMemcpyHostToDevice);

cudaMalloc(&device_dense_0_col_index_arr, sizeof(unsigned int) * 4649600);
cudaMemcpy(device_dense_0_col_index_arr, dense_block_0_template_data->dense_0_col_index_arr, sizeof(unsigned int) * 4649600, cudaMemcpyHostToDevice);

float *host_y_arr = (float *)malloc(sizeof(float) * 929920);
float *host_x_arr = (float *)malloc(sizeof(float) * 303645);

float *device_y_arr = NULL;
float *device_x_arr = NULL;

cudaMalloc(&device_y_arr, sizeof(float) * 929920);
cudaMalloc(&device_x_arr, sizeof(float) * 303645);

for (unsigned long i = 0; i < 929920; i++)
{
host_y_arr[i] = 0;
}

for (unsigned long i = 0; i < 303645; i++)
{
host_x_arr[i] = 100;
}

cudaMemcpy(device_y_arr, host_y_arr, sizeof(float) * 929920, cudaMemcpyHostToDevice);
cudaMemcpy(device_x_arr, host_x_arr, sizeof(float) * 303645, cudaMemcpyHostToDevice);

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

spmv_0<<<120, 512, 0, stream_arr[0]>>>(device_dense_0_val_arr, device_dense_0_col_index_arr, device_x_arr, device_y_arr);

cudaDeviceSynchronize();
}
gettimeofday(&end, NULL);

long timeuse = 1000000 * (end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
double gflops = ((double)2.0 * 4020750 * 1200 / ((double)timeuse / 1000000)) / 1000000000;

printf("time=%fms, gflops=%f\n",timeuse /1000.0, gflops);
cudaMemcpy(host_y_arr, device_y_arr, sizeof(float) * 929920, cudaMemcpyDeviceToHost);
print_arr_to_file_with_data_type(host_y_arr, FLOAT, 929920, "/home/duzhen/spmv_builder/data_source/test_result_3");

return 0;
}

