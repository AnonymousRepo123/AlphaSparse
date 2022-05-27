#include <cuda_runtime.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <string.h>
#include "sell_template.h"
#include <cuda.h>

__global__ void spmv_0(unsigned int * block_nz_begin_offset, unsigned char * thread_block_size_in_block, double * val_arr, unsigned int * col_index_arr, double * device_x_arr, double * device_y_arr)
{
int bid = blockIdx.x;
int tid_in_block = threadIdx.x;
int bnum = gridDim.x;
__shared__ unsigned int block_first_nz_index_shared[1];
__shared__ unsigned char thread_block_size_in_block_shared[1];
for(unsigned short block_level_block_id = bid; block_level_block_id < 1824; block_level_block_id = block_level_block_id + bnum)
{

unsigned int block_first_nz_of_this_block;
unsigned int first_thread_index_of_this_block;
unsigned int first_thread_index_of_next_block;
unsigned char thread_block_size_of_this_block;

__syncthreads();

if(tid_in_block == 0)
{
block_first_nz_index_shared[0] = block_nz_begin_offset[block_level_block_id];
thread_block_size_in_block_shared[0] = thread_block_size_in_block[block_level_block_id];
}

__syncthreads();

block_first_nz_of_this_block = block_first_nz_index_shared[0];
first_thread_index_of_this_block = block_level_block_id * 512;
thread_block_size_of_this_block = thread_block_size_in_block_shared[0];

unsigned int thread_level_block_num_of_block = 512;

{
unsigned int thread_level_block_id = first_thread_index_of_this_block + tid_in_block;
unsigned int thread_level_block_first_nz_index = block_first_nz_of_this_block + tid_in_block;

double thread_block_tmp_result = 0;

unsigned int global_nz_index = thread_level_block_first_nz_index;
for(unsigned char inner_thread_nz_level_id = 0; inner_thread_nz_level_id < thread_block_size_of_this_block; inner_thread_nz_level_id++)
{
thread_block_tmp_result = thread_block_tmp_result + val_arr[global_nz_index] * device_x_arr[col_index_arr[global_nz_index]];
global_nz_index = global_nz_index + thread_level_block_num_of_block;
}

unsigned int global_row_index;

global_row_index = thread_level_block_id;
device_y_arr[global_row_index] = thread_block_tmp_result;

}

}

}

int main()
{
compressed_dense_block_0_t *dense_block_0_template_data = read_dense_block_0_from_file("/home/duzhen/spmv_builder/data_source/0");

unsigned int* device_dense_0_block_nz_begin_offset;
unsigned char* device_dense_0_thread_block_size_in_block;
double* device_dense_0_val_arr;
unsigned int* device_dense_0_col_index_arr;

cudaMalloc(&device_dense_0_block_nz_begin_offset, sizeof(unsigned int) * 1824);
cudaMemcpy(device_dense_0_block_nz_begin_offset, dense_block_0_template_data->dense_0_block_nz_begin_offset, sizeof(unsigned int) * 1824, cudaMemcpyHostToDevice);

cudaMalloc(&device_dense_0_thread_block_size_in_block, sizeof(unsigned char) * 1824);
cudaMemcpy(device_dense_0_thread_block_size_in_block, dense_block_0_template_data->dense_0_thread_block_size_in_block, sizeof(unsigned char) * 1824, cudaMemcpyHostToDevice);

cudaMalloc(&device_dense_0_val_arr, sizeof(double) * 4027904);
cudaMemcpy(device_dense_0_val_arr, dense_block_0_template_data->dense_0_val_arr, sizeof(double) * 4027904, cudaMemcpyHostToDevice);

cudaMalloc(&device_dense_0_col_index_arr, sizeof(unsigned int) * 4027904);
cudaMemcpy(device_dense_0_col_index_arr, dense_block_0_template_data->dense_0_col_index_arr, sizeof(unsigned int) * 4027904, cudaMemcpyHostToDevice);

double *host_y_arr = (double *)malloc(sizeof(double) * 933888);
double *host_x_arr = (double *)malloc(sizeof(double) * 303645);

double *device_y_arr = NULL;
double *device_x_arr = NULL;

cudaMalloc(&device_y_arr, sizeof(double) * 933888);
cudaMalloc(&device_x_arr, sizeof(double) * 303645);

for (unsigned long i = 0; i < 933888; i++)
{
host_y_arr[i] = 0;
}

for (unsigned long i = 0; i < 303645; i++)
{
host_x_arr[i] = 100;
}

cudaMemcpy(device_y_arr, host_y_arr, sizeof(double) * 933888, cudaMemcpyHostToDevice);
cudaMemcpy(device_x_arr, host_x_arr, sizeof(double) * 303645, cudaMemcpyHostToDevice);

cudaStream_t stream_arr[1];
for(unsigned long i = 0; i < 1; i++)
{
cudaStreamCreate(&(stream_arr[i]));
}

cudaDeviceSynchronize();
struct timeval start,end;
gettimeofday(&start, NULL);
for (int i = 0; i < 2000; i++)
{

spmv_0<<<120, 512, 0, stream_arr[0]>>>(device_dense_0_block_nz_begin_offset, device_dense_0_thread_block_size_in_block, device_dense_0_val_arr, device_dense_0_col_index_arr, device_x_arr, device_y_arr);

cudaDeviceSynchronize();
}
gettimeofday(&end, NULL);

long timeuse = 1000000 * (end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
double gflops = ((double)2.0 * 4024718 * 2000 / (timeuse / 1000000)) / 1000000000;

printf("time=%fms, gflops=%f\n",timeuse /1000.0, gflops);
cudaMemcpy(host_y_arr, device_y_arr, sizeof(double) * 933888, cudaMemcpyDeviceToHost);
print_arr_to_file_with_data_type(host_y_arr, DOUBLE, 933888, "/home/duzhen/spmv_builder/data_source/test_result_3");

return 0;
}

