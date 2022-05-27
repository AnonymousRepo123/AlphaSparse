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


__global__ void spmv_0(unsigned int * global_first_row_index_of_warp_level_block, unsigned short * combine_meta_of_thread_level_block, float * val_arr, unsigned int * col_index_arr, float * device_x_arr, float * device_y_arr)
{
int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
int warp_id = global_tid / 32;
int tid_in_warp = global_tid % 32;
int warp_num = blockDim.x * gridDim.x / 32;

unsigned int WLB_id = warp_id;
if (WLB_id < 169314)
{
unsigned int WLB_first_row;
WLB_first_row = global_first_row_index_of_warp_level_block[WLB_id];
unsigned int global_TLB_id = WLB_id * 32 + tid_in_warp;

unsigned short combine_meta = combine_meta_of_thread_level_block[global_TLB_id];

unsigned short TLB_first_reduce_row = combine_meta << 11;
TLB_first_reduce_row = TLB_first_reduce_row >> 11;

unsigned int y_offset = TLB_first_reduce_row;

unsigned short reduce_offset = combine_meta << 7;
reduce_offset = reduce_offset >> 12;

bool through_sum_begin_bit = false;
float row_head_tmp_result = 0;
float row_other_tmp_result = 0;
float sum_tmp = 0;
unsigned int global_nz_index = WLB_id * 160 + tid_in_warp;

for (unsigned int TLB_nz_id = 0; TLB_nz_id < 5; TLB_nz_id++)
{
bool cur_sum_begin_bit = (combine_meta >> (15 - TLB_nz_id)) & 0x1;
if (cur_sum_begin_bit == true)
{
if (through_sum_begin_bit == true)
{
unsigned int global_row_index;
global_row_index = WLB_first_row + y_offset;
if (tid_in_warp == 0 && y_offset == TLB_first_reduce_row)
{
atomicAdd(&(device_y_arr[global_row_index]), sum_tmp);

}
else
{
device_y_arr[global_row_index] = sum_tmp;

}

}
else
{
row_other_tmp_result = sum_tmp;

}

}
y_offset = y_offset + (through_sum_begin_bit & cur_sum_begin_bit);

through_sum_begin_bit = through_sum_begin_bit || cur_sum_begin_bit;

sum_tmp = cur_sum_begin_bit ? 0 : sum_tmp;

sum_tmp = sum_tmp + val_arr[global_nz_index] * __ldg(&(device_x_arr[col_index_arr[global_nz_index]]));
global_nz_index = global_nz_index + 32;

}
row_other_tmp_result = through_sum_begin_bit ? row_other_tmp_result : sum_tmp;
row_head_tmp_result = sum_tmp;
row_other_tmp_result = __shfl_down_sync(0xFFFFFFFF, row_other_tmp_result, 1);
row_other_tmp_result = (tid_in_warp == 31) ? 0 : row_other_tmp_result;

float scan_tmp_sum = row_other_tmp_result;
float scan_tmp_sum_from_other_thread = __shfl_up_sync(0xFFFFFFFF, scan_tmp_sum, 1);
scan_tmp_sum = (tid_in_warp >= 1) ? scan_tmp_sum_from_other_thread + scan_tmp_sum : scan_tmp_sum;
scan_tmp_sum_from_other_thread = __shfl_up_sync(0xFFFFFFFF, scan_tmp_sum, 2);
scan_tmp_sum = (tid_in_warp >= 2) ? scan_tmp_sum_from_other_thread + scan_tmp_sum : scan_tmp_sum;
scan_tmp_sum_from_other_thread = __shfl_up_sync(0xFFFFFFFF, scan_tmp_sum, 4);
scan_tmp_sum = (tid_in_warp >= 4) ? scan_tmp_sum_from_other_thread + scan_tmp_sum : scan_tmp_sum;
scan_tmp_sum_from_other_thread = __shfl_up_sync(0xFFFFFFFF, scan_tmp_sum, 8);
scan_tmp_sum = (tid_in_warp >= 8) ? scan_tmp_sum_from_other_thread + scan_tmp_sum : scan_tmp_sum;
scan_tmp_sum_from_other_thread = __shfl_up_sync(0xFFFFFFFF, scan_tmp_sum, 16);
scan_tmp_sum = (tid_in_warp >= 16) ? scan_tmp_sum_from_other_thread + scan_tmp_sum : scan_tmp_sum;

scan_tmp_sum_from_other_thread = __shfl_down_sync(0xFFFFFFFF, scan_tmp_sum, reduce_offset);
row_other_tmp_result = scan_tmp_sum_from_other_thread - scan_tmp_sum + row_other_tmp_result;

row_head_tmp_result = through_sum_begin_bit ? row_head_tmp_result + row_other_tmp_result : 0;

if (through_sum_begin_bit == true)
{
unsigned int global_row_index;
global_row_index = WLB_first_row + y_offset;
if ((tid_in_warp == 0 && y_offset == TLB_first_reduce_row) || (tid_in_warp + reduce_offset == 31))
{
atomicAdd(&(device_y_arr[global_row_index]), sum_tmp);

}
else
{
device_y_arr[global_row_index] = sum_tmp;

}

}

}

}

int main()
{
int gpuDeviceCount = 0;
cudaGetDeviceCount(&gpuDeviceCount);
assert(0 <= (gpuDeviceCount - 1));

cudaSetDevice(0);

compressed_dense_block_0_t *dense_block_0_template_data = read_dense_block_0_from_file("/home/duzhen/spmv_builder/data_source/786855766_0");

unsigned int* device_dense_0_global_first_row_index_of_warp_level_block;
unsigned short* device_dense_0_combine_meta_of_thread_level_block;
float* device_dense_0_val_arr;
unsigned int* device_dense_0_col_index_arr;

cudaMalloc(&device_dense_0_global_first_row_index_of_warp_level_block, sizeof(unsigned int) * 169314);
cudaMemcpy(device_dense_0_global_first_row_index_of_warp_level_block, dense_block_0_template_data->dense_0_global_first_row_index_of_warp_level_block, sizeof(unsigned int) * 169314, cudaMemcpyHostToDevice);

cudaMalloc(&device_dense_0_combine_meta_of_thread_level_block, sizeof(unsigned short) * 5418048);
cudaMemcpy(device_dense_0_combine_meta_of_thread_level_block, dense_block_0_template_data->dense_0_combine_meta_of_thread_level_block, sizeof(unsigned short) * 5418048, cudaMemcpyHostToDevice);

cudaMalloc(&device_dense_0_val_arr, sizeof(float) * 27090240);
cudaMemcpy(device_dense_0_val_arr, dense_block_0_template_data->dense_0_val_arr, sizeof(float) * 27090240, cudaMemcpyHostToDevice);

cudaMalloc(&device_dense_0_col_index_arr, sizeof(unsigned int) * 27090240);
cudaMemcpy(device_dense_0_col_index_arr, dense_block_0_template_data->dense_0_col_index_arr, sizeof(unsigned int) * 27090240, cudaMemcpyHostToDevice);

float *host_y_arr = (float *)malloc(sizeof(float) * 1508065);
float *host_x_arr = (float *)malloc(sizeof(float) * 1508065);

float *device_y_arr = NULL;
float *device_x_arr = NULL;

cudaMalloc(&device_y_arr, sizeof(float) * 1508065);
cudaMalloc(&device_x_arr, sizeof(float) * 1508065);

for (unsigned long i = 0; i < 1508065; i++)
{
host_y_arr[i] = 0;
}

for (unsigned long i = 0; i < 1508065; i++)
{
host_x_arr[i] = 100;
}

cudaMemcpy(device_y_arr, host_y_arr, sizeof(float) * 1508065, cudaMemcpyHostToDevice);
cudaMemcpy(device_x_arr, host_x_arr, sizeof(float) * 1508065, cudaMemcpyHostToDevice);

cudaStream_t stream_arr[1];
for(unsigned long i = 0; i < 1; i++)
{
cudaStreamCreate(&(stream_arr[i]));
}

cudaDeviceSynchronize();
struct timeval start,end;
gettimeofday(&start, NULL);
for (int i = 0; i < 12883; i++)
{

spmv_0<<<21165, 256, 0, stream_arr[0]>>>(device_dense_0_global_first_row_index_of_warp_level_block, device_dense_0_combine_meta_of_thread_level_block, device_dense_0_val_arr, device_dense_0_col_index_arr, device_x_arr, device_y_arr);

cudaDeviceSynchronize();
}
gettimeofday(&end, NULL);

long timeuse = 1000000 * (end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
double gflops = ((double)2.0 * 27090195 * 12883 / ((double)timeuse / 1000000)) / 1000000000;

printf("time=%fms, gflops=%f\n",timeuse /1000.0, gflops);
cudaMemcpy(host_y_arr, device_y_arr, sizeof(float) * 1508065, cudaMemcpyDeviceToHost);
print_arr_to_file_with_data_type(host_y_arr, FLOAT, 1508065, "/home/duzhen/spmv_builder/data_source/test_result_3");
ofstream resultWrite("/home/duzhen/spmv_builder/cuda_code/perf_result", ios::out | ios::trunc);
resultWrite << timeuse /1000.0 << endl << gflops << endl;
resultWrite.close();

return 0;
}

