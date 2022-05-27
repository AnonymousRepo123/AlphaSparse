#include <bits/stdc++.h>
#include "utilities.h"
#include "io.h"
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <string.h>
#include <sys/time.h>

using namespace std;


// 当一个warp只能处理一行的时候，

__global__ void spmv(const float * values, const unsigned int * col_idx, unsigned int row_length, float* dvect, float * res, unsigned int m, unsigned int n, unsigned int nnz)
{
	// 当前线程号
	unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 线程的数量
	unsigned int total_thread_num = blockDim.x * gridDim.x;
	
	// 遍历所有行
	for (unsigned int i = global_tid; i < m; i = i + total_thread_num)
	{
		float thread_block_tmp_result = 0;
		// 当前当的第一个非零元索引
		unsigned int global_index_nz = i;
		for (unsigned int nz_index = 0; nz_index < row_length; nz_index++)
		{
			thread_block_tmp_result = thread_block_tmp_result + values[global_index_nz] * dvect[col_idx[global_index_nz]];
			global_index_nz = global_index_nz + m;
		}

		res[i] = thread_block_tmp_result;
	}
}


// Matrix : m x n
// Vector : n x 1
float *driver(float *values, unsigned int *col_idx, unsigned int row_length, float *x, float *y, unsigned int m, unsigned int n, unsigned int nnz, int repeat_num, float& exe_time, float& exe_gflops)
{
	unsigned int *dcol_idx;
    float *dvect, *dres, *dvalues;

	cudaMalloc((void **)&dcol_idx, (nnz) * sizeof(unsigned int));
    cudaMalloc((void **)&dvect, (n) * sizeof(float));
    cudaMalloc((void **)&dres, (m) * sizeof(float));
    cudaMalloc((void **)&dvalues, (nnz) * sizeof(float));

	// 将数据拷贝到显存
	cudaMemcpy(dcol_idx, col_idx, (nnz) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(dvect, x, (n) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvalues, values, (nnz) * sizeof(float), cudaMemcpyHostToDevice);

	// 这个数组要从内存拷贝回来
    cudaMemset(dres, 0, n * sizeof(float));

	cudaStream_t stream_arr[1];
	for(unsigned int i = 0; i < 1; i++)
	{
		cudaStreamCreate(&(stream_arr[i]));
	}

	cudaDeviceSynchronize();

	struct timeval start,end;
	gettimeofday(&start, NULL);

	// 遍历200次
	for (unsigned int i = 0; i < repeat_num; i++)
	{
		spmv<<<96, 512, 0, stream_arr[0]>>>(dvalues, dcol_idx, row_length, dvect, dres, m, n, nnz);
		cudaDeviceSynchronize();
	}
	gettimeofday(&end, NULL);

	long timeuse = 1000000 * (end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
	double gflops = ((double)2.0 * origin_nnz * repeat_num / ((double)timeuse / 1000000)) / 1000000000;

	exe_time = (float)timeuse / 1000.0;
	exe_gflops = gflops;

	// printf("time=%fms, gflops=%f\n", timeuse / 1000.0, gflops);

	float *kres = (float *)malloc(m * sizeof(float));
    cudaMemcpy(kres, dres, (m) * sizeof(float), cudaMemcpyDeviceToHost);
    return kres;
}

int main(int argc, char ** argv)
{
	unsigned int n, m, nnz = 0;
	unsigned int row_nnz_max;
	float *x;
	srand(time(NULL)); //Set current time as random seed.

	string file_name = argv[1];

	// conv("/home/duzhen/spmv_builder/data_source/para-10.mtx.coo", nnz, m, n, row_nnz_max);
	conv(file_name, nnz, m, n, row_nnz_max);
	x = vect_gen(n);
	float *y = (float *)malloc(m * sizeof(float));

	cout << "row_num:" << m << endl;
	cout << "col_num:" << n << endl;
	cout << "row_nnz_max:" << row_nnz_max << endl;
	cout << "nnz:" << nnz << endl;
	
	gettimeofday(&pre_end, NULL);

    double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;

    printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

	float exe_time = 99999999;
	float exe_gflops = 0;

	y = driver(values, col_idx, row_nnz_max, x, y, m, n, nnz, 5000, exe_time, exe_gflops);

	int final_repeat_num = 5000 * ((float)1000 / exe_time);

	y = driver(values, col_idx, row_nnz_max, x, y, m, n, nnz, final_repeat_num, exe_time, exe_gflops);

	printf("time=%fms, gflops=%f\n", exe_time, exe_gflops);

	// 将结果写到磁盘里
	// print_arr_to_file_with_data_type(y, FLOAT, m, "/home/duzhen/spmv_builder/data_source/test_result_3");

	// cout << "\n\n";
}