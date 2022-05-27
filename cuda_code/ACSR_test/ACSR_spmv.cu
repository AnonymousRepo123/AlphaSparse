#include <bits/stdc++.h>
#include "utilities.h"
#include "io.h"

using namespace std;

// 桶的数量
#define BIN_MAX 30
#define THREAD_LOAD 5

// 计算2的次方
unsigned int pow_2(unsigned int power)
{
	unsigned int result_num = 1;

	for (unsigned int i = 0; i < power; i++)
	{
		result_num = result_num * 2;
	}

	return result_num;
}

// 只需要warp层次归约的短行kernal，每一行分配的线程数量是2的(桶号+1)个线程
// 每个线程负责的非零元数量不超过两个
__global__ void spmv_short_row(unsigned int *row_offset_arr, unsigned int *col_index_arr, float *val_arr, float *x, float *y, unsigned int *row_index_of_this_bin, unsigned int row_num_of_this_bin, int m, int n, int nnz, int thread_num_of_each_row, int bin_index)
{
	// 获取当前线程的一系列基本数据
	// int bid = blockIdx.x;
	// int tid_in_block = threadIdx.x;
	// int wid_in_block = threadIdx.x / 32;
	// 一个向量负责一行
	int tid_in_vec = threadIdx.x % thread_num_of_each_row;

	// int tid_in_warp = threadIdx.x % 32;
	int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
	// int global_wid = global_tid / 32;
	int global_vid = global_tid / thread_num_of_each_row;

	// 线程的数量
	int thread_num = gridDim.x * blockDim.x;
	// 向量的数量
	int vec_num = thread_num / thread_num_of_each_row;

	float sum = 0;

	// assert(false);

	// 每个向量负责一行，遍历所有的行
	for (unsigned int bin_row_id = global_vid; bin_row_id < row_num_of_this_bin; bin_row_id = bin_row_id + vec_num)
	{
		// 获取真正的行号
		unsigned int row_id = row_index_of_this_bin[bin_row_id];

		// if (global_tid == 0)
		// {
		// 	printf("row_id:%u\n", row_id);
		// }

		// assert(false);
		// 获取当前行要遍历的非零元范围
		unsigned int this_row_begin_global_nz = row_offset_arr[row_id];
		unsigned int next_row_begin_global_nz = row_offset_arr[row_id + 1];

		// if (global_tid == 0)
		// {
		// 	printf("next_row_begin_global_nz:%u\n", next_row_begin_global_nz);
		// }

		// assert(false);

		// 执行一行中对应非零元和中间结果的计算
		sum = 0;

		for (unsigned int global_nz = this_row_begin_global_nz + tid_in_vec; global_nz < next_row_begin_global_nz; global_nz = global_nz + thread_num_of_each_row)
		{
			sum = sum + val_arr[global_nz] * __ldg(&(x[col_index_arr[global_nz]]));
		}

		// if (global_tid == 0)
		// {
		// 	printf("sum:%f\n", sum);
		// }

		// assert(false);

		// 根据当前桶号执行一个归约，对于树状归约来说，桶号就是树状归约的次数
		int offset = 16;
		for (int i = bin_index; i > 0; i--)
		{
			sum = sum + __shfl_down_sync(0xFFFFFFFF, sum, offset);
			offset = offset / 2;
		}

		// 写全局内存
		if (tid_in_vec == 0)
		{
			y[row_id] = sum;
		}

		sum = 0;
	}
}

// 行非零元数量超过63，使用一行一个block的方式，但是block中线程的数量是一个可调参数
__global__ void spmv_long_row(float *val_arr, unsigned int *col_index_arr, unsigned int *row_offset_arr, float *x, float *y, unsigned int *row_index_of_this_bin, unsigned int row_num_of_this_bin, int m, int n, int nnz)
{
	// 获取当前线程的一系列基本数据
	int bid = blockIdx.x;
	int tid_in_block = threadIdx.x;
	int wid_in_block = threadIdx.x / 32;

	int tid_in_warp = threadIdx.x % 32;
	// int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
	// int global_wid = global_tid / 32;

	// 线程的数量
	// int thread_num = gridDim.x * blockDim.x;

	float sum = 0;

	__shared__ float tmp_result_in_block[32];

	// 很据bid决定要处理的行号
	for (unsigned int bin_row_id = bid; bin_row_id < row_num_of_this_bin; bin_row_id = bin_row_id + gridDim.x)
	{
		__syncthreads();
		// 使用第一个warp初始化
		if (wid_in_block == 0)
		{
			tmp_result_in_block[tid_in_warp] = 0;
		}
		__syncthreads();

		// 获取真正的行号
		unsigned int row_id = row_index_of_this_bin[bin_row_id];
		// 获取当前行要遍历的非零元范围
		unsigned int this_row_begin_global_nz = row_offset_arr[row_id];
		unsigned int next_row_begin_global_nz = row_offset_arr[row_id + 1];

		// 执行一行中对应非零元和中间结果的计算
		sum = 0;

		// 处理一行
		for (unsigned int global_nz = this_row_begin_global_nz + tid_in_block; global_nz < next_row_begin_global_nz; global_nz = global_nz + blockDim.x)
		{
			sum = sum + val_arr[global_nz] * __ldg(&(x[col_index_arr[global_nz]]));
		}

		// warp内规约
		for (int offset = 16; offset > 0; offset = offset / 2)
		{
			sum = sum + __shfl_down_sync(0xFFFFFFFF, sum, offset);
		}

		// tid_in_warp
		if (tid_in_warp == 0)
		{
			tmp_result_in_block[wid_in_block] = sum;
		}

		// 块内规约
		__syncthreads();

		// 用一个线程算出所有结果
		if (tid_in_block == 0)
		{
			sum = 0;
			for (int i = 0; i < 32; i++)
			{
				sum = sum + tmp_result_in_block[i];
			}

			y[row_id] = sum;
		}

		__syncthreads();
	}
}

float *driver(float *val_arr, unsigned int *col_index_arr, unsigned int *row_offset_arr, float *x, float *y, int m, int n, int nnz, int row_nnz_max, int repeat_num, float &exe_time, float &exe_gflops)
{
	// print_arr_to_file_with_data_type(row_offset_arr, UNSIGNED_LONG, m + 1, "/home/duzhen/spmv_builder/data_source/test_result_3");

	// exit(-1);
	// 查看桶的数量
	int actual_bin_num = 0;

	// 一般不可能装不下
	bool is_found = false;

	for (unsigned int i = 0; i < BIN_MAX; i++)
	{
		// 每个桶的下界为2^i，每个桶的行非零元数量不超过2^(i+1)-1
		unsigned int row_num_low_bound = pow_2(i);
		unsigned int row_num_up_bound = pow_2(i + 1) - 1;

		if (row_nnz_max >= row_num_low_bound && row_nnz_max <= row_num_up_bound)
		{
			actual_bin_num = i + 1;
			is_found = true;
			break;
		}
	}

	assert(is_found);
	assert(actual_bin_num <= BIN_MAX);

	// cout << "actual_bin_num:" << actual_bin_num << endl;

	// 记录0行的数量
	unsigned int num_of_zero_row = 0;

	// 用一个数组不同桶的行号，空桶不启动kernal
	vector<vector<unsigned int>> row_index_of_each_bin(BIN_MAX);

	// 遍历所有的行，获取每一行的非零元数量
	for (unsigned int i = 0; i < m; i++)
	{
		assert(row_offset_arr[i + 1] >= row_offset_arr[i]);
		unsigned int cur_row_nnz = row_offset_arr[i + 1] - row_offset_arr[i];

		assert(cur_row_nnz <= row_nnz_max);

		is_found = false;
		// 遍历不同的bin范围，为其分类
		for (unsigned int j = 0; j < actual_bin_num; j++)
		{
			unsigned int row_num_low_bound = pow_2(j);
			unsigned int row_num_up_bound = pow_2(j + 1) - 1;

			if (cur_row_nnz >= row_num_low_bound && cur_row_nnz <= row_num_up_bound)
			{
				row_index_of_each_bin[j].push_back(i);
				is_found = true;
				break;
			}
		}

		if (is_found == false && cur_row_nnz != 0)
		{
			cout << "cur_row_nnz" << cur_row_nnz << endl;
			assert(false);
		}

		if (cur_row_nnz == 0)
		{
			num_of_zero_row++;
		}
	}

	// 所有桶中的行数量加起来，打印每个桶的非零元数量
	unsigned int row_num_sum = 0;
	for (unsigned int i = 0; i < row_index_of_each_bin.size(); i++)
	{
		row_num_sum = row_num_sum + row_index_of_each_bin[i].size();
		// cout << "bin" << i << " => " << row_index_of_each_bin[i].size() << endl;
	}

	assert(row_num_sum + num_of_zero_row == m);

	// 用一个数组存储每个bin要处理的行号的GPU数组，
	vector<unsigned int *> drow_index_ptr_of_each_bin(BIN_MAX);

	for (unsigned int i = 0; i < BIN_MAX; i++)
	{
		drow_index_ptr_of_each_bin[i] = NULL;
	}

	gettimeofday(&pre_end, NULL);

	double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;

	printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

	// 遍历host的行号
	for (unsigned int i = 0; i < row_index_of_each_bin.size(); i++)
	{
		unsigned int *drow_index_of_this_bin = NULL;

		if (row_index_of_each_bin[i].size() != 0)
		{
			// 创建一个GPU的数组
			cudaMalloc(&(drow_index_of_this_bin), sizeof(unsigned int) * row_index_of_each_bin[i].size());

			// 将行号拷贝GPU中
			cudaMemcpy(drow_index_of_this_bin, &(row_index_of_each_bin[i][0]), sizeof(unsigned int) * row_index_of_each_bin[i].size(), cudaMemcpyHostToDevice);

			drow_index_ptr_of_each_bin[i] = drow_index_of_this_bin;
		}
	}

	// 将CSR的三个主要数组拷贝到GPU
	unsigned int *drow_offset_arr;
	unsigned int *dcol_index_arr;
	float *dval_arr;
	float *d_y;
	float *d_x;

	cudaMalloc(&drow_offset_arr, sizeof(unsigned int) * (m + 1));
	cudaMalloc(&dcol_index_arr, sizeof(unsigned int) * nnz);
	cudaMalloc(&dval_arr, sizeof(float) * nnz);
	cudaMalloc(&d_y, sizeof(float) * m);
	cudaMalloc(&d_x, sizeof(float) * n);

	// 拷贝四个数组
	cudaMemcpy(drow_offset_arr, row_offset_arr, sizeof(unsigned int) * (m + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(dcol_index_arr, col_index_arr, sizeof(unsigned int) * nnz, cudaMemcpyHostToDevice);
	cudaMemcpy(dval_arr, val_arr, sizeof(float) * nnz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, sizeof(float) * n, cudaMemcpyHostToDevice);

	// 这里调用一系列kernal，使用流的方式
	cudaStream_t stream_arr[30];
	for (unsigned int i = 0; i < 30; i++)
	{
		cudaStreamCreate(&(stream_arr[i]));
	}

	cudaDeviceSynchronize();

	struct timeval start, end;
	gettimeofday(&start, NULL);

	for (unsigned int repeat = 0; repeat < repeat_num; repeat++)
	{
		// 如果行非零元数量不是0，就调用一个内核函运行桶中的数据
		for (unsigned int i = 0; i < 30; i++)
		{
			if (row_index_of_each_bin[i].size() != 0)
			{
				assert(drow_index_ptr_of_each_bin[i] != NULL);

				if (i <= 5)
				{
					// 短行
					unsigned int row_num_of_bin = row_index_of_each_bin[i].size();
					// 当前每一行的线程数量
					unsigned int thread_num_of_row = pow_2(i);
					// 总共的线程数量
					unsigned int total_thread_num = row_num_of_bin * thread_num_of_row;
					// block的数量
					int block_num = (total_thread_num / 128) + 1;
					// __global__ void spmv_short_row(unsigned int* row_offset_arr, unsigned int* col_index_arr, float* val_arr, float *x, float *y, unsigned int* row_index_of_this_bin, unsigned int row_num_of_this_bin, int m, int n, int nnz, int thread_num_of_each_row, int bin_index)
					spmv_short_row<<<block_num, 128, 0, stream_arr[i]>>>(drow_offset_arr, dcol_index_arr, dval_arr, d_x, d_y, drow_index_ptr_of_each_bin[i], row_index_of_each_bin[i].size(), m, n, nnz, thread_num_of_row, i);
				}
				else
				{
					// 长行，一行一个block
					unsigned int row_num_of_bin = row_index_of_each_bin[i].size();

					// 线程块的线程数量和bin的行非零元下界保持一致
					unsigned int thread_num_in_block = pow_2(i);

					if (thread_num_in_block > 1024)
					{
						thread_num_in_block = 1024;
					}
					// __global__ void spmv_long_row(float* val_arr, unsigned int* col_index_arr, unsigned int* row_offset_arr,  float *x, float *y, unsigned int* row_index_of_this_bin, unsigned int row_num_of_this_bin, int m, int n, int nnz)
					spmv_long_row<<<row_num_of_bin, thread_num_in_block, 0, stream_arr[i]>>>(dval_arr, dcol_index_arr, drow_offset_arr, d_x, d_y, drow_index_ptr_of_each_bin[i], row_index_of_each_bin[i].size(), m, n, nnz);
				}
			}
		}
		cudaDeviceSynchronize();
	}

	gettimeofday(&end, NULL);

	long timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
	double gflops = ((double)2.0 * nnz * repeat_num / ((double)timeuse / 1000000)) / 1000000000;

	exe_time = (float)timeuse / 1000.0;
	exe_gflops = gflops;

	printf("time=%fms, gflops=%f\n", timeuse / 1000.0, gflops);

	// 将数据从外面拷贝进来
	cudaMemcpy(y, d_y, sizeof(float) * m, cudaMemcpyDeviceToHost);

	// print_arr_to_file_with_data_type(y, FLOAT, m, "/home/duzhen/spmv_builder/data_source/test_result_3");

	return y;
}

int main(int argc, char ** argv)
{
	int n, m, nnz = 0;
	int nnz_max;
	float *x;
	srand(time(NULL)); //Set current time as random seed.

	string file_name = argv[1];

	// conv("/home/duzhen/spmv_builder/data_source/Si34H36.mtx.coo", nnz, m, n, nnz_max);
	conv(file_name, nnz, m, n, nnz_max);

	cout << "nnz:" << nnz << endl;
	cout << "m:" << m << endl;
	cout << "n:" << n << endl;
	cout << "nnz_max:" << nnz_max << endl;

	x = vect_gen(n);
	float *y = (float *)malloc(m * sizeof(float));
	float *res = new float[m];

	// 用两个变量分别把kernel的执行时间和gflops获得
	float exe_time = 99999999;
	float exe_gflops = 0;

	// 第一次运行5000个
	y = driver(values, col_idx, row_off, x, y, m, n, nnz, nnz_max, 5000, exe_time, exe_gflops);

	// 第二次运行的执行次数
	int final_repeat_time = 5000 * ((float)1000 / exe_time);

	exe_time = 99999999;
	exe_gflops = 0;

	y = driver(values, col_idx, row_off, x, y, m, n, nnz, nnz_max, final_repeat_time, exe_time, exe_gflops);

	// 最后打印输出
	printf("ACSR:time=%fms, gflops=%f\n", exe_time, exe_gflops);

	cout << "\n\n";
}