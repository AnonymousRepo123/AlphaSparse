// 本质上就是测试相似规模，但是irregular不同的矩阵在传统格式上的区别这里主要是处理COO
// 要比较的格式是circuit5M和Hardesty3，他们的大小类似，但是分别是regular和irregular的
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include <cuda_runtime.h>

using namespace std;

// 将COO文件读入，改变数组的内容
void get_coo_from_file(string file_name, vector<unsigned int> &row_index, vector<unsigned int> &col_index, vector<float> &val_arr, unsigned int& row_num, unsigned int& col_num, unsigned int& val_num)
{
    assert(row_index.size() == 0);
    assert(col_index.size() == 0);
    assert(val_arr.size() == 0);

    std::ifstream fin(file_name.c_str());
	if (!fin)
	{
		cout << "File Not found\n";
		exit(0);
	}

    while (fin.peek() == '%')
    {
        fin.ignore(2048, '\n');
    }

    fin >> row_num >> col_num >> val_num;

    // 读文件，主要是读行的和列
    for (unsigned int l = 0; l < val_num; l++)
    {
        // 将坐标和数值读出来
        long m, n;
        float value;

        // 假设每一行三个元素
        fin >> m >> n >> value;

        // 每一行三个元素
        // 增加一个元素
        assert(m > 0 && n > 0);
        assert(m <= row_num && n <= col_num);

        row_index.push_back(m - 1);
        col_index.push_back(n - 1);
        val_arr.push_back(1);
    }

    assert(val_arr.size() == val_num);

    fin.close();    
}

// 这里是核函数，每个线程一个非零元，最终按照行来处理归约
__global__ void spmv(const float * values, const unsigned int * row_idx, const unsigned int * col_idx, float* dvect, float * res, unsigned int row_num, unsigned int col_num, unsigned int nnz)
{
    // 查看当前的网格结构
    // 当前线程号
	unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
	// 线程的数量
	unsigned int total_thread_num = blockDim.x * gridDim.x;

    // 遍历所有的非零元
    for (unsigned int i = global_tid; i < nnz; i = i + total_thread_num)
    {
        unsigned int row_index = row_idx[i];
        unsigned int col_index = col_idx[i];
        float val = values[i];

        float temp_result = val * dvect[col_index];

        atomicAdd(&(res[row_index]), temp_result);
    }
}



int main()
{
    // 将一个文件读入
    string file_name = "/home/duzhen/matrix_suite/circuit5M/circuit5M.mtx";

    unsigned int repeat_num = 5000;

    // COO格式的属性
    vector<unsigned int> row_index;
    vector<unsigned int> col_index;
    vector<float> val_arr;
    unsigned int row_num;
    unsigned int col_num;
    unsigned int val_num;

    get_coo_from_file(file_name, row_index, col_index, val_arr, row_num, col_num, val_num);

    // 创造y矩阵和x矩阵
    vector<float> y;
    
    // y的大小为行的数量
    for (unsigned int i = 0; i < row_num; i++)
    {
        y.push_back(0);
    }

    vector<float> x;
    
    // x的长度为列的数量
    for (unsigned int i = 0; i < col_num; i++)
    {
        x.push_back(1);
    }

    // 创造一堆device
    unsigned int *drow_idx = NULL;
    unsigned int *dcol_idx = NULL;
    float *dvect = NULL;
    float *dres = NULL;
    float *dvalues = NULL;

    cudaMalloc((void **)&drow_idx, (val_num) * sizeof(unsigned int));
    cudaMalloc((void **)&dcol_idx, (val_num) * sizeof(unsigned int));
    cudaMalloc((void **)&dvect, (col_num) * sizeof(float));
    cudaMalloc((void **)&dres, (row_num) * sizeof(float));
    cudaMalloc((void **)&dvalues, (val_num) * sizeof(float));

    cudaMemcpy(drow_idx, &(row_index[0]), (val_num) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(dcol_idx, &(col_index[0]), (val_num) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(dvalues, &(val_arr[0]), (val_num) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dvect, &(x[0]), (col_num) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dres, &(y[0]), (row_num) * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaDeviceSynchronize();

	struct timeval start,end;
	gettimeofday(&start, NULL);
    // 运行核函数
    
    for (unsigned int i = 0; i < repeat_num; i++)
    {
        // __global__ void spmv(const float * values, const unsigned int * row_idx, const unsigned int * col_idx, float* dvect, float * res, unsigned int row_num, unsigned int col_num, unsigned int nnz)
        spmv<<<136, 128>>>(dvalues, drow_idx, dcol_idx, dvect, dres, row_num, col_num, val_num);
		cudaDeviceSynchronize();
    }

    gettimeofday(&end, NULL);

	long timeuse = 1000000 * (end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
	double gflops = ((double)2.0 * val_num * repeat_num / ((double)timeuse / 1000000)) / 1000000000;

	float exe_time = (float)timeuse / 1000.0;
	float exe_gflops = gflops;
    
    printf("time=%fms, gflops=%f\n", exe_time, exe_gflops);

    float *kres = (float *)malloc(val_num * sizeof(float));
    cudaMemcpy(kres, dres, (val_num) * sizeof(float), cudaMemcpyDeviceToHost);
}