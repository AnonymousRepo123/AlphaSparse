// 一个coo格式cuda程序
#include <cuda_runtime.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <string.h>

using namespace std;

__global__ void spmv(double* y, double* x, unsigned int* row_index_arr, unsigned int* col_index_arr, double* val_arr, unsigned int thread_num, unsigned int row_number, unsigned int col_number, unsigned int nnz){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 一个线程负责一位
    int i;
    // 本质上遍历colindex，如果没有进一步分块，那每个线程交错负责每一个块的非零元
    for(i = tid; i < nnz; i = i + thread_num){
        // 首先找出colindex，然后找出对应非零元的rowindex
        unsigned int col_index = col_index_arr[i];
        double val = val_arr[i];

        // 获取要乘的数据
        unsigned x_val = x[i];
        double temp_result = x_val * val;
        

        // 直接在全局内存中规约
        // 对于规约位置的计算
        unsigned int globel_reduce_position = row_index_arr[i];
        atomicAdd(&(y[globel_reduce_position]), temp_result);
    }
}

int main(){
    // 从外部读入coo文件，行索引和列索引，val和x的内容先瞎弄
    char buffer[1024];

    // 以读模式打开文件
    ifstream fin("/home/duzhen/spmv_builder/data_source/block_0_index_read_0", std::ios::in);
    ifstream fin2("/home/duzhen/spmv_builder/data_source/block_0_index_read_1",  std::ios::in);
    
    // 用一个数组存一下输入的数据
    vector<unsigned int> row_index_arr;
    vector<unsigned int> col_index_arr;

    if (fin.is_open() && fin2.is_open())
    {
        while (fin.good() && !fin.eof())
        {
            string line_str;
            memset(buffer, 0, 1024);
            fin.getline(buffer, 1024);
            line_str = buffer;

            // 碰到奇怪的输入就跳过
            if (isspace(line_str[0]) || line_str.empty())
            {
                continue;
            }

            row_index_arr.push_back(strtoul(line_str.c_str(), NULL, 10));
        }

        while(fin2.good() && !fin2.eof()){
            string line_str;
            memset(buffer, 0, 1024);
            fin2.getline(buffer, 1024);
            line_str = buffer;

            // 碰到奇怪的输入就跳过
            if (isspace(line_str[0]) || line_str.empty())
            {
                continue;
            }

            col_index_arr.push_back(strtoul(line_str.c_str(), NULL, 10));
        }
    }

    // infile >> data;
    fin.close();
    fin2.close();

    cout << row_index_arr.size() << "," << col_index_arr.size() << endl;

    // 随机的val和x以及空的y
    double* val_arr = new double[row_index_arr.size()];
    double* x = new double[200000];
    double* y = new double[826721];

    int i;
    for(i = 0; i < row_index_arr.size(); i++){
        val_arr[i] = 0.9;
    }

    for(i = 0; i < 200000; i++){
        x[i] = 1.1;
    }

    double* val_arr_device = NULL;
    double* x_arr_device = NULL;
    double* y_arr_device = NULL;
    unsigned int* row_index_arr_device = NULL;
    unsigned int* col_index_arr_device = NULL;
    
    cudaMalloc((void**)&val_arr_device, row_index_arr.size() * sizeof(double));
    cudaMalloc((void**)&x_arr_device, 200000 * sizeof(double));
    cudaMalloc((void**)&y_arr_device, 826721 * sizeof(double));
    cudaMalloc((void**)&row_index_arr_device, row_index_arr.size() * sizeof(unsigned int));
    cudaMalloc((void**)&col_index_arr_device, col_index_arr.size() * sizeof(unsigned int));
    
    cudaMemcpy(val_arr_device, val_arr, row_index_arr.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_arr_device, x, 200000 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_arr_device, y, 826721 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(row_index_arr_device, &(row_index_arr[0]), row_index_arr.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(col_index_arr_device, &(col_index_arr[0]), col_index_arr.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

    spmv<<<60,128>>>(y_arr_device, x_arr_device, row_index_arr_device, col_index_arr_device, val_arr_device, 60 * 128, 826721, 200000, col_index_arr.size());

    cout << y[400000] << endl;
    // 拷贝回来
    cudaMemcpy(y, y_arr_device, 826721 * sizeof(double), cudaMemcpyDeviceToHost);

    cout << y[400000] << endl;
}