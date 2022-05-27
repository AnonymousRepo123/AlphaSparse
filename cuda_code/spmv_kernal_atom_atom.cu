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
#include "spmv_header.h"
#include <cuda.h>

using namespace std;

// 将所有的内容装进去
__global__ void spmv_0(unsigned int *device_sorted_row_index, double *dense_0_staggered_padding_val_arr, unsigned int *dense_0_read_index_0_index_arr, unsigned int *dense_0_read_index_1_index_arr,
                    unsigned int *dense_0_read_index_2_index_arr, unsigned int *dense_0_read_index_3_index_arr, unsigned int *dense_0_read_index_5_index_arr, unsigned int *dense_0_read_index_6_index_arr, 
                    unsigned int *dense_0_read_index_2_begin_index_in_tmp_row_csr_arr_of_block, unsigned int *dense_0_read_index_2_child_tmp_row_csr_index_arr, unsigned int *dense_0_read_index_3_begin_index_in_tmp_row_csr_arr_of_block,
                    unsigned int *dense_0_read_index_3_child_tmp_row_csr_index_arr, unsigned int *dense_0_read_index_2_coo_begin_index_arr, unsigned int *dense_0_read_index_3_coo_begin_index_arr, 
                    unsigned int *dense_0_read_index_3_coo_block_size_arr, unsigned int *dense_0_read_index_4_coo_block_size_arr, unsigned int *dense_0_read_index_2_index_of_the_first_row_arr, 
                    unsigned int *dense_0_read_index_3_index_of_the_first_row_arr, unsigned int *dense_0_read_index_4_index_of_the_first_row_arr, unsigned int *dense_0_read_index_2_row_number_of_block_arr,
                    unsigned int *dense_0_read_index_3_row_number_of_block_arr, double *device_y_arr, double *device_x_arr)
{
    int tid_in_warp = threadIdx.x % 32;
    int bid = blockIdx.x;

    int wid_in_block = (int)(threadIdx.x / 32);

    unsigned long kernal_first_row_num = 0;

    int bnum = gridDim.x;
    
    int wnum = blockDim.x / 32;


    for(int block_level_block_id = bid; block_level_block_id < 465; block_level_block_id = block_level_block_id + bnum){
        
        unsigned int block_first_row = dense_0_read_index_2_index_of_the_first_row_arr[block_level_block_id];
        
        unsigned int block_first_nz = dense_0_read_index_2_coo_begin_index_arr[block_level_block_id];
        
        // 遍历相关
        int first_warp_index_in_this_block = dense_0_read_index_2_index_arr[block_level_block_id];
        int first_warp_index_in_next_block = dense_0_read_index_2_index_arr[block_level_block_id + 1];

        for(int warp_level_block_id = first_warp_index_in_this_block + wid_in_block; warp_level_block_id < first_warp_index_in_next_block; warp_level_block_id = warp_level_block_id + wnum){
            unsigned int warp_first_row =dense_0_read_index_3_index_of_the_first_row_arr[warp_level_block_id];
            
            unsigned int warp_first_nz = dense_0_read_index_3_coo_begin_index_arr[warp_level_block_id];
            
            unsigned int thread_block_size_in_warp = dense_0_read_index_4_coo_block_size_arr[warp_level_block_id];
            
            unsigned int first_thread_index_in_this_warp = dense_0_read_index_3_index_arr[warp_level_block_id];
            unsigned int first_thread_index_in_next_warp = dense_0_read_index_3_index_arr[warp_level_block_id + 1];
            
            
            unsigned int group_num_in_this_warp = (first_thread_index_in_next_warp - first_thread_index_in_this_warp) / 32;

            // 按照32的步长遍历所有的thread块，在warp块内部，线程块又以32为单位分成组，这一层遍历本质上是遍历每个组
            for(int thread_level_block_group_id = 0; thread_level_block_group_id < group_num_in_this_warp; thread_level_block_group_id = thread_level_block_group_id + 1){
                // warp内线程组的起始位置
                unsigned int thread_block_group_first_nz = block_first_nz + warp_first_nz + thread_level_block_group_id * 32 * thread_block_size_in_warp;

                unsigned int thread_level_block_index_inner_warp = thread_level_block_group_id * 32 + tid_in_warp;
                
                // 当前thread粒度块的全局总块号
                unsigned int global_thread_block_index = dense_0_read_index_3_index_arr[warp_level_block_id] + thread_level_block_index_inner_warp;
                
                // 用一个寄存器存储一个thread内部的计算结果
                double thread_block_tmp_result = 0;
                
                // 这里遍历的是一个group内部的thread块，本质上已经被交错存储了，这里每个tid负责一个group内的线程粒度的块，这里的遍历有问题，
                for(int nz_index_in_thread = 0; nz_index_in_thread < thread_block_size_in_warp; nz_index_in_thread = nz_index_in_thread + 1){
                    
                    // 当前非零元的
                    unsigned int global_nz_index = thread_block_group_first_nz + nz_index_in_thread * 32 + tid_in_warp;
                    // 最本质的代码，包含三个最本质的变量，当前非零元的索引、当前非零元的列号，之前的所有的铺垫都是为了这个
                    thread_block_tmp_result = thread_block_tmp_result + dense_0_staggered_padding_val_arr[global_nz_index] * device_x_arr[dense_0_read_index_6_index_arr[global_nz_index]];
                }
                
                // 获得全局的行号
                unsigned long global_row_index = device_sorted_row_index[kernal_first_row_num + block_first_row + warp_first_row + dense_0_read_index_4_index_of_the_first_row_arr[global_thread_block_index]];
                
                // 写结果，最本质的是电气概念非零元的全局行号。
                atomicAdd(&(device_y_arr[global_row_index]), thread_block_tmp_result);

            }
        }
    }
}

int main()
{
    all_compressed_block_t *total_matrix = read_matrix_from_file("/home/duzhen/spmv_builder/data_source/857772456");

    // 检查一下里面的内容是不是有空的
    assert(total_matrix->sorted_row_index != NULL);
    // 查看第一个和最后一个是不是正确
    assert(total_matrix->sorted_row_index[0] == 0 && total_matrix->sorted_row_index[total_matrix->size_of_sorted_row_index - 1] == 929900);

    // 第一个密集矩阵
    compressed_matrix_content_t *compressed_block = total_matrix->all_compressed_matrix_info;

    // 申请一系列显存，将所有内容扔给显存
    unsigned int *device_sorted_row_index;
    double *device_dense_0_staggered_padding_val_arr;
    unsigned int *device_dense_0_read_index_0_index_arr;
    unsigned int *device_dense_0_read_index_1_index_arr;
    unsigned int *device_dense_0_read_index_2_index_arr;
    unsigned int *device_dense_0_read_index_3_index_arr;
    unsigned int *device_dense_0_read_index_5_index_arr;
    unsigned int *device_dense_0_read_index_6_index_arr;

    unsigned int *device_dense_0_read_index_2_begin_index_in_tmp_row_csr_arr_of_block;
    unsigned int *device_dense_0_read_index_2_child_tmp_row_csr_index_arr;
    unsigned int *device_dense_0_read_index_3_begin_index_in_tmp_row_csr_arr_of_block;
    unsigned int *device_dense_0_read_index_3_child_tmp_row_csr_index_arr;

    unsigned int *device_dense_0_read_index_2_coo_begin_index_arr;
    unsigned int *device_dense_0_read_index_3_coo_begin_index_arr;

    unsigned int *device_dense_0_read_index_3_coo_block_size_arr;
    unsigned int *device_dense_0_read_index_4_coo_block_size_arr;

    unsigned int *device_dense_0_read_index_2_index_of_the_first_row_arr;
    unsigned int *device_dense_0_read_index_3_index_of_the_first_row_arr;
    unsigned int *device_dense_0_read_index_4_index_of_the_first_row_arr;

    unsigned int *device_dense_0_read_index_2_row_number_of_block_arr;
    unsigned int *device_dense_0_read_index_3_row_number_of_block_arr;

    // 申请对应的显存
    cudaMalloc(&device_sorted_row_index, sizeof(unsigned long) * total_matrix->size_of_sorted_row_index);
    cudaMalloc(&device_dense_0_staggered_padding_val_arr, sizeof(double) * compressed_block->size_of_staggered_padding_val_arr);
    cudaMalloc(&device_dense_0_read_index_0_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_0_index_arr);
    cudaMalloc(&device_dense_0_read_index_1_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_1_index_arr);
    cudaMalloc(&device_dense_0_read_index_2_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_index_arr);
    cudaMalloc(&device_dense_0_read_index_3_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_index_arr);
    cudaMalloc(&device_dense_0_read_index_5_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_5_index_arr);
    cudaMalloc(&device_dense_0_read_index_6_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_6_index_arr);

    cudaMalloc(&device_dense_0_read_index_2_begin_index_in_tmp_row_csr_arr_of_block, sizeof(unsigned int) * compressed_block->size_of_read_index_2_begin_index_in_tmp_row_csr_arr_of_block);
    cudaMalloc(&device_dense_0_read_index_2_child_tmp_row_csr_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_child_tmp_row_csr_index_arr);
    cudaMalloc(&device_dense_0_read_index_3_begin_index_in_tmp_row_csr_arr_of_block, sizeof(unsigned int) * compressed_block->size_of_read_index_3_begin_index_in_tmp_row_csr_arr_of_block);
    cudaMalloc(&device_dense_0_read_index_3_child_tmp_row_csr_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_child_tmp_row_csr_index_arr);

    cudaMalloc(&device_dense_0_read_index_2_coo_begin_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_coo_begin_index_arr);
    cudaMalloc(&device_dense_0_read_index_3_coo_begin_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_coo_begin_index_arr);

    cudaMalloc(&device_dense_0_read_index_3_coo_block_size_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_coo_block_size_arr);
    cudaMalloc(&device_dense_0_read_index_4_coo_block_size_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_4_coo_block_size_arr);

    cudaMalloc(&device_dense_0_read_index_2_index_of_the_first_row_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_index_of_the_first_row_arr);
    cudaMalloc(&device_dense_0_read_index_3_index_of_the_first_row_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_index_of_the_first_row_arr);
    cudaMalloc(&device_dense_0_read_index_4_index_of_the_first_row_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_4_index_of_the_first_row_arr);

    cudaMalloc(&device_dense_0_read_index_2_row_number_of_block_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_row_number_of_block_arr);
    cudaMalloc(&device_dense_0_read_index_3_row_number_of_block_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_row_number_of_block_arr);

    // 对对应的显存进行拷贝
    cudaMemcpy(device_sorted_row_index, total_matrix->sorted_row_index, sizeof(unsigned int) * total_matrix->size_of_sorted_row_index, cudaMemcpyHostToDevice);

    cudaMemcpy(device_dense_0_staggered_padding_val_arr, compressed_block->staggered_padding_val_arr, sizeof(double) * compressed_block->size_of_staggered_padding_val_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(device_dense_0_read_index_0_index_arr, compressed_block->read_index_0_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_0_index_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(device_dense_0_read_index_1_index_arr, compressed_block->read_index_1_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_1_index_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(device_dense_0_read_index_2_index_arr, compressed_block->read_index_2_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_index_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(device_dense_0_read_index_3_index_arr, compressed_block->read_index_3_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_index_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(device_dense_0_read_index_5_index_arr, compressed_block->read_index_5_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_5_index_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(device_dense_0_read_index_6_index_arr, compressed_block->read_index_6_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_6_index_arr, cudaMemcpyHostToDevice);

    cudaMemcpy(device_dense_0_read_index_2_begin_index_in_tmp_row_csr_arr_of_block, compressed_block->read_index_2_begin_index_in_tmp_row_csr_arr_of_block, sizeof(unsigned int) * compressed_block->size_of_read_index_2_begin_index_in_tmp_row_csr_arr_of_block, cudaMemcpyHostToDevice);
    cudaMemcpy(device_dense_0_read_index_2_child_tmp_row_csr_index_arr, compressed_block->read_index_2_child_tmp_row_csr_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_child_tmp_row_csr_index_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(device_dense_0_read_index_3_begin_index_in_tmp_row_csr_arr_of_block, compressed_block->read_index_3_begin_index_in_tmp_row_csr_arr_of_block, sizeof(unsigned int) * compressed_block->size_of_read_index_3_begin_index_in_tmp_row_csr_arr_of_block, cudaMemcpyHostToDevice);
    cudaMemcpy(device_dense_0_read_index_3_child_tmp_row_csr_index_arr, compressed_block->read_index_3_child_tmp_row_csr_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_child_tmp_row_csr_index_arr, cudaMemcpyHostToDevice);

    cudaMemcpy(device_dense_0_read_index_2_coo_begin_index_arr, compressed_block->read_index_2_coo_begin_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_coo_begin_index_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(device_dense_0_read_index_3_coo_begin_index_arr, compressed_block->read_index_3_coo_begin_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_coo_begin_index_arr, cudaMemcpyHostToDevice);

    cudaMemcpy(device_dense_0_read_index_3_coo_block_size_arr, compressed_block->read_index_3_coo_block_size_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_coo_block_size_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(device_dense_0_read_index_4_coo_block_size_arr, compressed_block->read_index_4_coo_block_size_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_4_coo_block_size_arr, cudaMemcpyHostToDevice);

    cudaMemcpy(device_dense_0_read_index_2_index_of_the_first_row_arr, compressed_block->read_index_2_index_of_the_first_row_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_index_of_the_first_row_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(device_dense_0_read_index_3_index_of_the_first_row_arr, compressed_block->read_index_3_index_of_the_first_row_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_index_of_the_first_row_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(device_dense_0_read_index_4_index_of_the_first_row_arr, compressed_block->read_index_4_index_of_the_first_row_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_4_index_of_the_first_row_arr, cudaMemcpyHostToDevice);

    cudaMemcpy(device_dense_0_read_index_2_row_number_of_block_arr, compressed_block->read_index_2_row_number_of_block_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_row_number_of_block_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(device_dense_0_read_index_3_row_number_of_block_arr, compressed_block->read_index_3_row_number_of_block_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_row_number_of_block_arr, cudaMemcpyHostToDevice);

    // 还有y数组和我们x数组
    double *host_y_arr = (double *)malloc(sizeof(double) * 929901);
    // bool* 
    double *host_x_arr = (double *)malloc(sizeof(double) * 303645);
    double *device_y_arr = NULL;
    double *device_x_arr = NULL;
    
    cudaMalloc(&device_y_arr, sizeof(double) * 929901);
    cudaMalloc(&device_x_arr, sizeof(double) * 303645);

    for (unsigned long i = 0; i < 929901; i++)
    {
        host_y_arr[i] = 0;
    }

    for (unsigned long i = 0; i < 303645; i++)
    {
        host_x_arr[i] = 100;
    }

    cudaMemcpy(device_y_arr, host_y_arr, sizeof(double) * 929901, cudaMemcpyHostToDevice);
    cudaMemcpy(device_x_arr, host_x_arr, sizeof(double) * 303645, cudaMemcpyHostToDevice);

    cudaStream_t stream_arr[1];
    
    for(unsigned long i = 0; i < 1; i++){
        cudaStreamCreate(&(stream_arr[i]));
    }

    cout << "begin kernal" << endl;
    // 3840和sp，30个SM，每个sm128个sp。启动120个block，每个block 512个线程
    spmv_0<<<120,512,0,stream_arr[0]>>>(device_sorted_row_index, device_dense_0_staggered_padding_val_arr, device_dense_0_read_index_0_index_arr, device_dense_0_read_index_1_index_arr, 
                       device_dense_0_read_index_2_index_arr, device_dense_0_read_index_3_index_arr, device_dense_0_read_index_5_index_arr, device_dense_0_read_index_6_index_arr, 
                       device_dense_0_read_index_2_begin_index_in_tmp_row_csr_arr_of_block, device_dense_0_read_index_2_child_tmp_row_csr_index_arr, device_dense_0_read_index_3_begin_index_in_tmp_row_csr_arr_of_block,
                       device_dense_0_read_index_3_child_tmp_row_csr_index_arr, device_dense_0_read_index_2_coo_begin_index_arr, device_dense_0_read_index_3_coo_begin_index_arr,
                       device_dense_0_read_index_3_coo_block_size_arr, device_dense_0_read_index_4_coo_block_size_arr, device_dense_0_read_index_2_index_of_the_first_row_arr, device_dense_0_read_index_3_index_of_the_first_row_arr,
                       device_dense_0_read_index_4_index_of_the_first_row_arr, device_dense_0_read_index_2_row_number_of_block_arr, device_dense_0_read_index_3_row_number_of_block_arr, device_y_arr, device_x_arr);




    cudaDeviceSynchronize();
    
    cout << "end kernal" << endl;

    // 将y数据拷贝出来
    cudaMemcpy(host_y_arr, device_y_arr, sizeof(double) * 929901, cudaMemcpyDeviceToHost);

    // 将y数据输出到文件中
    print_arr_to_file_with_data_type(host_y_arr, DOUBLE, 929901, "/home/duzhen/spmv_builder/data_source/test_result_1");
}