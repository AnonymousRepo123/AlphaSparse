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
__global__ void spmv(unsigned int *device_sorted_row_index, double *dense_0_staggered_padding_val_arr, unsigned int *dense_0_read_index_0_index_arr, unsigned int *dense_0_read_index_1_index_arr,
                    unsigned int *dense_0_read_index_2_index_arr, unsigned int *dense_0_read_index_3_index_arr, unsigned int *dense_0_read_index_5_index_arr, unsigned int *dense_0_read_index_6_index_arr, 
                    unsigned int *dense_0_read_index_2_begin_index_in_tmp_row_csr_arr_of_block, unsigned int *dense_0_read_index_2_child_tmp_row_csr_index_arr, unsigned int *dense_0_read_index_3_begin_index_in_tmp_row_csr_arr_of_block,
                    unsigned int *dense_0_read_index_3_child_tmp_row_csr_index_arr, unsigned int *dense_0_read_index_2_coo_begin_index_arr, unsigned int *dense_0_read_index_3_coo_begin_index_arr, 
                    unsigned int *dense_0_read_index_3_coo_block_size_arr, unsigned int *dense_0_read_index_4_coo_block_size_arr, unsigned int *dense_0_read_index_2_index_of_the_first_row_arr, 
                    unsigned int *dense_0_read_index_3_index_of_the_first_row_arr, unsigned int *dense_0_read_index_4_index_of_the_first_row_arr, unsigned int *dense_0_read_index_2_row_number_of_block_arr,
                    unsigned int *dense_0_read_index_3_row_number_of_block_arr, double *device_y_arr, double *device_x_arr)
{
    // 线程的全局索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_num_in_block = blockDim.x;
    // 线程的块内索引
    int tid_in_block = threadIdx.x;
    int tid_in_warp = threadIdx.x % 32;
    int bid = blockIdx.x;
    // warp的块内索引
    int wid_in_block = (int)(threadIdx.x / 32);

    int bnum = gridDim.x;
    
    // 一个线程块线程数量本来就是32的倍数，所以除一下就可以得出一个block中warp的数量
    int wnum = blockDim.x / 32;


    // shared memory只能容纳函数6000个double，
    // 产生的数据格式只有行分块
    // 因为在thread->warp，warp->block两个层次使用的是在共享内存中全局同步的规约，理论上需要两个共享内存，一个是warp内部的规约，一个是block内部的规约
    // 假设一个warp块最多64个线程，每个block最多32个warp，thread中间结果的数量和，warp中间结果的数量和每个warp行数量的总和
    // thread是按行分块，所以理论上不需要这一步
    __shared__ double thread_tmp_result[2048];

    // // warp层次只有按行分块，理论上不需要这一层规约，一个block最多2000行，warp之后行分块，所以这里是2000
    // // 不存在一行横跨两个warp的情况
    __shared__ double warp_tmp_result[2000];


    // thread->warp层次以及warp->block层次在共享内存内的同步归约，在block层次使用原子归约，因为block是按行分块，所以理论上不需要原子规约
    // 从当前bid开始，按照bnum的步长直到所有
    // 所有的内存访问都需要要assert
    for(int block_level_block_id = bid; block_level_block_id < 465; block_level_block_id = block_level_block_id + bnum){
        // 初始化完之后要带一个同步
        __syncthreads();

        // 当前块的首行
        assert(block_level_block_id < 465);
        unsigned int block_first_row = dense_0_read_index_2_index_of_the_first_row_arr[block_level_block_id];
        // 当前块要处理的第一个非零元
        assert(block_level_block_id < 466);
        unsigned int block_first_nz = dense_0_read_index_2_coo_begin_index_arr[block_level_block_id];
        
        // 要出的warp起始和结束的位置
        assert(block_level_block_id + 1 < 466);
        int first_warp_index_in_this_block = dense_0_read_index_2_index_arr[block_level_block_id];
        int first_warp_index_in_next_block = dense_0_read_index_2_index_arr[block_level_block_id + 1];
        
        // 当前block第一个thread块的索引
        assert(first_warp_index_in_this_block < 14880);
        int first_thread_index_in_this_block = dense_0_read_index_3_index_arr[first_warp_index_in_this_block];

        // 当前block的行数量
        assert(block_level_block_id < 465);
        int row_num_in_this_block = dense_0_read_index_2_row_number_of_block_arr[block_level_block_id];

        // warp_level_block_id本质上是这个warp对应的块的全局索引
        // 一个block的warp数量不超过32
        assert(first_warp_index_in_next_block - first_warp_index_in_this_block <= 32);
        for(int warp_level_block_id = first_warp_index_in_this_block + wid_in_block; warp_level_block_id < first_warp_index_in_next_block; warp_level_block_id = warp_level_block_id + wnum){
            // 遍历当前warp需要负责的范围
            assert(warp_level_block_id < 14879);
            unsigned int warp_first_row =dense_0_read_index_3_index_of_the_first_row_arr[warp_level_block_id];
            // 第一个非零元的索引
            assert(warp_level_block_id < 14879);
            unsigned int warp_first_nz = dense_0_read_index_3_coo_begin_index_arr[warp_level_block_id];
            // 当前warp内部thread块的大小
            assert(warp_level_block_id < 14879);
            unsigned int thread_block_size_in_warp = dense_0_read_index_4_coo_block_size_arr[warp_level_block_id];
            // 当前warp的第一个线程层次的块的全局索引
            assert(warp_level_block_id + 1 < 14880);
            unsigned int first_thread_index_in_this_warp = dense_0_read_index_3_index_arr[warp_level_block_id];
            unsigned int first_thread_index_in_next_warp = dense_0_read_index_3_index_arr[warp_level_block_id + 1];
            assert(first_thread_index_in_next_warp - first_thread_index_in_this_warp <= 64);


            assert((first_thread_index_in_next_warp - first_thread_index_in_this_warp) % 32 == 0);
            
            unsigned int group_num_in_this_warp = (first_thread_index_in_next_warp - first_thread_index_in_this_warp) / 32;
            
            // 当前warp负责的行数量
            assert(warp_level_block_id < 14879);
            unsigned int row_num_in_this_warp = dense_0_read_index_3_row_number_of_block_arr[warp_level_block_id];

            // 最后一个warp有32个线程，不用理会
            // assert(group_num_in_this_warp == 2);
            // 按照32的步长遍历所有的thread块，在warp块内部，线程块又以32为单位分成组，这一层遍历本质上是遍历每个组
            for(int thread_level_block_group_id = 0; thread_level_block_group_id < group_num_in_this_warp; thread_level_block_group_id = thread_level_block_group_id + 1){
                // warp内线程组的起始位置
                unsigned int thread_block_group_first_nz = block_first_nz + warp_first_nz + thread_level_block_group_id * 32 * thread_block_size_in_warp;

                // 当前线程粒度的块在warp块内的相对索引
                
                assert(tid_in_warp < 32);
                unsigned int thread_level_block_index_inner_warp = thread_level_block_group_id * 32 + tid_in_warp;
                
                // 用一个寄存器存储一个thread内部的计算结果
                double thread_block_tmp_result = 0;
                
                // 这里遍历的是一个group内部的thread块，本质上已经被交错存储了，这里每个tid负责一个group内的线程粒度的块，这里的遍历有问题，
                for(int nz_index_in_thread = 0; nz_index_in_thread < thread_block_size_in_warp; nz_index_in_thread = nz_index_in_thread + 1){
                    
                    // 当前非零元的全局索引
                    unsigned int global_nz_index = thread_block_group_first_nz + nz_index_in_thread * 32 + tid_in_warp;

                    // 进行计算，累加结果
                    assert(global_nz_index < 4118720);
                    assert(dense_0_read_index_6_index_arr[global_nz_index] < 303645);
                    thread_block_tmp_result = thread_block_tmp_result + dense_0_staggered_padding_val_arr[global_nz_index] * device_x_arr[dense_0_read_index_6_index_arr[global_nz_index]];
                }
                
                
                // 将当前thread块的结果写到共享内存中，写的位置与其在block内部的索引号有关，当前warp的第一个thread块在block的索引是可以计算的
                unsigned int thread_level_block_index_inner_block = thread_level_block_index_inner_warp + first_thread_index_in_this_warp - first_thread_index_in_this_block;

                assert(thread_level_block_index_inner_block < 2048);
                
                assert(thread_tmp_result[thread_level_block_index_inner_block] == 0 );

                thread_tmp_result[thread_level_block_index_inner_block] = thread_block_tmp_result;
            }
            
            // 当前warp规约信息的范围，这里一个warp_block中的内容已经完成
            assert(warp_level_block_id < 14879);
            unsigned int warp_reduce_index_begin = dense_0_read_index_3_begin_index_in_tmp_row_csr_arr_of_block[warp_level_block_id];
            // unsigned int next_warp_reduce_index_begin = dense_0_read_index_3_begin_index_in_tmp_row_csr_arr_of_block[warp_level_block_id + 1];
            
            // 这里是必然同步的，至少保证了一个warp block内的所有线程结果已经产生
            // 每个warp用和行数量相同数量的线程来规约，因为缺乏tmp_result_write_index_arr数组，这里理论上不应该这么写
            // 每个线程交错处理对应的行
            // 这里归约一个warp粒度的block内部的结果
            for(int row_index_in_warp = tid_in_warp; row_index_in_warp < row_num_in_this_warp; row_index_in_warp = row_index_in_warp + 32){
                // 要遍历的中间结果的起始位置
                assert(row_index_in_warp + warp_reduce_index_begin + 1 < 944780);
                // dense_0_read_index_3_child_tmp_row_csr_index_arr本质上是warp内thread中间结果的偏移量，所以先要算出次warp的第一个thread的在block层次内的索引，然后再加上这里的偏移量

                unsigned int child_result_begin_index = dense_0_read_index_3_child_tmp_row_csr_index_arr[row_index_in_warp + warp_reduce_index_begin];
                unsigned int next_child_result_begin_index = dense_0_read_index_3_child_tmp_row_csr_index_arr[row_index_in_warp + warp_reduce_index_begin + 1];
                
                unsigned int child_rsult_begin_index_inner_block = child_result_begin_index + first_thread_index_in_this_warp - first_thread_index_in_this_block;
                unsigned int next_child_result_begin_index_inner_block = next_child_result_begin_index + first_thread_index_in_this_warp - first_thread_index_in_this_block;
                
                double row_temp_result = 0;
                for(int index_of_inner_row_tmp_result = child_rsult_begin_index_inner_block; index_of_inner_row_tmp_result < next_child_result_begin_index_inner_block; index_of_inner_row_tmp_result++){
                    // 在warp中间结果中写的位置本质上要由tmp_result_write_index_arr数组决定，但是这里没有，所以就写到block块内行号对应的位置
                    assert(index_of_inner_row_tmp_result < 2048);
                    row_temp_result = row_temp_result + thread_tmp_result[index_of_inner_row_tmp_result];
                }

                assert(warp_first_row + row_index_in_warp < 2000);
                assert(warp_tmp_result[warp_first_row + row_index_in_warp] == 0);

                warp_tmp_result[warp_first_row + row_index_in_warp] = row_temp_result;
            }
        }

        __syncthreads();
        // 在block中规约warp层次的索引，然后直接原子加到全局内存中，遍历所有的warp产生的中间结果，加完的结果根据全局行号放到全局内存中
        // 归约信息数组的起始位置
        unsigned int block_reduce_index_begin = dense_0_read_index_2_begin_index_in_tmp_row_csr_arr_of_block[block_level_block_id];
        // 按行遍历各自的中间结果，每个线程负责一行，
        for(int row_index_in_block = tid_in_block; row_index_in_block < row_num_in_this_block; row_index_in_block = row_index_in_block + thread_num_in_block){
            // 一行在的中间结果的起始和结束位置
            unsigned int child_result_begin_index = dense_0_read_index_2_child_tmp_row_csr_index_arr[row_index_in_block + block_reduce_index_begin];
            unsigned int next_child_result_begin_index = dense_0_read_index_2_child_tmp_row_csr_index_arr[row_index_in_block + block_reduce_index_begin + 1];
            
            // 这一行的结果累加之后加到全局索引的对应位置
            double row_temp_result = 0;
            for(int index_of_inner_row_tmp_result = child_result_begin_index; index_of_inner_row_tmp_result < next_child_result_begin_index; index_of_inner_row_tmp_result++){
                row_temp_result = row_temp_result + warp_tmp_result[index_of_inner_row_tmp_result];
            }

            assert(block_first_row + row_index_in_block < 929901);
            unsigned int real_row_index = device_sorted_row_index[block_first_row + row_index_in_block];
            assert(real_row_index < 929901);
            // 之前不可能有人写这个位置吧
            assert(device_y_arr[real_row_index] == 0);

            atomicAdd(&(device_y_arr[real_row_index]), row_temp_result);
        }

        // 规约完了之后也需要一个同步，防止还没规约的部分被初始化
        __syncthreads();
    }
}

int main()
{
    // cuPrintInit();
    all_compressed_block_t *total_matrix = read_matrix_from_file("/home/duzhen/spmv_builder/data_source/857772456");

    // 检查一下里面的内容是不是有空的
    assert(total_matrix->compressed_matrix_vec.size() > 0);
    assert(total_matrix->sorted_row_index != NULL);
    // 查看第一个和最后一个是不是正确
    assert(total_matrix->sorted_row_index[0] == 0 && total_matrix->sorted_row_index[total_matrix->size_of_sorted_row_index - 1] == 929900);

    // 第一个密集矩阵
    compressed_matrix_content_t *compressed_block = total_matrix->compressed_matrix_vec[0];

    // 申请一系列显存，将所有内容扔给显存
    unsigned int *device_sorted_row_index;
    double *dense_0_staggered_padding_val_arr;
    unsigned int *dense_0_read_index_0_index_arr;
    unsigned int *dense_0_read_index_1_index_arr;
    unsigned int *dense_0_read_index_2_index_arr;
    unsigned int *dense_0_read_index_3_index_arr;
    unsigned int *dense_0_read_index_5_index_arr;
    unsigned int *dense_0_read_index_6_index_arr;

    unsigned int *dense_0_read_index_2_begin_index_in_tmp_row_csr_arr_of_block;
    unsigned int *dense_0_read_index_2_child_tmp_row_csr_index_arr;
    unsigned int *dense_0_read_index_3_begin_index_in_tmp_row_csr_arr_of_block;
    unsigned int *dense_0_read_index_3_child_tmp_row_csr_index_arr;

    unsigned int *dense_0_read_index_2_coo_begin_index_arr;
    unsigned int *dense_0_read_index_3_coo_begin_index_arr;

    unsigned int *dense_0_read_index_3_coo_block_size_arr;
    unsigned int *dense_0_read_index_4_coo_block_size_arr;

    unsigned int *dense_0_read_index_2_index_of_the_first_row_arr;
    unsigned int *dense_0_read_index_3_index_of_the_first_row_arr;
    unsigned int *dense_0_read_index_4_index_of_the_first_row_arr;

    unsigned int *dense_0_read_index_2_row_number_of_block_arr;
    unsigned int *dense_0_read_index_3_row_number_of_block_arr;

    // 申请对应的显存
    cudaMalloc(&device_sorted_row_index, sizeof(unsigned long) * total_matrix->size_of_sorted_row_index);
    cudaMalloc(&dense_0_staggered_padding_val_arr, sizeof(double) * compressed_block->size_of_staggered_padding_val_arr);
    cudaMalloc(&dense_0_read_index_0_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_0_index_arr);
    cudaMalloc(&dense_0_read_index_1_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_1_index_arr);
    cudaMalloc(&dense_0_read_index_2_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_index_arr);
    cudaMalloc(&dense_0_read_index_3_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_index_arr);
    cudaMalloc(&dense_0_read_index_5_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_5_index_arr);
    cudaMalloc(&dense_0_read_index_6_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_6_index_arr);

    cudaMalloc(&dense_0_read_index_2_begin_index_in_tmp_row_csr_arr_of_block, sizeof(unsigned int) * compressed_block->size_of_read_index_2_begin_index_in_tmp_row_csr_arr_of_block);
    cudaMalloc(&dense_0_read_index_2_child_tmp_row_csr_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_child_tmp_row_csr_index_arr);
    cudaMalloc(&dense_0_read_index_3_begin_index_in_tmp_row_csr_arr_of_block, sizeof(unsigned int) * compressed_block->size_of_read_index_3_begin_index_in_tmp_row_csr_arr_of_block);
    cudaMalloc(&dense_0_read_index_3_child_tmp_row_csr_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_child_tmp_row_csr_index_arr);

    cudaMalloc(&dense_0_read_index_2_coo_begin_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_coo_begin_index_arr);
    cudaMalloc(&dense_0_read_index_3_coo_begin_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_coo_begin_index_arr);

    cudaMalloc(&dense_0_read_index_3_coo_block_size_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_coo_block_size_arr);
    cudaMalloc(&dense_0_read_index_4_coo_block_size_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_4_coo_block_size_arr);

    cudaMalloc(&dense_0_read_index_2_index_of_the_first_row_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_index_of_the_first_row_arr);
    cudaMalloc(&dense_0_read_index_3_index_of_the_first_row_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_index_of_the_first_row_arr);
    cudaMalloc(&dense_0_read_index_4_index_of_the_first_row_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_4_index_of_the_first_row_arr);

    cudaMalloc(&dense_0_read_index_2_row_number_of_block_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_row_number_of_block_arr);
    cudaMalloc(&dense_0_read_index_3_row_number_of_block_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_row_number_of_block_arr);

    // 对对应的显存进行拷贝
    cudaMemcpy(device_sorted_row_index, total_matrix->sorted_row_index, sizeof(unsigned int) * total_matrix->size_of_sorted_row_index, cudaMemcpyHostToDevice);

    cudaMemcpy(dense_0_staggered_padding_val_arr, compressed_block->staggered_padding_val_arr, sizeof(double) * compressed_block->size_of_staggered_padding_val_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(dense_0_read_index_0_index_arr, compressed_block->read_index_0_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_0_index_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(dense_0_read_index_1_index_arr, compressed_block->read_index_1_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_1_index_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(dense_0_read_index_2_index_arr, compressed_block->read_index_2_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_index_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(dense_0_read_index_3_index_arr, compressed_block->read_index_3_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_index_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(dense_0_read_index_5_index_arr, compressed_block->read_index_5_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_5_index_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(dense_0_read_index_6_index_arr, compressed_block->read_index_6_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_6_index_arr, cudaMemcpyHostToDevice);

    cudaMemcpy(dense_0_read_index_2_begin_index_in_tmp_row_csr_arr_of_block, compressed_block->read_index_2_begin_index_in_tmp_row_csr_arr_of_block, sizeof(unsigned int) * compressed_block->size_of_read_index_2_begin_index_in_tmp_row_csr_arr_of_block, cudaMemcpyHostToDevice);
    cudaMemcpy(dense_0_read_index_2_child_tmp_row_csr_index_arr, compressed_block->read_index_2_child_tmp_row_csr_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_child_tmp_row_csr_index_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(dense_0_read_index_3_begin_index_in_tmp_row_csr_arr_of_block, compressed_block->read_index_3_begin_index_in_tmp_row_csr_arr_of_block, sizeof(unsigned int) * compressed_block->size_of_read_index_3_begin_index_in_tmp_row_csr_arr_of_block, cudaMemcpyHostToDevice);
    cudaMemcpy(dense_0_read_index_3_child_tmp_row_csr_index_arr, compressed_block->read_index_3_child_tmp_row_csr_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_child_tmp_row_csr_index_arr, cudaMemcpyHostToDevice);

    cudaMemcpy(dense_0_read_index_2_coo_begin_index_arr, compressed_block->read_index_2_coo_begin_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_coo_begin_index_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(dense_0_read_index_3_coo_begin_index_arr, compressed_block->read_index_3_coo_begin_index_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_coo_begin_index_arr, cudaMemcpyHostToDevice);

    cudaMemcpy(dense_0_read_index_3_coo_block_size_arr, compressed_block->read_index_3_coo_block_size_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_coo_block_size_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(dense_0_read_index_4_coo_block_size_arr, compressed_block->read_index_4_coo_block_size_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_4_coo_block_size_arr, cudaMemcpyHostToDevice);

    cudaMemcpy(dense_0_read_index_2_index_of_the_first_row_arr, compressed_block->read_index_2_index_of_the_first_row_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_index_of_the_first_row_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(dense_0_read_index_3_index_of_the_first_row_arr, compressed_block->read_index_3_index_of_the_first_row_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_index_of_the_first_row_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(dense_0_read_index_4_index_of_the_first_row_arr, compressed_block->read_index_4_index_of_the_first_row_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_4_index_of_the_first_row_arr, cudaMemcpyHostToDevice);

    cudaMemcpy(dense_0_read_index_2_row_number_of_block_arr, compressed_block->read_index_2_row_number_of_block_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_2_row_number_of_block_arr, cudaMemcpyHostToDevice);
    cudaMemcpy(dense_0_read_index_3_row_number_of_block_arr, compressed_block->read_index_3_row_number_of_block_arr, sizeof(unsigned int) * compressed_block->size_of_read_index_3_row_number_of_block_arr, cudaMemcpyHostToDevice);

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

    cout << "begin kernal" << endl;
    // 3840和sp，30个SM，每个sm128个sp。启动120个block，每个block 512个线程
    spmv<<<120,512>>>(device_sorted_row_index, dense_0_staggered_padding_val_arr, dense_0_read_index_0_index_arr, dense_0_read_index_1_index_arr, 
                       dense_0_read_index_2_index_arr, dense_0_read_index_3_index_arr, dense_0_read_index_5_index_arr, dense_0_read_index_6_index_arr, 
                       dense_0_read_index_2_begin_index_in_tmp_row_csr_arr_of_block, dense_0_read_index_2_child_tmp_row_csr_index_arr, dense_0_read_index_3_begin_index_in_tmp_row_csr_arr_of_block,
                       dense_0_read_index_3_child_tmp_row_csr_index_arr, dense_0_read_index_2_coo_begin_index_arr, dense_0_read_index_3_coo_begin_index_arr,
                       dense_0_read_index_3_coo_block_size_arr, dense_0_read_index_4_coo_block_size_arr, dense_0_read_index_2_index_of_the_first_row_arr, dense_0_read_index_3_index_of_the_first_row_arr,
                       dense_0_read_index_4_index_of_the_first_row_arr, dense_0_read_index_2_row_number_of_block_arr, dense_0_read_index_3_row_number_of_block_arr, device_y_arr, device_x_arr);

    cudaDeviceSynchronize();
    
    cout << "end kernal" << endl;

    // 将y数据拷贝出来
    cudaMemcpy(host_y_arr, device_y_arr, sizeof(double) * 929901, cudaMemcpyDeviceToHost);

    // 将y数据输出到文件中
    print_arr_to_file_with_data_type(host_y_arr, DOUBLE, 929901, "/home/duzhen/spmv_builder/data_source/test_result_0");
}