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

    // // 初始化共享内存，应该不用初始化，每一个位置在写之前都不会被读到，每个线程块每次处理一个线程块粒度的数据时，需要参与规约的内容就需要被覆盖
    // for(int i = tid_in_block; i < 1024; i = i + 32){
    //     thread_tmp_result[i] = 0;
    // }

    // // warp层次只有按行分块，理论上不需要这一层规约，一个block最多2000行，warp之后行分块，所以这里是2000
    // // 不存在一行横跨两个warp的情况
    __shared__ double warp_tmp_result[2000];

    // for(int i = tid_in_block; i < 2000; i = i + 32){
    //     warp_tmp_result[i] = 0;
    // }

    // thread->warp层次以及warp->block层次在共享内存内的同步归约，在block层次使用原子归约，因为block是按行分块，所以理论上不需要原子规约
    // 从当前bid开始，按照bnum的步长直到所有
    // 所有的内存访问都需要要assert
    // 
    for(int block_level_block_id = bid; block_level_block_id < 465; block_level_block_id = block_level_block_id + bnum){
        __syncthreads();
        // 每次一开头先初始化两个中间数组
        for(int i = tid_in_block; i < 2048; i = i + 32){
            thread_tmp_result[i] = 0;
        }
        
        for(int i = tid_in_block; i < 2000; i = i + 32){
            warp_tmp_result[i] = 0;
        }

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

            // if((first_thread_index_in_next_warp - first_thread_index_in_this_warp) % 32 != 0){
            //     printf("error, first_thread_index_in_next_warp = %u, first_thread_index_in_this_warp = %u\n", first_thread_index_in_next_warp, first_thread_index_in_this_warp);
            //     assert(0);
            // }


            assert((first_thread_index_in_next_warp - first_thread_index_in_this_warp) % 32 == 0);
            
            unsigned int group_num_in_this_warp = (first_thread_index_in_next_warp - first_thread_index_in_this_warp) / 32;
            
            // 当前warp负责的行数量
            assert(warp_level_block_id < 14879);
            unsigned int row_num_in_this_warp = dense_0_read_index_3_row_number_of_block_arr[warp_level_block_id];
            
            // if(group_num_in_this_warp != 2){
            //     printf("group_num_in_this_warp:%u, warp_level_block_id:%u\n", group_num_in_this_warp, warp_level_block_id);
            // }

            // 最后一个warp有32个线程，不用理会
            // assert(group_num_in_this_warp == 2);
            // 按照32的步长遍历所有的thread块，在warp块内部，线程块又以32为单位分成组，这一层遍历本质上是遍历每个组
            for(int thread_level_block_group_id = 0; thread_level_block_group_id < group_num_in_this_warp; thread_level_block_group_id = thread_level_block_group_id + 1){
                // warp内线程组的起始位置
                unsigned int thread_block_group_first_nz = block_first_nz + warp_first_nz + thread_level_block_group_id * 32 * thread_block_size_in_warp;

                // 当前线程粒度的块在warp块内的相对索引
                
                assert(tid_in_warp < 32);
                unsigned int thread_level_block_index_inner_warp = thread_level_block_group_id * 32 + tid_in_warp;

                // if(first_thread_index_in_this_warp == 0 && first_thread_index_in_this_block == 0 && thread_level_block_group_id == 0){
                //     printf("thread_level_block_index_inner_warp:%u\n", thread_level_block_index_inner_warp);
                // }

                // assert(thread_level_block_index_inner_warp < 64);
                
                
                // 用一个寄存器存储一个thread内部的计算结果
                double thread_block_tmp_result = 0;
                
                // 这里遍历的是一个group内部的thread块，本质上已经被交错存储了，这里每个tid负责一个group内的线程粒度的块，这里的遍历有问题，
                for(int nz_index_in_thread = 0; nz_index_in_thread < thread_block_size_in_warp; nz_index_in_thread = nz_index_in_thread + 1){
                    
                    // 当前非零元的全局索引
                    unsigned int global_nz_index = thread_block_group_first_nz + nz_index_in_thread * 32 + tid_in_warp;
                    // if(block_level_block_id == 464){
                    //     printf("1:global_nz_index:%u, dense_0_staggered_padding_val_arr[global_nz_index]:%f, device_x_arr[dense_0_read_index_6_index_arr[global_nz_index]:%f, thread_block_tmp_result:%f, %f\n", global_nz_index, dense_0_staggered_padding_val_arr[global_nz_index], device_x_arr[dense_0_read_index_6_index_arr[global_nz_index]], thread_block_tmp_result, dense_0_staggered_padding_val_arr[global_nz_index] * device_x_arr[dense_0_read_index_6_index_arr[global_nz_index]]);
                    // }
                    // 进行计算，累加结果
                    assert(global_nz_index < 4118720);
                    assert(dense_0_read_index_6_index_arr[global_nz_index] < 303645);
                    thread_block_tmp_result = thread_block_tmp_result + dense_0_staggered_padding_val_arr[global_nz_index] * device_x_arr[dense_0_read_index_6_index_arr[global_nz_index]];

                    // if(thread_level_block_group_id == 0 && tid_in_warp == 17 && block_level_block_id == 313 && )
                    // {

                    // }
                    // if(block_level_block_id == 464){
                    //     printf("2:global_nz_index:%u, dense_0_staggered_padding_val_arr[global_nz_index]:%f, device_x_arr[dense_0_read_index_6_index_arr[global_nz_index]:%f, thread_block_tmp_result:%f, %f\n", global_nz_index, dense_0_staggered_padding_val_arr[global_nz_index], device_x_arr[dense_0_read_index_6_index_arr[global_nz_index]], thread_block_tmp_result, dense_0_staggered_padding_val_arr[global_nz_index] * device_x_arr[dense_0_read_index_6_index_arr[global_nz_index]]);
                    // }
                }

                // if(first_thread_index_in_this_warp == 0 && first_thread_index_in_this_block == 0 && thread_level_block_group_id == 0){
                //     printf("thread_level_block_index_inner_warp2:%ld\n", thread_level_block_index_inner_warp);
                // }
                
                
                // 将当前thread块的结果写到共享内存中，写的位置与其在block内部的索引号有关，当前warp的第一个thread块在block的索引是可以计算的
                unsigned int thread_level_block_index_inner_block = thread_level_block_index_inner_warp + first_thread_index_in_this_warp - first_thread_index_in_this_block;
                // if(thread_level_block_index_inner_block>=4 && thread_level_block_index_inner_block < 32){

                // if(first_thread_index_in_this_warp == 0 && first_thread_index_in_this_block == 0 && thread_level_block_group_id == 0){
                //     printf("thread_level_block_index_inner_warp:%u, thread_level_block_index_inner_block:%u, thread_block_tmp_result:%f\n", thread_level_block_index_inner_warp, thread_level_block_index_inner_block, thread_block_tmp_result);
                // }
                    
                // }
                // 将结果写到共享内存
                // if(thread_level_block_index_inner_block >= 2048){
                //     printf("block_level_block_id:%u\n", block_level_block_id);
                //     printf("error, thread_level_block_index_inner_block = %u, thread_level_block_index_inner_warp = %u, first_thread_index_in_this_warp = %u, first_thread_index_in_this_block = %u\n", thread_level_block_index_inner_block, thread_level_block_index_inner_warp, first_thread_index_in_this_warp, first_thread_index_in_this_block);
                //     assert(0);
                // }
                assert(thread_level_block_index_inner_block < 2048);
                
                assert(thread_tmp_result[thread_level_block_index_inner_block] == 0 );
                // if(thread_level_block_index_inner_block == 1553 && block_level_block_id == 313){
                //     printf("thread_level_block_index_inner_block:%u, thread_block_tmp_result:%f, thread_level_block_index_inner_warp:%u, thread_level_block_group_id:%u, tid_in_warp:%ld\n", thread_level_block_index_inner_block, thread_block_tmp_result, thread_level_block_index_inner_warp, thread_level_block_group_id, tid_in_warp);
                //     printf("warp_level_block_id:%u\n", warp_level_block_id);
                //     assert(0);
                // }
                thread_tmp_result[thread_level_block_index_inner_block] = thread_block_tmp_result;

                // 查看有没有写超过4号位置的位置
                // if(block_level_block_id == 0 && warp_level_block_id == 0 && tid_in_warp == 0){
                //     if(thread_level_block_index_inner_block >= 4 && thread_level_block_index_inner_block < 32){
                //         printf("write to %ld, thread_block_tmp_result\n", thread_level_block_index_inner_block);
                //         assert(0);
                //     }
                // }
            }

            // 查看这个时候1553号位置
            // if(block_level_block_id == 313 && tid_in_block == 0 && warp_level_block_id == 10040){
            //     printf("thread_tmp_result[1553]:%f\n", thread_tmp_result[1553]);
            // }

            // 打印thread_tmp_result中的内容
            // if(tid_in_block < 64 && block_level_block_id == 0){
            //     printf("i:%d, thread_tmp_result[i]:%f\n", tid_in_block*2, thread_tmp_result[tid_in_block*2]);
            //     assert(0);
            // }

            
            // if(tid_in_block == 0 && block_level_block_id == 0){
            //     for(int i = 0; i < 2048; i++){
            //         printf("i:%d, thread_tmp_result[i]:%f\n", i, thread_tmp_result[i]);
            //     }
            //     // assert(0);
            // }
            
            // 当前warp规约信息的范围，这里一个warp_block中的内容已经完成
            assert(warp_level_block_id < 14879);
            
            unsigned int warp_reduce_index_begin = dense_0_read_index_3_begin_index_in_tmp_row_csr_arr_of_block[warp_level_block_id];
            // 在warp没有纵切块的基础上，warp_reduce_index_begin就是warp第一行的索引。但是如果有纵分块，warp中间结果的数量就有冗余，就没有办法进行这样的对应
            
            // unsigned int next_warp_reduce_index_begin = dense_0_read_index_3_begin_index_in_tmp_row_csr_arr_of_block[warp_level_block_id + 1];
            
            // 这里是必然同步的，至少保证了一个warp block内的所有线程结果已经产生
            // 每个warp用和行数量相同数量的线程来规约，因为缺乏tmp_result_write_index_arr数组，这里理论上不应该这么写
            // 每个线程交错处理对应的行
            // 这里归约一个warp粒度的block内部的结果
            for(int row_index_in_warp = tid_in_warp; row_index_in_warp < row_num_in_this_warp; row_index_in_warp = row_index_in_warp + 32){
                // 要遍历的中间结果的起始位置
                assert(row_index_in_warp + warp_reduce_index_begin + 1 < 944780);
                // dense_0_read_index_3_child_tmp_row_csr_index_arr本质上是warp内thread中间结果的偏移量，所以先要算出次warp的第一个thread的在block层次内的索引，然后再加上这里的偏移量
                // 首先先收到归约信息的起始位置，然后找到归约信息（每一行中间结果在warp对应的thread中间结果的起始和结束位置），然后加上一个偏移量（warp第一个thread的块内地址）
                unsigned int child_result_begin_index = dense_0_read_index_3_child_tmp_row_csr_index_arr[row_index_in_warp + warp_reduce_index_begin];
                unsigned int next_child_result_begin_index = dense_0_read_index_3_child_tmp_row_csr_index_arr[row_index_in_warp + warp_reduce_index_begin + 1];
                
                unsigned int child_rsult_begin_index_inner_block = child_result_begin_index + first_thread_index_in_this_warp - first_thread_index_in_this_block;
                unsigned int next_child_result_begin_index_inner_block = next_child_result_begin_index + first_thread_index_in_this_warp - first_thread_index_in_this_block;

                // if(warp_first_row + row_index_in_warp == 1528 && block_level_block_id == 313){
                //     // 打印要规约的位置
                //     printf("child_rsult_begin_index_inner_block:%u, next_child_result_begin_index_inner_block:%u, row_index_in_warp:%u, warp_reduce_index_begin:%u\n", child_rsult_begin_index_inner_block, next_child_result_begin_index_inner_block, row_index_in_warp, warp_reduce_index_begin);
                // }
                
                double row_temp_result = 0;
                for(int index_of_inner_row_tmp_result = child_rsult_begin_index_inner_block; index_of_inner_row_tmp_result < next_child_result_begin_index_inner_block; index_of_inner_row_tmp_result++){
                    // 在warp中间结果中写的位置本质上要由tmp_result_write_index_arr数组决定，但是这里没有，所以就写到block块内行号对应的位置
                    assert(index_of_inner_row_tmp_result < 2048);
                    row_temp_result = row_temp_result + thread_tmp_result[index_of_inner_row_tmp_result];
                }

                assert(warp_first_row + row_index_in_warp < 2000);
                assert(warp_tmp_result[warp_first_row + row_index_in_warp] == 0);
                // if(block_level_block_id == 464){
                //     // 一个warp的第一个块，一共要写63行的数据
                //     printf("warp_first_row + row_index_in_warp:%u, row_temp_result:%f\n", warp_first_row + row_index_in_warp, row_temp_result);
                // }
                warp_tmp_result[warp_first_row + row_index_in_warp] = row_temp_result;
            }

            // if(tid_in_block == 0 && block_level_block_id == 0){
            //     for(int i = 0; i < 2048; i++){
            //         printf("i:%d, thread_tmp_result[i]:%f\n", i, thread_tmp_result[i]);
            //     }

            //     for(int i = 0; i < 2000; i++){
            //         printf("i:%d, warp_tmp_result[i]:%f\n", i, warp_tmp_result[i]);
            //     }

            //     assert(0);
            // }
            
        }

        // 一个block粒度的块的数据到这里全部算完，这里带一个同步
        
        // printf("block_level_block_id:%u\n", block_level_block_id);
        __syncthreads();
        // 打印线程和warp的中间结果
        // if(tid_in_block == 0 && block_level_block_id == 313){
        //     for(int i = 0; i < 2048; i++){
        //         printf("i:%d, thread_tmp_result2[i]:%f\n", i, thread_tmp_result[i]);
        //     }

        //     for(int i = 0; i < 2000; i++){
        //         printf("i:%d, warp_tmp_result[i]:%f\n", i, warp_tmp_result[i]);
        //     }

        //     assert(0);
        // }

        // if(block_level_block_id == 464 && tid_in_block == 0){
        //     for(int i = 0; i < 2000; i++){
        //         printf("i:%ld,warp_tmp_result[i]:%f\n", i, warp_tmp_result[i]);
        //     }
        //     assert(0);
        // }
        
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
                // if(block_first_row + row_index_in_block == 627528){
                //     printf("index_of_inner_row_tmp_result:%ld, warp_tmp_result[index_of_inner_row_tmp_result]:%f, block_level_block_id:%u\n", index_of_inner_row_tmp_result, warp_tmp_result[index_of_inner_row_tmp_result], block_level_block_id);
                //     assert(0);
                // }
            }

            assert(block_first_row + row_index_in_block < 929901);
            unsigned int real_row_index = device_sorted_row_index[block_first_row + row_index_in_block];
            assert(real_row_index < 929901);
            // 之前不可能有人写这个位置吧
            assert(device_y_arr[real_row_index] == 0);
            // if(real_row_index == 3){
            //     printf("write to 3, row_temp_result:%f, block_level_block_id:%u\n", row_temp_result, block_level_block_id);
            //     assert(0);
            // }

            // if(real_row_index == 626254){
            //     printf("block_first_row + row_index_in_block:%u, row_temp_result:%f\n", block_first_row + row_index_in_block, row_temp_result);
            //     assert(0);
            // }
            assert(device_y_arr[real_row_index] == 0);
            atomicAdd(&(device_y_arr[real_row_index]), row_temp_result);
        }

        // 规约完了之后也需要一个同步，防止还没规约的部分被初始化
        __syncthreads();
    }

    if(tid == 0){
        printf("finish\n");
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