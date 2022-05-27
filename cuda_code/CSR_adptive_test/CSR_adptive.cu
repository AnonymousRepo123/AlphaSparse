#include <bits/stdc++.h>
#include "utilities.h"
#include <sys/time.h>
#include "io.h"

using namespace std;

// 每个块负责的首行行号要放进来
__global__ void spmv(const float *__restrict__ values, const unsigned int *__restrict__ col_idx, const unsigned int *__restrict__ row_off, const unsigned int *block_first_row_index, float *vector, float *__restrict__ res, unsigned int m, unsigned int n, unsigned int nnz, unsigned int block_num)
{
    __shared__ float shared_elements[1024];

    // 遍历所有的块
    for (unsigned int block_id = blockIdx.x; block_id < block_num; block_id += gridDim.x)
    {
        // 获取基本的数据，
        unsigned int row_start = block_first_row_index[block_id];
        unsigned int row_stop = block_first_row_index[block_id + 1];
        unsigned int element_start = row_off[row_start];
        unsigned int element_stop = row_off[row_stop];
        unsigned int rows_to_process = row_stop - row_start;

        if (rows_to_process > 1)
        {
            
            // if (element_stop - element_start > 1024)
            // {
            //     printf("element_stop - element_start:%ld\n", element_stop - element_start);
            //     assert(element_stop - element_start <= 1024);
            // }

            // 非零元数量没有超过1024，将每一位的内容算完之后
            for (unsigned int i = element_start + threadIdx.x; i < element_stop; i += blockDim.x)
            {
                // assert(i - element_start < 1024);
                // assert(col_idx[i] < 11284032);
                shared_elements[i - element_start] = values[i] * __ldg(&(vector[col_idx[i]]));
            }

            __syncthreads();

            // 归约
            for (unsigned int row = row_start + threadIdx.x; row < row_stop; row += blockDim.x)
            {
                // 遍历每一行
                float row_result = 0;
                // 非零元的索引
                unsigned int row_nz_begin = row_off[row] - element_start;
                unsigned int row_nz_stop = row_off[row + 1] - element_start;

                // 遍历非零元
                for (unsigned int i = row_nz_begin; i < row_nz_stop; ++i)
                {
                    row_result = row_result + shared_elements[i];
                }

                // 结果写到全局内存
                res[row] = row_result;
            }
        }
        else
        {
            shared_elements[threadIdx.x] = 0;

            // 交错处理一行的内容
            for (unsigned int i = element_start + threadIdx.x; i < element_stop; i += blockDim.x)
            {
                shared_elements[threadIdx.x] = shared_elements[threadIdx.x] + values[i] * vector[col_idx[i]];
            }

            // 归约
            for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
            {
                __syncthreads();
                if (threadIdx.x < stride)
                {
                    // assert(threadIdx.x+stride < 1024);
                    shared_elements[threadIdx.x] += shared_elements[threadIdx.x+stride];
                }
            }

            if (threadIdx.x == 0)
            {
                // 结果写回全局内存
                // assert(row_start < m);
                res[row_start] = shared_elements[0];
            }
        }

        __syncthreads();
    }
}


float *driver(float *values, unsigned int *col_idx, unsigned int *row_off, unsigned int *block_first_row_index, float *x, float *y, unsigned int m, unsigned int n, unsigned int nnz, unsigned int block_num, unsigned int repeat_num, float& exe_time, float& exe_gflops)
{
    unsigned int *dcol_idx, *drow_off;
    // 存储每一个块的首行地址
    unsigned int *dblock_first_row_index;
    float *dvect, *dres, *dvalues;

    cudaMalloc((void **)&dcol_idx, (nnz) * sizeof(unsigned int));
    cudaMalloc((void **)&drow_off, (m + 1) * sizeof(unsigned int));
    cudaMalloc((void **)&dblock_first_row_index, (block_num + 1) * sizeof(unsigned int));
    cudaMalloc((void **)&dvect, (n) * sizeof(float));
    cudaMalloc((void **)&dres, (m) * sizeof(float));
    cudaMalloc((void **)&dvalues, (nnz) * sizeof(float));

    // 将数据拷贝到显存
    cudaMemcpy(dcol_idx, col_idx, (nnz) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(drow_off, row_off, (m + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(dblock_first_row_index, block_first_row_index, (block_num + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(dvect, x, (n) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvalues, values, (nnz) * sizeof(float), cudaMemcpyHostToDevice);

    // 这个数组要从内存拷贝回来
    cudaMemset(dres, 0, n * sizeof(float));

    // 这里跑100遍，计算性能
    cudaDeviceSynchronize();
    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (unsigned int j = 0; j < repeat_num; j++)
    {
        spmv<<<32, 1024>>>(dvalues, dcol_idx, drow_off, dblock_first_row_index, dvect, dres, m, n, nnz, block_num);
        cudaDeviceSynchronize();
    }

    gettimeofday(&end, NULL);

    // cudaError_t cuda_err = cudaGetLastError();

    // if(cudaSuccess != cuda_err)
    // {
    //     fprintf(stderr,"%s\n", cudaGetErrorString(cuda_err));
    // }

    long timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    double gflops = ((double)2.0 * nnz * repeat_num / ((double)timeuse / 1000000)) / 1000000000;

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
    unsigned int block_num = 0;
    unsigned int nnz_max;
    float *x;
    srand(time(NULL)); //Set current time as random seed.

    // 真正的CSR adptive代码，有两个分支，太长的行由一个block负责，其他之后一个block处理1024个非零元

    // conv("/home/duzhen/spmv_builder/data_source/Si34H36.mtx.coo", nnz, m, n, nnz_max, block_num);
    string file_name = argv[1];

	// conv("/home/duzhen/spmv_builder/data_source/Si34H36.mtx.coo", nnz, m, n, nnz_max);
	conv(file_name, nnz, m, n, nnz_max, block_num);

    // 将预处理的数据打印出来
    cout << "nnz:" << nnz << endl;
    cout << "row_num:" << m << endl;
    cout << "col_num:" << n << endl;
    cout << "max_row_length:" << nnz_max << endl;
    cout << "block_num:" << block_num << endl;

    x = vect_gen(n);
    float *y = (float *)malloc(m * sizeof(float));

    gettimeofday(&pre_end, NULL);

    double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;

    printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

    float exe_time = 99999999;
    float exe_gflops = 0;

    y = driver(values, col_idx, row_off, block_first_row_index, x, y, m, n, nnz, block_num, 5000, exe_time, exe_gflops);

    // 新的重复数量
    int final_repeat_num = 5000 * ((float)1000 / exe_time);

    exe_time = 99999999;
	exe_gflops = 0;

	y = driver(values, col_idx, row_off, block_first_row_index, x, y, m, n, nnz, block_num, final_repeat_num, exe_time, exe_gflops);

    printf("CSR_Adaptive: time=%fms, gflops=%f\n", exe_time, exe_gflops);

    // 将结果写到文件中
    // print_arr_to_file_with_data_type(y, FLOAT, m, "/home/duzhen/spmv_builder/data_source/test_result_3");
    // print_arr_to_file_with_data_type(row_off, UNSIGNED_LONG, m + 1, "/home/duzhen/spmv_builder/data_source/test_result_4");
    // print_arr_to_file_with_data_type(values , FLOAT, m, "/home/duzhen/spmv_builder/data_source/test_result_5");
    // print_arr_to_file_with_data_type(block_first_row_index, UNSIGNED_LONG, block_num + 1, "/home/duzhen/spmv_builder/data_source/test_result_6");
    // print_arr_to_file_with_data_type(y, FLOAT, m, "/home/duzhen/spmv_builder/data_source/test_result_7");

    cout << "\n\n";
}

// __shared__ NumericT     shared_elements[1024];

//   for (unsigned int block_id = blockIdx.x; block_id < num_blocks; block_id += gridDim.x)
//   {
//     unsigned int row_start = row_blocks[block_id];
//     unsigned int row_stop  = row_blocks[block_id + 1];
//     unsigned int element_start = row_indices[row_start];
//     unsigned int element_stop = row_indices[row_stop];
//     unsigned int rows_to_process = row_stop - row_start;

//     if (rows_to_process > 1)  // CSR stream with one thread per row
//     {
//       // load to shared buffer:
//       for (unsigned int i = element_start + threadIdx.x; i < element_stop; i += blockDim.x)
//         shared_elements[i - element_start] = elements[i] * x[column_indices[i] * inc_x + start_x];

//       __syncthreads();

//       // use one thread per row to sum:
//       for (unsigned int row = row_start + threadIdx.x; row < row_stop; row += blockDim.x)
//       {
//         NumericT dot_prod = 0;
//         unsigned int thread_row_start = row_indices[row]     - element_start;
//         unsigned int thread_row_stop  = row_indices[row + 1] - element_start;
//         for (unsigned int i = thread_row_start; i < thread_row_stop; ++i)
//           dot_prod += shared_elements[i];
//         AlphaBetaHandlerT::apply(result[row * inc_result + start_result], alpha, dot_prod, beta);
//       }
//     }
//     // TODO here: Consider CSR vector for two to four rows (cf. OpenCL implementation. Experience on Fermi suggests that this may not be necessary)
//     else // CSR vector for a single row
//     {
//       // load and sum to shared buffer:
//       shared_elements[threadIdx.x] = 0;
//       for (unsigned int i = element_start + threadIdx.x; i < element_stop; i += blockDim.x)
//         shared_elements[threadIdx.x] += elements[i] * x[column_indices[i] * inc_x + start_x];

//       // reduction to obtain final result
//       for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
//       {
//         __syncthreads();
//         if (threadIdx.x < stride)
//           shared_elements[threadIdx.x] += shared_elements[threadIdx.x+stride];
//       }

//       if (threadIdx.x == 0)
//         AlphaBetaHandlerT::apply(result[row_start * inc_result + start_result], alpha, shared_elements[0], beta);
//     }

//     __syncthreads();  // avoid race conditions
//   }