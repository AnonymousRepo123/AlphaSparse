#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <string.h>
#include <sys/time.h>
#include <bits/stdc++.h>

using namespace std;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int main(int argc, char ** argv)
{
    string data_file_name = argv[1];
    unsigned long row_num;
    unsigned long column_num;
    unsigned long nnz;

    // 从外部读入数据
    string d;
	d = data_file_name;
	std::ifstream fin(d.c_str());
	if (!fin)
	{
		cout << "File Not found\n";
		exit(0);
	}

	//int row_length, column_length, nnz;

	// Ignore headers and comments:
	while (fin.peek() == '%')
		fin.ignore(2048, '\n');

	// Read defining parameters:
	fin >> row_num >> column_num >> nnz;

	// 在ELL中获取一个真正的nnz，真正的row_num
	// Create your matrix:
	int *row, *column;
	float *coovalues;
	row = new int[nnz];
	column = new int[nnz];
	coovalues = new float[nnz];

	// values = (float *)malloc(sizeof(float) * nnz);
	// col_idx = (unsigned int*)malloc(sizeof(unsigned int) * nnz);
	
	// Read the data
	for (int l = 0; l < nnz; l++)
	{
		int m, n;
		float data;


		fin >> m >> n >> data;

        assert(m >= 1 && n >= 1);
        assert(m <= row_num && n <= column_num);
		
		row[l] = m - 1;
		column[l] = n - 1;
        coovalues[l] = data;
	}

    // x向量和y向量，根据行数量申请
    float *X = new float[column_num];
    for (unsigned long i = 0; i < column_num; i++)
    {
        X[i] = 1;
    }

    float *Y = new float[row_num];
    for (unsigned long i = 0; i < row_num; i++)
    {
        Y[i] = 0;
    }
    
    
    float alpha        = 1.0f;
    float beta         = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_rows, *dA_columns;
    float *dA_values, *dX, *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_rows,    nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  nnz * sizeof(float))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         column_num * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         row_num * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_rows, row, nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, column, nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, coovalues, nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, X, column_num * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, Y, row_num * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in COO format
    CHECK_CUSPARSE( cusparseCreateCoo(&matA, row_num, column_num, nnz,
                                      dA_rows, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, column_num, dX, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, row_num, dY, CUDA_R_32F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // 计时，循环5000次
    cudaDeviceSynchronize();

	struct timeval start,end;
	gettimeofday(&start, NULL);

    int repeat_num = 5000;

    
	
	for (unsigned int i = 0; i < repeat_num; i++)
	{
        // execute SpMV
        CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                    CUSPARSE_MV_ALG_DEFAULT, dBuffer) )
        cudaDeviceSynchronize();
    }

    gettimeofday(&end, NULL);

	long timeuse = 1000000 * (end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
	double gflops = ((double)2.0 * nnz * repeat_num / ((double)timeuse / 1000000)) / 1000000000;

	float exe_time = (float)timeuse / 1000.0;
	float exe_gflops = gflops;

    printf("time=%fms, gflops=%f\n", exe_time, exe_gflops);

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(Y, dY, row_num * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    // int correct = 1;
    // for (int i = 0; i < A_num_rows; i++) {
    //     if (hY[i] != hY_result[i]) { // direct floating point comparison is not
    //         correct = 0;             // reliable
    //         break;
    //     }
    // }
    // if (correct)
    //     printf("spmv_coo_example test PASSED\n");
    // else
    //     printf("spmv_coo_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_rows) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )
    return 1;
}