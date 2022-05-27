#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h> 
#include <bits/stdc++.h>
#include "utilities.h"
#include <sys/time.h>
#include "io.h"

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

int main(int argc, char ** argv) {
    unsigned int n, m, nnz = 0;
    unsigned int block_num = 0;
    unsigned int nnz_max;
    float *X;

    string file_name = argv[1];

    conv(file_name, nnz, m, n, nnz_max, block_num);

    cout << "nnz:" << nnz << endl;
    cout << "row_num:" << m << endl;
    cout << "col_num:" << n << endl;
    cout << "max_row_length:" << nnz_max << endl;
    cout << "block_num:" << block_num << endl;

    X = vect_gen(n);
    float *Y = (float *)malloc(m * sizeof(float));

    for (int i = 0; i < m; i++)
    {
        Y[i] = 0;
    }

    float     alpha           = 1.0f;
    float     beta            = 0.0f;
    
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dX, *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (m + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  nnz * sizeof(float))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         n * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         m * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, row_off,
                           (m + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, col_idx, nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, values, nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, X, n * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, Y, m * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, m, n, nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, n, dX, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, m, dY, CUDA_R_32F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

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
    
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )
    return 1;
}