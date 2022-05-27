#include <iostream>

#include "anonymouslib_cuda.h"
#include <sys/time.h>
#include "mmio.h"

using namespace std;

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef NUM_RUN
#define NUM_RUN 1000
#endif

int call_anonymouslib(int m, int n, int nnzA,
                  int *csrRowPtrA, int *csrColIdxA, VALUE_TYPE *csrValA,
                  VALUE_TYPE *x, VALUE_TYPE *y, VALUE_TYPE alpha)
{
    int err = 0;
    cudaError_t err_cuda = cudaSuccess;

    // set device
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    cout << "Device [" <<  device_id << "] " << deviceProp.name << ", " << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " << endl;

    double gb = getB<int, VALUE_TYPE>(m, nnzA);
    double gflop = getFLOP<int>(nnzA);

    // Define pointers of matrix A, vector x and y
    int *d_csrRowPtrA;
    int *d_csrColIdxA;
    VALUE_TYPE *d_csrValA;
    VALUE_TYPE *d_x;
    VALUE_TYPE *d_y;

    // Matrix A
    checkCudaErrors(cudaMalloc((void **)&d_csrRowPtrA, (m+1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrColIdxA, nnzA  * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrValA,    nnzA  * sizeof(VALUE_TYPE)));

    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, csrRowPtrA, (m+1) * sizeof(int),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColIdxA, csrColIdxA, nnzA  * sizeof(int),   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrValA,    csrValA,    nnzA  * sizeof(VALUE_TYPE),   cudaMemcpyHostToDevice));

    // Vector x
    checkCudaErrors(cudaMalloc((void **)&d_x, n * 2 * sizeof(VALUE_TYPE)));
    checkCudaErrors(cudaMemcpy(d_x, x, n * 2 * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));

    // Vector y
    checkCudaErrors(cudaMalloc((void **)&d_y, m*2  * sizeof(VALUE_TYPE)));
    checkCudaErrors(cudaMemset(d_y, 0, m*2 * sizeof(VALUE_TYPE)));

    // 数组A
    anonymouslibHandle<int, unsigned int, VALUE_TYPE> A(m, n);
    err = A.inputCSR(nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA);
    //cout << "inputCSR err = " << err << endl;

    err = A.setX(d_x); // you only need to do it once!
    //cout << "setX err = " << err << endl;

    A.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);

    // warmup device
    // A.warmup();

    anonymouslib_timer asCSR5_timer;
    asCSR5_timer.start();

    err = A.asCSR5();

    cout << "CSR->CSR5 time = " << asCSR5_timer.stop() << " ms." << endl;
    //cout << "asCSR5 err = " << err << endl;

    // check correctness by running 1 time
    err = A.spmv(alpha, d_y);
    cout << "spmv err = " << err << endl;
    checkCudaErrors(cudaMemcpy(y, d_y, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));

    // warm up by running 50 times
    // if (NUM_RUN)
    // {
    //     for (int i = 0; i < 50; i++)
    //         err = A.spmv(alpha, d_y);
    // }

    err_cuda = cudaDeviceSynchronize();

    struct timeval start, end;
    gettimeofday(&start, NULL);
    anonymouslib_timer CSR5Spmv_timer;
    CSR5Spmv_timer.start();

    // time spmv by running NUM_RUN times
    for (int i = 0; i < NUM_RUN; i++)
    {
        err = A.spmv(alpha, d_y);
        err_cuda = cudaDeviceSynchronize();
    }

    double CSR5Spmv_time = CSR5Spmv_timer.stop() / (double)NUM_RUN;

    gettimeofday(&end, NULL);

    long timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    double gflops = ((double)2.0 * nnzA * NUM_RUN / ((double)timeuse / 1000000)) / 1000000000;

    // 时间和性能
    float exe_time = (float)timeuse / 1000.0;
    float exe_gflops = gflops;

    // 循环的次数
    int final_repeat_num = NUM_RUN * ((float)1000 / exe_time);

    gettimeofday(&start, NULL);

    for (int i = 0; i < final_repeat_num; i++)
    {
        err = A.spmv(alpha, d_y);
        err_cuda = cudaDeviceSynchronize();
    }
    
    gettimeofday(&end, NULL);

    timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    gflops = ((double)2.0 * nnzA * final_repeat_num / ((double)timeuse / 1000000)) / 1000000000;

    exe_time = (float)timeuse / 1000.0;
    exe_gflops = gflops;

    printf("CSR5:time=%fms, gflops=%f\n", exe_time, exe_gflops);

    A.destroy();

    checkCudaErrors(cudaFree(d_csrRowPtrA));
    checkCudaErrors(cudaFree(d_csrColIdxA));
    checkCudaErrors(cudaFree(d_csrValA));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));

    return err;
}

int main(int argc, char ** argv)
{
    int m, n, nnzA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;

    // report precision of floating-point
    cout << "------------------------------------------------------" << endl;
    char  *precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = "32-bit Single Precision";
    }
    else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = "64-bit Double Precision";
    }
    else
    {
        cout << "Wrong precision. Program exit!" << endl;
        return 0;
    }

    cout << "PRECISION = " << precision << endl;
    cout << "------------------------------------------------------" << endl;

    //ex: ./spmv webbase-1M.mtx
    int argi = 1;

    char  *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }
    cout << "--------------" << filename << "--------------" << endl;

    // read matrix from mtx file
    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
    {
        return -1;
    }

    // if (mm_read_banner(f, &matcode) != 0)
    // {
    //     cout << "Could not process Matrix Market banner." << endl;
    //     return -2;
    // }

    // if ( mm_is_complex( matcode ) )
    // {
    //     cout <<"Sorry, data type 'COMPLEX' is not supported. " << endl;
    //     return -3;
    // }

    // if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*cout << "type = Pattern" << endl;*/ }
    // if ( mm_is_real ( matcode) )     { isReal = 1; /*cout << "type = real" << endl;*/ }
    // if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*cout << "type = integer" << endl;*/ }

    // /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
    if (ret_code != 0)
    {
        return -4;
    }

    nnzA = nnzA_mtx_report;

    // if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    // {
    //     isSymmetric = 1;
    //     //cout << "symmetric = true" << endl;
    // }
    // else
    // {
    //     //cout << "symmetric = false" << endl;
    // }

    // int *csrRowPtrA_counter = (int *)malloc((m+1) * sizeof(int));
    // memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    // int *csrRowIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    // int *csrColIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    // VALUE_TYPE *csrValA_tmp    = (VALUE_TYPE *)malloc(nnzA_mtx_report * sizeof(VALUE_TYPE));

    // /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    // /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    // /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    // for (int i = 0; i < nnzA_mtx_report; i++)
    // {
    //     int idxi, idxj;
    //     double fval;
    //     int ival;

    //     if (isReal)
    //         fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
    //     else if (isInteger)
    //     {
    //         fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
    //         fval = ival;
    //     }
    //     else if (isPattern)
    //     {
    //         fscanf(f, "%d %d\n", &idxi, &idxj);
    //         fval = 1.0;
    //     }

    //     // adjust from 1-based to 0-based
    //     idxi--;
    //     idxj--;

    //     csrRowPtrA_counter[idxi]++;
    //     csrRowIdxA_tmp[i] = idxi;
    //     csrColIdxA_tmp[i] = idxj;
    //     csrValA_tmp[i] = fval;
    // }

    // if (f != stdin)
    //     fclose(f);

    // if (isSymmetric)
    // {
    //     for (int i = 0; i < nnzA_mtx_report; i++)
    //     {
    //         if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
    //             csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
    //     }
    // }

    // // exclusive scan for csrRowPtrA_counter
    // int old_val, new_val;

    // old_val = csrRowPtrA_counter[0];
    // csrRowPtrA_counter[0] = 0;
    // for (int i = 1; i <= m; i++)
    // {
    //     new_val = csrRowPtrA_counter[i];
    //     csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i-1];
    //     old_val = new_val;
    // }

    // nnzA = csrRowPtrA_counter[m];
    // csrRowPtrA = (int *)malloc((m+1) * sizeof(int));
    // memcpy(csrRowPtrA, csrRowPtrA_counter, (m+1) * sizeof(int));
    // memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    // csrColIdxA = (int *)malloc(nnzA * sizeof(int));
    // csrValA    = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE));

    // if (isSymmetric)
    // {
    //     for (int i = 0; i < nnzA_mtx_report; i++)
    //     {
    //         if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
    //         {
    //             int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
    //             csrColIdxA[offset] = csrColIdxA_tmp[i];
    //             csrValA[offset] = csrValA_tmp[i];
    //             csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

    //             offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
    //             csrColIdxA[offset] = csrRowIdxA_tmp[i];
    //             csrValA[offset] = csrValA_tmp[i];
    //             csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
    //         }
    //         else
    //         {
    //             int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
    //             csrColIdxA[offset] = csrColIdxA_tmp[i];
    //             csrValA[offset] = csrValA_tmp[i];
    //             csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
    //         }
    //     }
    // }
    // else
    // {
    //     for (int i = 0; i < nnzA_mtx_report; i++)
    //     {
    //         int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
    //         csrColIdxA[offset] = csrColIdxA_tmp[i];
    //         csrValA[offset] = csrValA_tmp[i];
    //         csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
    //     }
    // }

    // // free tmp space
    // free(csrColIdxA_tmp);
    // free(csrValA_tmp);
    // free(csrRowIdxA_tmp);
    // free(csrRowPtrA_counter);

    // srand(time(NULL));

    // 这里直接进行文件的读取，并且获得CSR的三个数组
    


    // set csrValA to 1, easy for checking floating-point results
    // for (int i = 0; i < nnzA; i++)
    // {
    //     csrValA[i] = rand() % 10;
    // }

    cout << " ( " << m << ", " << n << " ) nnz = " << nnzA << endl;


    int max_row_length;
    int *col_idx = NULL;
    int *row_off = NULL;
    
    // 强制使用单精度
    VALUE_TYPE *values = NULL;

    // 申请对应大小的数组
    col_idx = new int[nnzA];
    row_off = new int[m + 1];
    values = new VALUE_TYPE[nnzA];
    
    // 将数据读到三个矩阵中
    conv(filename, row_off, col_idx, values, max_row_length, true);

    cout << " ( " << m << ", " << n << " ) nnz = " << nnzA << ", max row length:" << max_row_length << endl;

    // exit(-1);
    csrRowPtrA = row_off;
    csrColIdxA = col_idx;
    csrValA = values;

    // 打印几个关键的数组
    for (int i; i < 10; i++)
    {
        cout << csrRowPtrA[i] << ",";
    }

    cout << endl;

    for (int i; i < 10; i++)
    {
        cout << csrColIdxA[i] << ",";
    }

    cout << endl;
    
    for (int i; i < 10; i++)
    {
        cout << csrValA[i] << ",";
    }

    cout << endl;

    // exit(-1);

    VALUE_TYPE *x = (VALUE_TYPE *)malloc(n * sizeof(VALUE_TYPE));
    for (int i = 0; i < n; i++)
        x[i] = 1;

    VALUE_TYPE *y = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));
    VALUE_TYPE *y_ref = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));

    double gb = getB<int, VALUE_TYPE>(m, nnzA);
    double gflop = getFLOP<int>(nnzA);

    VALUE_TYPE alpha = 1.0;

    // compute reference results on a cpu core
    anonymouslib_timer ref_timer;
    ref_timer.start();

    int ref_iter = 1;
    // for (int iter = 0; iter < ref_iter; iter++)
    // {
    //     for (int i = 0; i < m; i++)
    //     {
    //         VALUE_TYPE sum = 0;
    //         for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
    //             sum += x[csrColIdxA[j]] * csrValA[j] * alpha;
    //         y_ref[i] = sum;
    //     }
    // }

    double ref_time = ref_timer.stop() / (double)ref_iter;
    // cout << "cpu sequential time = " << ref_time
    //      << " ms. Bandwidth = " << gb/(1.0e+6 * ref_time)
    //      << " GB/s. GFlops = " << gflop/(1.0e+6 * ref_time)  << " GFlops." << endl << endl;

    // 在这之前都是预处理
    gettimeofday(&pre_end, NULL);

    // 预处理时间计算
    double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;

    printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);
    // launch compute
    call_anonymouslib(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, alpha);

    // compare reference and anonymouslib results
    if (false)
    {
        int error_count = 0;
        for (int i = 0; i < m; i++)
            if (abs(y_ref[i] - y[i]) > 0.01 * abs(y_ref[i]))
            {
                error_count++;
    //            cout << "ROW [ " << i << " ], NNZ SPAN: "
    //                 << csrRowPtrA[i] << " - "
    //                 << csrRowPtrA[i+1]
    //                 << "\t ref = " << y_ref[i]
    //                 << ", \t csr5 = " << y[i]
    //                 << ", \t error = " << y_ref[i] - y[i]
    //                 << endl;
    //            break;

    //            //if (abs(y_ref[i] - y[i]) > 0.00001)
    //            //    cout << ", \t error = " << y_ref[i] - y[i] << endl;
    //            //else
    //            //    cout << ". \t CORRECT!" << endl;
            }

        if (error_count == 0)
            cout << "Check... PASS!" << endl;
        else
            cout << "Check... NO PASS! #Error = " << error_count << " out of " << m << " entries." << endl;

        cout << "------------------------------------------------------" << endl;
    }

     for (int i; i < 10; i++)
    {
        cout << y[i] << ",";
    }

    

    cout << endl;

    free(csrRowPtrA);
    free(csrColIdxA);
    free(csrValA);
    free(x);
    free(y);
    free(y_ref);

    return 0;
}

