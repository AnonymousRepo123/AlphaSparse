#! /usr/bin/bash

/usr/bin/nvcc ELL_spmv.cu -arch=sm_60 -Xptxas -O3 -Xcompiler -O3 -std=c++11