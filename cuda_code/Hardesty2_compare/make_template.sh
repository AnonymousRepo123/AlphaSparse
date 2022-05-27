#! /usr/bin/bash

/usr/local/cuda/bin/nvcc ell_template.cu -o ell_template.out -arch=sm_60 -Xptxas -O3 -Xcompiler -O3 -std=c++11

/usr/local/cuda/bin/nvcc ell_spmv.cu -o ell_spmv.out -arch=sm_60 -Xptxas -O3 -Xcompiler -O3 -std=c++11

/usr/local/cuda/bin/nvcc sell_template.cu -o sell_template.out -arch=sm_60 -Xptxas -O3 -Xcompiler -O3 -std=c++11