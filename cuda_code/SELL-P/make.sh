#! /usr/bin/bash

/usr/local/cuda-11.6/bin/nvcc main.cu -arch=sm_60 -Xptxas -O3 -Xcompiler -O3 -std=c++11