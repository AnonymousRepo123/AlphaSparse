#! /usr/bin/bash

VALUE_TYPE=double
NUM_RUN=1

/usr/local/cuda/bin/nvcc -O3  -w -m64 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52 main.cu -o spmv -D VALUE_TYPE=${VALUE_TYPE} -D NUM_RUN=${NUM_RUN} -I/usr/local/cuda/samples/common/inc 