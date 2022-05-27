#! /usr/bin/bash

nvcc csr_cusparse.cu -lcudart -lcusparse -o csr_spmv