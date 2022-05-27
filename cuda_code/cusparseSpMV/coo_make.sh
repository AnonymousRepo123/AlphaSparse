#! /usr/bin/bash

nvcc coo_cusparse.cu -lcudart -lcusparse -o coo_spmv