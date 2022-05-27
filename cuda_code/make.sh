#! /usr/bin/bash

/usr/bin/nvcc spmv_kernal_atom_atom.cu -arch=sm_60 -O2 -std=c++11