#! /usr/bin/bash

nvcc template.cu -arch=sm_60 -Xptxas -O3 -Xcompiler -O3 -std=c++11
