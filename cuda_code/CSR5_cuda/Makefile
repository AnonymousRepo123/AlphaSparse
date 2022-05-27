#compilers

#GLOBAL_PARAMETERS
VALUE_TYPE = float
NUM_RUN = 5000

#ENVIRONMENT_PARAMETERS

#CUDA_PARAMETERS
#NVCC_FLAGS = -O3  -w -m64 -arch=sm_30
NVCC_FLAGS = -O3  -w -std=c++11 -arch=sm_60 -m64 -gencode=arch=compute_60,code=compute_60 -Xcompiler -fpermissive
# -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52
# CUDA_INSTALL_PATH = /usr/local/cuda
# CUDA_INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I/usr/local/cuda/samples/common/inc
# CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart

CUDA_INSTALL_PATH = /usr/local/cuda-11.5
CUDA_INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/samples/common/inc
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart


cuda:
	/usr/local/cuda-11.5/bin/nvcc $(NVCC_FLAGS) main.cu -o spmv $(CUDA_INCLUDES) $(CUDA_LIBS) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
# nvcc $(NVCC_FLAGS) main.cu -o spmv $(CUDA_INCLUDES) $(CUDA_LIBS) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
	

