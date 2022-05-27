# 从UF矩阵中处理所有的矩阵
import os
import sys
import numpy as np
import time
import pickle

from numpy.matrixlib.defmatrix import matrix

assert(len(sys.argv) == 2)

# 这里修改，告诉UF数据集的目录，里面全是tar.gz的UF数据
UF_DIR = "/home/duzhen/matrix_suite"

# 这里将名字重复的矩阵别名取出
if os.path.exists("repeat_matrix_name") == False:
    os.system("file:repeat_matrix_name is not existing")

cur_path = os.getcwd()

with open(cur_path + "/repeat_matrix_name", 'rb') as f:
    file_2_matrix_name_map = pickle.load(f)

# 用一个变量来存储当前的文件名
cur_path = os.getcwd()

# 输入的是矩阵文件的名称
matrix_file_name = str(sys.argv[1])

os.system("echo test matrix:" + matrix_file_name)

# 首先解压
os.system("cd " + UF_DIR + " && tar -zxvf " + matrix_file_name + ".tar.gz")

# 如果是重复的矩阵还需要找到一个真实的名字
matrix_name = matrix_file_name

if matrix_name in file_2_matrix_name_map.keys():
    matrix_name = file_2_matrix_name_map[matrix_name]


# 首先预处理
os.system("cd " + cur_path + "/data_source/matrix_figure && " + "python3 data_prepare.py " + UF_DIR + "/" + matrix_name + "/" + matrix_name + ".mtx " + cur_path + "/data_source/" + matrix_file_name + ".mtx.coo")

# 预处理之后的数据
preprocessed_file_name = cur_path + "/data_source/" + matrix_file_name + ".mtx.coo"

# 执行一般的几个数据结构
# /home/duzhen/spmv_builder/cuda_code/ACSR_test
os.system("echo =====ACSR=====")

os.system("cd " + cur_path + "/cuda_code/ACSR_test && " + "./a.out " + preprocessed_file_name)

os.system("echo =====ACSR=====")

os.system("echo =====CSR_Adaptive=====")

# /home/duzhen/spmv_builder/cuda_code/CSR_adptive_test
os.system("cd " + cur_path + "/cuda_code/CSR_adptive_test && " + "./a.out " + preprocessed_file_name)

os.system("echo =====CSR_Adaptive=====")

os.system("echo =====CSR5=====")

os.system("cd " + cur_path + "/cuda_code/CSR5_cuda && " + "./spmv " + preprocessed_file_name)

os.system("echo =====CSR5=====")

os.system("echo =====ELL=====")

os.system("cd " + cur_path + "/cuda_code/ELL_test && " + "./a.out " + preprocessed_file_name)

os.system("echo =====ELL=====")

# 还有merge-based的两个测试这里需要cuda 7.5才能测试
# os.system("cd " + cur_path + "/cuda_code/merge-spmv && " + "./gpu_spmv --mtx=" + preprocessed_file_name + " --fp32 --i=5000 --device=0 --hybmv --bsrmv")

time.sleep(3)

# 最后执行这里是跑自己的程序，换掉
# os.system("nohup ./main " + preprocessed_file_name + " > data_source/test.log 2>&1 &")
