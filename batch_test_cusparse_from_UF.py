import os
import sys
import time
import pickle

UF_DIR = "/home/duzhen/matrix_suite"

# 这里将名字重复的矩阵别名取出
if os.path.exists("repeat_matrix_name") == False:
    os.system("file:repeat_matrix_name is not existing")

cur_path = os.getcwd()

with open(cur_path + "/repeat_matrix_name", 'rb') as f:
    file_2_matrix_name_map = pickle.load(f)

# 当前目录
cur_path = os.getcwd()

# 矩阵列表
matrix_name_list_file = cur_path + "/matrix_name_list"

os.system("cd " + UF_DIR + " && find . -type d -exec rm -r {} +")

for line in open(matrix_name_list_file):
    matrix_file_name = line.strip()
    
    # 预处理之后做测试
    os.system("echo test matrix:" + matrix_file_name)

    # 首先解压
    os.system("cd " + UF_DIR + " && tar -zxvf " + matrix_file_name + ".tar.gz")

    # 如果是重复的矩阵还需要找到一个真实的名字
    matrix_name = matrix_file_name

    if matrix_name in file_2_matrix_name_map.keys():
        matrix_name = file_2_matrix_name_map[matrix_name]
    
    os.system("python3 data_prepare.py " + UF_DIR + "/" + matrix_name + "/" + matrix_name + ".mtx " + cur_path + "/data_source/" + matrix_file_name + ".mtx.coo")

    preprocessed_file_name = cur_path + "/data_source/" + matrix_file_name + ".mtx.coo"

    os.system("echo =====cusparseSpMV CSR=====")

    os.system("cd " + cur_path + "/cuda_code/cusparseSpMV && " + "./csr_spmv " + preprocessed_file_name)

    os.system("echo =====cusparseSpMV CSR=====")

    os.system("echo =====cusparseSpMV COO=====")

    os.system("cd " + cur_path + "/cuda_code/cusparseSpMV && " + "./coo_spmv " + preprocessed_file_name)

    os.system("echo =====cusparseSpMV COO=====")

    # 删除目录下的所有文件夹
    os.system("cd " + UF_DIR + " && find . -type d -exec rm -r {} +")