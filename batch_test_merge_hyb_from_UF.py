# 测试merge和hyb的函数，使用merge的开源实现
import os
import sys
import time
import pickle

from numpy import matrix

UF_DIR = "/home/duzhen/matrix_suite"

# 查看是不是有别名表
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

# 108816
for line in open(matrix_name_list_file):
    matrix_file_name = line.strip()
    
    matrix_name = matrix_file_name

    if matrix_name in file_2_matrix_name_map.keys():
        matrix_name = file_2_matrix_name_map[matrix_name]
    
    # 解压，然后直接运行原始的mtx文件
    os.system("echo test matrix:" + matrix_file_name)

    os.system("cd " + UF_DIR + " && tar -zxvf " + matrix_file_name + ".tar.gz")

    # 直接运行原始文件
    file_path = UF_DIR + "/" + matrix_name + "/" + matrix_name + ".mtx"

    # 运行merge测试
    os.system("cd cuda_code/merge-spmv && ./_gpu_spmv_driver --mtx=" + file_path + " --fp32 --i=5000 --device=0 --hybmv --bsrmv")

    time.sleep(3)
    # 删除目录下的所有文件夹
    os.system("cd " + UF_DIR + " && find . -type d -exec rm -r {} +")
    time.sleep(1)