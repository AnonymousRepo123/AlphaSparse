import os
import sys
import time

UF_DIR = "/home/duzhen/matrix_suite"

# 当前目录
cur_path = os.getcwd()

# 矩阵列表
matrix_name_list_file = cur_path + "/matrix_name_list"

# 输入的是矩阵的名称
# matrix_name = str(sys.argv[1])

# print(matrix_name_list_file)
# print(matrix_name)
os.system("cd " + UF_DIR + " && find . -type d -exec rm -r {} +")

# 6090
for line in open(matrix_name_list_file):
    matrix_name = line.strip()
    os.system("python3 test_from_UF_dataset.py " + matrix_name)
    # 删除目录下的所有文件夹
    os.system("cd " + UF_DIR + " && find . -type d -exec rm -r {} +")
    