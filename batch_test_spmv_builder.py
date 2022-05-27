import os
import sys
import time
import pickle

# 已经解压的matrix文件的目录
UF_DIR = "/home/duzhen/matrix_suite"

# 当前目录
cur_path = os.getcwd()

# 当前测试集的矩阵
matrix_name_list_file = cur_path + "/matrix_name_list"
os.system("cd " + UF_DIR + " && find . -type d -exec rm -r {} +")

# 考虑重名的别名
if os.path.exists("repeat_matrix_name") == False:
    os.system("file:repeat_matrix_name is not existing")

with open(cur_path + "/repeat_matrix_name", 'rb') as f:
    file_2_matrix_name_map = pickle.load(f)

# 2693822
for line in open(matrix_name_list_file):
    matrix_file_name = line.strip()

    # 真实的矩阵名
    matrix_name = matrix_file_name

    if matrix_name in file_2_matrix_name_map.keys():
        matrix_name = file_2_matrix_name_map[matrix_name]
    
    # 解压，然后直接运行原始的mtx文件
    os.system("echo test matrix:" + matrix_file_name)

    # 解压并且预处理
    os.system("cd " + UF_DIR + " && tar -zxvf " + matrix_file_name + ".tar.gz")

    # 首先预处理
    os.system("python3 data_prepare.py " + UF_DIR + "/" + matrix_name + "/" + matrix_name + ".mtx " + cur_path + "/data_source/" + matrix_file_name + ".mtx.coo")

    # 预处理之后的数据
    preprocessed_file_name = cur_path + "/data_source/" + matrix_file_name + ".mtx.coo"

    os.system("./main " + preprocessed_file_name)
    # 修改对应的文件名字
    os.system("mv data_source/best_result_of_strategy1 " + "data_source/" + matrix_file_name + "_best_result_of_strategy1")
    os.system("mv data_source/best_result_of_strategy2 " + "data_source/" + matrix_file_name + "_best_result_of_strategy2")
    os.system("mv data_source/best_result_of_strategy3 " + "data_source/" + matrix_file_name + "_best_result_of_strategy3")
    os.system("mv data_source/machine_learning_data_set " + "data_source/" + matrix_file_name + "_machine_learning_data_set")
    # 删除所有的中间数据
    os.system("cd data_source && find . -type d -name \'[0,1,2,3,4,5,6,7,8,9]*\' -exec rm -r {} +")
    os.system("cd " + UF_DIR + " && find . -type d -exec rm -r {} +")
    