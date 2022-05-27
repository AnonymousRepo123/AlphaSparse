# 一个新的函数，只看对称矩阵的性能
# import pandas as pd

# df1 = pd.DataFrame(data=[['A', 442],
#                          ['B', 998],
#                          ['C', 45],
#                          ['F', 777]]
#                    , columns=['KeyID', 'Info'])

# df2 = pd.DataFrame(data=[['A', 12],
#                         ['B', 77],
#                         ['C', 6556],
#                         ['D', 33],
#                         ['E', 876]
#                         ]
#                   ,columns=['KeyID', 'Info'])


# df_new = df2.append(df1).drop_duplicates(subset='KeyID')
# print(df_new)

# exit(-1)
import ssgetpy
import os
import sys
import pickle
import math

assert(len(sys.argv) == 2)

output_file_name = sys.argv[1]

UF_DIR = "/home/duzhen/matrix_suite"
# 删除目录下的所有文件夹
# os.system("cd " + UF_DIR + " && find . -type d -exec rm -r {} +")

cur_path = os.getcwd()

# 将个别矩阵的文件名和矩阵实际名字的映射建立起来
with open(cur_path + "/repeat_matrix_name", 'rb') as f:
    file_2_matrix_name_map = pickle.load(f)

with open(cur_path + "/matrix_id_map", 'rb') as f:
    file_2_matrix_id_map = pickle.load(f)

print(file_2_matrix_name_map)

existing_matrix_file_name_set = set()

# 遍历所有的目录中所有矩阵文件的名字
for file_name in os.listdir(UF_DIR):
    matrix_name = file_name[0:-7]
    existing_matrix_file_name_set.add(matrix_name)

print(len(existing_matrix_file_name_set))

# 用一个集合来整理已经记录feature的matrix文件名，在后面的统计中就可以忽略掉
exist_matrix_name_in_feature_file_set = set()

# 已经处理过的文件
if os.path.exists(output_file_name):
    os.system("echo " + output_file_name + " is existing")
    for line in open(output_file_name, "r"):
        line = line.strip()
        if line != "":
            line_str_arr = line.split(",")
            exist_matrix_name_in_feature_file_set.add(line_str_arr[0])

# print(exist_matrix_name_in_feature_file_set)

# matrix_feature_file.write("goddardRocketProblem_1,831,831,4457,0.0064541727667794735,5.363417569193743,1,161,114.95336690024438,30.01817365941216,830"+ "\n")

# exit(-1)

# 这里补一下
# existing_matrix_file_name_set = set()

# existing_matrix_file_name_set.add("kmer_V2a")
# existing_matrix_file_name_set.add("fem_hifreq_circuit")
# existing_matrix_file_name_set.add("plantstexture_10NN")

print(existing_matrix_file_name_set)

sym_matrix_name_set = set()
# 这里读一个矩阵，kmer_V2a还没有处理
# 出现错误的矩阵fem_hifreq_circuit，plantstexture_10NN
# 134510
for matrix_file_name in existing_matrix_file_name_set:
    if matrix_file_name == '':
        continue

    # 已经分析过了就下一个
    if matrix_file_name in exist_matrix_name_in_feature_file_set:
        os.system("echo " + matrix_file_name + " is existing")
        continue

    if (matrix_file_name in file_2_matrix_id_map) == False:
        continue
    
    # 查看矩阵的是不是对称的
    # assert(matrix_file_name in file_2_matrix_id_map.keys())
    
    # 查询这一矩阵
    result = ssgetpy.search(matid=file_2_matrix_id_map[matrix_file_name])

    # 必然可以查出来
    assert(len(result) == 1)

    
    if result[0].psym != 1.0 or result[0].nsym != 1.0:
        continue
    

    # help(result[0])
    # assert(result[0].isspd == True)
    # continue

    # print(result[0].psym)
    # print(result[0].nsym)
    # print(result[0].kind)

    # 解压
    os.system("cd " + UF_DIR + " && tar -zxvf " + matrix_file_name + ".tar.gz")

    os.system("echo finish tar")

    matrix_name = matrix_file_name
    # 实际的矩阵名字可能不一样
    if matrix_name in file_2_matrix_name_map.keys():
        matrix_name = file_2_matrix_name_map[matrix_name]

    # 打开对应文件
    # matrix_file = open(UF_DIR + "/" + matrix_name + "/" + matrix_name + ".mtx", "r")

    # 所有的特征，包含：
    # matrix,rows,cols,nnz,density,avr_nnz_row,min_nnz_row,max_nnz_row,var_nnz_row,ell_padding_ratio,empty_rows
    a_line_of_feature_data = matrix_file_name

    row_num_of_matrix = 0
    col_num_of_matrix = 0
    nnz = 0
    row_length_list = []

    is_first_line = True
    row_index_set = set()

    # 如果矩阵的大小大于60M，就忽视这个矩阵
    is_ignored_matrix = False

    for line in open(UF_DIR + "/" + matrix_name + "/" + matrix_name + ".mtx", "r"):
        # 用一个set来存储所有的row_index，以此来存储空行
        
        # 读文件中的内容，然后获取特征
        # 首先是略过所有的注释
        # 每一行的非零元数量
        if line[0] != "%":
            # 查看第一行，包含行数量，列数量和nnz
            if is_first_line == True:
                line_str_arr = line.split()
                row_num_of_matrix = eval(line_str_arr[0])
                col_num_of_matrix = eval(line_str_arr[1])
                nnz = eval(line_str_arr[2])

                # 如果这里查出来还不是对称的，那就忽视这个矩阵
                if nnz == result[0].nnz:
                    os.system("echo " + matrix_file_name + ":not symmetric, nnz:" + str(nnz) + ", real_nnz:" + str(result[0].nnz))
                    is_ignored_matrix = True
                    break

                os.system("echo " +  matrix_file_name + " nnz:" + str(nnz))
                if nnz >= 60000000:
                    os.system("echo " + matrix_file_name + ":too large to handle, nnz:" + str(nnz))
                    is_ignored_matrix = True
                    break

                a_line_of_feature_data = a_line_of_feature_data + \
                    "," + str(row_num_of_matrix)
                a_line_of_feature_data = a_line_of_feature_data + \
                    "," + str(col_num_of_matrix)
                a_line_of_feature_data = a_line_of_feature_data + \
                    "," + str(nnz)
                is_first_line = False
                row_length_list = [0] * row_num_of_matrix
            else:
                line_str_arr = line.split()
                row = eval(line_str_arr[0])
                row_index_set.add(row)
                col = eval(line_str_arr[1])
                if row < 1:
                    os.system("echo " + matrix_file_name + ":row index too small error, row:" + str(row))
                    is_ignored_matrix = True
                    break
                if col < 1:
                    os.system("echo " + matrix_file_name + ":col index too small error, col:" + str(col))
                    is_ignored_matrix = True
                    break
                
                # 如果这里有错误，就先跳过这里
                if row > row_num_of_matrix:
                    os.system("echo " + matrix_file_name + ":row index too large error, row:" + str(row) + ", row_num_of_matrix:" + str(row_num_of_matrix))
                    is_ignored_matrix = True
                    break
                if col > col_num_of_matrix:
                    os.system("echo " + matrix_file_name + ":col index too large error, col:" + str(col) + ", col_num_of_matrix:" + str(col_num_of_matrix))
                    is_ignored_matrix = True
                    break

                row_length_list[row - 1] = row_length_list[row - 1] + 1

    if is_ignored_matrix == True:
        continue
    
    # 计算空行的数量
    empty_line_num = row_num_of_matrix - len(row_index_set)
    # 计算密度，用非零元数量/(行数量*列数量)
    matrix_density = nnz / (row_num_of_matrix * col_num_of_matrix)
    # 平均行数量
    avg_row_length = nnz / row_num_of_matrix

    # 最大行数量，最小行数量，方差
    row_length_variance = 0
    min_row_length = 0
    max_row_length = 0

    is_first_element = True

    # 遍历每一行的行长度
    for row_len in row_length_list:
        if is_first_element == True:
            min_row_length = row_len
            max_row_length = row_len
            is_first_element = False
        else:
            if row_len < min_row_length:
                min_row_length = row_len
            if row_len > max_row_length:
                max_row_length = row_len
        row_length_variance = row_length_variance + math.pow(row_len - avg_row_length, 2)
    
    assert(max_row_length >= min_row_length)
    
    row_length_variance = row_length_variance / row_num_of_matrix
    ell_padding_rate = max_row_length * row_num_of_matrix / nnz

    a_line_of_feature_data = a_line_of_feature_data + "," + str(matrix_density)
    a_line_of_feature_data = a_line_of_feature_data + "," + str(avg_row_length)
    a_line_of_feature_data = a_line_of_feature_data + "," + str(min_row_length)
    a_line_of_feature_data = a_line_of_feature_data + "," + str(max_row_length)
    a_line_of_feature_data = a_line_of_feature_data + "," + str(row_length_variance)
    a_line_of_feature_data = a_line_of_feature_data + "," + str(ell_padding_rate)
    a_line_of_feature_data = a_line_of_feature_data + "," + str(empty_line_num)

    # python一次只能打开一个文件句柄
    matrix_feature_file = open(output_file_name, "a+")
    # 打印对应的一条数据
    # print(a_line_of_feature_data)
    matrix_feature_file.write(a_line_of_feature_data+"\n")
    
    # print(a_line_of_feature_data)
    matrix_feature_file.close()
    
    # 删除产生的数据
    # os.system("cd " + UF_DIR + " && find . -type d -exec rm -r {} +")

# matrix_feature_file.close()
print(len(sym_matrix_name_set))
