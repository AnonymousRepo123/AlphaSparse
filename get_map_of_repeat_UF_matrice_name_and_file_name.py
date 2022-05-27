import ssgetpy
import os
import sys
import pickle


UF_DIR = "/home/duzhen/matrix_suite"
# 删除目录下的所有文件夹
os.system("cd " + UF_DIR + " && find . -type d -exec rm -r {} +")

# 检查有重复名字的矩阵名字
result = ssgetpy.search(limit=3000)
repeat_name_set = set()

# 所有矩阵的名字
existing_matrix_name_set = set()
all_matrix_name_set = set()

# 有些矩阵名字一样，只是数据类型不一样，比如nasa*，还有一些名字一样，但是内容不一样
for matrix_item in result:
    if matrix_item.name in all_matrix_name_set:
        print("already exist:" + matrix_item.name)
        repeat_name_set.add(matrix_item.name)
    all_matrix_name_set.add(matrix_item.name)

repeat_matrix_name_map = {}
# 来一个map，将文件名称和其id对应起来
matrix_id_map = {}
repeat_matrix_cover_map = {}

# 遍历所有的重复矩阵明，找到其混合名字，并且建立混合名字和真实名字之间的映射
for repeat_matrix_name in repeat_name_set:
    # 遍历result中所有的名字
    for matrix_item in result:
        matrix_id_map_key = matrix_item.name
        # 查看名字是不是重复矩阵的
        if matrix_item.name == repeat_matrix_name:
            # 建立映射
            repeat_matrix_name_map[matrix_item.name + matrix_item.group + str(matrix_item.id)] = matrix_item.name
            repeat_matrix_cover_map[matrix_item.name + matrix_item.group + str(matrix_item.id)] = False
            matrix_id_map_key = matrix_item.name + matrix_item.group + str(matrix_item.id)
        matrix_id_map[matrix_id_map_key] = matrix_item.id

# print(repeat_matrix_name_map)
repeat_num = 0

for file_name in os.listdir(UF_DIR):  # 不仅仅是文件，当前目录下的文件夹也会被认为遍历到
    matrix_name = file_name[0:-7]
    existing_matrix_name_set.add(matrix_name)
    if matrix_name in repeat_matrix_name_map.keys():
        # print(matrix_name)
        repeat_matrix_cover_map[matrix_name] = True
        repeat_num = repeat_num + 1

# print(repeat_num)
# print(repeat_matrix_cover_map)

# 将映射关系写到文件中
cur_path = os.getcwd()

print(cur_path + "/repeat_matrix_name")

with open(cur_path + "/repeat_matrix_name", 'wb') as f:
    pickle.dump(repeat_matrix_name_map, f, 0)

with open(cur_path + "/matrix_id_map", 'wb') as f:
    pickle.dump(matrix_id_map, f, 0)

# print(len(existing_matrix_name_set))

# 解压所有的文件
