# coding=utf-8

# 处理科学计数法

print("将输入文件重新按行排序")

import os
import sys
import time
start = time.time()

cur_path = os.getcwd()

# 从外部传入一个matrix market的文件到sys.argv[1]
input_matrix_file_name = str(sys.argv[1])

# 从外部传入一个输出文件
output_matrix_file_name = str(sys.argv[2])

print("input_matrix_file_name:" + input_matrix_file_name)
print("output_matrix_file_name:" + output_matrix_file_name)

write_file = open(output_matrix_file_name, "w")

all_data = []

is_first_line = True

row_num_of_matrix = 0

row_index_set = set()


for line in open(input_matrix_file_name):
    # 只要第一行开头是百分号，就跳过
    if line[0] != "%":
        if is_first_line == True:
            write_file.write(line)
            line_str_arr = line.split()
            row_num_of_matrix = eval(line_str_arr[0])
            is_first_line = False
        else:
            line_str_arr = line.split()
            row = eval(line_str_arr[0])
            row_index_set.add(row)
            col = eval(line_str_arr[1])
            # val = eval(line_str_arr[2])
            val = 1
            new_tuple = (row,col,val)
            all_data.append(new_tuple)


if len(row_index_set) != row_num_of_matrix:
    print("has empty line")
    os._exit()

# 排序
all_data = sorted(all_data)

for tuple_item in all_data:
    write_file.write(str(tuple_item[0]))
    write_file.write(" ")
    write_file.write(str(tuple_item[1]))
    write_file.write(" ")
    write_file.write(str(tuple_item[2]))
    write_file.write("\n")

end = time.time()
print("run for " + str(end - start) + " second")