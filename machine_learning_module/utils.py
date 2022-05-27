import os
import sys

# 将特定开头的数据从文件中取出
def get_complete_data_source_from_file(prefix : str, file_name : str):
    # 从文件中读入内容，放到一个数组中
    return_line_arr = []
    for line in open(file_name):
        line = line.strip()
        if line.startswith(prefix):
            # 将头部的字符去除
            line = line.lstrip(prefix)
            return_line_arr.append(line)
    
    return return_line_arr

# 将逗号分割的字符串转化为一个float类型的列表
def convert_str_2_float_arr(arr_str : str):
    str_of_each_item = arr_str.split(",")
    assert(len(str_of_each_item) > 0)
    return_arr = []
    # 遍历所有的内容，并且将内容转化为一个真正的float数组
    for item in str_of_each_item:
        return_arr.append(float(item))
    return return_arr
    

# 将特定开头的数据从文件中取出，并且将x输出为array的
def get_complete_numpy_x_and_y(data_set_item_str_list : list):
    # 需要两个东西，一个是二维的x，一个是每组x对应的y
    x_list = []
    y = []
    for data_str_item in data_set_item_str_list:
        # 将一个str类型的，用逗号分开的数据，变成一个float类型的数组
        a_float_type_arr = convert_str_2_float_arr(data_str_item)
        x_list.append(a_float_type_arr[0:-1].copy())
        y.append(a_float_type_arr[-1])
    
    return x_list, y
    
        