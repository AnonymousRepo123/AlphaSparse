# 从UF中下载数据
# 首先搜出来满足要求的矩阵，然后一个个下载
import ssgetpy
import os
import time

UF_DIR = "/home/duzhen/matrix_suite_test"
matrix_name_list_file_name = "/home/duzhen/spmv_builder/matrix_name_list"

# help(ssgetpy.search)

# exit(-1)

# 按照一定的查看需要的矩阵id
result = ssgetpy.search(limit = 3000, nzbounds=(50000, 60000000), rowbounds=(9000, None))

print(len(result))

# help(result[0])

# exit(-1)

#  |  `id`   : The unique identifier for the matrix in the database.
#  |  `group`: The name of the group this matrix belongs to.
#  |  `name` : The name of this matrix.
#  |  `rows` : The number of rows.
#  |  `cols` : The number of columns.
#  |  `nnz`  : The number of non-zero elements.
#  |  `dtype`: The datatype of non-zero elements: `real`, `complex` or `binary`
#  |  `is2d3d`: True if this matrix comes from a 2D or 3D discretization.
#  |  `isspd` : True if this matrix is symmetric, positive definite
#  |  `kind`  : The underlying problem domain

# 遍历所有的矩阵，将重复名字的矩阵取出来
existing_matrix_name_set = set()
repeat_matrix_name_set = set()
# 查看矩阵的id
existing_matrix_id_set = set()

for result_item in result:
    matrix_name = result_item.name
    matrix_id = result_item.id
    # 查看矩阵名称是不是存在的
    if matrix_name in existing_matrix_name_set:
        # 当前矩阵是重名的
        repeat_matrix_name_set.add(matrix_name)
    existing_matrix_name_set.add(matrix_name)
    existing_matrix_id_set.add(matrix_id)

# print(len(repeat_matrix_name_set))
# print(len(existing_matrix_name_set))
# print(len(existing_matrix_id_set))

print(repeat_matrix_name_set)
print(existing_matrix_name_set)
print(existing_matrix_id_set)

# 处理矩阵列表
for matrix_result in result:
    assert(matrix_result.id in existing_matrix_id_set)
    group_name = matrix_result.group
    matrix_name = matrix_result.name
    assert(matrix_name in existing_matrix_name_set)
    if matrix_name in repeat_matrix_name_set:
        with open(matrix_name_list_file_name, 'a+') as f:
            f.write(matrix_name + group_name + '\n')
    else:
        with open(matrix_name_list_file_name, 'a+') as f:
            f.write(matrix_name+'\n')
    

# 遍历所有的矩阵，按照要求下载对应的矩阵，然后根据组号重命名
for matrix_result in result:
    # 查看当前矩阵的id在不在
    assert(matrix_result.id in existing_matrix_id_set)
    # 下载对应的矩阵
    # https://suitesparse-collection-website.herokuapp.com/MM/HB/1138_bus.tar.gz
    # 查看当前矩阵的组号
    group_name = matrix_result.group
    matrix_name = matrix_result.name
    assert(matrix_name in existing_matrix_name_set)
    os.system("cd " + UF_DIR + " && wget https://suitesparse-collection-website.herokuapp.com/MM/" + group_name + "/" + matrix_name + ".tar.gz")
    # 如果是重复的，那就改一个名
    time.sleep(1)
    if matrix_name in repeat_matrix_name_set:
        os.system("cd " + UF_DIR + " && mv " + matrix_name + ".tar.gz " + matrix_name + group_name + ".tar.gz")
    #     with open(matrix_name_list_file_name, 'a+') as f:
    #         f.write(matrix_name + group_name + '\n')
    # else:
    #     with open(matrix_name_list_file_name, 'a+') as f:
    #         f.write(matrix_name+'\n')
    
