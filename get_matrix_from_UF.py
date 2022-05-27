from time import process_time_ns
import ssgetpy
import os

# 要将矩阵下载到文件夹
UF_DIR = "/home/duzhen/matrix_suite"
# 删除目录下的所有文件夹
os.system("cd " + UF_DIR + " && find . -type d -exec rm -r {} +")

# name group id 三个属性决定一个矩阵名字
result = ssgetpy.search(limit=3000)

print(len(result))

# help(result[0])

all_matrix_name_set = set()
repeat_name_set = set()

# 有些矩阵名字一样，只是数据类型不一样，比如nasa*，还有一些名字一样，但是内容不一样
for matrix_item in result:
    if matrix_item.name in all_matrix_name_set:
        print("already exist:" + matrix_item.name)
        repeat_name_set.add(matrix_item.name)
    all_matrix_name_set.add(matrix_item.name)

# 对于名字重复的矩阵先删一下
for repeat_matrix_name in repeat_name_set:
    os.system("rm -rf " + UF_DIR + "/" + repeat_matrix_name + ".tar.gz")

# 收集目录中矩阵的名字
existing_matrix_name_set = set()

for file_name in os.listdir(UF_DIR):  # 不仅仅是文件，当前目录下的文件夹也会被认为遍历到
    matrix_name = file_name[0:-7]
    existing_matrix_name_set.add(matrix_name)

needed_matrix_name_set = all_matrix_name_set - existing_matrix_name_set

print(len(all_matrix_name_set))
print(len(needed_matrix_name_set))
print(len(repeat_name_set))
print(len(existing_matrix_name_set))

# 针对每一个类型，找出对应的Matrix对象准备下载
download_matrix_list = []

# 针对每一个矩阵名字，遍历整个矩阵集，找出对应的矩阵名字、并记录下来矩阵对象
for needed_matrix_name in needed_matrix_name_set:
    for matrix_item in result:
        if matrix_item.name == needed_matrix_name:
            download_matrix_list.append(matrix_item)

# print(download_matrix_list)
print(len(download_matrix_list))
# 下载、命名、并且拷贝到目录的最外面
for download_matrix in download_matrix_list:
    print("downloading " + download_matrix.group + "/" + download_matrix.name)
    
    # print(download_matrix.download(destpath=UF_DIR))
    # # 将下载之后的数据拷贝出来，重复的数据需要额外加上group名和matrix名字
    if download_matrix.name in repeat_name_set:
        print("is repeat")
    #     os.system("mv " + UF_DIR + "/" + download_matrix.name + ".tar.gz " + UF_DIR + "/" + download_matrix.name + download_matrix.group + download_matrix.id + ".tar.gz")
        



# 获取已经有的所有矩阵的名字
# 太大没下的：GAP-twitter，AGATHA_2015,mycielskian20,mycielskian19,kmer_P1a，GAP-urand,MOLIERE_2016,GAP-web,mawi_201512020330,twitter7,com-Friendster,GAP-kron,mawi_201512020130,stokes
# vas_stokes_4M,kmer_A2a,mawi_201512020030


# print(result[0].name)



