#ifndef MATRIX_INFO_H
#define MATRIX_INFO_H

// 用一个工具函数来获取一个矩阵的一些基本信息
#include "struct.hpp"
// 收集完基本信息之后就将矩阵删除
#include "memory_garbage_manager.hpp"
#include "exe_graph.hpp"

typedef struct matrix_info
{
    vector<unsigned long> row_nnz;

    unsigned long nnz;
    unsigned long col_num;
    unsigned long row_num;

    // 查看是不是有全局的排序
    bool is_sorted;

    // 最小和最大行非零元数量
    unsigned long max_row_nnz;
    unsigned long min_row_nnz;
    
    // 行非零元数量的平均值
    unsigned long avg_row_nnz;
}matrix_info_t;

// 根据一个输入节点给出矩阵的基本信息
matrix_info_t get_global_matrix_info_from_input_node(exe_begin_memory_cache_input_file_param_t input_matrix_node_param);

// 从一个coo文件中获得基本的信息
matrix_info_t get_matrix_info_from_matrix_coo_file(string coo_file_name);

// 从一个矩阵的指针中获得基本的信息
matrix_info_t get_matrix_info_from_sparse_matrix_ptr(sparse_struct_t* matrix);

// 从压缩之后的子图中获得某些子图的基本信息
matrix_info_t get_sub_matrix_info_from_compressed_matrix_block(dense_block_table_item_t* sub_matrix);

#endif