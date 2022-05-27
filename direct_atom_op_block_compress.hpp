// 去除block级别的遍历，保留warp层次的遍历，很据实际开启的线程网格结构，一个block的前几个线程出来取元数据
#ifndef DIRECT_ATOM_TEMPLATE_BLOCK_COMPRESS_H
#define DIRECT_ATOM_TEMPLATE_BLOCK_COMPRESS_H

#include "direct_atom_op.hpp"

typedef struct direct_atom_template_block_compress
{
    // 模板对应的稠密矩阵号
    unsigned long dense_block_index;
    // 对应的密集矩阵
    sparse_struct_t *matrix = NULL;
    // 当前密集子块的首行行号
    unsigned long kernal_first_row_index = 0;
    unsigned long kernal_first_col_index = 0;

    // 用一个变量存是否要用原子加来归约
    bool is_atom_add = false;

    // 重构一个新的矩阵，用来直接存储线程粒度的块所对应的全局行号
    void *global_row_index_of_thread_level_block = NULL;
    data_type data_type_of_global_row_index_of_thread_level_block;
    unsigned long size_of_global_row_index_of_thread_level_block;

    //

} direct_atom_template_block_compress_t;

#endif