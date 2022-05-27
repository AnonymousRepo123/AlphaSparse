// 这个宏定义只能防止单个.cc重读引用多次这个头文件。所以不加这个dataset_builder会在一个.cc文件中重定义。
// 但是即便加了这个宏定义，函数的定义和声明依旧要分开，因为结构体的定义可以在多个.c文件中重复而不报错
// 但是函数在多个.c文件之间的重复定义会导致代码空间的问题，即便加了这个宏定义，还是要声明与实现分离
#ifndef DATASET_BUILDER_H
#define DATASET_BUILDER_H

#include "struct.hpp"

// 一个coo格式的矩阵，包含三个vector
typedef struct dataset_builder
{
    // 逻辑上最大行索引号
    unsigned long max_row_index;
    unsigned long max_col_index;

    // 行非零元的数量，默认没有空行
    unsigned long nnz_in_row;
} dataset_builder_t;

dataset_builder_t get_dataset_builder(unsigned long max_row_index, unsigned long max_col_index, unsigned long nnz_in_row);

vector<unsigned long> get_row_index_of_dataset_builder(dataset_builder_t builder);

vector<unsigned long> get_col_index_of_dataset_builder(dataset_builder_t builder);

vector<double> get_double_val_of_dataset_builder(dataset_builder_t builder);

vector<float> get_float_val_of_dataset_builder(dataset_builder_t builder);

#endif