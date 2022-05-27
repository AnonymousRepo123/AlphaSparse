#ifndef CODE_BUILDER_H
#define CODE_BUILDER_H

#include "config.hpp"
#include "struct.hpp"
#include "op_manager.hpp"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include "arr_optimization.hpp"

// 模板的种类
enum template_type
{
    DIRECT_ATOM_TEMPLATE,
    DIRECT_ATOM_TEMPLATE_WARP_COMPRESS,
    DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS,
    SHARED_MEMORY_TEMPLATE,
    SHARED_MEMORY_TEMPLATE_WARP_COMPRESS,
    SHARED_MEMORY_LONG_ROW_TEMPLATE,
    SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE,
    DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE,
    UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE,
    UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE,
    EMPTY_TEMPLATE,
    NONE_TEMPLATE
};

enum reduce_type
{
    ATOMIC_ADD,
    SYNC_BY_SHARED_MEM,
    // 在某一个层次不归约
    NONE
};

// 每个子块可以从两个位置数组中看到行的原始地址，需要标记一下
enum sort_type
{
    GLOBAL_SORT,
    SUB_BLOCK_SORT,
    NO_SORT
};

typedef struct code_builder
{
    operator_manager_t *op_manager = NULL;

    // 记录每一个子块的归约策略，查看需要归约的层次，如果三个层次都不归约，就直接对全局内存的特定位置复制
    // 更细一层对行的共享决定了上一层是不是需要规约。
    // 原子型规约需要更细一层直接将结果写到下一层的输出中，共享内存型归约需要将更细一层的结果全部保留下来
    // 相邻层次的原子型归约可以合并，如果全都是原子性，那就合并为对全局显存的操作。
    vector<bool> is_reduce_in_warp_level_vec;
    vector<bool> is_reduce_in_block_level_vec;
    // 这一层只支持原子性归约
    vector<bool> is_reduce_in_global_level_vec;

    // 不同层次的归约手段
    vector<reduce_type> reduce_type_in_warp_level_vec;
    vector<reduce_type> reduce_type_in_block_level_vec;
    vector<reduce_type> reduce_type_in_global_level_vec;

    // 剩余共享内存的大小
    int bytes_of_shared_mem_remain_size = get_config()["SHARED_MEM_TOTAL_SIZE"].as_integer();

    // 记录不同的密集子块是的实际行号的位置
    vector<sort_type> sub_block_sort_type_vec;

    // 记录每个核函数的线程块数量和线程块内部的线程数量
    vector<unsigned int> kernal_block_num_vec;
    vector<unsigned int> kernal_thread_num_in_block_vec;

    // 用一个void指针数组来存储所有的模板，
    vector<void *> template_vec;

    // 用一个数组来存储所有的模板的类型
    vector<template_type> template_type_vec;

} code_builder_t;

// 用操作管理器搞出来代码生成器
code_builder_t *init_code_builder(operator_manager_t *op_manager);

// 代码生成器，只将一部分子矩阵生成代码，子矩阵号存在sub_matrix_id_vec中
code_builder_t *init_code_builder(operator_manager_t *op_manager, vector<int> sub_matrix_id_vec);

// 用一个大函数来生成头文件的字符串和实际的字符串
string build_header_file(code_builder_t *builder);

// 只将一部分头文件代码生成出来
string build_header_file(code_builder_t *builder, vector<int> sub_matrix_id_vec);

string build_main_file(code_builder_t *builder, unsigned long kernal_repeat = get_config()["KERNAL_REPEAT_TIME"].as_integer());

// 将一部分main文件的代码生成出来
string build_main_file(code_builder_t *builder, vector<int> sub_matrix_id_vec, unsigned long kernal_repeat = get_config()["KERNAL_REPEAT_TIME"].as_integer());

// 将代码生成器的元数据转化成字符串
string convert_code_builder_to_str(code_builder_t *builder);

string convert_sort_type_to_str(sort_type type);

string convert_reduce_type_to_str(reduce_type type);

// kernal生成器操作，将warp层次的归约设定为全局同步的方式

// kernal生成器操作，将block层次的归约设定为全局同步的方式

// kernal生成器操作，压缩一些索引

// 输出模板中某一个索引对变量的赋值代码
// 要赋给的变量名
// 这个数组在数据类型中的实际的位置，需要标明其子块号和索引号
// 针对每一个在index中的数组都需要这个东西
string code_of_index_arr_assignment(operator_manager_t *op_manager, unsigned long index_of_dense_block, unsigned long index_of_read_index, string target_var_name, string index_arr_var_name, string index_of_index_arr_var_name);

//从文件读入到string里
string readFileIntoString(string filename);

string code_of_kernal_define(code_builder_t *builder);

string code_of_kernal_define(code_builder_t *builder, vector<int> sub_matrix_id_vec);

// 获得包含了稀疏矩阵所有数据结构的声明
string code_of_compressed_matrix_content_define(code_builder_t *code_builder);

string code_of_compressed_matrix_content_define(code_builder_t *code_builder, vector<int> sub_matrix_id_vec);

string code_of_all_compressed_block_define(code_builder_t *code_builder);

string code_of_data_type(data_type type);

string code_of_arr_var_name(int index_of_dense_block, int index_of_read_index, string arr_name);

string code_of_y_write_arr_var_name(int index_of_dense_block, int index_of_y_write_index, string arr_name);

string code_of_matrix_file_read(code_builder_t *code_builder);

string code_of_matrix_file_read(code_builder_t *code_builder, vector<int> sub_matrix_id_vec);

// 给对应子块加入一个模板
void add_template_to_builder(code_builder_t *builder, void *template_ptr, template_type type, unsigned long dense_block_id);

// 将某一个模板进行压缩
bool compress_template_in_builder(code_builder_t *builder, template_type type, unsigned long dense_block_id);

// 打印所有模板的数据
void store_code_builder_data(code_builder_t *builder, string output_dir = get_config()["ROOT_PATH_STR"].as_string() + "/data_source");

// 打印特定模板的数据
void store_code_builder_data(code_builder_t *builder, vector<int> sub_matrix_id_vec, string output_dir = get_config()["ROOT_PATH_STR"].as_string() + "/data_source");

// 将string写到文件
void write_string_to_file(string file_name, string output_str);

// 默认遍历2000遍，然后加一个计时函数，然后判断是都需要计算性能
string code_of_main_function(code_builder_t *code_builder, unsigned long kernal_repeat = get_config()["KERNAL_REPEAT_TIME"].as_integer(), bool perf_test = true);

// 对特定子块生成main函数
string code_of_main_function(code_builder_t *code_builder, vector<int> sub_matrix_id_vec, unsigned long kernal_repeat = get_config()["KERNAL_REPEAT_TIME"].as_integer(), bool perf_test = true);

string code_line_of_pointer_define(data_type type, string var_name);

string code_line_of_cuda_malloc(data_type type, string code_of_size, string arr_name);

string code_line_of_cuda_memcpy(string var_name_of_dest_arr, string var_name_of_source_arr, data_type type, string size_var_str, string copy_direct_str);

string code_of_kernal_func_call(code_builder_t *code_builder, unsigned long dense_block_index);

string code_of_a_formal_param_declare(data_type type, string var_name);

string code_of_kernal_func_define(code_builder_t *code_builder, unsigned long index_of_dense_block);

string code_of_template_data_struct(void *output_template, template_type type, unsigned long dense_block_id);

// 末尾的参数用来记录是不是要禁止排序行索引的共享内存
string code_of_read_template_data_from_file_func_define(void *output_template, template_type type, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

// 末尾的参数用来记录是不是要禁止排序行索引的共享内存
string code_of_write_template_data_to_gpu(void *output_template, template_type type, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

string code_of_template_kernal(void *output_template, template_type type, unsigned long dense_block_id);

string code_of_kernal_function_call(void *output_template, template_type type, unsigned long dense_block_id);

// 将模板类型转化为string
string convert_template_type_to_string(template_type type);

#endif