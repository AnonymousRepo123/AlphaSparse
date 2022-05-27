// 类似于CSR5的模板，相比unaligned_warp_reduce_same_TLB_size_template引入了新的warp全局的归约
// 也就是当一个整个WLB都在一行内的时候，直接做一个树状归约
// 这就需要每个warp知道自己和下一个warp的起始行号。warp_first_row数组内容的数量是WLB_num + 1
// 需要引入unaligned_warp_reduce_same_TLB_size_template的头文件，需要共用其一些工具函数
#ifndef UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE_H
#define UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE_H

#include "struct.hpp"
#include "config.hpp"
#include "arr_optimization.hpp"
#include "code_builder.hpp"
#include "unaligned_warp_reduce_same_TLB_size_op.hpp"

typedef struct unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce
{
    // 模板对应的稠密矩阵号
    unsigned long dense_block_index;
    // 对应的密集矩阵
    sparse_struct_t *matrix = NULL;
    // 当前密集子块的首行行号
    unsigned long kernal_first_row_index = 0;
    unsigned long kernal_first_col_index = 0;

    // 用4个变量分别存储合并之后的线程粒度元数据
    int bit_num_of_thread_level_combine_meta;
    int bit_num_of_sum_begin_bit_flag;
    int bit_num_of_first_relative_reduce_row_of_thread_level_block;
    int bit_num_of_tmp_result_reduce_offset_of_thread_level_block;

    // 用一个变量来决定是不是一定要强制所有的显存写都是原子加
    bool is_all_force_atom_add = false;

    // 全局的TLB大小
    unsigned long global_thread_level_block_size = 0;

    // 用一个数组来存储所有warp的首行索引
    void* global_first_row_index_of_warp_level_block = NULL;
    data_type data_type_of_global_first_row_index_of_warp_level_block;
    // 大小是WLB的数量
    unsigned long size_of_global_first_row_index_of_warp_level_block;

    // 用一个数组来存储所有加和起始位置，主要来自于WLB的第一个元素，和行第一个元素
    vector<vector<bool>> sum_bool_flag_of_sum_begin;

    // 用一个数组来存储所有TLB的相对行号
    void* first_relative_reduce_row_of_thread_level_block = NULL;
    data_type data_type_of_first_relative_reduce_row_of_thread_level_block;
    // TLB的数量
    unsigned long size_of_first_relative_reduce_row_of_thread_level_block;

    // 每个线程的归约偏移量，行的身子分布在多个线程中，所以行的身子要先做一次归约，然后在和行脑袋拼在一起
    void* tmp_result_reduce_offset_of_thread_level_block = NULL;
    data_type data_type_of_tmp_result_reduce_offset_of_thread_level_block;
    unsigned long size_of_tmp_result_reduce_offset_of_thread_level_block;

    // 用一个变量合并上面所有元数据的存储
    void* combine_meta_of_thread_level_block = NULL;
    data_type data_type_of_combine_meta_of_thread_level_block;
    unsigned long size_of_combine_meta_of_thread_level_block;

    // 用一个可能存在的数组存储排序之后的输出，可能有全局的和局部的两种情况
    bool global_sort_index = false;
    bool local_sort_index = false;
    void *row_index_before_sort = NULL;
    data_type data_type_of_row_index_before_sort;
    unsigned long size_of_row_index_before_sort;

    // 当前稠密视图子块的所有值，经过padding和交错存储
    void *val_arr = NULL;
    data_type data_type_of_val_arr;
    unsigned long size_of_val_arr;

    // 当前稠密视图子块的所有列号，经过padding和交错存储
    void *col_index_arr = NULL;
    data_type data_type_of_col_index_arr;
    unsigned long size_of_col_index_arr;
    
    // 压缩warp起始行
    arr_compress_type global_first_row_index_of_warp_level_block_compress = NONE_COMPRESS;
    void *global_first_row_index_of_warp_level_block_compress_meta = NULL;
    
    // 排序数组的压缩，对于矩阵分解的情况可能存在
    arr_compress_type row_index_before_sort_compress = NONE_COMPRESS;
    void *row_index_before_sort_compress_meta = NULL;

    // 这个模板中，线程的数量一定要比TLB数量要多，线程块的数量不能自定义，线程块数量的定义需要使得线程数量刚好比TLB数量多
    unsigned long thread_num_in_block = get_config()["DEFAULT_THREAD_NUM_IN_BLOCK"].as_integer();

    // 用一个数存储一个模板的id的哈希
    unsigned long hash_of_this_template;
} unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t;

unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t* init_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce(code_builder_t *builder, unsigned long dense_block_id);

// 判断当前是矩阵是不是支持当前模板
bool is_supported_by_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce(code_builder_t* builder, unsigned long dense_block_id);

bool is_supported_by_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce(sparse_struct_t* matrix, unsigned long dense_block_id);

// 将相关数据存到磁盘中
void store_template_data(unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *output_template, string output_dir, bool force_not_share_global_sort_index = false);

// 执行compress
bool compress_global_first_row_index_of_warp_level_block(unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

// 压缩排序行索引
bool compress_row_index_before_sort(unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

// 构造数据结构
string code_of_template_data_struct(unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *output_template, unsigned long dense_block_id);

// 从文件中读取数据的代码
string code_of_read_template_data_from_file_func_define(unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

// 在main函数中将模板的数据读出来，并且拷贝到对应的显存中
string code_of_write_template_data_to_gpu(unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

string code_of_template_kernal(unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *output_template, unsigned long dense_block_id);

string code_of_kernal_function_call(unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *output_template, unsigned long dense_block_id);

void try_all_compress(unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *output_template);

#endif