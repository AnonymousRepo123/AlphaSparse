#ifndef DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE_H
#define DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE_H

#include "struct.hpp"
#include "config.hpp"
#include "arr_optimization.hpp"
#include "code_builder.hpp"

typedef struct direct_atom_total_warp_reduce_template
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

    // 如果有在压缩子图中执行了padding，那么尾部的WLB是无效的，所以需要一个变量来存储有效
    unsigned long effective_WLB_num;

    // 存储一个warp粒度的块对应的行号
    void *global_row_index_of_warp_level_block = NULL;
    data_type data_type_of_global_row_index_of_warp_level_block;
    unsigned long size_of_global_row_index_of_warp_level_block;

    // 每个warp的非零元起始位置
    void *global_warp_nz_begin_offset = NULL;
    data_type data_type_of_global_warp_nz_begin_offset;
    unsigned long size_of_global_warp_nz_begin_offset;

    // 用一个可能存在的数组存储排序之后的输出，可能有全局的和局部的两种情况
    bool global_sort_index = false;
    bool local_sort_index = false;
    void *row_index_before_sort = NULL;
    data_type data_type_of_row_index_before_sort;
    unsigned long size_of_row_index_before_sort;

    // 当前稠密视图子块的所有值，值数组和列号要重新padding，让整个线程块padding起来，每个thread块从thread块的线程块内的索引开始遍历，每次自增一个block中thread块大小的长度
    void *val_arr = NULL;
    data_type data_type_of_val_arr;
    unsigned long size_of_val_arr;

    // 当前稠密视图子块的所有列号
    void *col_index_arr = NULL;
    data_type data_type_of_col_index_arr;
    unsigned long size_of_col_index_arr;

    arr_compress_type global_row_index_of_warp_level_block_compress = NONE_COMPRESS;
    void *global_row_index_of_warp_level_block_compress_meta = NULL;

    // block首个非零元偏移，可以使用线性压缩
    arr_compress_type global_warp_nz_begin_offset_compress = NONE_COMPRESS;
    void *global_warp_nz_begin_offset_compress_meta = NULL;

    // 排序产生的数组不太可能压缩
    arr_compress_type row_index_before_sort_compress = NONE_COMPRESS;
    void *row_index_before_sort_compress_meta = NULL;

    // 当前内核使用的线程块数量和线程块内的线程数量
    unsigned long tblock_num = get_config()["DEFAULT_THREAD_BLOCK_NUM"].as_integer();
    unsigned long thread_num_in_block = get_config()["DEFAULT_THREAD_NUM_IN_BLOCK"].as_integer();

    // 用一个数存储一个模板的id的哈希
    unsigned long hash_of_this_template;

} direct_atom_total_warp_reduce_template_t;

direct_atom_total_warp_reduce_template_t *init_direct_atom_total_warp_reduce_template(code_builder_t *builder, unsigned long dense_block_id);

bool is_supported_by_direct_atom_total_warp_reduce_template(code_builder_t *builder, unsigned long dense_block_id);

bool is_supported_by_direct_atom_total_warp_reduce_template(sparse_struct_t *matrix, unsigned long dense_block_id);

void store_template_data(direct_atom_total_warp_reduce_template_t *output_template, string output_dir, bool force_not_share_global_sort_index = false);

string code_of_template_data_struct(direct_atom_total_warp_reduce_template_t *output_template, unsigned long dense_block_id);

string code_of_read_template_data_from_file_func_define(direct_atom_total_warp_reduce_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

string code_of_write_template_data_to_gpu(direct_atom_total_warp_reduce_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

string code_of_kernal_function_call(direct_atom_total_warp_reduce_template_t *output_template, unsigned long dense_block_id);

string code_of_template_kernal(direct_atom_total_warp_reduce_template_t *output_template, unsigned long dense_block_id);

// 循环增加的行号和线性增加的行号
bool compress_global_row_index_of_warp_level_block(direct_atom_total_warp_reduce_template_t *output_template, bool need_check, arr_compress_type type);

// warp索引，线性压缩
bool compress_global_warp_nz_begin_offset(direct_atom_total_warp_reduce_template_t *output_template, bool need_check, arr_compress_type type);

void try_all_compress(direct_atom_total_warp_reduce_template_t *output_template);

#endif