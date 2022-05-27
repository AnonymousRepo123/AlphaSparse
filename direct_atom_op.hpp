#ifndef DIRECT_ATOM_TEMPLATE_H
#define DIRECT_ATOM_TEMPLATE_H

#include "struct.hpp"
#include "config.hpp"
#include "arr_optimization.hpp"
#include "code_builder.hpp"

// 一个模板，从线程粒度的块，直接规约到底到显存
typedef struct direct_atom_template
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

    // 用两个数组存储warp号和thread号的CSR压缩，直接拷贝指针
    void *block_begin_warp_index_offset = NULL;
    data_type data_type_of_block_begin_warp_index_offset;
    unsigned long size_of_block_begin_warp_index_offset;

    void *warp_begin_thread_index_offset = NULL;
    data_type data_type_of_warp_begin_thread_index_offset;
    unsigned long size_of_warp_begin_thread_index_offset;

    // 用一个数组存储每个warp的thread块大小
    void *thread_block_size_in_warp = NULL;
    data_type data_type_of_thread_block_size_in_warp;
    unsigned long size_of_thread_block_size_in_warp;

    // 用一个可能存在的数组存储排序之后的输出，可能有全局的和局部的两种情况
    bool global_sort_index = false;
    bool local_sort_index = false;
    void *row_index_before_sort = NULL;
    data_type data_type_of_row_index_before_sort;
    unsigned long size_of_row_index_before_sort;

    // block和warp的偏移量，以及遍历的范围可以尝试使用共享内存的方式来解决。其中block和warp的偏移量的偏移量可以使用
    // 大小都和不同层次的block size相同
    void *block_nz_begin_offset = NULL;
    data_type data_type_of_block_nz_begin_offset;
    unsigned long size_of_block_nz_begin_offset;

    // 相对偏移
    void *warp_nz_begin_offset = NULL;
    data_type data_type_of_warp_nz_begin_offset;
    unsigned long size_of_warp_nz_begin_offset;

    // 当前稠密视图子块的所有值
    void *val_arr = NULL;
    data_type data_type_of_val_arr;
    unsigned long size_of_val_arr;

    // 当前稠密视图子块的所有列号
    void *col_index_arr = NULL;
    data_type data_type_of_col_index_arr;
    unsigned long size_of_col_index_arr;

    // 线程粒度块的全局行号的压缩，可以使用线性压缩，加上可能的压缩元数据，也可以使用周期线性压缩
    arr_compress_type global_row_index_compress = NONE_COMPRESS;
    void *global_row_index_compress_meta = NULL;
    // 线程块粒度的块的第一个warp粒度块号，可以使用线性压缩
    arr_compress_type block_begin_warp_index_compress = NONE_COMPRESS;
    void *block_begin_warp_index_compress_meta = NULL;
    // warp粒度的块的第一个thread粒度块号，可以使用线性压缩
    arr_compress_type warp_begin_thread_index_compress = NONE_COMPRESS;
    void *warp_begin_thread_index_compress_meta = NULL;
    // warp中线程粒度的块的大小，可以使用常值压缩
    arr_compress_type thread_block_size_compress = NONE_COMPRESS;
    void *thread_block_size_compress_meta = NULL;

    // 排序产生的数组不太可能压缩
    arr_compress_type row_index_before_sort_compress = NONE_COMPRESS;
    void *row_index_before_sort_compress_meta = NULL;

    // 块非零元偏移和warp非零元偏移的的压缩
    arr_compress_type block_nz_begin_offset_compress = NONE_COMPRESS;
    void *block_nz_begin_offset_compress_meta = NULL;
    // 可以使用周期线性压缩，作作为相对索引可以使用的压缩方式
    arr_compress_type warp_nz_begin_offset_compress = NONE_COMPRESS;
    void *warp_nz_begin_offset_compress_meta = NULL;

    // 当前内核使用的线程块数量和线程块内的线程数量
    unsigned long tblock_num = get_config()["DEFAULT_THREAD_BLOCK_NUM"].as_integer();
    unsigned long thread_num_in_block = get_config()["DEFAULT_THREAD_NUM_IN_BLOCK"].as_integer();

    // 用一个数存储一个模板的id的哈希
    unsigned long hash_of_this_template;
} direct_atom_template_t;

// 初始化，用code_build初始化，并且注明对应的稠密矩阵号
direct_atom_template_t *init_direct_atom_template(code_builder_t *builder, unsigned long dense_block_id);

// 用一个函数判断一个子矩阵是否可以匹配当前模板
bool is_supported_by_direct_atom_template(code_builder_t *builder, unsigned long dense_block_id);

bool is_supported_by_direct_atom_template(sparse_struct_t *matrix, unsigned long dense_block_id);

// 持久化一个模板内所需要的数据，被压缩的向量不需要持久化
void store_template_data(direct_atom_template_t *output_template, string output_dir, bool force_not_share_global_sort_index = false);

// 构造数据结构
string code_of_template_data_struct(direct_atom_template_t *output_template, unsigned long dense_block_id);

string code_of_read_template_data_from_file_func_define(direct_atom_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

// 在main函数中将模板的数据读出来，并且拷贝到对应的显存中
string code_of_write_template_data_to_gpu(direct_atom_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

string code_of_template_kernal(direct_atom_template_t *output_template, unsigned long dense_block_id);

string code_of_kernal_function_call(direct_atom_template_t *output_template, unsigned long dense_block_id);

// 压缩每个线程粒度的子块的全局行号，一般使用线性压缩
bool compress_global_row_index_of_thread_level_block(direct_atom_template_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

// 压缩block的首warp索引，使用线性压缩
bool compress_block_begin_warp_index_offset(direct_atom_template_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

// 压缩warp的首thread索引，使用线性压缩，当每一个warp的thread数量相同时出现
bool compress_warp_begin_thread_index_offset(direct_atom_template_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

// 压缩每个warp内部的thread块大小，可以使用常值压缩
bool compress_thread_block_size_in_warp(direct_atom_template_t *output_template, bool need_check = true, arr_compress_type type = CONSTANT_COMPRESS);

// 压缩block的首个非零元索引，使用线性压缩
bool compress_block_nz_begin_offset(direct_atom_template_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

// 压缩warp的首个非零元相对索引，使用周期线性压缩
bool compress_warp_nz_begin_offset(direct_atom_template_t *output_template, bool need_check = true, arr_compress_type type = CYCLE_LINEAR_COMPRESS);

// 执行全部的压缩
void try_all_compress(direct_atom_template_t *output_template);

#endif