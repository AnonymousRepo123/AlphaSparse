// 将warp级别的遍历去掉，前提条件的是block内所有线程粒度的块的大小相等
#ifndef SHARED_MEMORY_TEMPLATE_WARP_COMPRESS_H
#define SHARED_MEMORY_TEMPLATE_WARP_COMPRESS_H

#include "struct.hpp"
#include "config.hpp"
#include "arr_optimization.hpp"
#include "code_builder.hpp"
#include "shared_memory_op.hpp"

typedef struct shared_memory_template_warp_compress
{
    // 模板对应的稠密矩阵号
    unsigned long dense_block_index;
    // 对应的密集矩阵
    sparse_struct_t *matrix = NULL;
    // 当前密集子块的首行行号
    unsigned long kernal_first_row_index = 0;
    unsigned long kernal_first_col_index = 0;

    // 用一个变量来存储有效行数量
    unsigned long effective_row_num;

    // 用一个变量存是否要用原子加来归约
    bool is_atom_add = false;

    // 重新构建一个数组，存储每一行的在thread计算结果中的偏移量
    void *row_offset_in_thread_tmp_result = NULL;
    data_type data_type_of_row_offset_in_thread_tmp_result;
    unsigned long size_of_row_offset_in_thread_tmp_result;

    // 每个block的首行行号，用来进行归约
    void *block_first_row_index = NULL;
    data_type data_type_of_block_first_row_index;
    unsigned long size_of_block_first_row_index;

    // 每个block的第一个线程粒度的块的索引
    void *block_begin_thread_index_offset = NULL;
    data_type data_type_of_block_begin_thread_index_offset;
    unsigned long size_of_block_begin_thread_index_offset;

    // 每个block内线程粒度的块的大小
    void *thread_block_size_in_block = NULL;
    data_type data_type_of_thread_block_size_in_block;
    unsigned long size_of_thread_block_size_in_block;

    // 排序相关
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

    // 当前稠密视图子块的所有值
    void *val_arr = NULL;
    data_type data_type_of_val_arr;
    unsigned long size_of_val_arr;

    // 当前稠密视图子块的所有列号
    void *col_index_arr = NULL;
    data_type data_type_of_col_index_arr;
    unsigned long size_of_col_index_arr;

    // 线程的计算结果在结果中的行偏移量
    arr_compress_type row_offset_in_thread_tmp_result_compress = NONE_COMPRESS;
    void *row_offset_in_thread_tmp_result_compress_meta = NULL;
    // 每个block的行起始位置的压缩
    arr_compress_type block_first_row_index_compress = NONE_COMPRESS;
    void *block_first_row_index_compress_meta = NULL;

    // 线程块粒度的块的第一个warp粒度块号，可以使用线性压缩
    arr_compress_type block_begin_thread_index_offset_compress = NONE_COMPRESS;
    void *block_begin_thread_index_offset_compress_meta = NULL;

    // warp中线程粒度的块的大小，可以使用常值压缩
    arr_compress_type thread_block_size_in_block_compress = NONE_COMPRESS;
    void *thread_block_size_in_block_compress_meta = NULL;

    // 排序产生的数组不太可能压缩
    arr_compress_type row_index_before_sort_compress = NONE_COMPRESS;
    void *row_index_before_sort_compress_meta = NULL;

    // 块非零元偏移和warp非零元偏移的的压缩
    arr_compress_type block_nz_begin_offset_compress = NONE_COMPRESS;
    void *block_nz_begin_offset_compress_meta = NULL;

    // 当前内核使用的线程块数量和线程块内的线程数量
    unsigned long tblock_num = get_config()["DEFAULT_THREAD_BLOCK_NUM"].as_integer();
    unsigned long thread_num_in_block = get_config()["DEFAULT_THREAD_NUM_IN_BLOCK"].as_integer();

    // 当前模板每一行的树状规约并行度
    unsigned long thread_num_of_row_reduce = 1;

    // 用一个数存储一个模板的id的哈希
    unsigned long hash_of_this_template;

} shared_memory_template_warp_compress_t;

shared_memory_template_warp_compress_t *init_shared_memory_template_warp_compress(code_builder_t *builder, unsigned long dense_block_id);

shared_memory_template_warp_compress_t *init_shared_memory_template_warp_compress(shared_memory_template_t *old_template);

bool is_supported_by_shared_memory_template_warp_compress(code_builder_t *builder, unsigned long dense_block_id);

bool is_supported_by_shared_memory_template_warp_compress(sparse_struct_t *matrix, unsigned long dense_block_id);

void store_template_data(shared_memory_template_warp_compress_t *output_template, string output_dir, bool force_not_share_global_sort_index = false);

string code_of_template_data_struct(shared_memory_template_warp_compress_t *output_template, unsigned long dense_block_id);

string code_of_read_template_data_from_file_func_define(shared_memory_template_warp_compress_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

string code_of_template_kernal(shared_memory_template_warp_compress_t *output_template, unsigned long dense_block_id);

string code_of_write_template_data_to_gpu(shared_memory_template_warp_compress_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

string code_of_kernal_function_call(shared_memory_template_warp_compress_t *output_template, unsigned long dense_block_id);

// 一系列压缩
bool compress_block_begin_thread_index_offset(shared_memory_template_warp_compress_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

bool compress_thread_block_size_in_block(shared_memory_template_warp_compress_t *output_template, bool need_check = true, arr_compress_type type = CONSTANT_COMPRESS);

bool compress_block_nz_begin_offset(shared_memory_template_warp_compress_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

// 每个block首行行号的压缩
bool compress_block_first_row_index(shared_memory_template_warp_compress_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

// 每一行中间结果的压缩，可以线性压缩
bool compress_row_offset_in_thread_tmp_result(shared_memory_template_warp_compress_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

// 尝试所有的压缩
void try_all_compress(shared_memory_template_warp_compress_t *output_template);

// 设定每一行归约的并行度
bool set_row_reduce_thread_num(shared_memory_template_warp_compress_t *output_template, unsigned long row_reduce_thread_num);

#endif
