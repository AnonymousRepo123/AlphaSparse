// 用来处理一行一个或者多个block的情况，必须要求当前子块的最少一个block一行，并且线程粒度的块的大小等于1。在这种情况下，要处理的非零元数量是32的倍数（padding），并且通过使用warp层次的归约，进一步提升性能
// warp reduce https://blog.csdn.net/Bruce_0712/article/details/64926471
// 从shared_memory_template_warp_compress进一步压缩而来，也可以直接从生成出来

// 先不进行正确性测试

#ifndef SHARED_MEMORY_LONG_ROW_TEMPLATE_H
#define SHARED_MEMORY_LONG_ROW_TEMPLATE_H

#include "struct.hpp"
#include "config.hpp"
#include "arr_optimization.hpp"
#include "code_builder.hpp"
#include "shared_memory_op.hpp"
#include "shared_memory_op_warp_compress.hpp"

typedef struct shared_memory_long_row_template
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

    // 每一个线程块粒度的块所处的行号，
    void *row_index_of_block_level_block = NULL;
    data_type data_type_of_row_index_of_block_level_block;
    unsigned long size_of_row_index_of_block_level_block;

    // 每个线程块粒度的块的起始非零元数量
    void *block_nz_begin_offset = NULL;
    data_type data_type_of_block_nz_begin_offset;
    unsigned long size_of_block_nz_begin_offset;

    // 排序相关
    // 用一个可能存在的数组存储排序之后的输出，可能有全局的和局部的两种情况
    bool global_sort_index = false;
    bool local_sort_index = false;
    void *row_index_before_sort = NULL;
    data_type data_type_of_row_index_before_sort;
    unsigned long size_of_row_index_before_sort;

    // 当前稠密视图子块的所有值
    void *val_arr = NULL;
    data_type data_type_of_val_arr;
    unsigned long size_of_val_arr;

    // 当前稠密视图子块的所有列号
    void *col_index_arr = NULL;
    data_type data_type_of_col_index_arr;
    unsigned long size_of_col_index_arr;

    // 压缩每一行的行号
    arr_compress_type row_index_of_block_level_block_compress = NONE_COMPRESS;
    void *row_index_of_block_level_block_compress_meta = NULL;

    // 压缩块非零元起始
    arr_compress_type block_nz_begin_offset_compress = NONE_COMPRESS;
    void *block_nz_begin_offset_compress_meta = NULL;

    arr_compress_type row_index_before_sort_compress = NONE_COMPRESS;
    void *row_index_before_sort_compress_meta = NULL;

    // 当前内核使用的线程块数量和线程块内的线程数量
    unsigned long tblock_num = get_config()["DEFAULT_THREAD_BLOCK_NUM"].as_integer();
    unsigned long thread_num_in_block = get_config()["DEFAULT_THREAD_NUM_IN_BLOCK"].as_integer();

    // 当前模板每一行的树状规约并行度
    // unsigned long thread_num_of_row_reduce = 1;

    // 用一个数存储一个模板的id的哈希
    unsigned long hash_of_this_template;

} shared_memory_long_row_template_t;

// 初始化一个新的模板，用矩阵的压缩视图初始化
shared_memory_long_row_template_t *init_shared_memory_long_row_template(code_builder_t *builder, unsigned long dense_block_id);

bool is_supported_by_shared_memory_long_row_template(code_builder_t *builder, unsigned long dense_block_id);

bool is_supported_by_shared_memory_long_row_template(sparse_struct_t *matrix, unsigned long dense_block_id);

void store_template_data(shared_memory_long_row_template_t *output_template, string output_dir, bool force_not_share_global_sort_index = false);

// 构造数据结构
string code_of_template_data_struct(shared_memory_long_row_template_t *output_template, unsigned long dense_block_id);

string code_of_read_template_data_from_file_func_define(shared_memory_long_row_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

string code_of_write_template_data_to_gpu(shared_memory_long_row_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

string code_of_template_kernal(shared_memory_long_row_template_t *output_template, unsigned long dense_block_id);

string code_of_kernal_function_call(shared_memory_long_row_template_t *output_template, unsigned long dense_block_id);

// 压缩每个线程粒度的子块的全局行号，一般使用线性压缩
bool compress_row_index_of_block_level_block(shared_memory_long_row_template_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

// 压缩子块的起始
bool compress_block_nz_begin_offset(shared_memory_long_row_template_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

void try_all_compress(shared_memory_long_row_template_t *output_template);

#endif