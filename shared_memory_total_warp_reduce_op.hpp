// 一个带warp reduce模板，其中warp块是按照行对齐的，一行包含多个warp，一个warp不能包含多行。
// 需要满足每一行的块为32的倍数，每个warp将自己的内容放到block中，然后在block中做一次归约。
// 因为每个warp负责的内容都在一行内，所以行列分块没有意义，这里本质上要一个线程一个非零元就好了，所以无视thread层级是怎么分块的
// 都让一个线程负责一个非零元即可
// 在树状归约问题上，先可以让每次归约的结果放在共享内存中
// 这个版本的warp结果可以在sharedmemory中规约，还有一个原子加的版本，和thread的版本一样，都有两个版本

#ifndef SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE_H
#define SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE_H

#include "struct.hpp"
#include "config.hpp"
#include "arr_optimization.hpp"
#include "code_builder.hpp"

typedef struct shared_memory_total_warp_reduce_template
{
    // 稠密矩阵号
    unsigned long dense_block_index;

    // 对应的密集矩阵
    sparse_struct_t *matrix = NULL;

    // 当前密集子块的首行行号
    unsigned long kernal_first_row_index = 0;
    unsigned long kernal_first_col_index = 0;

    // 存在compressed row padding，只有有效的compressed内相对行号才需要找到原索引，并写回结果。
    unsigned long effective_row_num;

    // 用一个变量存是否要用原子加来归约
    bool is_atom_add = false;
    
    // 重新构建一个数组，每一行在warp计算结果中的偏移量
    void *row_offset_in_warp_tmp_result = NULL;
    data_type data_type_of_row_offset_in_warp_tmp_result;
    unsigned long size_of_row_offset_in_warp_tmp_result;

    // 每个block的首行行号，用来进行归约
    void *block_first_row_index = NULL;
    data_type data_type_of_block_first_row_index;
    unsigned long size_of_block_first_row_index;

    // 每个block的第一个warp粒度的块的索引
    void *block_begin_warp_index_offset = NULL;
    data_type data_type_of_block_begin_warp_index_offset;
    unsigned long size_of_block_begin_warp_index_offset;

    // 排序相关
    // 用一个可能存在的数组存储排序之后的输出，可能有全局的和局部的两种情况
    bool global_sort_index = false;
    bool local_sort_index = false;
    void *row_index_before_sort = NULL;
    data_type data_type_of_row_index_before_sort;
    unsigned long size_of_row_index_before_sort;

    // 每个warp粒度的块的第一个非零元的索引。不需要block索引
    void *global_warp_block_first_nz = NULL;
    data_type data_type_of_global_warp_block_first_nz;
    unsigned long size_of_global_warp_block_first_nz;

    // 当前稠密视图子块的所有值
    void *val_arr = NULL;
    data_type data_type_of_val_arr;
    unsigned long size_of_val_arr;

    // 当前稠密视图子块的所有列号
    void *col_index_arr = NULL;
    data_type data_type_of_col_index_arr;
    unsigned long size_of_col_index_arr;

    // warp计算结果的行偏移，可以线性压缩
    arr_compress_type row_offset_in_warp_tmp_result_compress = NONE_COMPRESS;
    void *row_offset_in_warp_tmp_result_compress_meta = NULL;
    // 每个block的行起始位置的压缩，可以线性压缩
    arr_compress_type block_first_row_index_compress = NONE_COMPRESS;
    void *block_first_row_index_compress_meta = NULL;

    // 每个block的第一个warp的索引
    arr_compress_type block_begin_warp_index_offset_compress = NONE_COMPRESS;
    void *block_begin_warp_index_offset_compress_meta = NULL;

    // warp块的第一个非零元
    arr_compress_type global_warp_block_first_nz_compress = NONE_COMPRESS;
    void *global_warp_block_first_nz_compress_meta = NULL;

    // 排序原索引
    arr_compress_type row_index_before_sort_compress = NONE_COMPRESS;
    void *row_index_before_sort_compress_meta = NULL;

    // 当前内核使用的线程块数量和线程块内的线程数量
    unsigned long tblock_num = get_config()["DEFAULT_THREAD_BLOCK_NUM"].as_integer();
    unsigned long thread_num_in_block = get_config()["DEFAULT_THREAD_NUM_IN_BLOCK"].as_integer();

    // 当前模板每一行的树状规约并行度
    unsigned long thread_num_of_row_reduce = 1;

    // 用一个数存储一个模板的id的哈希
    unsigned long hash_of_this_template;

} shared_memory_total_warp_reduce_template_t;

shared_memory_total_warp_reduce_template_t *init_shared_memory_total_warp_reduce_template(code_builder_t *builder, unsigned long dense_block_id);

bool is_supported_by_shared_memory_total_warp_reduce_template(code_builder_t *builder, unsigned long dense_block_id);

bool is_supported_by_shared_memory_total_warp_reduce_template(sparse_struct_t *matrix, unsigned long dense_block_id);

// 打印所有数据
void store_template_data(shared_memory_total_warp_reduce_template_t *output_template, string output_dir, bool force_not_share_global_sort_index = false);

string code_of_template_data_struct(shared_memory_total_warp_reduce_template_t *output_template, unsigned long dense_block_id);

string code_of_read_template_data_from_file_func_define(shared_memory_total_warp_reduce_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

string code_of_template_kernal(shared_memory_total_warp_reduce_template_t *output_template, unsigned long dense_block_id);

string code_of_kernal_function_call(shared_memory_total_warp_reduce_template_t *output_template, unsigned long dense_block_id);

string code_of_write_template_data_to_gpu(shared_memory_total_warp_reduce_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

bool compress_block_begin_warp_index_offset(shared_memory_total_warp_reduce_template_t *output_template, bool need_check, arr_compress_type type);

bool compress_row_offset_in_warp_tmp_result(shared_memory_total_warp_reduce_template_t *output_template, bool need_check, arr_compress_type type);

bool compress_block_first_row_index(shared_memory_total_warp_reduce_template_t *output_template, bool need_check, arr_compress_type type);

bool compress_global_warp_block_first_nz(shared_memory_total_warp_reduce_template_t *output_template, bool need_check, arr_compress_type type);

// 尝试所有的压缩
void try_all_compress(shared_memory_total_warp_reduce_template_t *output_template);

// 归约每一行结果的线程
bool set_row_reduce_thread_num(shared_memory_total_warp_reduce_template_t *output_template, unsigned long row_reduce_thread_num);

#endif
