#ifndef DIRECT_ATOM_TEMPLATE_WRAP_COMPRESS_H
#define DIRECT_ATOM_TEMPLATE_WRAP_COMPRESS_H

#include "direct_atom_op.hpp"

// 取出warp层次所有元数据的模板， 每个block内部所有线程粒度的块一样大就可以压缩
typedef struct direct_atom_template_warp_compress
{
    // 模板对应的稠密矩阵号
    unsigned long dense_block_index;
    // 对应的密集矩阵
    sparse_struct_t *matrix = NULL;
    // 当前密集子块的首行行号
    unsigned long kernal_first_row_index = 0;
    unsigned long kernal_first_col_index = 0;

    // 因为row padding的存在，压缩子矩阵最后的一些TLB可能是无效的，这些TLB集中在稠密子块的尾部。
    // 需要一个变量来存储有效的TLB数量。
    unsigned long effective_TLB_num;

    // 用一个变量存是否要用原子加来归约
    bool is_atom_add = false;

    // 重构一个新的矩阵，用来直接存储线程粒度的块所对应的全局行号
    void *global_row_index_of_thread_level_block = NULL;
    data_type data_type_of_global_row_index_of_thread_level_block;
    unsigned long size_of_global_row_index_of_thread_level_block;

    // 只需要block和thread两个层次的索引，包括block的thread索引偏移，block的nz起始位置，block内所有thread块的大小
    // 首先第一个是thread块的大小
    void *block_begin_thread_index_offset = NULL;
    data_type data_type_of_block_begin_thread_index_offset;
    unsigned long size_of_block_begin_thread_index_offset;

    // block中首个非零元偏移
    void *block_nz_begin_offset = NULL;
    data_type data_type_of_block_nz_begin_offset;
    unsigned long size_of_block_nz_begin_offset;

    // block中每个线程粒度的块的非零元数量，每个block的这个值是相同的
    void *thread_block_size_in_block = NULL;
    data_type data_type_of_thread_block_size_in_block;
    unsigned long size_of_thread_block_size_in_block;

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

    // 线程粒度块的全局行号的压缩，可以使用线性压缩，加上可能的压缩元数据，也可以使用周期线性压缩
    arr_compress_type global_row_index_compress = NONE_COMPRESS;
    void *global_row_index_compress_meta = NULL;
    // 线程块粒度的块中所有线程粒度的块的偏移的压缩，可以使用线性压缩
    arr_compress_type block_begin_thread_index_offset_compress = NONE_COMPRESS;
    void *block_begin_thread_index_offset_compress_meta = NULL;
    // block首个非零元偏移，可以使用线性压缩
    arr_compress_type block_nz_begin_offset_compress = NONE_COMPRESS;
    void *block_nz_begin_offset_compress_meta = NULL;
    // 每个线程块粒度的块中线程粒度块的大小，可以常值压缩
    arr_compress_type thread_block_size_in_block_compress = NONE_COMPRESS;
    void *thread_block_size_in_block_compress_meta = NULL;

    // 排序产生的数组不太可能压缩
    arr_compress_type row_index_before_sort_compress = NONE_COMPRESS;
    void *row_index_before_sort_compress_meta = NULL;

    // 当前内核使用的线程块数量和线程块内的线程数量
    unsigned long tblock_num = get_config()["DEFAULT_THREAD_BLOCK_NUM"].as_integer();
    unsigned long thread_num_in_block = get_config()["DEFAULT_THREAD_NUM_IN_BLOCK"].as_integer();

    // 用一个数存储一个模板的id的哈希
    unsigned long hash_of_this_template;
} direct_atom_template_warp_compress_t;

// 用老的初始化新的、压缩之后的模板，并且析构原来的模板
direct_atom_template_warp_compress_t *init_direct_atom_template_warp_compress(direct_atom_template_t *old_template);

direct_atom_template_warp_compress_t *init_direct_atom_template_warp_compress(code_builder_t *builder, unsigned long dense_block_id);

bool is_supported_by_direct_atom_template_warp_compress(code_builder_t *builder, unsigned long dense_block_id);

bool is_supported_by_direct_atom_template_warp_compress(sparse_struct_t* matrix, unsigned long dense_block_id);

// 持久化一个模板内所需要的数据，被压缩的向量不需要持久化
void store_template_data(direct_atom_template_warp_compress_t *output_template, string output_dir, bool force_not_share_global_sort_index = false);

string code_of_template_data_struct(direct_atom_template_warp_compress_t *output_template, unsigned long dense_block_id);

string code_of_read_template_data_from_file_func_define(direct_atom_template_warp_compress_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

string code_of_write_template_data_to_gpu(direct_atom_template_warp_compress_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

string code_of_kernal_function_call(direct_atom_template_warp_compress_t *output_template, unsigned long dense_block_id);

string code_of_template_kernal(direct_atom_template_warp_compress_t *output_template, unsigned long dense_block_id);

// 压缩每个线程粒度的子块的全局行号，一般使用线性压缩
bool compress_global_row_index_of_thread_level_block(direct_atom_template_warp_compress_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

// 压缩block的首thread索引，使用线性压缩
bool compress_block_begin_thread_index_offset(direct_atom_template_warp_compress_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

// 压缩每个block内部的thread块大小，可以使用常值压缩
bool compress_thread_block_size_in_block(direct_atom_template_warp_compress_t *output_template, bool need_check = true, arr_compress_type type = CONSTANT_COMPRESS);

// 压缩block的首个非零元索引，使用线性压缩
bool compress_block_nz_begin_offset(direct_atom_template_warp_compress_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

// 尝试所有的压缩
void try_all_compress(direct_atom_template_warp_compress_t *output_template);

#endif