#ifndef DIRECT_ATOM_TEMPLATE_WRAP_BLOCK_COMPRESS_H
#define DIRECT_ATOM_TEMPLATE_WRAP_BLOCK_COMPRESS_H

#include "direct_atom_op.hpp"

// 取出warp层次所有元数据的模板， 每个block内部所有线程粒度的块一样大就可以压缩
typedef struct direct_atom_template_warp_block_compress
{
    // 模板对应的稠密矩阵号
    unsigned long dense_block_index;
    // 对应的密集矩阵
    sparse_struct_t *matrix = NULL;
    // 当前密集子块的首行行号
    unsigned long kernal_first_row_index = 0;
    unsigned long kernal_first_col_index = 0;

    // 有效的线程数量，因为线程在执行完之后会直接写回自己的结果。而因为压缩视图的row padding产生的TLB都集中在压缩子图的最后
    // 所以对于所有的TLB来说，一开始的那些TLB都是有效的，之后的TLB都是无效的所以需要一个变量存储有效的TLB数量
    unsigned long effective_TLB_num;

    // 用一个变量存是否要用原子加来归约
    bool is_atom_add = false;

    // 重构一个新的矩阵，用来直接存储线程粒度的块所对应的全局行号
    void *global_row_index_of_thread_level_block = NULL;
    data_type data_type_of_global_row_index_of_thread_level_block;
    unsigned long size_of_global_row_index_of_thread_level_block;

    unsigned long thread_block_size_in_block;

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

    // 排序产生的数组不太可能压缩
    arr_compress_type row_index_before_sort_compress = NONE_COMPRESS;
    void *row_index_before_sort_compress_meta = NULL;

    // 当前内核使用的线程块数量和线程块内的线程数量
    unsigned long tblock_num = get_config()["DEFAULT_THREAD_BLOCK_NUM"].as_integer();
    unsigned long thread_num_in_block = get_config()["DEFAULT_THREAD_NUM_IN_BLOCK"].as_integer();

    // 用一个数存储一个模板的id的哈希
    unsigned long hash_of_this_template;
} direct_atom_template_warp_block_compress_t;

// 用老的初始化新的、压缩之后的模板，并且析构原来的模板
direct_atom_template_warp_block_compress_t *init_direct_atom_template_warp_block_compress(direct_atom_template_t *old_template);

// 直接构造一个压缩了block和warp的模板
direct_atom_template_warp_block_compress_t *init_direct_atom_template_warp_block_compress(code_builder_t *builder, unsigned long dense_block_id);

// 传入稀疏矩阵的模板支持检查
bool is_supported_by_direct_atom_template_warp_block_compress(sparse_struct_t* matrix, unsigned long dense_block_id);

bool is_supported_by_direct_atom_template_warp_block_compress(code_builder_t *builder, unsigned long dense_block_id);

// 持久化一个模板内所需要的数据，被压缩的向量不需要持久化
void store_template_data(direct_atom_template_warp_block_compress_t *output_template, string output_dir, bool force_not_share_global_sort_index = false);

string code_of_template_data_struct(direct_atom_template_warp_block_compress_t *output_template, unsigned long dense_block_id);

string code_of_read_template_data_from_file_func_define(direct_atom_template_warp_block_compress_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

string code_of_write_template_data_to_gpu(direct_atom_template_warp_block_compress_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

string code_of_kernal_function_call(direct_atom_template_warp_block_compress_t *output_template, unsigned long dense_block_id);

string code_of_template_kernal(direct_atom_template_warp_block_compress_t *output_template, unsigned long dense_block_id);

// 压缩每个线程粒度的子块的全局行号，一般使用线性压缩
bool compress_global_row_index_of_thread_level_block(direct_atom_template_warp_block_compress_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

// 尝试所有的压缩
void try_all_compress(direct_atom_template_warp_block_compress_t *output_template);

#endif