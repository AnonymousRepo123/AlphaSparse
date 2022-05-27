#ifndef UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_H
#define UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_H

#include "struct.hpp"
#include "config.hpp"
#include "arr_optimization.hpp"
#include "code_builder.hpp"

// 类似于CSR5的模板
typedef struct unaligned_warp_reduce_same_TLB_size_template
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

    // 当前内核使用的线程块数量和线程块内的线程数量
    unsigned long tblock_num = get_config()["DEFAULT_THREAD_BLOCK_NUM"].as_integer();
    unsigned long thread_num_in_block = get_config()["DEFAULT_THREAD_NUM_IN_BLOCK"].as_integer();

    // 用一个数存储一个模板的id的哈希
    unsigned long hash_of_this_template;
} unaligned_warp_reduce_same_TLB_size_template_t;

unaligned_warp_reduce_same_TLB_size_template_t* init_unaligned_warp_reduce_same_TLB_size_template(code_builder_t *builder, unsigned long dense_block_id);

// 判断当前是矩阵是不是支持当前模板
bool is_supported_by_unaligned_warp_reduce_same_TLB_size_template(code_builder_t* builder, unsigned long dense_block_id);

bool is_supported_by_unaligned_warp_reduce_same_TLB_size_template(sparse_struct_t* matrix, unsigned long dense_block_id);

// 传入一个COO格式的矩阵，获得经过空行补齐和把nnz padding到32*TLB_size的倍数的COO矩阵
void fill_empty_and_padding_to_align_warp(void *source_row_index_arr, void *source_col_index_arr, void *source_val_arr, data_type data_type_of_row_index_arr,
                                          data_type data_type_of_col_index_arr, data_type data_type_of_val_arr, unsigned long nnz, vector<unsigned long> &dest_row_index_vec, 
                                          vector<unsigned long> &dest_col_index_vec, vector<double> &dest_val_vec, unsigned long global_thread_level_block_size);

// 将相关数据存到磁盘中
void store_template_data(unaligned_warp_reduce_same_TLB_size_template_t *output_template, string output_dir, bool force_not_share_global_sort_index = false);

// 执行compress
bool compress_global_first_row_index_of_warp_level_block(unaligned_warp_reduce_same_TLB_size_template_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

// 压缩排序行索引
bool compress_row_index_before_sort(unaligned_warp_reduce_same_TLB_size_template_t *output_template, bool need_check = true, arr_compress_type type = LINEAR_COMPRESS);

// 输入行号，来判断加和的起始位置
vector<vector<bool>> get_sum_begin_bool_flag_of_each_thread(vector<unsigned long> row_index_vec, int nnz_in_thread);

// WLB的块的首行索引
vector<unsigned long> get_first_global_row_index_of_each_warp(vector<unsigned long> row_index_vec, int nnz_in_thread);

// 针对多个行身子中间结果先进行一次归约的偏移量，记录一个最大的偏移量，用来在之后确定需要的空间。
vector<unsigned long> get_tmp_result_reduce_offset_vec(vector<vector<bool>> begin_sum_bool_flag, unsigned long* max_tmp_result_reduce_offset);

vector<unsigned long> get_first_relative_reduce_row_of_thread_level_block_vec(vector<unsigned long> row_index_vec, vector<unsigned long> WLB_first_global_row_index, vector<vector<bool>> row_begin_bool_flag, int nnz_in_thread, unsigned long *max_first_relative_reduce_row);

void store_bool_flag_of_sum_begin_to_file(vector<vector<bool>> bool_vec, string input_file);

// 构造数据结构
string code_of_template_data_struct(unaligned_warp_reduce_same_TLB_size_template_t *output_template, unsigned long dense_block_id);

// 从文件中读取数据的代码
string code_of_read_template_data_from_file_func_define(unaligned_warp_reduce_same_TLB_size_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

// 在main函数中将模板的数据读出来，并且拷贝到对应的显存中
string code_of_write_template_data_to_gpu(unaligned_warp_reduce_same_TLB_size_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index = false);

string code_of_template_kernal(unaligned_warp_reduce_same_TLB_size_template_t *output_template, unsigned long dense_block_id);

string code_of_kernal_function_call(unaligned_warp_reduce_same_TLB_size_template_t *output_template, unsigned long dense_block_id);

// 根据一个元数据的最大值，查看这个元数据需要多少的bit来存储
int get_max_bit_num_of_meta_data(unsigned long max_meta_data);

template<typename T> T combine_meta_data(vector<bool> sum_begin_bool_flag, unsigned long reduce_offset, unsigned long reduce_relative_row_index, int bit_num_of_sum_begin_flag, int bit_num_of_reduce_offset, int bit_num_of_row_index);

// 将几个元数据合并到一起，生成一个对应数据类型的合并之后的数据，最终得到一个合并之后数据
unsigned long combine_meta_data_to_unsigned_long(vector<bool> sum_begin_bool_flag, unsigned long reduce_offset, unsigned long reduce_relative_row_index, int bit_num_of_sum_begin_flag, int bit_num_of_reduce_offset, int bit_num_of_row_index);

unsigned int combine_meta_data_to_unsigned_int(vector<bool> sum_begin_bool_flag, unsigned long reduce_offset, unsigned long reduce_relative_row_index, int bit_num_of_sum_begin_flag, int bit_num_of_reduce_offset, int bit_num_of_row_index);

unsigned short combine_meta_data_to_unsigned_short(vector<bool> sum_begin_bool_flag, unsigned long reduce_offset, unsigned long reduce_relative_row_index, int bit_num_of_sum_begin_flag, int bit_num_of_reduce_offset, int bit_num_of_row_index);

unsigned char combine_meta_data_to_unsigned_char(vector<bool> sum_begin_bool_flag, unsigned long reduce_offset, unsigned long reduce_relative_row_index, int bit_num_of_sum_begin_flag, int bit_num_of_reduce_offset, int bit_num_of_row_index);

// 将一个合并之后的数据转化为二进制bit的字符串
template<typename T> string convert_meta_data_to_bit_flag_string(T compress_bit);

template<typename T> string convert_meta_data_to_bit_flag_string_with_data_type(T compress_bit, data_type type);

// 将combine_meta写到文件中，按照bit的形式
void write_combine_meta_data_to_file(void* combine_meta_arr, data_type type, unsigned long length, string file_name);


void try_all_compress(unaligned_warp_reduce_same_TLB_size_template_t *output_template);

#endif