#ifndef PARAMETER_SET_STRATEGY_H
#define PARAMETER_SET_STRATEGY_H

#include "exe_graph.hpp"
#include "graph_enumerate.hpp"

// 为所有的执行图节点增加一个参数设定的策略，然后策略也提供一些参数。本质上调整的是策略的参数。
// 所有的图节点都必须的调参策略，即便没有必要加什么调参策略，也加一个缺省的直接参数赋值策略，用来保证逻辑上的统一

enum exe_node_param_set_strategy
{
    COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY,
    COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY,
    COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY,
    COMPRESSED_TBLOCK_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY,
    COMPRESSED_WARP_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY,
    COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY,
    COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY,
    COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY,
    COMPRESSED_THREAD_LEVEL_NNZ_DIV_DIRECT_PARAM_STRATEGY,
    DENSE_ROW_COARSE_SORT_FIXED_PARAM_STRATEGY,
    DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY,
    COMPRESS_NONE_PARAM_STRATEGY,
    DENSE_ROW_DIV_ACC_TO_EXPONENTIAL_INCREASE_ROW_NNZ_PARAM_STRATEGY,
};

// 压缩视图下的rowpadding的参数策略
typedef struct compressed_row_padding_direct_param_strategy
{
    long multiply;
    long padding_row_length;
} compressed_row_padding_direct_param_strategy_t;

// 压缩视图下的线程块的行切分，采用均匀切分的方式
// 参数为切分的宽度
typedef struct compressed_tblock_level_row_div_evenly_param_strategy
{
    long block_row_num;
} compressed_tblock_level_row_div_evenly_param_strategy_t;

// 按照非零元数量切分
// 在行切分的基础上，让每个行条带的非零元数量不能低于某个值（除了尾部分边角料）
typedef struct compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy
{
    long nnz_low_bound;
} compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t;

// BLB的列分块的分块，每个列块的长度是固定的
typedef struct compressed_tblock_level_col_div_fixed_param_strategy
{
    long col_block_nnz_num;
} compressed_tblock_level_col_div_fixed_param_strategy_t;

// WLB的行分块，在BLB子块的基础上进一步分块，每个BLB子块内的WLB块的行数量是相同的
// 其中，最后一个块的行数量可能不到一个行条带的大小
typedef struct compressed_warp_level_row_div_evenly_param_strategy
{
    long warp_row_num_of_each_BLB;
} compressed_warp_level_row_div_evenly_param_strategy_t;

// 执行WLB的列分块，对每一行执行一个列分块
// 在之前WLB会先对每个BLB执行一个行分块，而因为空块会被完全忽视，也就是空行会被完全忽视
typedef struct compressed_warp_level_col_div_fixed_param_strategy
{
    long col_block_nnz_num;
} compressed_warp_level_col_div_fixed_param_strategy_t;

// 执行TLB的行分块，没有要执行内容
typedef struct compressed_thread_level_row_div_none_param_strategy
{

} compressed_thread_level_row_div_none_param_strategy_t;

// 执行TLB全局固定大小的纵分块，参数是全局的纵分块大小
typedef struct compressed_thread_level_col_div_fixed_param_strategy
{
    long col_block_nnz_num;
} compressed_thread_level_col_div_fixed_param_strategy_t;

// 执行TLB级别的nnz分块，直接赋值
typedef struct compressed_thread_level_nnz_div_direct_param_strategy
{
    long block_nnz_num;
} compressed_thread_level_nnz_div_direct_param_strategy_t;

// 按照固定的行非零元范围进行行排序，行非零元的步长越大说明排序越快，但是排的越不严谨
typedef struct dense_row_coarse_sort_fixed_param_strategy
{
    long row_nnz_low_bound_step_size;
} dense_row_coarse_sort_fixed_param_strategy_t;

// 用直接赋值的方式给begin_memory_cache_input_file赋值
typedef struct dense_begin_memory_cache_input_file_direct_param_strategy
{
    vector<unsigned long> row_index_cache;
    vector<unsigned long> col_index_cache;

    vector<double> double_val_cache;
    vector<float> float_val_cache;

    // 最大行号和最大列号
    unsigned long col_index_max;
    unsigned long row_index_max;

    data_type val_data_type;
} dense_begin_memory_cache_input_file_direct_param_strategy_t;

// 直接从文件中读出对应的参数
dense_begin_memory_cache_input_file_direct_param_strategy_t get_begin_memory_cache_input_file_direct_param_strategy_from_coo_file(string file_name, data_type type);

// 压缩视图的策略，压缩操作没有任何参数，策略本身是空的
typedef struct compress_none_param_strategy
{
    
} compress_none_param_strategy_t;

// 执行压缩视图的行切分，按照指数增加的行非零元数量
// 三个参数，行非零元数量边界的最小值，以及行非零元数量的边界的增加倍数
// 最后一个参数用来规定一个上界，
// 还有一个参数，是要进一步分块的子块编号
typedef struct dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy
{
    long lowest_nnz_bound_of_row;
    long highest_nnz_bound_of_row;
    long expansion_rate;
    long sub_dense_block_id;
} dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t;

// 一个参数设定节点，由参数节点的类型和参数设定的参数策略
typedef struct param_strategy_node
{
    exe_node_param_set_strategy strategy_type;
    exe_node_type node_type;
    void *param_strategy = NULL;
    void *param = NULL;
} param_strategy_node_t;

// 压缩和稠密视图的参数策略
typedef struct param_strategy_of_sub_graph
{
    vector<param_strategy_node_t> param_strategy_vec;
} param_strategy_of_sub_graph_t;


param_strategy_of_sub_graph_t val_copy_from_old_param_strategy_of_sub_graph(param_strategy_of_sub_graph_t old_param_strategy_of_compressed_sub_graph);

// 执行某个节点的参数设定策略，当参数依赖于进一步分块的结果时，需要提前执行一些缺省的操作

// 执行稠密视图的策略执行，不需要matrix指针一定存在，并且不需要子块的编号
void execute_param_strategy_node_of_dense_matrix(param_strategy_node_t* node, sparse_struct_t* matrix);


// 执行参数设定策略
void execute_param_strategy_node_of_sub_compressed_matrix(param_strategy_node_t* node, sparse_struct_t* matrix, unsigned long sub_matrix_id);

// 执行一个参数策略，传入的形参有矩阵，执行图节点，策略节点
void execute_compressed_row_padding_direct_param_strategy(compressed_row_padding_direct_param_strategy_t *param_strategy, exe_compress_row_padding_param_t *param, sparse_struct_t *matrix, unsigned long sub_matrix_id);

// BLB级别的分块，这个级别的分块会忽视没有非零元的BLB，所以最后的非零元数量
void execute_compressed_tblock_level_row_div_evenly_param_strategy(compressed_tblock_level_row_div_evenly_param_strategy_t *param_strategy, exe_compress_tblock_level_row_div_param_t *param, sparse_struct_t *matrix, unsigned long sub_matrix_id);
void execute_compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy(compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t *param_strategy, exe_compress_tblock_level_row_div_param_t *param, sparse_struct_t *matrix, unsigned long sub_matrix_id);
void execute_compressed_tblock_level_col_div_fixed_param_strategy(compressed_tblock_level_col_div_fixed_param_strategy_t *param_strategy, exe_compress_tblock_level_col_div_param_t *param, sparse_struct_t *matrix, unsigned long sub_matrix_id);

// WLB级别的分块，行分块在BLB分块的基础上进一步分块，主要的执行取决于每个BLB的行数量
// WLB级别的列分块，先对每一行都执行一次WLB的行分块，然后再在这个结果的基础上执行一个列分块
void execute_compressed_warp_level_row_div_evenly_param_strategy(compressed_warp_level_row_div_evenly_param_strategy_t *param_strategy, exe_compress_warp_level_row_div_param_t *param, sparse_struct_t *matrix, unsigned long sub_matrix_id);
void execute_compressed_warp_level_col_div_fixed_param_strategy(compressed_warp_level_col_div_fixed_param_strategy_t *param_strategy, exe_compress_warp_level_col_div_param_t *param, sparse_struct_t *matrix, unsigned long sub_matrix_id);

// TLB级别的分块，总体上策略比较简单
void execute_compressed_thread_level_row_div_none_param_strategy(compressed_thread_level_row_div_none_param_strategy_t *param_strategy, exe_compress_thread_level_row_div_param_t *param, sparse_struct_t *matrix, unsigned long sub_matrix_id);
void execute_compressed_thread_level_col_div_fixed_param_strategy(compressed_thread_level_col_div_fixed_param_strategy_t *param_strategy, exe_compress_thread_level_col_div_param_t *param, sparse_struct_t *matrix, unsigned long sub_matrix_id);
void execute_compressed_thread_level_nnz_div_direct_param_strategy(compressed_thread_level_nnz_div_direct_param_strategy_t *param_strategy, exe_compress_thread_level_nnz_div_param_t *param, sparse_struct_t *matrix, unsigned long sub_matrix_id);

void execute_dense_row_coarse_sort_fixed_param_strategy(dense_row_coarse_sort_fixed_param_strategy_t* param_strategy, exe_dense_row_coarse_sort_param_t* param, sparse_struct_t *matrix);
void execute_dense_begin_memory_cache_input_file_direct_param_strategy(dense_begin_memory_cache_input_file_direct_param_strategy_t* param_strategy, exe_begin_memory_cache_input_file_param_t* param, sparse_struct_t* matrix);
void execute_compress_none_param_strategy(compress_none_param_strategy_t* param_strategy, exe_compress_param_t* param, sparse_struct_t* matrix);
void execute_dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy(dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t* param_strategy, exe_dense_row_div_param_t* param, sparse_struct_t* matrix);

// 析构参数策略节点的策略
void del_strategy_of_param_strategy_node(param_strategy_node_t* node);

// 创造各种类型的策略节点
param_strategy_node_t init_compressed_row_padding_direct_param_strategy(compressed_row_padding_direct_param_strategy_t param_strategy, exe_compress_row_padding_param_t* param);
param_strategy_node_t init_compressed_tblock_level_row_div_evenly_param_strategy(compressed_tblock_level_row_div_evenly_param_strategy_t param_strategy, exe_compress_tblock_level_row_div_param_t* param);
param_strategy_node_t init_compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy(compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t param_strategy, exe_compress_tblock_level_row_div_param_t* param);
param_strategy_node_t init_compressed_tblock_level_col_div_fixed_param_strategy(compressed_tblock_level_col_div_fixed_param_strategy_t param_strategy, exe_compress_tblock_level_col_div_param_t* param);

param_strategy_node_t init_compressed_warp_level_row_div_evenly_param_strategy(compressed_warp_level_row_div_evenly_param_strategy_t param_strategy, exe_compress_warp_level_row_div_param_t* param);
param_strategy_node_t init_compressed_warp_level_col_div_fixed_param_strategy(compressed_warp_level_col_div_fixed_param_strategy_t param_strategy, exe_compress_warp_level_col_div_param_t* param);

param_strategy_node_t init_compressed_thread_level_row_div_none_param_strategy(compressed_thread_level_row_div_none_param_strategy_t param_strategy, exe_compress_thread_level_row_div_param_t* param);
param_strategy_node_t init_compressed_thread_level_col_div_fixed_param_strategy(compressed_thread_level_col_div_fixed_param_strategy_t param_strategy, exe_compress_thread_level_col_div_param_t* param);
param_strategy_node_t init_compressed_thread_level_nnz_div_direct_param_strategy(compressed_thread_level_nnz_div_direct_param_strategy_t param_strategy, exe_compress_thread_level_nnz_div_param_t* param);

param_strategy_node_t init_dense_row_coarse_sort_fixed_param_strategy(dense_row_coarse_sort_fixed_param_strategy_t param_strategy, exe_dense_row_coarse_sort_param_t* param);
param_strategy_node_t init_dense_begin_memory_cache_input_file_direct_param_strategy(dense_begin_memory_cache_input_file_direct_param_strategy_t param_strategy, exe_begin_memory_cache_input_file_param_t* param);
param_strategy_node_t init_compress_none_param_strategy(compress_none_param_strategy_t param_strategy, exe_compress_param_t* param);
param_strategy_node_t init_dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy(dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t param_strategy, exe_dense_row_div_param_t* param);

// 对子图的所有参数执行析构，同时给析构之后的指针赋值为NULL
void del_strategy_of_param_strategy_node_in_sub_matrix(param_strategy_of_sub_graph_t* param_strategy_of_sub_matrix);

// 打印所有的参数
string convert_strategy_param_to_string(void *param_strategy, exe_node_param_set_strategy type);

// 将策略类型直接转化为字符串
string convert_param_set_strategy_to_string(exe_node_param_set_strategy type);

// 将策略参数的节点直接打印出来
string convert_stategy_node_to_string(param_strategy_node_t node);

// 将整个子图转化为字符串
string convert_all_stategy_node_of_sub_matrix_to_string(param_strategy_of_sub_graph_t strategy_skeleon_of_sub_matrix);

#endif