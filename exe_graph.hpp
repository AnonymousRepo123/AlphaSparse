#ifndef EXE_GRAPH_H
#define EXE_GRAPH_H

#include <iostream>
#include "struct.hpp"
#include "op_manager.hpp"
#include <assert.h>
#include "code_builder.hpp"
#include <sys/time.h>
#include <string.h>
#include "arr_optimization.hpp"
#include "direct_atom_op.hpp"
#include "direct_atom_op_warp_compress.hpp"
#include "direct_atom_op_warp_block_compress.hpp"
#include "shared_memory_op.hpp"
#include "shared_memory_op_warp_compress.hpp"
#include "shared_memory_long_row_op.hpp"
#include "shared_memory_total_warp_reduce_op.hpp"
#include "direct_atom_total_warp_reduce_op.hpp"
#include "unaligned_warp_reduce_same_TLB_size_op.hpp"
#include "unaligned_warp_reduce_same_TLB_size_op_with_warp_reduce.hpp"
#include "dataset_builder.hpp"
// #include "matrix_info.hpp"
#include <set>
#include <vector>

#define GRAPH_END -1

// 子图的类型，有两类，一类是稠密子图，一个是压缩子图
enum exe_sub_graph_type
{
    EXE_DENSE_SUB_GRAPH,
    EXE_COMPRESSED_SUB_GRAPH,
};

enum exe_node_type
{
    BEGIN_INPUT_FILE,
    BEGIN_ARTIFICIAL_INPUT,
    BEGIN_MEMORY_CACHE_INPUT_FILE,
    DENSE_ROW_COARSE_SORT,
    DENSE_FINE_SORT,
    DENSE_TOTAL_ROW_LEVEL_PADDING,
    DENSE_BLOCK_SORT,
    DENSE_ROW_DIV,
    DENSE_FIXED_COL_DIV,
    COMPRESS,
    COMPRESSED_ROW_PADDING,
    COMPRESSED_BLOCK_SORT,
    COMPRESSED_THREAD_LEVEL_ROW_DIV,
    COMPRESSED_THREAD_LEVEL_COL_DIV,
    COMPRESSED_WARP_LEVEL_ROW_DIV,
    COMPRESSED_WARP_LEVEL_COL_DIV,
    COMPRESSED_TBLOCK_LEVEL_ROW_DIV,
    COMPRESSED_TBLOCK_LEVEL_COL_DIV,
    COMPRESSED_THREAD_LEVEL_NNZ_DIV,
};

// 对某一个压缩子图执行分块操作，相关的操作有主要是要被padding为多少行数量的整数倍，padding的每行非零元数量
typedef struct exe_compress_row_padding_param
{
    long multiply;
    long padding_row_length;
} exe_compress_row_padding_param_t;

// 不对齐的线程粒度分块的节点，参数是TLB块的大小
typedef struct exe_compress_thread_level_nnz_div_param
{
    long TLB_nnz_num;
} exe_compress_thread_level_nnz_div_param_t;

// 线程级别的列分块，这里的分块策略和之前的是不同的，TLB针对每个父块都只有一个统一的TLB宽度
// 所以节点参数的数组大小为WLB的数量相同。
typedef struct exe_compress_thread_level_col_div_param
{
    vector<unsigned long> col_num_of_TLB_in_each_parent_block;
} exe_compress_thread_level_col_div_param_t;

// 线程级别的行分块
typedef struct exe_compress_thread_level_row_div_param
{
} exe_compress_thread_level_row_div_param_t;

// 压缩之后的warp级别的分块，用一个数组存储每个tblock内的进一步分块的行分块的大小。
// 在一个BLB的所有WLB分块的行数量加起来和BLB的行号是一致的
typedef struct exe_compress_warp_level_row_div_param
{
    vector<vector<unsigned int>> row_num_of_each_WLB_in_BLB;
} exe_compress_warp_level_row_div_param_t;

// 增加一个列分块的操作，在列分块之前都会有一个warp级别的行分块，在block分块的基础上在BLB将每一行切成一块
// 节点的参数是一个二维数组，二维数组外层的大小为BLB经过进一步行分块之后的大小，二维数组内层的大小是进一步执行列分块的大小
// 列分块的执行策略和其他列分块的策略是一致的，即不要求所有列块的大小相加等于父块，在块边界和列块边界都会执行分块
typedef struct exe_compress_warp_level_col_div_param
{
    vector<vector<unsigned int>> col_num_of_WLB_in_each_parent_row_block_or_BLB;
} exe_compress_warp_level_col_div_param_t;

// 压缩之后的行分块和列分块
typedef struct exe_compress_tblock_level_row_div_param
{
    // 每一个tblock的粒度的块的行数量
    vector<unsigned int> row_num_of_each_BLB;
} exe_compress_tblock_level_row_div_param_t;

// 压缩完之后的列分块，输入的参数是每一行列分块的粒度
typedef struct exe_compress_tblock_level_col_div_param
{
    // 针对每一行的纵分块大小，每一行的每一个纵分块的长度是可以不一样的。所以这里采用一个二维数组，每一行都对应一个一维数组，用来对应这一行中
    // 不同列分块的长度。注意列分块的长度不用和行边界对齐，行边界和列块边界都在切分非零元。
    // 外层数组的大小为非空行的数量，也就是空行的分块不被包括在其中
    vector<vector<unsigned int>> col_block_nnz_num_of_each_BLB;
} exe_compress_tblock_level_col_div_param_t;

typedef struct exe_dense_row_coarse_sort_param
{
    // 分块的粒度，也就是分块的每一个桶包含的行非零元数量，包含每个桶行非零元数量的下界，
    vector<unsigned long> bin_row_nnz_low_bound;
} exe_dense_row_coarse_sort_param_t;

// 执行行分块的节点的参数
typedef struct exe_dense_row_div_param
{
    // 分块点
    vector<unsigned long> row_div_position;
    // 针对分块的子块，一开始没有分块的时候子块默认为0
    unsigned long dense_sub_block_id;
} exe_dense_row_div_param_t;

// 空的压缩节点参数
typedef struct exe_compress_param
{
    // 暂时是空的
} exe_compress_param_t;

// padding的执行节点
// void total_row_level_padding_add(operator_manager_t *op_manager, unsigned long add_size, unsigned padding_col_num = 1, global_padding_position padding_type = END_PADDING, unsigned long input_col_index = 0)
typedef struct exe_row_level_padding_node_param
{
    // 要padding的位置
    global_padding_position position;

    // 要padding的行数量
    long added_row_num;

    // padding的非零元数量
    long nnz_of_padding_row;

    // 暂时不需要的padding的列号
    long input_col_index = 0;
} exe_row_level_padding_node_param_t;

typedef struct exe_dense_fixed_col_div_param
{
    // 定长分块长度
    long fixed_col_block_size;
    // 针对分块的子块，一开始没有分块的时候子块默认为0
    long dense_sub_block_id;
} exe_dense_fixed_col_div_param_t;

// 第一个节点，从内存中初始化一个矩阵
typedef struct exe_begin_memory_cache_input_file_param
{
    vector<unsigned long> row_index_cache;
    vector<unsigned long> col_index_cache;

    vector<double> double_val_cache;
    vector<float> float_val_cache;

    // 最大行号和最大列号
    unsigned long col_index_max;
    unsigned long row_index_max;

    data_type val_data_type;
} exe_begin_memory_cache_input_file_param_t;

// 第一个节点，读入数据
typedef struct exe_begin_input_file_param
{
    // 文件名称，以及数据的类型
    string input_file_name;

    // 值的数据类型
    data_type val_data_type;

} exe_begin_input_file_param_t;

// 第一个节点，创造一个人工矩阵
typedef struct exe_begin_artificial_input_param
{
    // 最大行索引
    long max_col_index;

    // 最大列索引
    long max_row_index;

    // 行非零元数量
    long nnz_of_each_row;

    // 值的数据类型
    data_type val_data_type;
} exe_begin_artificial_input_param_t;

// 执行节点
typedef struct exe_node
{
    // 节点类型，以及这个节点类型所对应的参数
    exe_node_type type;

    // 存储参数
    void *param = NULL;
} exe_node_t;

// 整个graph一共分为三个阶段，分别使用三个不同的结构
// 密集子图。一开始是padding，然后是排序类的操作，最后是分块类操作。
// 然后最后是compress操作。
typedef struct exe_dense_sub_graph
{
    // 存储压缩之前的所有操作
    vector<exe_node_t> exe_node_vec;
    // 用一个变量存储已经加入的前序节点的种类
    set<exe_node_type> preorder_node_set;
} exe_dense_sub_graph_t;

// 一个压缩子矩阵的线性子图，在compress之后实际产生的
typedef struct exe_compressed_sub_graph
{
    vector<exe_node_t> exe_node_vec;
    // 用一个变量存储压缩包子图中所有的前序节点
    set<exe_node_type> preorder_node_set;
} exe_compressed_sub_graph_t;

// 所有子矩阵的压缩子图
typedef struct exe_total_compressed_graph
{
    vector<exe_compressed_sub_graph_t> compressed_sub_graph_vec;
} exe_total_compressed_graph_t;

// 执行图和逻辑图是不一样的，执行图是一个一分多的形状，逻辑图是一个树状图
typedef struct exe_graph
{
    // 用来执行操作的结构
    operator_manager_t *op_manager = NULL;
    // 用来生成模板的东西
    code_builder_t *builder = NULL;

    exe_dense_sub_graph_t dense_sub_graph;
    exe_total_compressed_graph total_compressed_sub_graph;

} exe_graph_t;

// 传入一个子矩阵，判断这个矩阵适合哪些模板
set<template_type> supported_template_of_sub_matrix(sparse_struct_t *matrix, unsigned long dense_block_id);

// 根据子图中产生
sparse_struct_t *get_sparse_matrix_from_exe_sub_graph(exe_graph_t *graph);

// 传入一个执行图，判断这个图适合哪些模板，
set<template_type> supported_template_of_matrix(exe_graph_t *graph);

// 从一个coo文件中直接得出exe_begin_memory_cache_input_file_node的参数
exe_begin_memory_cache_input_file_param_t get_exe_begin_memory_cache_input_file_param_from_coo_file(string file_name, data_type type);

// 执行一个节点的依赖检查，首先是输入节点的依赖检查，输入节点的参数和节点插入的位置，用一个bool来标记节点的参数是不是被初始化过了
// 一个节点的参数搜寻策略存在依赖关系，即一个节点参数的制定依赖于其他节点的参数，要从依赖的顶端开始调参。
bool dependence_of_exe_begin_artificial_input_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_begin_artificial_input_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_begin_memory_cache_input_file_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_begin_memory_cache_input_file_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_compress_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_begin_input_file_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_begin_input_file_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_dense_row_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_dense_row_div_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_dense_fixed_col_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_dense_fixed_col_div_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_dense_row_coarse_sort_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_dense_row_coarse_sort_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_compress_BLB_row_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_tblock_level_row_div_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_compress_BLB_col_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_tblock_level_col_div_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_compress_WLB_row_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_warp_level_row_div_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_compress_WLB_col_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_warp_level_col_div_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_compress_TLB_row_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_thread_level_row_div_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_compress_TLB_col_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_thread_level_col_div_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_compress_thread_level_nnz_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_thread_level_nnz_div_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_compress_row_padding_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_row_padding_param_t param, int sub_graph, int input_index);

// 所有节点的参数全部清零，使得在第二次迭代的时候不受影响
// 先析构掉所有节点的参数，在重新申请所有节点的参数
void reset_param_of_all_sub_compressed_graph(exe_compressed_sub_graph_t* sub_compressed_graph);
// 重新为每个节点申请新的参数，旧的参数不析构
void malloc_param_of_all_sub_compressed_graph(exe_compressed_sub_graph_t* sub_compressed_graph);

// 上面两个函数的稠密子矩阵版
void reset_param_of_all_sub_dense_graph(exe_dense_sub_graph* sub_dense_graph);
// 重新为每个节点申请新的参数，旧的参数不析构
void malloc_param_of_all_sub_dense_graph(exe_dense_sub_graph* sub_dense_graph);

// 将节点添加到，对应的位置，在执行这一步之前必须先做检查
void add_exe_begin_artificial_input_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_begin_artificial_input_param_t param, int sub_graph, int input_index);
void add_exe_begin_memory_cache_input_file_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_begin_memory_cache_input_file_param_t param, int sub_graph, int input_index);
void add_exe_compress_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_param_t param, int sub_graph, int input_index);
void add_exe_begin_input_file_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_begin_input_file_param_t param, int sub_graph, int input_index);
void add_exe_dense_row_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_dense_row_div_param_t param, int sub_graph, int input_index);
void add_exe_dense_fixed_col_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_dense_fixed_col_div_param_t param, int sub_graph, int input_index);
void add_exe_dense_row_coarse_sort_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_dense_row_coarse_sort_param_t param, int sub_graph, int input_index);
void add_exe_compress_BLB_row_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_tblock_level_row_div_param_t param, int sub_graph, int input_index);
void add_exe_compress_BLB_col_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_tblock_level_col_div_param_t param, int sub_graph, int input_index);
void add_exe_compress_WLB_row_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_warp_level_row_div_param_t param, int sub_graph, int input_index);
void add_exe_compress_WLB_col_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_warp_level_col_div_param_t param, int sub_graph, int input_index);
void add_exe_compress_TLB_row_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_thread_level_row_div_param_t param, int sub_graph, int input_index);
void add_exe_compress_TLB_col_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_thread_level_col_div_param_t param, int sub_graph, int input_index);
void add_exe_compress_thread_level_nnz_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_thread_level_nnz_div_param_t param, int sub_graph, int input_index);
void add_exe_compress_row_padding_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_row_padding_param_t param, int sub_graph, int input_index);

// 给节点增加执行参数，
void add_param_to_exe_begin_artificial_input_node(exe_node_t *node, exe_begin_artificial_input_param_t param);
void add_param_to_exe_begin_memory_cache_input_file_node(exe_node_t *node, exe_begin_memory_cache_input_file_param param);
void add_param_to_exe_compress_node(exe_node_t *node, exe_compress_param_t param);
void add_param_to_exe_begin_input_file_node(exe_node_t *node, exe_begin_input_file_param_t param);
void add_param_to_exe_dense_row_div_node(exe_node_t *node, exe_dense_row_div_param_t param);
void add_param_to_exe_dense_fixed_col_div_node(exe_node_t *node, exe_dense_fixed_col_div_param_t param);
void add_param_to_exe_dense_row_coarse_sort_node(exe_node_t *node, exe_dense_row_coarse_sort_param_t param);
void add_param_to_exe_compress_BLB_row_div_node(exe_node_t *node, exe_compress_tblock_level_row_div_param_t param);
void add_param_to_exe_compress_BLB_col_div_node(exe_node_t *node, exe_compress_tblock_level_col_div_param_t param);
void add_param_to_exe_compress_WLB_row_div_node(exe_node_t *node, exe_compress_warp_level_row_div_param_t param);
void add_param_to_exe_compress_WLB_col_div_node(exe_node_t *node, exe_compress_warp_level_col_div_param_t param);
void add_param_to_exe_compress_TLB_row_div_node(exe_node_t *node, exe_compress_thread_level_row_div_param_t param);
void add_param_to_exe_compress_TLB_col_div_node(exe_node_t *node, exe_compress_thread_level_col_div_param_t param);
void add_param_to_exe_compress_thread_level_nnz_div_node(exe_node_t *node, exe_compress_thread_level_nnz_div_param_t param);
void add_param_to_exe_compress_row_padding_node(exe_node_t *node, exe_compress_row_padding_param_t param);

// 单个节点的执行
void execute_exe_begin_artificial_input_node(exe_graph_t *graph, exe_node_t node);
void execute_exe_begin_memory_cache_input_file_node(exe_graph_t *graph, exe_node_t node);
void execute_exe_compress_node(exe_graph_t *graph, exe_node_t node);
void execute_exe_begin_input_file_node(exe_graph_t *graph, exe_node_t node);
void execute_exe_dense_row_div_node(exe_graph_t *graph, exe_node_t node);
void execute_exe_dense_fixed_col_div_node(exe_graph_t *graph, exe_node_t node);
void execute_exe_dense_row_coarse_sort_node(exe_graph_t *graph, exe_node_t node);

void execute_exe_compress_BLB_row_div_node(exe_graph_t *graph, exe_node_t node, unsigned long sub_matrix_id);
void execute_exe_compress_BLB_col_div_node(exe_graph_t *graph, exe_node_t node, unsigned long sub_matrix_id);
void execute_exe_compress_WLB_row_div_node(exe_graph_t *graph, exe_node_t node, unsigned long sub_matrix_id);
void execute_exe_compress_WLB_col_div_node(exe_graph_t *graph, exe_node_t node, unsigned long sub_matrix_id);
void execute_exe_compress_TLB_row_div_node(exe_graph_t *graph, exe_node_t node, unsigned long sub_matrix_id);
void execute_exe_compress_TLB_col_div_node(exe_graph_t *graph, exe_node_t node, unsigned long sub_matrix_id);
void execute_exe_compress_thread_level_nnz_div_node(exe_graph_t *graph, exe_node_t node, unsigned long sub_matrix_id);
void execute_exe_compress_row_padding_node(exe_graph_t *graph, exe_node_t node, unsigned long sub_matrix_id);

// 另外一组函数，让压缩视图的子块执行对应的节点
void execute_exe_compress_BLB_row_div_node(sparse_struct_t* matrix, exe_node_t node, unsigned long sub_matrix_id);
void execute_exe_compress_BLB_col_div_node(sparse_struct_t* matrix, exe_node_t node, unsigned long sub_matrix_id);
void execute_exe_compress_WLB_row_div_node(sparse_struct_t* matrix, exe_node_t node, unsigned long sub_matrix_id);
void execute_exe_compress_WLB_col_div_node(sparse_struct_t* matrix, exe_node_t node, unsigned long sub_matrix_id);
void execute_exe_compress_TLB_row_div_node(sparse_struct_t* matrix, exe_node_t node, unsigned long sub_matrix_id);
void execute_exe_compress_TLB_col_div_node(sparse_struct_t* matrix, exe_node_t node, unsigned long sub_matrix_id);
void execute_exe_compress_thread_level_nnz_div_node(sparse_struct_t* matrix, exe_node_t node, unsigned long sub_matrix_id);
void execute_exe_compress_row_padding_node(sparse_struct_t* matrix, exe_node_t node, unsigned long sub_matrix_id);

// 向一个子块中增加一个节点

// 执行图的密集矩阵部分
void execute_graph(exe_graph_t *graph);

// 执行密集子图的部分
void execute_graph_dense_part(exe_graph_t *graph);
// 执行压缩子图的部分
void execute_graph_compress_part(exe_graph_t *graph);

// 执行图的一些优化操作，将padding作为优化操作的一种
void optimize_graph(exe_graph_t *graph);

// 执行稠密视图的某一个节点
void execute_node_of_dense_sub_graph(exe_graph_t *graph, int exe_node_of_sub_dense_graph_id);

// 执行压缩视图的某一个节点
void execute_node_of_compressed_sub_graph(exe_graph_t *graph, int sub_graph_index, int exe_node_index_of_sub_graph);

// 对一个子矩阵执行压缩视图的一个节点
void execute_exe_node_in_compressed_sub_matrix(sparse_struct_t* matrix, int sub_matrix_index, exe_node_t node);

// 对于一个压缩子图的优化序列执行值拷贝，关键在于重新申请一系列的节点参数
exe_compressed_sub_graph_t val_copy_from_old_compressed_sub_matrix(exe_compressed_sub_graph_t old_exe_compressed_sub_graph);

// 对于一个压缩视图的优化路径的值拷贝
exe_dense_sub_graph_t val_copy_from_old_dense_sub_matrix(exe_dense_sub_graph_t old_exe_dense_sub_graph);

// 析构一个执行节点的参数
void del_param_of_exe_node(exe_node_t *node);

// 析构子块执行节点的所有参数
// void del_exe_node_param_of_compress_sub_matrix(exe_compressed_sub_graph_t compress_sub_graph);

// 析构子块执行节点的所有参数，同时给对应的参数赋值为NULL
void del_exe_node_param_of_compress_sub_matrix(exe_compressed_sub_graph_t *compress_sub_graph);

// 析构稠密视图所有节点的所有参数
void del_exe_node_param_of_dense_view_matrix(exe_dense_sub_graph_t *dense_sub_graph);

// 根据稠密视图的优化路径给出一个仅仅压缩过的matrix
sparse_struct_t* get_matrix_dense_view_graph(exe_dense_sub_graph_t* dense_graph);

// 将优化节点的类型打印出来
string convert_exe_node_type_to_string(exe_node_type node_type);

// 以下是和模板相关的东西，一些模板相关的节点和参数exe_compressed_sub_graph


// 增加模板节点
typedef struct template_node
{
    template_type type;

    void *template_param = NULL;
} template_node_t;

// 一系列模板的参数
typedef struct direct_atom_template_warp_block_compress_node_param
{
    long tblock_num;
    long thread_num_in_block;
} direct_atom_template_warp_block_compress_node_param_t;

typedef struct direct_atom_template_warp_compress_node_param
{
    long tblock_num;
    long thread_num_in_block;
} direct_atom_template_warp_compress_node_param_t;

typedef struct direct_atom_template_node_param
{
    long tblock_num;
    long thread_num_in_block;
} direct_atom_template_node_param_t;

typedef struct direct_atom_total_warp_reduce_template_node_param
{
    long tblock_num;
    long thread_num_in_block;
} direct_atom_total_warp_reduce_template_node_param_t;

typedef struct shared_memory_long_row_template_node_param
{
    long tblock_num;
    long thread_num_in_block;
} shared_memory_long_row_template_node_param_t;

// shared memory多加了一个归约的并行度。这个归约的并行度大小不能超过每一行中间结果的数量
typedef struct shared_memory_template_warp_compress_node_param
{
    long tblock_num;
    long thread_num_in_block;
    long thread_num_of_row_reduce;
} shared_memory_template_warp_compress_node_param_t;

typedef struct shared_memory_template_node_param
{
    long tblock_num;
    long thread_num_in_block;
    long thread_num_of_row_reduce;
} shared_memory_template_node_param_t;

// 最多不超过32
typedef struct shared_memory_total_warp_reduce_template_node_param
{
    long tblock_num;
    long thread_num_in_block;
    long thread_num_of_row_reduce;
} shared_memory_total_warp_reduce_template_node_param_t;

typedef struct unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_node_param
{
    long thread_num_in_block;
} unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_node_param_t;

typedef struct unaligned_warp_reduce_same_TLB_size_template_node_param
{
    long tblock_num;
    long thread_num_in_block;
} unaligned_warp_reduce_same_TLB_size_template_node_param_t;

// 这里面的参数析构
void del_param_of_template_node(template_node_t* node);
// 打印一个节点
void print_template_node(template_node_t* node);
// 将模板节点转化为字符串
string convert_template_node_to_string(template_node_t* node);

// 值拷贝相同模板节点，主要是将new出来的参数重新new一个并且做一个赋值。
template_node_t val_copy_from_old_template_node(template_node_t old_template);

// 执行对应的模板节点，根据这个节点为code_builder中的节点设定参数
void execute_template_node_and_update_code_builder(code_builder_t* builder, unsigned long sub_matrix_id, template_node_t node);

// 检查已有的输入节点，查看是不是按照行和列分别自增
bool check_begin_memory_cache_input_file(exe_begin_memory_cache_input_file_param_t input_node);

// 查看已有输入节点，查看是不是有空行
bool has_empty_line_in_begin_memory_cache_input_file(exe_begin_memory_cache_input_file_param_t input_node);

#endif