#ifndef DEFAULT_AUTO_TUNER_H
#define DEFAULT_AUTO_TUNER_H

// 默认的自动调优器
#include <iostream>
#include "struct.hpp"
#include "op_manager.hpp"
#include <assert.h>
#include "code_builder.hpp"
#include <sys/time.h>
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
#include "memory_garbage_manager.hpp"
#include "exe_graph.hpp"
#include "graph_enumerate.hpp"
#include "param_enumerater.hpp"
#include "executor.hpp"
#include "parameter_set_strategy.hpp"
#include "matrix_info.hpp"
#include "search_strategy.hpp"
#include <memory>
#include "machine_learning_data_set_collector.hpp"

using namespace std;


// 子块的最优的组合
typedef struct compressed_sub_block_exe_graph_and_template
{
    // 模板
    template_node_t temp_node;
    // 子图的优化路径
    exe_compressed_sub_graph_t sub_graph;
    // 子图的参数策略
    param_strategy_of_sub_graph_t sub_graph_param_strategy;
} compressed_sub_block_exe_graph_and_template_t;

// 稠密子图的最优组合，包含了调参策略
typedef struct dense_view_matrix_exe_graph_and_param_strategy
{
    // 稠密图的优化路径
    exe_dense_sub_graph_t dense_sub_graph;
    // 稠密图的策略路径
    param_strategy_of_sub_graph_t dense_sub_graph_param_strategy;
} dense_view_matrix_exe_graph_and_param_strategy_t;

// 全局分块的组合
typedef struct dense_view_matrix_and_compressed_sub_block_exe_graph_and_template
{
    // 所有子块的优化路径和模板
    vector<compressed_sub_block_exe_graph_and_template_t> compressed_sub_block_exe_graph_and_template_vec;
    // 稠密视图路径
    dense_view_matrix_exe_graph_and_param_strategy dense_sub_graph_and_param_strategy;
} dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t;

// 用一个数据结构来记录不同的子图的参数和最终结果之间的关系，每一条数据包含了稠密图的节点类型的数组，参数策略类型数组，稠密策略的id
// 子图的节点类型的数组，子图的id，




typedef struct default_auto_tuner
{
    
    
} default_auto_tuner_t;

// 找出一个子块的默认最优


// 从一个模板集中获取最优的模板和对应参数
template_node_t find_best_template_node_of_specific_sub_matrix_from_template_set(sparse_struct_t* matrix, int sub_matrix_id, set<template_type> template_set, float& best_time, float& best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

// 根据感觉当前子块的现状，执行默认的分块操作，补完各个
void execute_default_div_to_complete_each_level_blocking(sparse_struct_t* matrix, int sub_matrix_id);

// 找出一个子块的一个模板的最佳参数
template_node_t find_best_param_of_specific_template_node_of_sub_matrix(sparse_struct_t* matrix, int sub_matrix_id, template_type type, float& best_time, float& best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

// 针对每个模板执行参数查找工作
template_node_t find_best_param_of_direct_atom_template_warp_block_compress(code_builder_t* builder, int sub_matrix_id, float& best_time, float& best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

template_node_t find_best_param_of_direct_atom_template_warp_compress(code_builder_t* builder, int sub_matrix_id, float& best_time, float& best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

template_node_t find_best_param_of_direct_atom_template(code_builder_t* builder, int sub_matrix_id, float& best_time, float& best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

template_node_t find_best_param_of_direct_atom_total_warp_reduce_template(code_builder_t* builder, int sub_matrix_id, float& best_time, float& best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

template_node_t find_best_param_of_shared_memory_long_row_template(code_builder_t* builder, int sub_matrix_id, float& best_time, float& best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

template_node_t find_best_param_of_shared_memory_template_warp_compress(code_builder_t* builder, int sub_matrix_id, float& best_time, float& best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

template_node_t find_best_param_of_shared_memory_template(code_builder_t* builder, int sub_matrix_id, float& best_time, float& best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

template_node_t find_best_param_of_shared_memory_total_warp_reduce_template(code_builder_t* builder, int sub_matrix_id, float& best_time, float& best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

template_node_t find_best_param_of_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce(code_builder_t* builder, int sub_matrix_id, float& best_time, float& best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

template_node_t find_best_param_of_unaligned_warp_reduce_same_TLB_size_template(code_builder_t* builder, int sub_matrix_id, float& best_time, float& best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

// 根据参数策略和子矩阵优化节点对某一个子矩阵执行优化操作，执行完之后修改matrix对应内容
void execute_sub_matrix_exe_graph_with_param_strategy(sparse_struct_t* matrix, unsigned long sub_matrix_id, exe_compressed_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy);

// 根据参数策略和稠密视图的优化节点，输出一个经过压缩的稀疏矩阵
sparse_struct_t* execute_dense_matrix_exe_graph_with_param_strategy(exe_dense_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy);

// 向稠密视图中增加一个节点以及其对应的策略
void add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(exe_dense_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy, exe_node_type node_type, exe_node_param_set_strategy strategy_type, void* strategy_param_ptr);

// 向子图中增加一个节点，传入的是参数调节策略。不需要的是节点的参数，因为那由调参策略决定
void add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(sparse_struct_t* matrix, unsigned long sub_matrix_id, exe_compressed_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy, exe_node_type node_type, exe_node_param_set_strategy strategy_type, void* strategy_param_ptr);
void add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(exe_compressed_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy, exe_node_type node_type, exe_node_param_set_strategy strategy_type, void* strategy_param_ptr);

// 给出稠密视图的优化路径，找出某一个子矩阵的最优优化路径
// compressed_sub_block_exe_graph_and_template_t find_best_sub_matrix_optimization_path(exe_dense_sub_graph_t *dense_graph, unsigned long sub_matrix_id);

// 析构子图+模板节点的所有参数
void del_param_of_compressed_sub_block_exe_graph_and_template(compressed_sub_block_exe_graph_and_template_t* sub_graph_and_template);

// 重置优化骨架和策略子骨架，主要体现在重置优化骨架，然后重新绑定优化骨架和策略骨架
void reset_exe_node_param_and_param_strategy_of_sub_graph(exe_compressed_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy);
void reset_exe_node_param_and_param_strategy_of_sub_graph(exe_dense_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy);

// 重新绑定策略骨架和优化骨架的函数，有时候优化骨架的参数暂时不析构，只需要重新初始化一些新的参数，并且重新绑定即可
void malloc_exe_node_param_and_param_strategy_of_sub_graph(exe_compressed_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy);

// 绑定优化骨架和策略骨架（绑定压缩子图）
void bind_exe_node_param_param_strategy_of_sub_graph(exe_compressed_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy);
// 绑定稠密子图
void bind_exe_node_param_param_strategy_of_sub_graph(exe_dense_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy);

// 安全地析构整个图的参数，需要考虑有些参数是空指针的情况
void del_param_of_total_exe_graph_and_strategy_graph_safely(dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t* total_graph);

// 设计一个执行的白名单，将根据已有的剪枝策略产生特定的几种优化路径
// 判断在特定的参数组合下判断一个路径是不是要被尝试，还有一个就是对于对应子块的实际执行
// 每个子图的策略有三个版本、传入稠密优化子图，传入稠密优化加策略子图，传入一个刚刚做完压缩的matrix结构
// 前两个版需要在一开始通过图得到真正的matrix，最后一个版本需要对matrix执行一个值拷贝，然后对值拷贝的matrix进行进一步优化
// 最后一个版本的意义是节省同一个稠密优化的不同子图优化的运行时间，让稠密视图的优化可以在不同的子图优化中复用

// 提供三种接口，核心的接口是最后一个，需要matrix在每次尝试的时候执行一个值拷贝

// row_padding => evenly_BLB_row => one_TLB_row => direct_atom_template_warp_compress
// 只要不修改参数的指针的指向，就可以不用传指针进去
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy1(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy1(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy1(sparse_struct_t* matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

// row_padding => evenly_BLB_row => TLB_col => shared_memory_template_warp_compress
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy2(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy2(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy2(sparse_struct_t* matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

// nnz_BLB_row => TLB_col => shared_memory_template_warp_compress
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy3(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy3(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy3(sparse_struct_t* matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

// WLB_col => direct_atom_total_warp_reduce_template
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy4(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy4(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy4(sparse_struct_t* matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

// row_padding32 => TLB_row => direct_atom_template_warp_block_compress
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy5(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy5(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy5(sparse_struct_t* matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

// row_padding => evenly_BLB_row => WLB_col => shared_memory_total_warp_reduce
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy6(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy6(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy6(sparse_struct_t* matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

// nnz_BLB_row => WLB_col => shared_memory_total_warp_reduce
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy7(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy7(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy7(sparse_struct_t* matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

// BLB_col => long_row_shared_memory
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy8(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy8(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy8(sparse_struct_t* matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

// TLB_nnz => CSR5_like
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy9(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy9(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy9(sparse_struct_t* matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

// 通过一个子图，找出所有策略中最优的子图
// 提供三种接口，一种实现
compressed_sub_block_exe_graph_and_template_t find_best_path_of_compressed_sub_matrix(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_compressed_sub_matrix(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);
compressed_sub_block_exe_graph_and_template_t find_best_path_of_compressed_sub_matrix(sparse_struct_t* matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);


// 将整个图的所有整个路径打印出来
string convert_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_to_string(dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t graph);
// 将整个图的整个路径打印出来，自动跳过参数是空的路径
string convert_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_to_string_safety(dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t graph);

// 稠密子图一共四个策略，一个是仅仅进行压缩，一个是排序、行分块压缩，最后是行分块和压缩

// 第一个策略，读入之后再执行一个压缩，输入的是在内存中的矩阵COO信息，用执行图节点的方式表现出来，最后一个dataset收集的指针来代表是不是要收集机器学习的训练集。
dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t find_best_graph_of_white_list_strategy1(exe_begin_memory_cache_input_file_param_t input_matrix_node, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

// 对于行分块压缩来说，如果切完之后行条带的数量多于8个，那就不需要进一步测试
// 直接在行分块之后进行压缩
dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t find_best_graph_of_white_list_strategy2(exe_begin_memory_cache_input_file_param_t input_matrix_node, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t find_best_graph_of_white_list_strategy3(exe_begin_memory_cache_input_file_param_t input_matrix_node, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

// 将所有的稠密视图策略整合到一起
dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t find_best_graph_of_white_list_strategy(exe_begin_memory_cache_input_file_param_t input_matrix_node, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr = NULL, shared_ptr<machine_learning_data_set_collector> data_set_collector = NULL);

// 用完整的图得到一个matrix
sparse_struct_t* get_matrix_from_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template(dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t* total_graph);

// 执行一个完整的视图，并且得出最后的结果
void execute_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template(dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t* total_graph, float& gflops, float& time, int repeat_time);

// 临时记录当前性能最高的图结构和对应的性能
void write_graph_structure_and_performance_to_file(float gflops, float time, dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t total_graph, string file_name);


#endif