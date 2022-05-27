// 这个文件提供用户的剪枝策略包含模板策略的剪枝和节点选择的剪枝
#ifndef USER_PRUNING_STRATEGY_H
#define USER_PRUNING_STRATEGY_H

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
#include "dataset_builder.hpp"
#include <set>
#include "exe_graph.hpp"

// 用户自定义的剪枝策略，对于模板来说传入矩阵，对于操作来做传入前趋操作集
bool is_supported_by_direct_atom_template_warp_block_compress_with_user_strategy(sparse_struct_t* matrix, unsigned long dense_block_id);

bool is_supported_by_direct_atom_template_warp_compress_with_user_strategy(sparse_struct_t* matrix, unsigned long dense_block_id);

bool is_supported_by_direct_atom_template_with_user_strategy(sparse_struct_t* matrix, unsigned long dense_block_id);

bool is_supported_by_direct_atom_total_warp_reduce_template_with_user_strategy(sparse_struct_t* matrix, unsigned long dense_block_id);

bool is_supported_by_shared_memory_long_row_template_with_user_strategy(sparse_struct_t* matrix, unsigned long dense_block_id);

bool is_supported_by_shared_memory_template_warp_compress_with_user_strategy(sparse_struct_t* matrix, unsigned long dense_block_id);

bool is_supported_by_shared_memory_template_with_user_strategy(sparse_struct_t* matrix, unsigned long dense_block_id);

bool is_supported_by_shared_memory_total_warp_reduce_template_with_user_strategy(sparse_struct_t* matrix, unsigned long dense_block_id);

bool is_supported_by_unaligned_warp_reduce_same_TLB_size_template_with_user_strategy(sparse_struct_t* matrix, unsigned long dense_block_id);

bool is_supported_be_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_with_user_strategy(sparse_struct_t* matrix, unsigned long dense_block_id);

// 根据一个模板的集合再做一波筛选，形成一个新的模板集合
set<template_type> filter_from_existing_template_set(set<template_type> old_temp_set);

// 用户定义的插入节点的依赖性检查
bool dependence_of_exe_begin_artificial_input_node_with_user_strategy(exe_graph_t *graph, exe_begin_artificial_input_param_t param, int input_index);
bool dependence_of_exe_compress_node_with_user_strategy(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_begin_input_file_node_with_user_strategy(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_begin_input_file_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_dense_row_div_node_with_user_strategy(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_dense_row_div_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_dense_fixed_col_div_node_with_user_strategy(exe_graph_t* graph, exe_sub_graph_type graph_type, exe_dense_fixed_col_div_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_dense_row_coarse_sort_node_with_user_strategy(exe_graph_t* graph, exe_sub_graph_type graph_type, exe_dense_row_coarse_sort_param_t param, int sub_graph, int input_index);
bool dependence_of_exe_compress_WLB_row_div_node_with_user_strategy(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_warp_level_row_div_param_t param, int sub_graph, int input_index);


#endif