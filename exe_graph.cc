#include "exe_graph.hpp"
#include "user_pruning_strategy.hpp"

set<template_type> supported_template_of_sub_matrix(sparse_struct_t *matrix, unsigned long dense_block_id)
{
    assert(matrix != NULL);

    assert(matrix->block_coor_table.item_arr.size() > 0 && dense_block_id < matrix->block_coor_table.item_arr.size());

    set<template_type> return_template_type_set;

    // 子矩阵的表格项
    dense_block_table_item_t *sub_matrix = matrix->block_coor_table.item_arr[dense_block_id];

    assert(sub_matrix != NULL);
    assert(sub_matrix->compressed_block_ptr != NULL);
    assert(sub_matrix->compressed_block_ptr->read_index.size() == 7);

    // 所有的检查，压缩的版本可以覆盖不压缩的版本
    bool is_supported = false;

    is_supported = is_supported_by_unaligned_warp_reduce_same_TLB_size_template(matrix, dense_block_id) && is_supported_by_unaligned_warp_reduce_same_TLB_size_template_with_user_strategy(matrix, dense_block_id);

    if (is_supported == true)
    {
        // 一般来说，UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE和UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE的支持条件是一样的
        // cout << "UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE" << endl;
        return_template_type_set.insert(UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE);
    }

    // UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE多了一个tblock数量的限制，因为UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE要求
    // 在thread in block的数量一定的前提下，为了让总线程的数量将将多于TLB的数量，需要做一个检查
    is_supported = is_supported_by_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce(matrix, dense_block_id) && is_supported_be_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_with_user_strategy(matrix, dense_block_id);

    if (is_supported == true)
    {
        return_template_type_set.insert(UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE);
    }

    // 极度压缩，首先是warp和block的压缩
    is_supported = is_supported_by_direct_atom_template_warp_block_compress(matrix, dense_block_id) && is_supported_by_direct_atom_template_warp_block_compress_with_user_strategy(matrix, dense_block_id);

    // 小的不行检查大的
    if (is_supported == false)
    {
        is_supported = is_supported_by_direct_atom_template_warp_compress(matrix, dense_block_id) && is_supported_by_direct_atom_template_warp_compress_with_user_strategy(matrix, dense_block_id);

        if (is_supported == false)
        {
            is_supported = is_supported_by_direct_atom_template(matrix, dense_block_id) && is_supported_by_direct_atom_template_with_user_strategy(matrix, dense_block_id);

            if (is_supported == false)
            {
                cout << "cannot pass check in direct_atom_template" << endl;
            }
            else
            {
                return_template_type_set.insert(DIRECT_ATOM_TEMPLATE);
            }
        }
        else
        {
            // 通过了检查，将对应的模板类型记录下来
            return_template_type_set.insert(DIRECT_ATOM_TEMPLATE_WARP_COMPRESS);
        }
    }
    else
    {
        return_template_type_set.insert(DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS);
    }

    // shared memory的压缩
    is_supported = is_supported_by_shared_memory_template_warp_compress(matrix, dense_block_id) && is_supported_by_shared_memory_template_warp_compress_with_user_strategy(matrix, dense_block_id);

    if (is_supported == false)
    {
        is_supported = is_supported_by_shared_memory_template(matrix, dense_block_id) && is_supported_by_shared_memory_template_with_user_strategy(matrix, dense_block_id);

        if (is_supported == true)
        {
            return_template_type_set.insert(SHARED_MEMORY_TEMPLATE);
        }
    }
    else
    {
        return_template_type_set.insert(SHARED_MEMORY_TEMPLATE_WARP_COMPRESS);
    }

    // 带上warp归约的原子加
    is_supported = is_supported_by_direct_atom_total_warp_reduce_template(matrix, dense_block_id) && is_supported_by_direct_atom_total_warp_reduce_template_with_user_strategy(matrix, dense_block_id);

    if (is_supported == true)
    {
        return_template_type_set.insert(DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE);
    }

    // 所有的归约层次全部使用
    is_supported = is_supported_by_shared_memory_long_row_template(matrix, dense_block_id) && is_supported_by_shared_memory_long_row_template_with_user_strategy(matrix, dense_block_id);

    if (is_supported == true)
    {
        return_template_type_set.insert(SHARED_MEMORY_LONG_ROW_TEMPLATE);
    }

    // warp不跨行
    is_supported = is_supported_by_shared_memory_total_warp_reduce_template(matrix, dense_block_id) && is_supported_by_shared_memory_total_warp_reduce_template_with_user_strategy(matrix, dense_block_id);

    if (is_supported == true)
    {
        return_template_type_set.insert(SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE);
    }

    if (return_template_type_set.size() == 0)
    {
        cout << "this matrix can not be supported by all existing template" << endl;
    }
    else
    {
        return_template_type_set = filter_from_existing_template_set(return_template_type_set);
    }

    return return_template_type_set;
}

exe_begin_memory_cache_input_file_param_t get_exe_begin_memory_cache_input_file_param_from_coo_file(string file_name, data_type type)
{
    assert(type == DOUBLE || type == FLOAT);

    vector<float> float_val_vec;
    vector<double> double_val_vec;
    unsigned long max_col_index;
    unsigned long max_row_index;
    vector<unsigned long> col_index_vec;
    vector<unsigned long> row_index_vec;

    get_matrix_index_and_val_from_file(file_name, row_index_vec, col_index_vec, float_val_vec, double_val_vec, type, max_row_index, max_col_index);

    exe_begin_memory_cache_input_file_param_t param;
    param.col_index_cache = col_index_vec;
    param.row_index_cache = row_index_vec;
    param.float_val_cache = float_val_vec;
    param.double_val_cache = double_val_vec;
    param.col_index_max = max_col_index;
    param.row_index_max = max_row_index;
    param.val_data_type = type;

    // 将结果直接返回
    return param;
}

bool dependence_of_exe_begin_artificial_input_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_begin_artificial_input_param_t param, int sub_graph, int input_index)
{
    // 只判断输入的合法性
    assert(graph != NULL && (input_index == GRAPH_END || input_index <= graph->dense_sub_graph.exe_node_vec.size()));

    if (graph_type != EXE_DENSE_SUB_GRAPH)
    {
        return false;
    }

    if (sub_graph != 0)
    {
        return false;
    }

    if (graph_type == EXE_DENSE_SUB_GRAPH)
    {
        assert(sub_graph == 0);
    }

    // 全场只能有一个输入节点
    for (int dense_node_id = 0; dense_node_id < graph->dense_sub_graph.exe_node_vec.size(); dense_node_id++)
    {
        exe_node_t node = graph->dense_sub_graph.exe_node_vec[dense_node_id];
        assert(node.param != NULL);

        if (node.type == BEGIN_ARTIFICIAL_INPUT || node.type == BEGIN_INPUT_FILE || node.type == BEGIN_MEMORY_CACHE_INPUT_FILE)
        {
            return false;
        }
    }

    // 检查一下依赖，只能放在开头，或者在啥都没有的时候放在末尾
    if (input_index == 0)
    {
    }
    else if (input_index == GRAPH_END && graph->dense_sub_graph.exe_node_vec.size() == 0)
    {
    }
    else
    {
        return false;
    }

    return true;
}

bool dependence_of_exe_compress_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_param_t param, int sub_graph, int input_index)
{
    assert(graph != NULL && (input_index == GRAPH_END || input_index <= graph->dense_sub_graph.exe_node_vec.size()));

    // compress的插入位置是稠密视图
    if (graph_type != EXE_DENSE_SUB_GRAPH)
    {
        return false;
    }

    if (sub_graph != 0)
    {
        return false;
    }

    // 节点的位置和当前图的状态都会影响依赖，在加入这个节点的时候，已经加入了输入，插入的位置是稠密矩阵的最后一个位置
    if (graph->dense_sub_graph.exe_node_vec.size() == 0)
    {
        return false;
    }

    // 第一个位置已经有输入节点
    if (graph->dense_sub_graph.exe_node_vec[0].type != BEGIN_INPUT_FILE && graph->dense_sub_graph.exe_node_vec[0].type != BEGIN_ARTIFICIAL_INPUT && graph->dense_sub_graph.exe_node_vec[0].type != BEGIN_MEMORY_CACHE_INPUT_FILE)
    {
        return false;
    }

    // 插入在稠密矩阵视图的最后
    if (input_index != graph->dense_sub_graph.exe_node_vec.size() && input_index != GRAPH_END)
    {
        return false;
    }

    // 实际上压缩视图是空的
    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() != 0)
    {
        return false;
    }

    // 在之前不能出现compress
    for (int dense_node_id = 0; dense_node_id < graph->dense_sub_graph.exe_node_vec.size(); dense_node_id++)
    {
        exe_node_t node = graph->dense_sub_graph.exe_node_vec[dense_node_id];
        assert(node.param != NULL);

        if (node.type == COMPRESS)
        {
            return false;
        }
    }

    return true;
}

// 从内存中初始化一个矩阵的节点
bool dependence_of_exe_begin_memory_cache_input_file_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_begin_memory_cache_input_file_param_t param, int sub_graph, int input_index)
{
    assert(graph != NULL && (input_index == GRAPH_END || input_index <= graph->dense_sub_graph.exe_node_vec.size()));

    if (graph_type != EXE_DENSE_SUB_GRAPH)
    {
        return false;
    }

    if (sub_graph != 0)
    {
        return false;
    }

    if (graph_type == EXE_DENSE_SUB_GRAPH)
    {
        assert(sub_graph == 0);
    }

    // 全场只能有一个输入节点
    for (int dense_node_id = 0; dense_node_id < graph->dense_sub_graph.exe_node_vec.size(); dense_node_id++)
    {
        exe_node_t node = graph->dense_sub_graph.exe_node_vec[dense_node_id];
        assert(node.param != NULL);

        if (node.type == BEGIN_ARTIFICIAL_INPUT || node.type == BEGIN_INPUT_FILE || node.type == BEGIN_MEMORY_CACHE_INPUT_FILE)
        {
            return false;
        }
    }

    // 检查一下依赖，只能放在开头，或者在啥都没有的时候放在末尾
    // 
    if (input_index == 0)
    {
    }
    else if (input_index == GRAPH_END && graph->dense_sub_graph.exe_node_vec.size() == 0)
    {
    }
    else
    {
        return false;
    }

    return true;
}

// 依赖和人工输入节点差不多
bool dependence_of_exe_begin_input_file_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_begin_input_file_param_t param, int sub_graph, int input_index)
{
    // 只判断输入的合法性
    assert(graph != NULL && (input_index == GRAPH_END || input_index <= graph->dense_sub_graph.exe_node_vec.size()));

    if (graph_type != EXE_DENSE_SUB_GRAPH)
    {
        return false;
    }

    if (sub_graph != 0)
    {
        return false;
    }

    if (graph_type == EXE_DENSE_SUB_GRAPH)
    {
        assert(sub_graph == 0);
    }

    // 全场只能有一个输入节点
    for (int dense_node_id = 0; dense_node_id < graph->dense_sub_graph.exe_node_vec.size(); dense_node_id++)
    {
        exe_node_t node = graph->dense_sub_graph.exe_node_vec[dense_node_id];
        assert(node.param != NULL);

        if (node.type == BEGIN_ARTIFICIAL_INPUT || node.type == BEGIN_INPUT_FILE || node.type == BEGIN_MEMORY_CACHE_INPUT_FILE)
        {
            return false;
        }
    }

    // 检查一下依赖，只能放在开头，或者在啥都没有的时候放在末尾
    if (input_index == 0)
    {
    }
    else if (input_index == GRAPH_END && graph->dense_sub_graph.exe_node_vec.size() == 0)
    {
    }
    else
    {
        return false;
    }

    return true;
}

bool dependence_of_exe_dense_row_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_dense_row_div_param_t param, int sub_graph, int input_index)
{
    // 判断行分块是不是是不是可以被枚举。行分块只能添加到操作序列的尾部。
    assert(graph != NULL && (input_index == GRAPH_END || input_index == graph->dense_sub_graph.exe_node_vec.size()));
    assert(param.row_div_position.size() > 0);

    if (graph_type != EXE_DENSE_SUB_GRAPH)
    {
        return false;
    }

    if (sub_graph != 0)
    {
        return false;
    }

    // 并且保证之前出现了输入节点
    if (graph->dense_sub_graph.exe_node_vec.size() > 0)
    {
        if (graph->dense_sub_graph.exe_node_vec[0].type != BEGIN_INPUT_FILE && graph->dense_sub_graph.exe_node_vec[0].type != BEGIN_ARTIFICIAL_INPUT && graph->dense_sub_graph.exe_node_vec[0].type != BEGIN_MEMORY_CACHE_INPUT_FILE)
        {
            return false;
        }
    }
    else
    {
        return false;
    }

    // 依赖通过
    return true;
}

bool dependence_of_exe_dense_fixed_col_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_dense_fixed_col_div_param_t param, int sub_graph, int input_index)
{
    assert(graph != NULL && (input_index == GRAPH_END || input_index == graph->dense_sub_graph.exe_node_vec.size()));
    assert(param.fixed_col_block_size > 0);

    if (graph_type != EXE_DENSE_SUB_GRAPH)
    {
        return false;
    }

    if (sub_graph != 0)
    {
        return false;
    }

    // 并且保证之前出现了输入节点
    if (graph->dense_sub_graph.exe_node_vec.size() > 0)
    {
        if (graph->dense_sub_graph.exe_node_vec[0].type != BEGIN_INPUT_FILE && graph->dense_sub_graph.exe_node_vec[0].type != BEGIN_ARTIFICIAL_INPUT && graph->dense_sub_graph.exe_node_vec[0].type != BEGIN_MEMORY_CACHE_INPUT_FILE)
        {
            return false;
        }
    }
    else
    {
        return false;
    }

    // 保证之前没有出现compress
    if (graph->dense_sub_graph.preorder_node_set.count(COMPRESS) != 0)
    {
        return false;
    }

    return true;
}

bool dependence_of_exe_dense_row_coarse_sort_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_dense_row_coarse_sort_param_t param, int sub_graph, int input_index)
{
    // 只能加到尾部
    assert(graph != NULL && (input_index == GRAPH_END || input_index == graph->dense_sub_graph.exe_node_vec.size()));
    assert(param.bin_row_nnz_low_bound.size() > 0);
    assert(param.bin_row_nnz_low_bound[0] == 0);

    // 之前必须有输入节点
    if (graph->dense_sub_graph.preorder_node_set.count(BEGIN_ARTIFICIAL_INPUT) == 0 && graph->dense_sub_graph.preorder_node_set.count(BEGIN_INPUT_FILE) == 0 && graph->dense_sub_graph.preorder_node_set.count(BEGIN_MEMORY_CACHE_INPUT_FILE) == 0)
    {
        // 没有出现输入节点，不能通过
        return false;
    }

    // 之前不能有分块节点，主要是因为还没实现
    if (graph->dense_sub_graph.preorder_node_set.count(DENSE_FIXED_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->dense_sub_graph.preorder_node_set.count(DENSE_ROW_DIV) != 0)
    {
        return false;
    }

    // 排序不能排两次
    if (graph->dense_sub_graph.preorder_node_set.count(DENSE_ROW_COARSE_SORT) != 0)
    {
        return false;
    }

    if (graph->dense_sub_graph.preorder_node_set.count(DENSE_FINE_SORT) != 0)
    {
        return false;
    }

    if (graph->dense_sub_graph.preorder_node_set.count(DENSE_BLOCK_SORT) != 0)
    {
        return false;
    }

    // 之前不能出现compress
    if (graph->dense_sub_graph.preorder_node_set.count(COMPRESS) != 0)
    {
        return false;
    }

    return true;
}

bool dependence_of_exe_compress_BLB_row_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_tblock_level_row_div_param_t param, int sub_graph, int input_index)
{
    // 必须在执行完稠密子图的部分之后再执行
    assert(graph != NULL && graph->op_manager->matrix != NULL);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 子图的编号小于子块的数量
    assert(sub_graph < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size());
    // 节点添加的位置只能加到尾部
    assert(input_index == GRAPH_END || input_index == graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.size());

    // 保证所有所有线程块粒度的行号加起来正好是对应压缩子块的行数量
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->min_dense_row_index);

    unsigned long row_num_of_sub_matrix = graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->max_dense_row_index - graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->min_dense_row_index + 1;

    unsigned long sum_tmp = 0;
    for (auto item : param.row_num_of_each_BLB)
    {
        sum_tmp = sum_tmp + item;
    }

    assert(sum_tmp == row_num_of_sub_matrix);

    // 在之前不能出现其他任何类型的分块
    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_WARP_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_WARP_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_NNZ_DIV) != 0)
    {
        return false;
    }

    return true;
}

bool dependence_of_exe_compress_BLB_col_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_tblock_level_col_div_param_t param, int sub_graph, int input_index)
{
    // 必须在执行完稠密子图的部分之后再执行
    assert(graph != NULL && graph->op_manager->matrix != NULL);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 子图的编号小于子块的数量
    assert(sub_graph < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size());
    // 节点添加的位置只能加到尾部
    assert(input_index == GRAPH_END || input_index == graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.size());

    // 保证所有所有线程块粒度的行号加起来正好是对应压缩子块的行数量
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->min_dense_row_index);
    
    // 列块数组的大小肯定大于0的
    assert(param.col_block_nnz_num_of_each_BLB.size() > 0 && param.col_block_nnz_num_of_each_BLB[0].size() > 0);

    // 在之前不能出现其他任何类型的分块
    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_WARP_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_WARP_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_NNZ_DIV) != 0)
    {
        return false;
    }

    return true;
}

bool dependence_of_exe_compress_WLB_row_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_warp_level_row_div_param_t param, int sub_graph, int input_index)
{
    // 必须在执行完稠密子图的部分之后再执行
    assert(graph != NULL && graph->op_manager->matrix != NULL);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 子图的编号小于子块的数量
    assert(sub_graph < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size());
    // 节点添加的位置只能加到尾部
    assert(input_index == GRAPH_END || input_index == graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.size());

    // 保证所有所有线程块粒度的行号加起来正好是对应压缩子块的行数量
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->min_dense_row_index);
    
    // 列块数组的大小肯定大于0的
    assert(param.row_num_of_each_WLB_in_BLB.size() > 0 && param.row_num_of_each_WLB_in_BLB[0].size() > 0);

    // 在之前不能出现warp级别和thread级别的所有分块
    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_WARP_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_WARP_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_NNZ_DIV) != 0)
    {
        return false;
    }

    return true;
}

bool dependence_of_exe_compress_WLB_col_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_warp_level_col_div_param_t param, int sub_graph, int input_index)
{
    // 必须在执行完稠密子图的部分之后再执行
    assert(graph != NULL && graph->op_manager->matrix != NULL);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 子图的编号小于子块的数量
    assert(sub_graph < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size());
    // 节点添加的位置只能加到尾部
    assert(input_index == GRAPH_END || input_index == graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.size());

    // 保证所有所有线程块粒度的行号加起来正好是对应压缩子块的行数量
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->min_dense_row_index);
    
    // 列块数组的大小肯定大于0的
    assert(param.col_num_of_WLB_in_each_parent_row_block_or_BLB.size() > 0 && param.col_num_of_WLB_in_each_parent_row_block_or_BLB[0].size() > 0);

    // 列切分块的大小是32的倍数
    for (unsigned long i = 0; i < param.col_num_of_WLB_in_each_parent_row_block_or_BLB.size(); i++)
    {
        for (unsigned long j = 0; j < param.col_num_of_WLB_in_each_parent_row_block_or_BLB[i].size(); j++)
        {
            if (param.col_num_of_WLB_in_each_parent_row_block_or_BLB[i][j] % 32 != 0)
            {
                cout << "nnz of WLB must be multiples of 32" << endl;
                assert(false);
            }
        }
    }

    // 在之前不能出现warp级别和thread级别的所有分块
    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_WARP_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_WARP_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_NNZ_DIV) != 0)
    {
        return false;
    }

    return true;
}

bool dependence_of_exe_compress_TLB_row_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_thread_level_row_div_param_t param, int sub_graph, int input_index)
{
    // 必须在执行完稠密子图的部分之后再执行
    assert(graph != NULL && graph->op_manager->matrix != NULL);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 子图的编号小于子块的数量
    assert(sub_graph < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size());
    // 节点添加的位置只能加到尾部
    assert(input_index == GRAPH_END || input_index == graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.size());

    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->min_dense_row_index);

    // 之前不能出现所有thread级别的排序方式
    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_NNZ_DIV) != 0)
    {
        return false;
    }

    return true;
}

bool dependence_of_exe_compress_TLB_col_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_thread_level_col_div_param_t param, int sub_graph, int input_index)
{
    // 必须在执行完稠密子图的部分之后再执行
    assert(graph != NULL && graph->op_manager->matrix != NULL);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 子图的编号小于子块的数量
    assert(sub_graph < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size());
    // 节点添加的位置只能加到尾部
    assert(input_index == GRAPH_END || input_index == graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.size());

    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->min_dense_row_index);

    // 之前不能出现所有thread级别的排序方式
    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_NNZ_DIV) != 0)
    {
        return false;
    }

    return true;
}

bool dependence_of_exe_compress_thread_level_nnz_div_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_thread_level_nnz_div_param_t param, int sub_graph, int input_index)
{
    // 必须在执行完稠密子图的部分之后再执行
    assert(graph != NULL && graph->op_manager->matrix != NULL);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 子图的编号小于子块的数量
    assert(sub_graph < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size());

    // 之前不能出现任何种类的分块
    // 在之前不能出现其他任何类型的分块
    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_WARP_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_WARP_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_NNZ_DIV) != 0)
    {
        return false;
    }

    // 之前不能出现压缩视图下的padding
    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_ROW_PADDING) != 0)
    {
        return false;        
    }

    return true;
}

bool dependence_of_exe_compress_row_padding_node(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_row_padding_param_t param, int sub_graph, int input_index)
{
    // 必须在执行完稠密子图的部分之后再执行
    assert(graph != NULL && graph->op_manager->matrix != NULL);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 子图的编号小于子块的数量
    assert(sub_graph < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size());

    // 在之前不能出现其他任何类型的分块
    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_WARP_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_WARP_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_THREAD_LEVEL_NNZ_DIV) != 0)
    {
        return false;
    }

    // 之前不能出现row padding
    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_ROW_PADDING) != 0)
    {
        return false;
    }

    return true;
}

void reset_param_of_all_sub_compressed_graph(exe_compressed_sub_graph_t* sub_compressed_graph)
{
    assert(sub_compressed_graph != NULL);
    assert(sub_compressed_graph->exe_node_vec.size() > 0 && sub_compressed_graph->preorder_node_set.size() > 0);

    for (unsigned long i = 0; i < sub_compressed_graph->exe_node_vec.size(); i++)
    {
        // 当前节点的大小
        if (sub_compressed_graph->exe_node_vec[i].type == COMPRESSED_ROW_PADDING)
        {
            assert(sub_compressed_graph->exe_node_vec[i].param != NULL);

            delete (exe_compress_row_padding_param_t *)sub_compressed_graph->exe_node_vec[i].param;
            sub_compressed_graph->exe_node_vec[i].param = new exe_compress_row_padding_param_t();
        }
        else if (sub_compressed_graph->exe_node_vec[i].type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV)
        {
            assert(sub_compressed_graph->exe_node_vec[i].param != NULL);
            
            delete (exe_compress_tblock_level_row_div_param_t *)sub_compressed_graph->exe_node_vec[i].param;
            sub_compressed_graph->exe_node_vec[i].param = new exe_compress_tblock_level_row_div_param_t();
        }
        else if (sub_compressed_graph->exe_node_vec[i].type == COMPRESSED_TBLOCK_LEVEL_COL_DIV)
        {
            
            assert(sub_compressed_graph->exe_node_vec[i].param != NULL);

            delete (exe_compress_tblock_level_col_div_param_t *)sub_compressed_graph->exe_node_vec[i].param;
            sub_compressed_graph->exe_node_vec[i].param = new exe_compress_tblock_level_col_div_param_t();
        }
        else if (sub_compressed_graph->exe_node_vec[i].type == COMPRESSED_WARP_LEVEL_ROW_DIV)
        {
            assert(sub_compressed_graph->exe_node_vec[i].param != NULL);

            delete (exe_compress_warp_level_row_div_param_t *)sub_compressed_graph->exe_node_vec[i].param;
            sub_compressed_graph->exe_node_vec[i].param = new exe_compress_warp_level_row_div_param_t();
        }
        else if (sub_compressed_graph->exe_node_vec[i].type == COMPRESSED_WARP_LEVEL_COL_DIV)
        {
            assert(sub_compressed_graph->exe_node_vec[i].param != NULL);
            
            delete (exe_compress_warp_level_col_div_param_t *)sub_compressed_graph->exe_node_vec[i].param;
            sub_compressed_graph->exe_node_vec[i].param = new exe_compress_warp_level_col_div_param_t();
        }
        else if (sub_compressed_graph->exe_node_vec[i].type == COMPRESSED_THREAD_LEVEL_ROW_DIV)
        {
            assert(sub_compressed_graph->exe_node_vec[i].param != NULL);

            delete (exe_compress_thread_level_row_div_param_t *)sub_compressed_graph->exe_node_vec[i].param;
            sub_compressed_graph->exe_node_vec[i].param = new exe_compress_thread_level_row_div_param_t();
        }
        else if (sub_compressed_graph->exe_node_vec[i].type == COMPRESSED_THREAD_LEVEL_COL_DIV)
        {
            assert(sub_compressed_graph->exe_node_vec[i].param != NULL);

            delete (exe_compress_thread_level_col_div_param_t *)sub_compressed_graph->exe_node_vec[i].param;
            sub_compressed_graph->exe_node_vec[i].param = new exe_compress_thread_level_col_div_param_t();
        }
        else if (sub_compressed_graph->exe_node_vec[i].type == COMPRESSED_THREAD_LEVEL_NNZ_DIV)
        {
            assert(sub_compressed_graph->exe_node_vec[i].param != NULL);

            delete (exe_compress_thread_level_nnz_div_param_t *)sub_compressed_graph->exe_node_vec[i].param;
            sub_compressed_graph->exe_node_vec[i].param = new exe_compress_thread_level_nnz_div_param_t();
        }
        else
        {
            cout << "reset_param_of_all_sub_compressed_graph: exe node type is not supported" << endl;
            assert(false);
        }
    }
}


void reset_param_of_all_sub_dense_graph(exe_dense_sub_graph* sub_dense_graph)
{
    assert(sub_dense_graph != NULL);
    assert(sub_dense_graph->exe_node_vec.size() > 0 && sub_dense_graph->preorder_node_set.size() > 0);

    for (unsigned long i = 0; i < sub_dense_graph->exe_node_vec.size(); i++)
    {
        // 根据当前节点的类型来重置其参数
        if (sub_dense_graph->exe_node_vec[i].type == BEGIN_MEMORY_CACHE_INPUT_FILE)
        {
            assert(sub_dense_graph->exe_node_vec[i].param != NULL);
            delete (exe_begin_memory_cache_input_file_param_t *)sub_dense_graph->exe_node_vec[i].param;
            sub_dense_graph->exe_node_vec[i].param = new exe_begin_memory_cache_input_file_param_t();
        }
        else if (sub_dense_graph->exe_node_vec[i].type == DENSE_ROW_COARSE_SORT)
        {
            assert(sub_dense_graph->exe_node_vec[i].param != NULL);
            delete (exe_dense_row_coarse_sort_param_t *)sub_dense_graph->exe_node_vec[i].param;
            sub_dense_graph->exe_node_vec[i].param = new exe_dense_row_coarse_sort_param_t();
        }
        else if (sub_dense_graph->exe_node_vec[i].type == DENSE_ROW_DIV)
        {
            assert(sub_dense_graph->exe_node_vec[i].param != NULL);
            delete (exe_dense_row_div_param_t *)sub_dense_graph->exe_node_vec[i].param;
            sub_dense_graph->exe_node_vec[i].param = new exe_dense_row_div_param_t();
        }
        else if (sub_dense_graph->exe_node_vec[i].type == COMPRESS)
        {
            assert(sub_dense_graph->exe_node_vec[i].param != NULL);
            delete (exe_compress_param_t *)sub_dense_graph->exe_node_vec[i].param;
            sub_dense_graph->exe_node_vec[i].param = new exe_compress_param_t();
        }
        else
        {
            cout << "reset_param_of_all_sub_dense_graph: exe node type is not supported" << endl;
            assert(false);
        }
    }
}

void malloc_param_of_all_sub_compressed_graph(exe_compressed_sub_graph_t* sub_compressed_graph)
{
    assert(sub_compressed_graph != NULL);
    assert(sub_compressed_graph->exe_node_vec.size() > 0 && sub_compressed_graph->preorder_node_set.size() > 0);

    for (unsigned long i = 0; i < sub_compressed_graph->exe_node_vec.size(); i++)
    {
        // 当前节点的大小
        if (sub_compressed_graph->exe_node_vec[i].type == COMPRESSED_ROW_PADDING)
        {
            assert(sub_compressed_graph->exe_node_vec[i].param != NULL);
            sub_compressed_graph->exe_node_vec[i].param = new exe_compress_row_padding_param_t();
        }
        else if (sub_compressed_graph->exe_node_vec[i].type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV)
        {
            assert(sub_compressed_graph->exe_node_vec[i].param != NULL);
            sub_compressed_graph->exe_node_vec[i].param = new exe_compress_tblock_level_row_div_param_t();
        }
        else if (sub_compressed_graph->exe_node_vec[i].type == COMPRESSED_TBLOCK_LEVEL_COL_DIV)
        {
            assert(sub_compressed_graph->exe_node_vec[i].param != NULL);
            sub_compressed_graph->exe_node_vec[i].param = new exe_compress_tblock_level_col_div_param_t();
        }
        else if (sub_compressed_graph->exe_node_vec[i].type == COMPRESSED_WARP_LEVEL_ROW_DIV)
        {
            assert(sub_compressed_graph->exe_node_vec[i].param != NULL);
            sub_compressed_graph->exe_node_vec[i].param = new exe_compress_warp_level_row_div_param_t();
        }
        else if (sub_compressed_graph->exe_node_vec[i].type == COMPRESSED_WARP_LEVEL_COL_DIV)
        {
            assert(sub_compressed_graph->exe_node_vec[i].param != NULL);
            sub_compressed_graph->exe_node_vec[i].param = new exe_compress_warp_level_col_div_param_t();
        }
        else if (sub_compressed_graph->exe_node_vec[i].type == COMPRESSED_THREAD_LEVEL_ROW_DIV)
        {
            assert(sub_compressed_graph->exe_node_vec[i].param != NULL);
            sub_compressed_graph->exe_node_vec[i].param = new exe_compress_thread_level_row_div_param_t();
        }
        else if (sub_compressed_graph->exe_node_vec[i].type == COMPRESSED_THREAD_LEVEL_COL_DIV)
        {
            assert(sub_compressed_graph->exe_node_vec[i].param != NULL);
            sub_compressed_graph->exe_node_vec[i].param = new exe_compress_thread_level_col_div_param_t();
        }
        else if (sub_compressed_graph->exe_node_vec[i].type == COMPRESSED_THREAD_LEVEL_NNZ_DIV)
        {
            assert(sub_compressed_graph->exe_node_vec[i].param != NULL);
            sub_compressed_graph->exe_node_vec[i].param = new exe_compress_thread_level_nnz_div_param_t();
        }
        else
        {
            cout << "reset_param_of_all_sub_compressed_graph: exe node type is not supported" << endl;
            assert(false);
        }
    }
}

void malloc_param_of_all_sub_dense_graph(exe_dense_sub_graph* sub_dense_graph)
{
    assert(sub_dense_graph != NULL);
    assert(sub_dense_graph->exe_node_vec.size() > 0 && sub_dense_graph->preorder_node_set.size() > 0);

    for (unsigned long i = 0; i < sub_dense_graph->exe_node_vec.size(); i++)
    {
        assert(sub_dense_graph->exe_node_vec[i].param != NULL);

        // 根据当前节点的类型来重置其参数
        if (sub_dense_graph->exe_node_vec[i].type == BEGIN_MEMORY_CACHE_INPUT_FILE)
        {
            sub_dense_graph->exe_node_vec[i].param = new exe_begin_memory_cache_input_file_param_t();
        }
        else if (sub_dense_graph->exe_node_vec[i].type == DENSE_ROW_COARSE_SORT)
        {
            sub_dense_graph->exe_node_vec[i].param = new exe_dense_row_coarse_sort_param_t();
        }
        else if (sub_dense_graph->exe_node_vec[i].type == DENSE_ROW_DIV)
        {
            sub_dense_graph->exe_node_vec[i].param = new exe_dense_row_div_param_t();
        }
        else if (sub_dense_graph->exe_node_vec[i].type == COMPRESS)
        {
            sub_dense_graph->exe_node_vec[i].param = new exe_compress_param_t();
        }
        else
        {
            cout << "malloc_param_of_all_sub_dense_graph: exe node type is not supported" << endl;
            assert(false);
        }
    }
}

void add_exe_begin_artificial_input_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_begin_artificial_input_param_t param, int sub_graph, int input_index)
{
    // 只判断输入的合法性
    assert(graph != NULL && (input_index == GRAPH_END || input_index <= graph->dense_sub_graph.exe_node_vec.size()));

    exe_node_t node;

    node.type = BEGIN_ARTIFICIAL_INPUT;

    assert(node.param == NULL);
    add_param_to_exe_begin_artificial_input_node(&node, param);
    assert(node.param != NULL);

    // 将节点放到执行图中
    if (input_index == GRAPH_END)
    {
        graph->dense_sub_graph.exe_node_vec.push_back(node);
    }
    else
    {
        graph->dense_sub_graph.exe_node_vec.insert(graph->dense_sub_graph.exe_node_vec.begin() + input_index, node);
        assert(graph->dense_sub_graph.exe_node_vec[input_index].type == BEGIN_ARTIFICIAL_INPUT);
    }

    graph->dense_sub_graph.preorder_node_set.insert(BEGIN_ARTIFICIAL_INPUT);
}

// 将压缩类型的节点放到图中
void add_exe_compress_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_param_t param, int sub_graph, int input_index)
{
    assert(graph != NULL && (input_index == GRAPH_END || input_index <= graph->dense_sub_graph.exe_node_vec.size()));

    exe_node_t node;

    node.type = COMPRESS;

    assert(node.param == NULL);
    add_param_to_exe_compress_node(&node, param);
    assert(node.param != NULL);

    if (input_index == GRAPH_END)
    {
        graph->dense_sub_graph.exe_node_vec.push_back(node);
    }
    else
    {
        graph->dense_sub_graph.exe_node_vec.insert(graph->dense_sub_graph.exe_node_vec.begin() + input_index, node);
        assert(graph->dense_sub_graph.exe_node_vec[input_index].type == COMPRESS);
    }

    graph->dense_sub_graph.preorder_node_set.insert(COMPRESS);
}

// 将文件输入节点放到对应的位置
void add_exe_begin_input_file_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_begin_input_file_param_t param, int sub_graph, int input_index)
{
    assert(graph != NULL && (input_index == GRAPH_END || input_index <= graph->dense_sub_graph.exe_node_vec.size()));

    exe_node_t node;

    node.type = BEGIN_INPUT_FILE;

    assert(node.param == NULL);
    add_param_to_exe_begin_input_file_node(&node, param);
    assert(node.param != NULL);

    if (input_index == GRAPH_END)
    {
        graph->dense_sub_graph.exe_node_vec.push_back(node);
    }
    else
    {
        graph->dense_sub_graph.exe_node_vec.insert(graph->dense_sub_graph.exe_node_vec.begin() + input_index, node);
        assert(graph->dense_sub_graph.exe_node_vec[input_index].type == BEGIN_INPUT_FILE);
    }

    graph->dense_sub_graph.preorder_node_set.insert(BEGIN_INPUT_FILE);
}

void add_exe_dense_row_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_dense_row_div_param_t param, int sub_graph, int input_index)
{
    assert(graph != NULL && (input_index == GRAPH_END || input_index == graph->dense_sub_graph.exe_node_vec.size()));
    assert(param.row_div_position.size() > 0);

    exe_node_t node;

    node.type = DENSE_ROW_DIV;
    assert(node.param == NULL);
    add_param_to_exe_dense_row_div_node(&node, param);
    assert(node.param != NULL);

    // 都是加在末尾的
    graph->dense_sub_graph.exe_node_vec.push_back(node);

    graph->dense_sub_graph.preorder_node_set.insert(DENSE_ROW_DIV);
}

void add_exe_dense_fixed_col_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_dense_fixed_col_div_param_t param, int sub_graph, int input_index)
{
    assert(graph != NULL && (input_index == GRAPH_END || input_index == graph->dense_sub_graph.exe_node_vec.size()));
    assert(param.fixed_col_block_size > 0);

    exe_node_t node;

    node.type = DENSE_FIXED_COL_DIV;
    assert(node.param == NULL);
    add_param_to_exe_dense_fixed_col_div_node(&node, param);
    assert(node.param != NULL);

    // 只能加到末尾
    graph->dense_sub_graph.exe_node_vec.push_back(node);

    graph->dense_sub_graph.preorder_node_set.insert(DENSE_FIXED_COL_DIV);
}

// 将粗粒度排序的节点加入到对应位置
void add_exe_dense_row_coarse_sort_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_dense_row_coarse_sort_param_t param, int sub_graph, int input_index)
{
    assert(graph != NULL && (input_index == GRAPH_END || input_index == graph->dense_sub_graph.exe_node_vec.size()));
    assert(param.bin_row_nnz_low_bound.size() > 0);

    exe_node_t node;

    node.type = DENSE_ROW_COARSE_SORT;
    assert(node.param == NULL);
    add_param_to_exe_dense_row_coarse_sort_node(&node, param);
    assert(node.param != NULL);

    // 加到末尾
    graph->dense_sub_graph.exe_node_vec.push_back(node);

    graph->dense_sub_graph.preorder_node_set.insert(DENSE_ROW_COARSE_SORT);
}

void add_exe_compress_BLB_row_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_tblock_level_row_div_param_t param, int sub_graph, int input_index)
{
    // 必须在执行完稠密子图的部分之后再执行
    assert(graph != NULL && graph->op_manager->matrix != NULL);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 子图的编号小于子块的数量
    assert(sub_graph < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size());
    // 节点添加的位置只能加到尾部
    assert(input_index == GRAPH_END || input_index == graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.size());
    assert(param.row_num_of_each_BLB.size() > 0);

    // 在对应的位置添加一个操作
    exe_node_t node;

    node.type = COMPRESSED_TBLOCK_LEVEL_ROW_DIV;
    assert(node.param == NULL);
    add_param_to_exe_compress_BLB_row_div_node(&node, param);
    assert(node.param != NULL);

    // 加到子图的结尾
    graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.push_back(node);

    graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.insert(COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
}

void add_exe_compress_BLB_col_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_tblock_level_col_div_param_t param, int sub_graph, int input_index)
{
    // 必须在执行完稠密子图的部分之后再执行
    assert(graph != NULL && graph->op_manager->matrix != NULL);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 子图的编号小于子块的数量
    assert(sub_graph < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size());
    // 节点添加的位置只能加到尾部
    assert(input_index == GRAPH_END || input_index == graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.size());

    // 保证所有所有线程块粒度的行号加起来正好是对应压缩子块的行数量
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->min_dense_row_index);
    
    // 列块数组的大小肯定大于0的
    assert(param.col_block_nnz_num_of_each_BLB.size() > 0 && param.col_block_nnz_num_of_each_BLB[0].size() > 0);

    // 在对应位置添加一个操作
    exe_node_t node;

    node.type = COMPRESSED_TBLOCK_LEVEL_COL_DIV;
    assert(node.param == NULL);
    add_param_to_exe_compress_BLB_col_div_node(&node, param);
    assert(node.param != NULL);

    // 添加到子图结尾
    graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.push_back(node);

    graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.insert(COMPRESSED_TBLOCK_LEVEL_COL_DIV);
}

void add_exe_compress_WLB_row_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_warp_level_row_div_param_t param, int sub_graph, int input_index)
{
    // 必须在执行完稠密子图的部分之后再执行
    assert(graph != NULL && graph->op_manager->matrix != NULL);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 子图的编号小于子块的数量
    assert(sub_graph < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size());
    // 节点添加的位置只能加到尾部
    assert(input_index == GRAPH_END || input_index == graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.size());

    // 保证所有所有线程块粒度的行号加起来正好是对应压缩子块的行数量
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->min_dense_row_index);
    
    // 列块数组的大小肯定大于0的
    assert(param.row_num_of_each_WLB_in_BLB.size() > 0 && param.row_num_of_each_WLB_in_BLB[0].size() > 0);

    exe_node_t node;

    node.type = COMPRESSED_WARP_LEVEL_ROW_DIV;
    assert(node.param == NULL);
    add_param_to_exe_compress_WLB_row_div_node(&node, param);
    assert(node.param != NULL);

    // 将子图拷贝到结尾
    graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.push_back(node);
    
    graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.insert(COMPRESSED_WARP_LEVEL_ROW_DIV);
}

void add_exe_compress_WLB_col_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_warp_level_col_div_param_t param, int sub_graph, int input_index)
{
    // 必须在执行完稠密子图的部分之后再执行
    assert(graph != NULL && graph->op_manager->matrix != NULL);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 子图的编号小于子块的数量
    assert(sub_graph < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size());
    // 节点添加的位置只能加到尾部
    assert(input_index == GRAPH_END || input_index == graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.size());

    // 保证所有所有线程块粒度的行号加起来正好是对应压缩子块的行数量
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->min_dense_row_index);

    assert(param.col_num_of_WLB_in_each_parent_row_block_or_BLB.size() > 0 && param.col_num_of_WLB_in_each_parent_row_block_or_BLB[0].size() > 0);

    // 加入节点
    exe_node_t node;

    node.type = COMPRESSED_WARP_LEVEL_COL_DIV;
    assert(node.param == NULL);
    add_param_to_exe_compress_WLB_col_div_node(&node, param);
    assert(node.param != NULL);
    
    // 将子图拷贝到结尾
    graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.push_back(node);
    graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.insert(COMPRESSED_WARP_LEVEL_COL_DIV);
}

void add_exe_compress_TLB_row_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_thread_level_row_div_param_t param, int sub_graph, int input_index)
{
    // 必须在执行完稠密子图的部分之后再执行
    assert(graph != NULL && graph->op_manager->matrix != NULL);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 子图的编号小于子块的数量
    assert(sub_graph < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size());
    // 节点添加的位置只能加到尾部
    assert(input_index == GRAPH_END || input_index == graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.size());

    // 保证所有所有线程块粒度的行号加起来正好是对应压缩子块的行数量
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->min_dense_row_index);

    exe_node_t node;

    node.type = COMPRESSED_THREAD_LEVEL_ROW_DIV;
    assert(node.param == NULL);
    add_param_to_exe_compress_TLB_row_div_node(&node, param);
    assert(node.param != NULL);

    // 将子图拷贝到结尾
    graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.push_back(node);
    graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.insert(COMPRESSED_THREAD_LEVEL_ROW_DIV);
}

void add_exe_compress_TLB_col_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_thread_level_col_div_param_t param, int sub_graph, int input_index)
{
    // 必须在执行完稠密子图的部分之后再执行
    assert(graph != NULL && graph->op_manager->matrix != NULL);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 子图的编号小于子块的数量
    assert(sub_graph < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size());
    // 节点添加的位置只能加到尾部
    assert(input_index == GRAPH_END || input_index == graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.size());

    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->min_dense_row_index);

    exe_node_t node;

    node.type = COMPRESSED_THREAD_LEVEL_COL_DIV;
    assert(node.param == NULL);
    add_param_to_exe_compress_TLB_col_div_node(&node, param);
    assert(node.param != NULL);

    // 将子图拷贝到结尾
    graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.push_back(node);
    graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.insert(COMPRESSED_THREAD_LEVEL_COL_DIV);
}

// 增加一个不对齐的加操作
void add_exe_compress_thread_level_nnz_div_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_thread_level_nnz_div_param_t param, int sub_graph, int input_index)
{
    // 必须在执行完稠密子图的部分之后再执行
    assert(graph != NULL && graph->op_manager->matrix != NULL);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 子图的编号小于子块的数量
    assert(sub_graph < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size());
    // 节点添加的位置只能加到尾部
    assert(input_index == GRAPH_END || input_index == graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.size());

    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_graph]->min_dense_row_index);

    exe_node_t node;
    
    node.type = COMPRESSED_THREAD_LEVEL_NNZ_DIV;
    assert(node.param == NULL);
    add_param_to_exe_compress_thread_level_nnz_div_node(&node, param);
    assert(node.param != NULL);

    // 将子图拷贝到结尾
    graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.push_back(node);
    graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.insert(COMPRESSED_THREAD_LEVEL_NNZ_DIV);
}

void add_exe_compress_row_padding_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_row_padding_param_t param, int sub_graph, int input_index)
{
     // 必须在执行完稠密子图的部分之后再执行
    assert(graph != NULL && graph->op_manager->matrix != NULL);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 子图的编号小于子块的数量
    assert(sub_graph < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size());
    // 节点添加的位置只能加到尾部
    assert(input_index == GRAPH_END || input_index == graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.size());

    exe_node_t node;
    node.type = COMPRESSED_ROW_PADDING;
    assert(node.param == NULL);
    add_param_to_exe_compress_row_padding_node(&node, param);
    assert(node.param != NULL);

    // 将子图拷贝到结尾
    graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].exe_node_vec.push_back(node);
    graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.insert(COMPRESSED_ROW_PADDING);
}

void add_exe_begin_memory_cache_input_file_node_to_exe_graph(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_begin_memory_cache_input_file_param_t param, int sub_graph, int input_index)
{
    assert(graph != NULL && (input_index == GRAPH_END || input_index <= graph->dense_sub_graph.exe_node_vec.size()));
    
    exe_node_t node;

    node.type = BEGIN_MEMORY_CACHE_INPUT_FILE;

    assert(node.param == NULL);
    add_param_to_exe_begin_memory_cache_input_file_node(&node, param);
    assert(node.param != NULL);

    if (input_index == GRAPH_END)
    {
        graph->dense_sub_graph.exe_node_vec.push_back(node);
    }
    else
    {
        graph->dense_sub_graph.exe_node_vec.insert(graph->dense_sub_graph.exe_node_vec.begin() + input_index, node);
        assert(graph->dense_sub_graph.exe_node_vec[input_index].type == BEGIN_MEMORY_CACHE_INPUT_FILE);
    }

    graph->dense_sub_graph.preorder_node_set.insert(BEGIN_MEMORY_CACHE_INPUT_FILE);
}

void execute_exe_begin_memory_cache_input_file_node(exe_graph_t *graph, exe_node_t node)
{
    assert(graph != NULL && node.param != NULL && node.type == BEGIN_MEMORY_CACHE_INPUT_FILE);
    exe_begin_memory_cache_input_file_param_t *param_ptr = (exe_begin_memory_cache_input_file_param_t *)node.param;

    // 查看参数类型是不是正确
    assert(param_ptr->val_data_type == DOUBLE || param_ptr->val_data_type == FLOAT);

    assert(param_ptr->col_index_cache.size() > 0);

    assert(param_ptr->col_index_cache.size() == param_ptr->row_index_cache.size());

    assert(param_ptr->col_index_max >= param_ptr->col_index_cache[param_ptr->col_index_cache.size() - 1]);
    assert(param_ptr->row_index_max >= param_ptr->row_index_cache[param_ptr->row_index_cache.size() - 1]);

    if (param_ptr->val_data_type == DOUBLE)
    {
        assert(param_ptr->float_val_cache.size() == 0);
        assert(param_ptr->double_val_cache.size() == param_ptr->col_index_cache.size());
    }

    if (param_ptr->val_data_type == FLOAT)
    {
        assert(param_ptr->double_val_cache.size() == 0);
        assert(param_ptr->float_val_cache.size() == param_ptr->col_index_cache.size());
    }

    // 执行
    // sparse_struct_t *init_sparse_struct_by_coo_vector(vector<unsigned long> row_arr, vector<unsigned long> col_arr,
    //                                               vector<float> val_arr_float, vector<double> val_arr_double, data_type value_data_type,
    //                                               unsigned long col_index_max, unsigned long row_index_max);
    sparse_struct_t *matrix = NULL;

    matrix = init_sparse_struct_by_coo_vector(param_ptr->row_index_cache, param_ptr->col_index_cache, param_ptr->float_val_cache, param_ptr->double_val_cache, param_ptr->val_data_type, param_ptr->col_index_max, param_ptr->row_index_max);

    operator_manager_t *op_manager = init_op_manager(matrix);
    
    assert(op_manager != NULL && matrix != NULL);

    graph->op_manager = op_manager;
}

// 执行对应的节点
void execute_exe_begin_artificial_input_node(exe_graph_t *graph, exe_node_t node)
{
    assert(graph != NULL && node.param != NULL && node.type == BEGIN_ARTIFICIAL_INPUT);
    exe_begin_artificial_input_param_t *param_ptr = (exe_begin_artificial_input_param_t *)node.param;

    assert(param_ptr->val_data_type == DOUBLE || param_ptr->val_data_type == FLOAT);

    sparse_struct_t *matrix = NULL;

    // 产生一个新的矩阵，并且初始化自己
    if (param_ptr->val_data_type == FLOAT)
    {
        dataset_builder_t data_builder = get_dataset_builder(param_ptr->max_row_index, param_ptr->max_col_index, param_ptr->nnz_of_each_row);
        vector<double> none_vec;
        matrix = init_sparse_struct_by_coo_vector(get_row_index_of_dataset_builder(data_builder), get_col_index_of_dataset_builder(data_builder), get_float_val_of_dataset_builder(data_builder), none_vec, FLOAT, param_ptr->max_col_index, param_ptr->max_row_index);
    }

    if (param_ptr->val_data_type == DOUBLE)
    {
        dataset_builder_t data_builder = get_dataset_builder(param_ptr->max_row_index, param_ptr->max_col_index, param_ptr->nnz_of_each_row);
        vector<float> none_vec;
        matrix = init_sparse_struct_by_coo_vector(get_row_index_of_dataset_builder(data_builder), get_col_index_of_dataset_builder(data_builder), none_vec, get_double_val_of_dataset_builder(data_builder), DOUBLE, param_ptr->max_col_index, param_ptr->max_row_index);
    }

    operator_manager_t *op_manager = init_op_manager(matrix);

    assert(op_manager != NULL);

    graph->op_manager = op_manager;
}

// 执行compress
void execute_exe_compress_node(exe_graph_t *graph, exe_node_t node)
{
    assert(graph != NULL && node.param != NULL && node.type == COMPRESS);

    operator_manager_t *op_manager = graph->op_manager;

    assert(op_manager != NULL);

    sparse_struct_t *matrix = op_manager->matrix;

    assert(matrix != NULL);

    compress_dense_view(graph->op_manager);

    // 子矩阵的数量
    int sub_compressed_matrix_num = matrix->block_coor_table.item_arr.size();

    assert(sub_compressed_matrix_num > 0);

    for (int sub_compressed_matrix_id = 0; sub_compressed_matrix_id < sub_compressed_matrix_num; sub_compressed_matrix_id++)
    {
        exe_compressed_sub_graph_t sub_graph;

        // 插入到矩阵中
        graph->total_compressed_sub_graph.compressed_sub_graph_vec.push_back(sub_graph);
    }

    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == sub_compressed_matrix_num);
}

void execute_exe_begin_input_file_node(exe_graph_t *graph, exe_node_t node)
{
    assert(graph != NULL && node.param != NULL && node.type == BEGIN_INPUT_FILE);
    assert(graph->op_manager == NULL);

    exe_begin_input_file_param_t *param_ptr = (exe_begin_input_file_param_t *)node.param;

    // 执行
    assert(param_ptr->val_data_type == DOUBLE || param_ptr->val_data_type == FLOAT);

    sparse_struct_t *matrix = NULL;

    // 执行产生对应的矩阵
    // matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/rail4284.mtx.coo", FLOAT);
    // cout << "param_ptr->input_file_name:" << param_ptr->input_file_name << endl;
    matrix = init_sparse_struct_by_coo_file(param_ptr->input_file_name, param_ptr->val_data_type);

    assert(matrix != NULL);

    operator_manager_t *op_manager = init_op_manager(matrix);

    assert(op_manager != NULL);

    graph->op_manager = op_manager;
}

void execute_exe_dense_row_div_node(exe_graph_t *graph, exe_node_t node)
{
    assert(graph != NULL && node.param != NULL && node.type == DENSE_ROW_DIV);
    assert(graph->op_manager != NULL);
    assert(graph->op_manager->matrix != NULL);

    exe_dense_row_div_param_t *param_ptr = (exe_dense_row_div_param_t *)node.param;

    // 如果分块的数量只有一个，就代表不用分块
    if (param_ptr->row_div_position.size() <= 2)
    {
        return;
    }

    // 执行
    assert(param_ptr->dense_sub_block_id <= graph->op_manager->matrix->block_coor_table.item_arr.size());
    assert(param_ptr->row_div_position.size() > 1);
    assert(param_ptr->row_div_position[0] == 0);

    // 执行分块操作
    // var_len_row_div(op_manager->matrix, NULL, block_begin_row);

    // 如果之前没有分块过那就直接NULL分块
    if (graph->op_manager->matrix->block_coor_table.item_arr.size() == 0)
    {
        assert(param_ptr->dense_sub_block_id == 0);
        assert(param_ptr->row_div_position[param_ptr->row_div_position.size() - 1] == graph->op_manager->matrix->dense_row_number);
        // 这里执行分块操作
        var_len_row_div(graph->op_manager->matrix, NULL, param_ptr->row_div_position);
        assert(graph->op_manager->matrix->block_coor_table.item_arr.size() > 0);
    }
    else
    {
        assert(param_ptr->dense_sub_block_id < graph->op_manager->matrix->block_coor_table.item_arr.size());
        // cout << "param_ptr->row_div_position[param_ptr->row_div_position.size() - 1]:" << param_ptr->row_div_position[param_ptr->row_div_position.size() - 1] << endl;
        // cout << "(graph->op_manager->matrix->block_coor_table.item_arr[param_ptr->dense_sub_block_id]->max_dense_row_index - graph->op_manager->matrix->block_coor_table.item_arr[param_ptr->dense_sub_block_id]->min_dense_row_index + 1):" << (graph->op_manager->matrix->block_coor_table.item_arr[param_ptr->dense_sub_block_id]->max_dense_row_index - graph->op_manager->matrix->block_coor_table.item_arr[param_ptr->dense_sub_block_id]->min_dense_row_index + 1) << endl;
        assert(param_ptr->row_div_position[param_ptr->row_div_position.size() - 1] == (graph->op_manager->matrix->block_coor_table.item_arr[param_ptr->dense_sub_block_id]->max_dense_row_index - graph->op_manager->matrix->block_coor_table.item_arr[param_ptr->dense_sub_block_id]->min_dense_row_index + 1));
        var_len_row_div(graph->op_manager->matrix, graph->op_manager->matrix->block_coor_table.item_arr[param_ptr->dense_sub_block_id], param_ptr->row_div_position);
    }
}

void execute_exe_dense_fixed_col_div_node(exe_graph_t *graph, exe_node_t node)
{
    assert(graph != NULL && node.param != NULL && node.type == DENSE_FIXED_COL_DIV);
    assert(graph->op_manager != NULL);
    assert(graph->op_manager->matrix != NULL);

    exe_dense_fixed_col_div_param_t *param_ptr = (exe_dense_fixed_col_div_param_t *)node.param;

    // 执行
    assert(param_ptr->dense_sub_block_id <= graph->op_manager->matrix->block_coor_table.item_arr.size());
    assert(param_ptr->fixed_col_block_size > 0);

    // 如果之前没有分过块，就执行全局分块
    if (graph->op_manager->matrix->block_coor_table.item_arr.size() == 0)
    {
        assert(param_ptr->dense_sub_block_id == 0);
        assert(param_ptr->fixed_col_block_size > 0);
        fixed_len_col_div(graph->op_manager->matrix, NULL, param_ptr->fixed_col_block_size);
    }
    else
    {
        assert(param_ptr->dense_sub_block_id < graph->op_manager->matrix->block_coor_table.item_arr.size());
        assert(param_ptr->fixed_col_block_size > 0);
        fixed_len_col_div(graph->op_manager->matrix, graph->op_manager->matrix->block_coor_table.item_arr[param_ptr->dense_sub_block_id], param_ptr->fixed_col_block_size);
    }
}

void execute_exe_dense_row_coarse_sort_node(exe_graph_t *graph, exe_node_t node)
{
    assert(graph != NULL && node.param != NULL && node.type == DENSE_ROW_COARSE_SORT);
    assert(graph->op_manager != NULL);
    assert(graph->op_manager->matrix != NULL);

    exe_dense_row_coarse_sort_param_t *param_ptr = (exe_dense_row_coarse_sort_param_t *)node.param;

    // 之前没有排序过
    assert(graph->op_manager->matrix->is_sorted == false && graph->op_manager->matrix->sorted_row_index == NULL);

    // 执行粗粒度排序
    total_dense_block_coarse_sort(graph->op_manager, param_ptr->bin_row_nnz_low_bound);

    assert(graph->op_manager->matrix->is_sorted == true && graph->op_manager->matrix->sorted_row_index != NULL);
}

void execute_exe_compress_BLB_row_div_node(exe_graph_t *graph, exe_node_t node, unsigned long sub_matrix_id)
{
    assert(graph != NULL && node.param != NULL && node.type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 已经被压缩过
    assert(graph->dense_sub_graph.exe_node_vec[graph->dense_sub_graph.exe_node_vec.size() - 1].type == COMPRESS);

    // 子图的索引小于子图的数量
    assert(sub_matrix_id < graph->op_manager->matrix->block_coor_table.item_arr.size());

    // 压缩已经被执行
    for (auto item : graph->op_manager->matrix->block_coor_table.item_arr)
    {
        assert(item->compressed_block_ptr != NULL);
    }

    exe_compress_tblock_level_row_div_param_t *param_ptr = (exe_compress_tblock_level_row_div_param_t *)node.param;

    // 从矩阵中看看之前有没有给对应的矩阵增加一个tblock分块的操作
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    // 执行行分块
    sep_tblock_level_row_csr(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr, param_ptr->row_num_of_each_BLB);

    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 3);
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2]->level_of_this_index == TBLOCK_LEVEL);
}

void execute_exe_compress_BLB_row_div_node(sparse_struct_t* matrix, exe_node_t node, unsigned long sub_matrix_id)
{
    assert(matrix != NULL && node.param != NULL && node.type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
    // 子图索引满足要求
    assert(sub_matrix_id < matrix->block_coor_table.item_arr.size());
    // 子图已经被压缩
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    
    exe_compress_tblock_level_row_div_param_t *param_ptr = (exe_compress_tblock_level_row_div_param_t *)node.param;

    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    // 创建一个临时操作管理器
    operator_manager_t* op_manager = init_op_manager(matrix);
    
    // 执行行分块
    sep_tblock_level_row_csr(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr, param_ptr->row_num_of_each_BLB);
    // 析构操作管理器的最外层指针
    delete op_manager;

    // 检查
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 3);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2]->level_of_this_index == TBLOCK_LEVEL);
}

void execute_exe_compress_BLB_col_div_node(exe_graph_t *graph, exe_node_t node, unsigned long sub_matrix_id)
{
    assert(graph != NULL && node.param != NULL && node.type == COMPRESSED_TBLOCK_LEVEL_COL_DIV);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 已经被压缩过
    assert(graph->dense_sub_graph.exe_node_vec[graph->dense_sub_graph.exe_node_vec.size() - 1].type == COMPRESS);
    // 子图的索引小于子图的数量
    assert(sub_matrix_id < graph->op_manager->matrix->block_coor_table.item_arr.size());

    // 压缩已经被执行
    for (auto item : graph->op_manager->matrix->block_coor_table.item_arr)
    {
        assert(item->compressed_block_ptr != NULL);
    }

    exe_compress_tblock_level_col_div_param_t *param_ptr = (exe_compress_tblock_level_col_div_param_t *)node.param;

    // 之前矩阵没有进行过tblock级别的分块，
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    one_row_sep_tblock_level_row_csr(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr);

    // 现在tblock的数量和列分块的外层数组大小是一致的
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2]->block_num == param_ptr->col_block_nnz_num_of_each_BLB.size());

    // 这里执行一个列切分，遍历每一行，一行执行一个列分块，这个数组记录要进一步列分块的tblock编号，按照从0到block_num的数字代表所有tblock都需要进一步列分块
    vector<unsigned long> sub_block_index_vec;

    for (unsigned long i = 0; i < param_ptr->col_block_nnz_num_of_each_BLB.size(); i++)
    {
        sub_block_index_vec.push_back(i);
    }

    // 执行列分块
    sep_tblock_level_col_csr(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr, sub_block_index_vec, param_ptr->col_block_nnz_num_of_each_BLB);

    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 3);
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2]->level_of_this_index == TBLOCK_LEVEL);
}

void execute_exe_compress_BLB_col_div_node(sparse_struct_t* matrix, exe_node_t node, unsigned long sub_matrix_id)
{
    assert(matrix != NULL && node.param != NULL && node.type == COMPRESSED_TBLOCK_LEVEL_COL_DIV);
    // 子块的索引要满足要求
    assert(sub_matrix_id < matrix->block_coor_table.item_arr.size());
    // 对应子块存在并且被压缩
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    exe_compress_tblock_level_col_div_param_t *param_ptr = (exe_compress_tblock_level_col_div_param_t *)node.param;

    operator_manager_t* op_manager = init_op_manager(matrix);

    // 之前矩阵没有进行过tblock级别的分块，
    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    // 执行一行一个的分块
    one_row_sep_tblock_level_row_csr(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr);

    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2]->block_num == param_ptr->col_block_nnz_num_of_each_BLB.size());

    // 然后执行列切分
    vector<unsigned long> sub_block_index_vec;
    for (unsigned long i = 0; i < param_ptr->col_block_nnz_num_of_each_BLB.size(); i++)
    {
        sub_block_index_vec.push_back(i);
    }

    // 执行列分块
    sep_tblock_level_col_csr(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr, sub_block_index_vec, param_ptr->col_block_nnz_num_of_each_BLB);

    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 3);
    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2]->level_of_this_index == TBLOCK_LEVEL);

    // 析构对应的操作管理器
    delete op_manager;
}

// 执行WLB级别的行分块
void execute_exe_compress_WLB_row_div_node(exe_graph_t *graph, exe_node_t node, unsigned long sub_matrix_id)
{
    assert(graph != NULL && node.param != NULL && node.type == COMPRESSED_WARP_LEVEL_ROW_DIV);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 已经被压缩过
    assert(graph->dense_sub_graph.exe_node_vec[graph->dense_sub_graph.exe_node_vec.size() - 1].type == COMPRESS);
    // 子图的索引小于子图的数量
    assert(sub_matrix_id < graph->op_manager->matrix->block_coor_table.item_arr.size());

    // 压缩已经被执行
    for (auto item : graph->op_manager->matrix->block_coor_table.item_arr)
    {
        assert(item->compressed_block_ptr != NULL);
    }

    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->min_dense_row_index);

    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2 || graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 3);

    // 如果之前BLB级别没有切分，这里要补一个默认的BLB切分，将整个矩阵划分为一个BLB
    if (graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2)
    {
        default_sep_tblock_level_row_csr(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr);
    }

    // 执行完之后检查
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 3);

    exe_compress_warp_level_row_div_param_t* param_ptr = (exe_compress_warp_level_row_div_param_t*)node.param;

    // 实际子块的数量和节点参数子块的数量相吻合
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2]->block_num == param_ptr->row_num_of_each_WLB_in_BLB.size());

    // 每一个tblock子块都需要被进一步分块
    vector<unsigned long> sep_block_id_arr;

    for (unsigned long i = 0; i < param_ptr->row_num_of_each_WLB_in_BLB.size(); i++)
    {
        sep_block_id_arr.push_back(i);
    }

    // 执行进一步分块，每一个子块的大小由节点中的参数决定
    sep_warp_level_row_csr(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr, sep_block_id_arr, param_ptr->row_num_of_each_WLB_in_BLB);

    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 4);
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[3]->level_of_this_index == WRAP_LEVEL);
}

void execute_exe_compress_WLB_row_div_node(sparse_struct_t* matrix, exe_node_t node, unsigned long sub_matrix_id)
{
    assert(matrix != NULL && node.param != NULL && node.type == COMPRESSED_WARP_LEVEL_ROW_DIV);
    // 子图索引满足要求
    assert(sub_matrix_id < matrix->block_coor_table.item_arr.size());
    // 子图存在并且已经被压缩
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    // 准备一个操作管理器
    operator_manager_t* op_manager = init_op_manager(matrix);

    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->max_dense_row_index >= op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->min_dense_row_index);

    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2 || op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 3);

    // 如果之前BLB级别没有切分，这里要补一个默认的BLB切分，将整个矩阵划分为一个BLB
    if (op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2)
    {
        default_sep_tblock_level_row_csr(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr);
    }

    // 执行完之后检查
    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 3);

    exe_compress_warp_level_row_div_param_t* param_ptr = (exe_compress_warp_level_row_div_param_t*)node.param;

    // 实际子块的数量和节点参数子块的数量相吻合
    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2]->block_num == param_ptr->row_num_of_each_WLB_in_BLB.size());

    // 每一个tblock子块都需要被进一步分块
    vector<unsigned long> sep_block_id_arr;

    for (unsigned long i = 0; i < param_ptr->row_num_of_each_WLB_in_BLB.size(); i++)
    {
        sep_block_id_arr.push_back(i);
    }

    // 执行进一步分块，每一个子块的大小由节点中的参数决定
    sep_warp_level_row_csr(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr, sep_block_id_arr, param_ptr->row_num_of_each_WLB_in_BLB);

    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 4);
    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[3]->level_of_this_index == WRAP_LEVEL);

    delete op_manager;
}

void execute_exe_compress_WLB_col_div_node(exe_graph_t *graph, exe_node_t node, unsigned long sub_matrix_id)
{
    // 执行WLB的纵分块
    assert(graph != NULL && node.param != NULL && node.type == COMPRESSED_WARP_LEVEL_COL_DIV);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 已经被压缩过
    assert(graph->dense_sub_graph.exe_node_vec[graph->dense_sub_graph.exe_node_vec.size() - 1].type == COMPRESS);
    // 子图的索引小于子图的数量
    assert(sub_matrix_id < graph->op_manager->matrix->block_coor_table.item_arr.size());

    // 压缩已经被执行
    for (auto item : graph->op_manager->matrix->block_coor_table.item_arr)
    {
        assert(item->compressed_block_ptr != NULL);
    }

    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->min_dense_row_index);

    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2 || graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 3);
    
    // 如果之前BLB级别没有切分，这里要补一个默认的BLB切分，将整个矩阵划分为一个BLB
    if (graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2)
    {
        // 默认的BLB的分块
        default_sep_tblock_level_row_csr(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr);
    }

    // 执行完之后检查
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 3);
    
    compressed_block_t* compressed_block_view = graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 这里需要补一个行切分，将每个BLB都执行一次行切分
    one_row_sep_warp_level_row_csr(compressed_block_view);
    
    assert(compressed_block_view->read_index.size() == 4);

    // 这个时候warp级别的列分块，节点列分块的参数和当前WLB的数量一致
    exe_compress_warp_level_col_div_param_t* param_ptr = (exe_compress_warp_level_col_div_param_t*) node.param;
    assert(param_ptr != NULL && param_ptr->col_num_of_WLB_in_each_parent_row_block_or_BLB.size() == compressed_block_view->read_index[3]->block_num);

    // 申请一个数组，包含所有WLB的分块，然后执行列分块
    vector<unsigned long> sep_WLB_id;
    for (unsigned long i = 0; i < param_ptr->col_num_of_WLB_in_each_parent_row_block_or_BLB.size(); i++)
    {
        sep_WLB_id.push_back(i);
    }

    sep_warp_level_col_csr(compressed_block_view, sep_WLB_id, param_ptr->col_num_of_WLB_in_each_parent_row_block_or_BLB);

    // 分割完之后，检查一下
    assert(compressed_block_view->read_index.size() == 4);
}

void execute_exe_compress_WLB_col_div_node(sparse_struct_t* matrix, exe_node_t node, unsigned long sub_matrix_id)
{
    // 执行WLB的纵分块
    assert(matrix != NULL && node.param != NULL && node.type == COMPRESSED_WARP_LEVEL_COL_DIV);
    assert(sub_matrix_id < matrix->block_coor_table.item_arr.size());
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    operator_manager_t* op_manager = init_op_manager(matrix);

    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->max_dense_row_index >= op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->min_dense_row_index);
    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2 || op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 3);

    // 如果之前BLB级别没有切分，这里要补一个默认的BLB切分，将整个矩阵划分为一个BLB
    if (op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2)
    {
        // 默认的BLB的分块
        default_sep_tblock_level_row_csr(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr);
    }

    // 执行完之后检查
    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 3);

    compressed_block_t* compressed_block_ptr = op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 这里需要补一个行切分，将每个BLB都执行一次行切分
    one_row_sep_warp_level_row_csr(compressed_block_ptr);
    
    assert(compressed_block_ptr->read_index.size() == 4);

    // 这个时候warp级别的列分块，节点列分块的参数和当前WLB的数量一致
    exe_compress_warp_level_col_div_param_t* param_ptr = (exe_compress_warp_level_col_div_param_t*) node.param;
    assert(param_ptr != NULL && param_ptr->col_num_of_WLB_in_each_parent_row_block_or_BLB.size() == compressed_block_ptr->read_index[3]->block_num);

    // 申请一个数组，包含所有WLB的分块，然后执行列分块
    vector<unsigned long> sep_WLB_id;
    for (unsigned long i = 0; i < param_ptr->col_num_of_WLB_in_each_parent_row_block_or_BLB.size(); i++)
    {
        sep_WLB_id.push_back(i);
    }

    sep_warp_level_col_csr(compressed_block_ptr, sep_WLB_id, param_ptr->col_num_of_WLB_in_each_parent_row_block_or_BLB);

    // 分割完之后，检查一下
    assert(compressed_block_ptr->read_index.size() == 4);

    delete op_manager;
}

void execute_exe_compress_TLB_row_div_node(exe_graph_t *graph, exe_node_t node, unsigned long sub_matrix_id)
{
    // 执行WLB的纵分块
    assert(graph != NULL && node.param != NULL && node.type == COMPRESSED_THREAD_LEVEL_ROW_DIV);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 已经被压缩过
    assert(graph->dense_sub_graph.exe_node_vec[graph->dense_sub_graph.exe_node_vec.size() - 1].type == COMPRESS);
    // 子图的索引小于子图的数量
    assert(sub_matrix_id < graph->op_manager->matrix->block_coor_table.item_arr.size());

    // 压缩已经被执行
    for (auto item : graph->op_manager->matrix->block_coor_table.item_arr)
    {
        assert(item->compressed_block_ptr != NULL);
    }

    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->min_dense_row_index);

    compressed_block_t* compressed_block_ptr = graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    assert(compressed_block_ptr->read_index.size() == 2 || compressed_block_ptr->read_index.size() == 3 || compressed_block_ptr->read_index.size() == 4);

    // 如果之前BLB级别没有切分，这里要补一个默认的BLB切分，将整个矩阵划分为一个BLB
    if (compressed_block_ptr->read_index.size() == 2)
    {
        // 执行默认的BLB级别的分块
        default_sep_tblock_level_row_csr(compressed_block_ptr);

        // 执行完之后检查
        assert(compressed_block_ptr->read_index.size() == 3);
    }

    // 只有BLB级别的切分，但是没有WLB级别的切分
    if (compressed_block_ptr->read_index.size() == 3)
    {
        default_sep_warp_level_row_csr(compressed_block_ptr);
    }

    assert(compressed_block_ptr->read_index.size() == 4);

    // 最后执行一个thread级别的行切分，遍历所有的WLB，分别执行一个列切分。
    // 不对列切分TLB传入任何参数就可以默认激活一个TLB的行切分
    vector<unsigned long> futher_thread_block_vec;
    vector<unsigned long> futher_thread_col_block_size;

    // 列分块
    sep_thread_level_col_ell_with_padding(compressed_block_ptr, futher_thread_block_vec, futher_thread_col_block_size);

    // TLB的切分会产生3个索引
    assert(compressed_block_ptr->read_index.size() == 7);
}

void execute_exe_compress_TLB_row_div_node(sparse_struct_t* matrix, exe_node_t node, unsigned long sub_matrix_id)
{
    assert(matrix != NULL && node.param != NULL && node.type == COMPRESSED_THREAD_LEVEL_ROW_DIV);
    assert(sub_matrix_id < matrix->block_coor_table.item_arr.size());
    // 子块存在，并且已经被压缩
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    operator_manager_t* op_manager = init_op_manager(matrix);

    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->max_dense_row_index >= op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->min_dense_row_index);

    compressed_block_t* compressed_block_ptr = op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 可能仅仅经过压缩，也可能经过了BLB和WLB分块
    assert(compressed_block_ptr->read_index.size() == 2 || compressed_block_ptr->read_index.size() == 3 || compressed_block_ptr->read_index.size() == 4);

    // 如果之前BLB级别没有切分，这里要补一个默认的BLB切分，将整个矩阵划分为一个BLB
    if (compressed_block_ptr->read_index.size() == 2)
    {
        // 执行默认的BLB级别的分块
        default_sep_tblock_level_row_csr(compressed_block_ptr);

        // 执行完之后检查
        assert(compressed_block_ptr->read_index.size() == 3);
    }

    // 只有BLB级别的切分，但是没有WLB级别的切分
    if (compressed_block_ptr->read_index.size() == 3)
    {
        default_sep_warp_level_row_csr(compressed_block_ptr);
    }

    assert(compressed_block_ptr->read_index.size() == 4);

    // 最后执行一个thread级别的行切分，遍历所有的WLB，分别执行一个列切分。
    // 不对列切分TLB传入任何参数就可以默认激活一个TLB的行切分
    vector<unsigned long> futher_thread_block_vec;
    vector<unsigned long> futher_thread_col_block_size;

    // 列分块
    sep_thread_level_col_ell_with_padding(compressed_block_ptr, futher_thread_block_vec, futher_thread_col_block_size);

    // TLB的切分会产生3个索引
    assert(compressed_block_ptr->read_index.size() == 7);
    
    delete op_manager;
}

void execute_exe_compress_TLB_col_div_node(exe_graph_t *graph, exe_node_t node, unsigned long sub_matrix_id)
{
    // 执行WLB的纵分块
    assert(graph != NULL && node.param != NULL && node.type == COMPRESSED_THREAD_LEVEL_COL_DIV);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 已经被压缩过
    assert(graph->dense_sub_graph.exe_node_vec[graph->dense_sub_graph.exe_node_vec.size() - 1].type == COMPRESS);
    // 子图的索引小于子图的数量
    assert(sub_matrix_id < graph->op_manager->matrix->block_coor_table.item_arr.size());

    // 压缩已经被执行
    for (auto item : graph->op_manager->matrix->block_coor_table.item_arr)
    {
        assert(item->compressed_block_ptr != NULL);
    }

    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->min_dense_row_index);

    compressed_block_t* compressed_block_ptr = graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    assert(compressed_block_ptr->read_index.size() == 2 || compressed_block_ptr->read_index.size() == 3 || compressed_block_ptr->read_index.size() == 4);

    // 如果之前BLB级别没有切分，这里要补一个默认的BLB切分，将整个矩阵划分为一个BLB
    if (compressed_block_ptr->read_index.size() == 2)
    {
        // 默认的BLB分块
        default_sep_tblock_level_row_csr(compressed_block_ptr);
    }

    // 只有BLB级别的切分，但是没有WLB级别的切分
    if (compressed_block_ptr->read_index.size() == 3)
    {
        // 如果没有warp层次的切分，需要补一个默认的warp层次的切分，将WLB和BLB合为一体
        default_sep_warp_level_row_csr(compressed_block_ptr);
    }

    assert(compressed_block_ptr->read_index.size() == 4);

    // 参数
    exe_compress_thread_level_col_div_param_t* param_ptr = (exe_compress_thread_level_col_div_param_t*)node.param;
    assert(param_ptr != NULL);

    // 节点参数的数量和当前BLB的数量一致
    assert(param_ptr->col_num_of_TLB_in_each_parent_block.size() == compressed_block_ptr->read_index[3]->block_num);

    // 用一个数组记录所有需要进一步分块的
    vector<unsigned long> WLB_id_need_to_div;

    for (unsigned long i = 0; i < param_ptr->col_num_of_TLB_in_each_parent_block.size(); i++)
    {
        WLB_id_need_to_div.push_back(i);
    }

    // 执行ell分块
    sep_thread_level_col_ell_with_padding(compressed_block_ptr, WLB_id_need_to_div, param_ptr->col_num_of_TLB_in_each_parent_block);

    // TLB的切分会产生3个索引
    assert(compressed_block_ptr->read_index.size() == 7);
}

void execute_exe_compress_TLB_col_div_node(sparse_struct_t* matrix, exe_node_t node, unsigned long sub_matrix_id)
{
    assert(matrix != NULL && node.param != NULL && node.type == COMPRESSED_THREAD_LEVEL_COL_DIV);
    assert(sub_matrix_id < matrix->block_coor_table.item_arr.size());
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    operator_manager_t* op_manager = init_op_manager(matrix);

    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->max_dense_row_index >= op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->min_dense_row_index);

    compressed_block_t* compressed_block_ptr = op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    assert(compressed_block_ptr->read_index.size() == 2 || compressed_block_ptr->read_index.size() == 3 || compressed_block_ptr->read_index.size() == 4);

    // 如果之前BLB级别没有切分，这里要补一个默认的BLB切分，将整个矩阵划分为一个BLB
    if (compressed_block_ptr->read_index.size() == 2)
    {
        // 默认的BLB分块
        default_sep_tblock_level_row_csr(compressed_block_ptr);
    }

    // 只有BLB级别的切分，但是没有WLB级别的切分
    if (compressed_block_ptr->read_index.size() == 3)
    {
        // 如果没有warp层次的切分，需要补一个默认的warp层次的切分，将WLB和BLB合为一体
        default_sep_warp_level_row_csr(compressed_block_ptr);
    }

    assert(compressed_block_ptr->read_index.size() == 4);

    // 参数
    exe_compress_thread_level_col_div_param_t* param_ptr = (exe_compress_thread_level_col_div_param_t*)node.param;
    assert(param_ptr != NULL);

    // 节点参数的数量和当前BLB的数量一致
    assert(param_ptr->col_num_of_TLB_in_each_parent_block.size() == compressed_block_ptr->read_index[3]->block_num);

    // 用一个数组记录所有需要进一步分块的
    vector<unsigned long> WLB_id_need_to_div;

    for (unsigned long i = 0; i < param_ptr->col_num_of_TLB_in_each_parent_block.size(); i++)
    {
        WLB_id_need_to_div.push_back(i);
    }

    // 执行ell分块
    sep_thread_level_col_ell_with_padding(compressed_block_ptr, WLB_id_need_to_div, param_ptr->col_num_of_TLB_in_each_parent_block);

    // TLB的切分会产生3个索引
    assert(compressed_block_ptr->read_index.size() == 7);

    delete op_manager;
}

void execute_exe_compress_thread_level_nnz_div_node(exe_graph_t *graph, exe_node_t node, unsigned long sub_matrix_id)
{
    // 之前完全什么都没有执行
    // 执行WLB的纵分块
    assert(graph != NULL && node.param != NULL && node.type == COMPRESSED_THREAD_LEVEL_NNZ_DIV);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 已经被压缩过
    assert(graph->dense_sub_graph.exe_node_vec[graph->dense_sub_graph.exe_node_vec.size() - 1].type == COMPRESS);
    // 子图的索引小于子图的数量
    assert(sub_matrix_id < graph->op_manager->matrix->block_coor_table.item_arr.size());

    // 压缩已经被执行
    for (auto item : graph->op_manager->matrix->block_coor_table.item_arr)
    {
        assert(item->compressed_block_ptr != NULL);
    }

    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->max_dense_row_index >= graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->min_dense_row_index);

    compressed_block_t* compressed_block_ptr = graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 之前什么分块都没有执行
    assert(compressed_block_ptr->read_index.size() == 2);

    // block级别的分块
    default_sep_tblock_level_row_csr(compressed_block_ptr);

    assert(compressed_block_ptr->read_index.size() == 3);

    default_sep_warp_level_row_csr(compressed_block_ptr);

    assert(compressed_block_ptr->read_index.size() == 4);

    exe_compress_thread_level_nnz_div_param_t* param_ptr = (exe_compress_thread_level_nnz_div_param_t*) node.param;

    sep_thread_level_acc_to_nnz(compressed_block_ptr, param_ptr->TLB_nnz_num);

    // 执行完之后有7个读索引
    assert(compressed_block_ptr->read_index.size() == 7);
}

void execute_exe_compress_thread_level_nnz_div_node(sparse_struct_t* matrix, exe_node_t node, unsigned long sub_matrix_id)
{
    // 矩阵执行
    assert(matrix != NULL && node.param != NULL && node.type == COMPRESSED_THREAD_LEVEL_NNZ_DIV);
    // 子块索引满足要求
    assert(sub_matrix_id < matrix->block_coor_table.item_arr.size());
    // 已经压缩
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    operator_manager_t* op_manager = init_op_manager(matrix);

    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->max_dense_row_index >= op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->min_dense_row_index);

    compressed_block_t* compressed_block_ptr = op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 之前什么分块都没有执行
    assert(compressed_block_ptr->read_index.size() == 2);

    // block级别的分块
    default_sep_tblock_level_row_csr(compressed_block_ptr);

    assert(compressed_block_ptr->read_index.size() == 3);

    default_sep_warp_level_row_csr(compressed_block_ptr);

    assert(compressed_block_ptr->read_index.size() == 4);

    exe_compress_thread_level_nnz_div_param_t* param_ptr = (exe_compress_thread_level_nnz_div_param_t*) node.param;

    sep_thread_level_acc_to_nnz(compressed_block_ptr, param_ptr->TLB_nnz_num);

    // 执行完之后有7个读索引
    assert(compressed_block_ptr->read_index.size() == 7);

    delete op_manager;
}

void execute_exe_compress_row_padding_node(exe_graph_t* graph, exe_node_t node, unsigned long sub_matrix_id)
{
    // 之前完全什么都没有执行
    // 执行WLB的纵分块
    assert(graph != NULL && node.param != NULL && node.type == COMPRESSED_ROW_PADDING);
    // 子块的大小和子图的数量是一致的
    assert(graph->total_compressed_sub_graph.compressed_sub_graph_vec.size() == graph->op_manager->matrix->block_coor_table.item_arr.size());
    // 已经被压缩过
    assert(graph->dense_sub_graph.exe_node_vec[graph->dense_sub_graph.exe_node_vec.size() - 1].type == COMPRESS);
    // 子图的索引小于子图的数量
    assert(sub_matrix_id < graph->op_manager->matrix->block_coor_table.item_arr.size());

    // 之前没有 执行任何分块操作
    assert(graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    exe_compress_row_padding_param_t* param_ptr = (exe_compress_row_padding_param_t*) node.param;

    assert(param_ptr != NULL);

    compress_block_end_block_multiple_padding(graph->op_manager, sub_matrix_id, param_ptr->multiply, param_ptr->padding_row_length);

    unsigned long row_num_of_compressed_sub_block = graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[0]->max_row_index -  graph->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[0]->min_row_index + 1;
    
    assert(row_num_of_compressed_sub_block % param_ptr->multiply == 0);
}

void execute_exe_compress_row_padding_node(sparse_struct_t* matrix, exe_node_t node, unsigned long sub_matrix_id)
{
    assert(matrix != NULL && node.param != NULL && node.type == COMPRESSED_ROW_PADDING);
    // 子块满足要求
    assert(sub_matrix_id < matrix->block_coor_table.item_arr.size());
    // 子块存在
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    operator_manager_t* op_manager = init_op_manager(matrix);

    assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    exe_compress_row_padding_param_t* param_ptr = (exe_compress_row_padding_param_t*) node.param;

    assert(param_ptr != NULL);

    compress_block_end_block_multiple_padding(op_manager, sub_matrix_id, param_ptr->multiply, param_ptr->padding_row_length);

    unsigned long row_num_of_compressed_sub_block = op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[0]->max_row_index -  op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[0]->min_row_index + 1;
    
    assert(row_num_of_compressed_sub_block % param_ptr->multiply == 0);

    delete op_manager;
}

void add_param_to_exe_begin_memory_cache_input_file_node(exe_node_t *node, exe_begin_memory_cache_input_file_param param)
{
    assert(node != NULL && node->param == NULL);

    node->param = new exe_begin_memory_cache_input_file_param_t();
    exe_begin_memory_cache_input_file_param_t* param_ptr = (exe_begin_memory_cache_input_file_param_t *)node->param;
    param_ptr->col_index_cache = param.col_index_cache;
    param_ptr->row_index_cache = param.row_index_cache;
    param_ptr->col_index_max = param.col_index_max;
    param_ptr->row_index_max = param.row_index_max;
    param_ptr->float_val_cache = param.float_val_cache;
    param_ptr->double_val_cache = param.double_val_cache;
    
    param_ptr->val_data_type = param.val_data_type;
}

void add_param_to_exe_begin_artificial_input_node(exe_node_t *node, exe_begin_artificial_input_param_t param)
{
    assert(node != NULL && node->param == NULL);

    node->param = new exe_begin_artificial_input_param_t();
    // 将参数拷贝
    memcpy(node->param, &param, sizeof(exe_begin_artificial_input_param_t));
}

void add_param_to_exe_compress_node(exe_node_t *node, exe_compress_param_t param)
{
    assert(node != NULL && node->param == NULL);

    node->param = new exe_compress_param_t();

    // 将参数拷贝
    memcpy(node->param, &param, sizeof(exe_compress_param_t));
}

void add_param_to_exe_begin_input_file_node(exe_node_t *node, exe_begin_input_file_param_t param)
{
    assert(node != NULL && node->param == NULL);

    node->param = new exe_begin_input_file_param_t();

    // 因为有字符串这样的复杂结构体，所以一个个拷贝
    exe_begin_input_file_param_t *node_param_ptr = (exe_begin_input_file_param_t *)node->param;
    node_param_ptr->input_file_name = param.input_file_name;
    node_param_ptr->val_data_type = param.val_data_type;
}

void add_param_to_exe_dense_row_div_node(exe_node_t *node, exe_dense_row_div_param_t param)
{
    assert(node != NULL && node->param == NULL);

    node->param = new exe_dense_row_div_param_t();

    // 一个个拷贝
    exe_dense_row_div_param_t *node_param_ptr = (exe_dense_row_div_param_t *)node->param;
    node_param_ptr->dense_sub_block_id = param.dense_sub_block_id;
    node_param_ptr->row_div_position = param.row_div_position;
}

void add_param_to_exe_dense_fixed_col_div_node(exe_node_t *node, exe_dense_fixed_col_div_param_t param)
{
    assert(node != NULL && node->param == NULL);

    node->param = new exe_dense_fixed_col_div_param_t();

    // 一个个拷贝
    exe_dense_fixed_col_div_param_t *node_param_ptr = (exe_dense_fixed_col_div_param_t *)node->param;
    node_param_ptr->dense_sub_block_id = param.dense_sub_block_id;
    node_param_ptr->fixed_col_block_size = param.fixed_col_block_size;
}

void add_param_to_exe_dense_row_coarse_sort_node(exe_node_t *node, exe_dense_row_coarse_sort_param_t param)
{
    assert(node != NULL && node->param == NULL);

    node->param = new exe_dense_row_coarse_sort_param_t();

    // 一个个拷贝
    exe_dense_row_coarse_sort_param_t *node_param_ptr = (exe_dense_row_coarse_sort_param_t *)node->param;
    node_param_ptr->bin_row_nnz_low_bound = param.bin_row_nnz_low_bound;
}

void add_param_to_exe_compress_BLB_row_div_node(exe_node_t *node, exe_compress_tblock_level_row_div_param_t param)
{
    assert(node != NULL && node->param == NULL);
    assert(param.row_num_of_each_BLB.size() > 0);

    node->param = new exe_compress_tblock_level_row_div_param_t();

    // 拷贝内容
    exe_compress_tblock_level_row_div_param_t *node_param_ptr = (exe_compress_tblock_level_row_div_param_t *)node->param;
    node_param_ptr->row_num_of_each_BLB = param.row_num_of_each_BLB;
    
    assert(node_param_ptr->row_num_of_each_BLB.size() > 0);
}

void add_param_to_exe_compress_BLB_col_div_node(exe_node_t *node, exe_compress_tblock_level_col_div_param_t param)
{
    assert(node != NULL && node->param == NULL);
    assert(param.col_block_nnz_num_of_each_BLB.size() > 0 && param.col_block_nnz_num_of_each_BLB[0].size() > 0);
    
    node->param = new exe_compress_tblock_level_col_div_param_t();

    // 拷贝内容
    exe_compress_tblock_level_col_div_param_t *node_param_ptr = (exe_compress_tblock_level_col_div_param_t *)node->param;
    node_param_ptr->col_block_nnz_num_of_each_BLB = param.col_block_nnz_num_of_each_BLB;

    assert(node_param_ptr->col_block_nnz_num_of_each_BLB.size() > 0 && node_param_ptr->col_block_nnz_num_of_each_BLB[0].size() > 0);
}

void add_param_to_exe_compress_WLB_row_div_node(exe_node_t *node, exe_compress_warp_level_row_div_param_t param)
{
    assert(node != NULL && node->param == NULL);
    assert(param.row_num_of_each_WLB_in_BLB.size() > 0 && param.row_num_of_each_WLB_in_BLB[0].size() > 0);

    node->param = new exe_compress_warp_level_row_div_param_t();

    // 拷贝内容
    exe_compress_warp_level_row_div_param_t *node_param_ptr = (exe_compress_warp_level_row_div_param_t *)node->param;
    node_param_ptr->row_num_of_each_WLB_in_BLB = param.row_num_of_each_WLB_in_BLB;

    assert(node_param_ptr->row_num_of_each_WLB_in_BLB.size() > 0);
}

void add_param_to_exe_compress_WLB_col_div_node(exe_node_t *node, exe_compress_warp_level_col_div_param_t param)
{
    assert(node != NULL && node->param == NULL);
    assert(param.col_num_of_WLB_in_each_parent_row_block_or_BLB.size() > 0 && param.col_num_of_WLB_in_each_parent_row_block_or_BLB[0].size());

    node->param = new exe_compress_warp_level_col_div_param_t();

    // 拷贝内容
    exe_compress_warp_level_col_div_param_t *node_param_ptr = (exe_compress_warp_level_col_div_param_t *)node->param;
    node_param_ptr->col_num_of_WLB_in_each_parent_row_block_or_BLB = param.col_num_of_WLB_in_each_parent_row_block_or_BLB;

    assert(node_param_ptr->col_num_of_WLB_in_each_parent_row_block_or_BLB.size() > 0);
}

void add_param_to_exe_compress_TLB_row_div_node(exe_node_t *node, exe_compress_thread_level_row_div_param_t param)
{
    // 这个参数是空的
    assert(node != NULL && node->param == NULL);
    node->param = new exe_compress_thread_level_row_div_param_t();
}

void add_param_to_exe_compress_TLB_col_div_node(exe_node_t *node, exe_compress_thread_level_col_div_param_t param)
{
    assert(node != NULL && node->param == NULL);
    assert(param.col_num_of_TLB_in_each_parent_block.size() > 0);

    node->param = new exe_compress_thread_level_col_div_param_t();

    exe_compress_thread_level_col_div_param_t* node_param_ptr = (exe_compress_thread_level_col_div_param_t*)node->param;
    node_param_ptr->col_num_of_TLB_in_each_parent_block = param.col_num_of_TLB_in_each_parent_block;

    assert(node_param_ptr->col_num_of_TLB_in_each_parent_block.size() > 0);
}

void add_param_to_exe_compress_thread_level_nnz_div_node(exe_node_t *node, exe_compress_thread_level_nnz_div_param_t param)
{
    assert(node != NULL && node->param == NULL);
    assert(param.TLB_nnz_num > 0);

    node->param = new exe_compress_thread_level_nnz_div_param_t();

    exe_compress_thread_level_nnz_div_param_t* node_param_ptr = (exe_compress_thread_level_nnz_div_param_t*)node->param;
    node_param_ptr->TLB_nnz_num = param.TLB_nnz_num;
}

void add_param_to_exe_compress_row_padding_node(exe_node_t *node, exe_compress_row_padding_param_t param)
{
    assert(node != NULL && node->param == NULL);
    assert(param.multiply >= 1 && param.padding_row_length > 0);

    node->param = new exe_compress_row_padding_param_t();
    
    exe_compress_row_padding_param_t* node_param_ptr = (exe_compress_row_padding_param_t*)node->param;
    node_param_ptr->multiply = param.multiply;
    node_param_ptr->padding_row_length = param.padding_row_length;
}

void optimize_graph(exe_graph_t *graph)
{
}

void execute_graph(exe_graph_t *graph)
{
    assert(graph != NULL);

    // 首先执行密集矩阵部分
    for (int dense_matrix_node_id = 0; dense_matrix_node_id < graph->dense_sub_graph.exe_node_vec.size(); dense_matrix_node_id++)
    {
        // 执行密集视图的节点
        execute_node_of_dense_sub_graph(graph, dense_matrix_node_id);
    }

    // 执行压缩视图的部分
}

void execute_graph_dense_part(exe_graph_t *graph)
{
    assert(graph != NULL);

    // 首先执行密集矩阵部分
    for (int dense_matrix_node_id = 0; dense_matrix_node_id < graph->dense_sub_graph.exe_node_vec.size(); dense_matrix_node_id++)
    {
        // 执行密集视图的节点
        execute_node_of_dense_sub_graph(graph, dense_matrix_node_id);
    }
}

sparse_struct_t* get_matrix_dense_view_graph(exe_dense_sub_graph_t* dense_graph)
{
    assert(dense_graph != NULL);
    // 检查
    for (unsigned long i = 0; i < dense_graph->exe_node_vec.size(); i++)
    {
        assert(dense_graph->exe_node_vec[i].param != NULL);
    }

    // 创造一个临时的图
    exe_graph_t graph;
    graph.dense_sub_graph = *dense_graph;
    graph.builder = NULL;

    // 执行临时的图
    execute_graph_dense_part(&graph);

    assert(graph.op_manager != NULL);
    assert(graph.builder == NULL);

    sparse_struct_t* matrix = graph.op_manager->matrix;

    delete graph.op_manager;
    
    return matrix;
}

string convert_exe_node_type_to_string(exe_node_type node_type)
{
    if (node_type == BEGIN_INPUT_FILE)
    {
        return "BEGIN_INPUT_FILE";
    }
    else if (node_type == BEGIN_ARTIFICIAL_INPUT)
    {
        return "BEGIN_ARTIFICIAL_INPUT";
    }
    else if (node_type == BEGIN_MEMORY_CACHE_INPUT_FILE)
    {
        return "BEGIN_MEMORY_CACHE_INPUT_FILE";
    }
    else if (node_type == DENSE_ROW_COARSE_SORT)
    {
        return "DENSE_ROW_COARSE_SORT";
    }
    else if (node_type == DENSE_FINE_SORT)
    {
        return "DENSE_FINE_SORT";
    }
    else if (node_type == DENSE_TOTAL_ROW_LEVEL_PADDING)
    {
        return "DENSE_TOTAL_ROW_LEVEL_PADDING";
    }
    else if (node_type == DENSE_BLOCK_SORT)
    {
        return "DENSE_BLOCK_SORT";
    }
    else if (node_type == DENSE_ROW_DIV)
    {
        return "DENSE_ROW_DIV";
    }
    else if (node_type == DENSE_FIXED_COL_DIV)
    {
        return "DENSE_FIXED_COL_DIV";
    }
    else if (node_type == COMPRESS)
    {
        return "COMPRESS";
    }
    else if (node_type == COMPRESSED_ROW_PADDING)
    {
        return "COMPRESSED_ROW_PADDING";
    }
    else if (node_type == COMPRESSED_BLOCK_SORT)
    {
        return "COMPRESSED_BLOCK_SORT";
    }
    else if (node_type == COMPRESSED_THREAD_LEVEL_ROW_DIV)
    {
        return "COMPRESSED_THREAD_LEVEL_ROW_DIV";
    }
    else if (node_type == COMPRESSED_THREAD_LEVEL_COL_DIV)
    {
        return "COMPRESSED_THREAD_LEVEL_COL_DIV";
    }
    else if (node_type == COMPRESSED_WARP_LEVEL_ROW_DIV)
    {
        return "COMPRESSED_WARP_LEVEL_ROW_DIV";
    }
    else if (node_type == COMPRESSED_WARP_LEVEL_COL_DIV)
    {
        return "COMPRESSED_WARP_LEVEL_COL_DIV";
    }
    else if (node_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV)
    {
        return "COMPRESSED_TBLOCK_LEVEL_ROW_DIV";
    }
    else if (node_type == COMPRESSED_TBLOCK_LEVEL_COL_DIV)
    {
        return "COMPRESSED_TBLOCK_LEVEL_COL_DIV";
    }
    else if (node_type == COMPRESSED_THREAD_LEVEL_NNZ_DIV)
    {
        return "COMPRESSED_THREAD_LEVEL_NNZ_DIV";
    }
    else
    {
        cout << "convert_exe_node_type_to_string: exe node type is not supported" << endl;
        assert(false);
    }
}

void execute_graph_compress_part(exe_graph_t *graph)
{
    assert(graph != NULL);

    // 遍历所有子图
    for (int sub_graph_id = 0; sub_graph_id < graph->total_compressed_sub_graph.compressed_sub_graph_vec.size(); sub_graph_id++)
    {
        // cout << "sub_graph_id:" << sub_graph_id << endl;
        // 遍历一个子图中每一个节点，并且执行。
        for (int sub_graph_node_id = 0; sub_graph_node_id < graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph_id].exe_node_vec.size(); sub_graph_node_id++)
        {
            // 执行子图中的一个节点
            // cout << "sub_graph_node_id:" << sub_graph_node_id << endl;
            execute_node_of_compressed_sub_graph(graph, sub_graph_id, sub_graph_node_id);
        }
    }
}

void execute_node_of_dense_sub_graph(exe_graph_t *graph, int exe_node_of_sub_dense_graph_id)
{
    assert(graph != NULL);

    // 获取当前节点
    exe_node_t node = graph->dense_sub_graph.exe_node_vec[exe_node_of_sub_dense_graph_id];

    assert(node.param != NULL);

    if (node.type == BEGIN_ARTIFICIAL_INPUT)
    {
        // 执行这个节点
        execute_exe_begin_artificial_input_node(graph, node);
        return;
    }

    if (node.type == BEGIN_MEMORY_CACHE_INPUT_FILE)
    {
        execute_exe_begin_memory_cache_input_file_node(graph, node);
        return;
    }

    if (node.type == COMPRESS)
    {
        execute_exe_compress_node(graph, node);
        return;
    }

    if (node.type == BEGIN_INPUT_FILE)
    {
        execute_exe_begin_input_file_node(graph, node);
        return;
    }

    if (node.type == DENSE_ROW_DIV)
    {
        execute_exe_dense_row_div_node(graph, node);
        return;
    }

    if (node.type == DENSE_FIXED_COL_DIV)
    {
        execute_exe_dense_fixed_col_div_node(graph, node);
        return;
    }

    if (node.type == DENSE_ROW_COARSE_SORT)
    {
        execute_exe_dense_row_coarse_sort_node(graph, node);
        return;
    }

    cout << "execute_node_of_dense_sub_graph: exe_node is not supported" << endl;
    assert(false);
}

void execute_node_of_compressed_sub_graph(exe_graph_t *graph, int sub_graph_index, int exe_node_index_of_sub_graph)
{
    assert(graph != NULL);

    // 获取当前节点
    exe_node_t node = graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph_index].exe_node_vec[exe_node_index_of_sub_graph];

    assert(node.param != NULL);

    // cout << node.type << endl;
    if (node.type == COMPRESSED_ROW_PADDING)
    {
        execute_exe_compress_row_padding_node(graph, node, sub_graph_index);
        return;
    }

    if (node.type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV)
    {
        execute_exe_compress_BLB_row_div_node(graph, node, sub_graph_index);
        return;
    }

    if (node.type == COMPRESSED_TBLOCK_LEVEL_COL_DIV)
    {
        execute_exe_compress_BLB_col_div_node(graph, node, sub_graph_index);
        return;
    }

    if (node.type == COMPRESSED_WARP_LEVEL_ROW_DIV)
    {
        execute_exe_compress_WLB_row_div_node(graph, node, sub_graph_index);
        return;
    }

    if (node.type == COMPRESSED_WARP_LEVEL_COL_DIV)
    {
        execute_exe_compress_WLB_col_div_node(graph, node, sub_graph_index);
        return;
    }

    if (node.type == COMPRESSED_THREAD_LEVEL_ROW_DIV)
    {
        execute_exe_compress_TLB_row_div_node(graph, node, sub_graph_index);
        return;
    }

    if (node.type == COMPRESSED_THREAD_LEVEL_COL_DIV)
    {
        execute_exe_compress_TLB_col_div_node(graph, node, sub_graph_index);
        return;
    }

    if (node.type == COMPRESSED_THREAD_LEVEL_NNZ_DIV)
    {
        execute_exe_compress_thread_level_nnz_div_node(graph, node, sub_graph_index);
        return;
    }

    cout << "execute_node_of_compressed_sub_graph: exe_node is not supported" << endl;
    assert(false);
}

void execute_exe_node_in_compressed_sub_matrix(sparse_struct_t* matrix, int sub_matrix_index, exe_node_t node)
{
    assert(matrix != NULL);
    assert(node.param != NULL);
    assert(sub_matrix_index < matrix->block_coor_table.item_arr.size());

    // 执行对应的
    if (node.type == COMPRESSED_ROW_PADDING)
    {
        execute_exe_compress_row_padding_node(matrix, node, sub_matrix_index);
        return;
    }

    if (node.type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV)
    {
        execute_exe_compress_BLB_row_div_node(matrix, node, sub_matrix_index);
        return;
    }

    if (node.type == COMPRESSED_TBLOCK_LEVEL_COL_DIV)
    {
        execute_exe_compress_BLB_col_div_node(matrix, node, sub_matrix_index);
        return;
    }

    if (node.type == COMPRESSED_WARP_LEVEL_ROW_DIV)
    {
        execute_exe_compress_WLB_row_div_node(matrix, node, sub_matrix_index);
        return;
    }

    if (node.type == COMPRESSED_WARP_LEVEL_COL_DIV)
    {
        execute_exe_compress_WLB_col_div_node(matrix, node, sub_matrix_index);
        return;
    }

    if (node.type == COMPRESSED_THREAD_LEVEL_ROW_DIV)
    {
        execute_exe_compress_TLB_row_div_node(matrix, node, sub_matrix_index);
        return;
    }

    if (node.type == COMPRESSED_THREAD_LEVEL_COL_DIV)
    {
        execute_exe_compress_TLB_col_div_node(matrix, node, sub_matrix_index);
        return;
    }

    if (node.type == COMPRESSED_THREAD_LEVEL_NNZ_DIV)
    {
        execute_exe_compress_thread_level_nnz_div_node(matrix, node, sub_matrix_index);
        return;
    }
    
    cout << "execute_node_of_compressed_sub_graph: exe_node is not supported" << endl;
    assert(false);
}

exe_compressed_sub_graph_t val_copy_from_old_compressed_sub_matrix(exe_compressed_sub_graph_t old_exe_compressed_sub_graph)
{
    // 检查子图的内容是不是规范
    assert(old_exe_compressed_sub_graph.exe_node_vec.size() > 0 && old_exe_compressed_sub_graph.preorder_node_set.size() > 0);

    for (unsigned long i = 0; i < old_exe_compressed_sub_graph.exe_node_vec.size(); i++)
    {
        assert(old_exe_compressed_sub_graph.exe_node_vec[i].param != NULL);
    }

    exe_compressed_sub_graph_t return_sub_compressed_graph;
    
    return_sub_compressed_graph.preorder_node_set = old_exe_compressed_sub_graph.preorder_node_set;

    for (unsigned long i = 0; i < old_exe_compressed_sub_graph.exe_node_vec.size(); i++)
    {
        // 增加一个执行节点
        exe_node_t new_node;
        exe_node_t old_node = old_exe_compressed_sub_graph.exe_node_vec[i];

        // 执行类型的拷贝
        new_node.type = old_node.type;

        if (new_node.type == COMPRESSED_ROW_PADDING)
        {
            new_node.param = new exe_compress_row_padding_param_t();

            // 新的节点参数
            exe_compress_row_padding_param_t* new_exe_param_ptr = (exe_compress_row_padding_param_t*)new_node.param;
            exe_compress_row_padding_param_t* old_exe_param_ptr = (exe_compress_row_padding_param_t*)old_node.param;

            new_exe_param_ptr->multiply = old_exe_param_ptr->multiply;
            new_exe_param_ptr->padding_row_length = old_exe_param_ptr->padding_row_length;
        }
        else if (new_node.type == COMPRESSED_THREAD_LEVEL_ROW_DIV)
        {
            new_node.param = new exe_compress_thread_level_row_div_param_t();

            // 新的节点参数
            exe_compress_thread_level_row_div_param_t* new_exe_param_ptr = (exe_compress_thread_level_row_div_param_t*)new_node.param;
            exe_compress_thread_level_row_div_param_t* old_exe_param_ptr = (exe_compress_thread_level_row_div_param_t*)old_node.param;
        }
        else if (new_node.type == COMPRESSED_THREAD_LEVEL_COL_DIV)
        {
            new_node.param = new exe_compress_thread_level_col_div_param_t();

            // 新的节点参数
            exe_compress_thread_level_col_div_param_t* new_exe_param_ptr = (exe_compress_thread_level_col_div_param_t*)new_node.param;
            exe_compress_thread_level_col_div_param_t* old_exe_param_ptr = (exe_compress_thread_level_col_div_param_t*)old_node.param;

            // 参数拷贝
            new_exe_param_ptr->col_num_of_TLB_in_each_parent_block = old_exe_param_ptr->col_num_of_TLB_in_each_parent_block;
        }
        else if (new_node.type == COMPRESSED_WARP_LEVEL_ROW_DIV)
        {
            new_node.param = new exe_compress_warp_level_row_div_param_t();

            // 新的节点
            exe_compress_warp_level_row_div_param_t* new_exe_param_ptr = (exe_compress_warp_level_row_div_param_t*)new_node.param;
            exe_compress_warp_level_row_div_param_t* old_exe_param_ptr = (exe_compress_warp_level_row_div_param_t*)old_node.param;

            new_exe_param_ptr->row_num_of_each_WLB_in_BLB = old_exe_param_ptr->row_num_of_each_WLB_in_BLB;
        }
        else if (new_node.type == COMPRESSED_WARP_LEVEL_COL_DIV)
        {
            new_node.param = new exe_compress_warp_level_col_div_param_t();

            // 新的节点
            exe_compress_warp_level_col_div_param_t* new_exe_param_ptr = (exe_compress_warp_level_col_div_param_t*)new_node.param;
            exe_compress_warp_level_col_div_param_t* old_exe_param_ptr = (exe_compress_warp_level_col_div_param_t*)old_node.param;

            new_exe_param_ptr->col_num_of_WLB_in_each_parent_row_block_or_BLB = old_exe_param_ptr->col_num_of_WLB_in_each_parent_row_block_or_BLB;
        }
        else if (new_node.type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV)
        {
            new_node.param = new exe_compress_tblock_level_row_div_param_t();

            // 新的节点
            exe_compress_tblock_level_row_div_param_t* new_exe_param_ptr = (exe_compress_tblock_level_row_div_param_t*)new_node.param;
            exe_compress_tblock_level_row_div_param_t* old_exe_param_ptr = (exe_compress_tblock_level_row_div_param_t*)old_node.param;

            new_exe_param_ptr->row_num_of_each_BLB = old_exe_param_ptr->row_num_of_each_BLB;
        }
        else if (new_node.type == COMPRESSED_TBLOCK_LEVEL_COL_DIV)
        {
            new_node.param = new exe_compress_tblock_level_col_div_param_t();

            // 新的节点
            exe_compress_tblock_level_col_div_param_t* new_exe_param_ptr = (exe_compress_tblock_level_col_div_param_t*)new_node.param;
            exe_compress_tblock_level_col_div_param_t* old_exe_param_ptr = (exe_compress_tblock_level_col_div_param_t*)old_node.param;

            new_exe_param_ptr->col_block_nnz_num_of_each_BLB = old_exe_param_ptr->col_block_nnz_num_of_each_BLB;
        }
        else if (new_node.type == COMPRESSED_THREAD_LEVEL_NNZ_DIV)
        {
            new_node.param = new exe_compress_thread_level_nnz_div_param_t();

            // 新的节点
            exe_compress_thread_level_nnz_div_param_t* new_exe_param_ptr = (exe_compress_thread_level_nnz_div_param_t*)new_node.param;
            exe_compress_thread_level_nnz_div_param_t* old_exe_param_ptr = (exe_compress_thread_level_nnz_div_param_t*)old_node.param;

            new_exe_param_ptr->TLB_nnz_num = old_exe_param_ptr->TLB_nnz_num;
        }
        else
        {
            cout << "val_copy_from_old_compressed_sub_matrix: node type is not supported" << endl;
            assert(false);
        }

        // 将新的节点放到当前数组中
        return_sub_compressed_graph.exe_node_vec.push_back(new_node);
    }

    // 搞定之后新的子图检查一下
    assert(return_sub_compressed_graph.exe_node_vec.size() > 0 && return_sub_compressed_graph.preorder_node_set.size() > 0);
    assert(return_sub_compressed_graph.exe_node_vec.size() == old_exe_compressed_sub_graph.exe_node_vec.size());

    for (unsigned long i = 0; i < return_sub_compressed_graph.exe_node_vec.size(); i++)
    {
        assert(return_sub_compressed_graph.exe_node_vec[i].param != NULL);
    }

    return return_sub_compressed_graph;
}

exe_dense_sub_graph_t val_copy_from_old_dense_sub_matrix(exe_dense_sub_graph_t old_exe_dense_sub_graph)
{
    // 检查子图的内容是不是规范
    assert(old_exe_dense_sub_graph.exe_node_vec.size() > 0 && old_exe_dense_sub_graph.preorder_node_set.size() > 0);

    for (unsigned long i = 0; i < old_exe_dense_sub_graph.exe_node_vec.size(); i++)
    {
        assert(old_exe_dense_sub_graph.exe_node_vec[i].param != NULL);
    }

    exe_dense_sub_graph_t return_sub_dense_graph;

    return_sub_dense_graph.preorder_node_set = old_exe_dense_sub_graph.preorder_node_set;

    for (unsigned long i = 0; i < old_exe_dense_sub_graph.exe_node_vec.size(); i++)
    {
        // 增加一个执行节点
        exe_node_t new_node;
        exe_node_t old_node = old_exe_dense_sub_graph.exe_node_vec[i];

        new_node.type = old_node.type;

        if (new_node.type == BEGIN_MEMORY_CACHE_INPUT_FILE)
        {
            new_node.param = new exe_begin_memory_cache_input_file_param_t();

            // 新旧节点的参数
            exe_begin_memory_cache_input_file_param_t* new_param_ptr = (exe_begin_memory_cache_input_file_param_t*)new_node.param;
            exe_begin_memory_cache_input_file_param_t* old_param_ptr = (exe_begin_memory_cache_input_file_param_t*)old_node.param;

            new_param_ptr->col_index_cache = old_param_ptr->col_index_cache;
            new_param_ptr->col_index_max = old_param_ptr->col_index_max;
            new_param_ptr->double_val_cache = old_param_ptr->double_val_cache;
            new_param_ptr->float_val_cache = old_param_ptr->float_val_cache;
            new_param_ptr->row_index_cache = old_param_ptr->row_index_cache;
            new_param_ptr->row_index_max = old_param_ptr->row_index_max;
            new_param_ptr->val_data_type = old_param_ptr->val_data_type;
        }
        else if (new_node.type == DENSE_ROW_COARSE_SORT)
        {
            new_node.param = new exe_dense_row_coarse_sort_param_t();

            // 新旧节点的参数
            exe_dense_row_coarse_sort_param_t* new_param_ptr = (exe_dense_row_coarse_sort_param_t*)new_node.param;
            exe_dense_row_coarse_sort_param_t* old_param_ptr = (exe_dense_row_coarse_sort_param_t*)old_node.param;

            new_param_ptr->bin_row_nnz_low_bound = old_param_ptr->bin_row_nnz_low_bound;
        }
        else if (new_node.type == DENSE_ROW_DIV)
        {
            new_node.param = new exe_dense_row_div_param_t();

            // 新旧节点参数
            exe_dense_row_div_param_t* new_param_ptr = (exe_dense_row_div_param_t*)new_node.param;
            exe_dense_row_div_param_t* old_param_ptr = (exe_dense_row_div_param_t*)old_node.param;

            new_param_ptr->dense_sub_block_id = old_param_ptr->dense_sub_block_id;
            new_param_ptr->row_div_position = old_param_ptr->row_div_position;
        }
        else if (new_node.type == COMPRESS)
        {
            new_node.param = new exe_compress_param_t();

            // 新旧节点参数
            exe_compress_param_t* new_param_ptr = (exe_compress_param_t*)new_node.param;
            exe_compress_param_t* old_param_ptr = (exe_compress_param_t*)old_node.param;
        }
        else
        {
            cout << "val_copy_from_old_dense_sub_matrix: node type is not supported" << endl;
            assert(false);
        }

        // 将新的节点放到稠密子图中
        return_sub_dense_graph.exe_node_vec.push_back(new_node);
    }

    // 搞定之后新的子图检查一下
    assert(return_sub_dense_graph.exe_node_vec.size() > 0 && return_sub_dense_graph.preorder_node_set.size() > 0);
    assert(return_sub_dense_graph.exe_node_vec.size() == old_exe_dense_sub_graph.exe_node_vec.size());

    for (unsigned long i = 0; i < return_sub_dense_graph.exe_node_vec.size(); i++)
    {
        assert(return_sub_dense_graph.exe_node_vec[i].param != NULL);
    }

    return return_sub_dense_graph;
}

void del_param_of_exe_node(exe_node_t *node)
{
    assert(node != NULL && node->param != NULL);

    if (node->type == BEGIN_INPUT_FILE)
    {
        delete (exe_begin_input_file_param_t *)node->param;
        node->param = NULL;
        return;
    }

    if (node->type == BEGIN_ARTIFICIAL_INPUT)
    {
        delete (exe_begin_artificial_input_param_t *)node->param;
        node->param = NULL;
        return;
    }

    // 析构内存输入节点
    if (node->type == BEGIN_MEMORY_CACHE_INPUT_FILE)
    {
        delete (exe_begin_memory_cache_input_file_param_t *)node->param;
        node->param = NULL;
        return;
    }

    if (node->type == DENSE_TOTAL_ROW_LEVEL_PADDING)
    {
        delete (exe_row_level_padding_node_param_t *)node->param;
        node->param = NULL;
        return;
    }

    if (node->type == DENSE_FIXED_COL_DIV)
    {
        delete (exe_dense_fixed_col_div_param_t *)node->param;
        node->param = NULL;
        return;
    }

    if (node->type == DENSE_ROW_DIV)
    {
        delete (exe_dense_row_div_param_t *)node->param;
        node->param = NULL;
        return;
    }

    if (node->type == DENSE_ROW_COARSE_SORT)
    {
        delete (exe_dense_row_coarse_sort_param_t *)node->param;
        node->param = NULL;
        return;
    }

    if (node->type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV)
    {
        delete (exe_compress_tblock_level_row_div_param_t *)node->param;
        node->param = NULL;
        return;
    }

    if (node->type == COMPRESS)
    {
        delete (exe_compress_param_t *)node->param;
        node->param = NULL;
        return;
    }

    if (node->type == COMPRESSED_ROW_PADDING)
    {
        delete (exe_compress_row_padding_param_t *)node->param;
        node->param = NULL;
        return;
    }

    if (node->type == COMPRESSED_TBLOCK_LEVEL_COL_DIV)
    {
        delete (exe_compress_tblock_level_col_div_param_t *)node->param;
        node->param = NULL;
        return;
    }

    if (node->type == COMPRESSED_WARP_LEVEL_ROW_DIV)
    {
        delete (exe_compress_warp_level_row_div_param_t *)node->param;
        node->param = NULL;
        return;
    }

    if (node->type == COMPRESSED_WARP_LEVEL_COL_DIV)
    {
        delete (exe_compress_warp_level_col_div_param_t *)node->param;
        node->param = NULL;
        return;
    }

    if (node->type == COMPRESSED_THREAD_LEVEL_ROW_DIV)
    {
        delete (exe_compress_thread_level_row_div_param_t *)node->param;
        node->param = NULL;
        return;
    }

    if (node->type == COMPRESSED_THREAD_LEVEL_COL_DIV)
    {
        delete (exe_compress_thread_level_col_div_param_t *)node->param;
        node->param = NULL;
        return;
    }

    if (node->type == COMPRESSED_THREAD_LEVEL_NNZ_DIV)
    {
        delete (exe_compress_thread_level_nnz_div_param_t *)node->param;
        node->param = NULL;
        return;
    }

    cout << "del_param_of_exe_node: exe_node_type is not supported" << endl;
    assert(false);
}

// void del_exe_node_param_of_compress_sub_matrix(exe_compressed_sub_graph_t compress_sub_graph)
// {
//     // assert(compress_sub_graph.exe_node_vec.size() == compress_sub_graph.preorder_node_set.size());
    
//     for (unsigned int i = 0; i < compress_sub_graph.exe_node_vec.size(); i++)
//     {
//         assert(compress_sub_graph.exe_node_vec[i].param != NULL);
//         del_param_of_exe_node(&(compress_sub_graph.exe_node_vec[i]));
//     }
// }

void del_exe_node_param_of_compress_sub_matrix(exe_compressed_sub_graph_t *compress_sub_graph)
{
    assert(compress_sub_graph != NULL);
    for (unsigned int i = 0; i < compress_sub_graph->exe_node_vec.size(); i++)
    {
        assert(compress_sub_graph->exe_node_vec[i].param != NULL);
        del_param_of_exe_node(&(compress_sub_graph->exe_node_vec[i]));
        assert(compress_sub_graph->exe_node_vec[i].param == NULL);
    }
}

void del_exe_node_param_of_dense_view_matrix(exe_dense_sub_graph_t* dense_sub_graph)
{
    assert(dense_sub_graph != NULL);
    for (unsigned int i = 0; i < dense_sub_graph->exe_node_vec.size(); i++)
    {
        assert(dense_sub_graph->exe_node_vec[i].param != NULL);
        del_param_of_exe_node(&(dense_sub_graph->exe_node_vec[i]));
        assert(dense_sub_graph->exe_node_vec[i].param == NULL);
    }
}

void del_param_of_template_node(template_node_t* node)
{
    assert(node != NULL && node->template_param != NULL);

    // 如果参数不存在，就不需要析构，这种情况一般出现在搜索过程一开始的时候，或者出现整个模板全部运行失败的时候
    // if (node->template_param == NULL)
    // {
    //     cout << "param of template is empty, don't need to delete" << endl;
    //     return;
    // }

    cout << "del_param_of_template_node: del template type:" << convert_template_type_to_string(node->type) << endl;

    if (node->type == DIRECT_ATOM_TEMPLATE)
    {
        delete (direct_atom_template_node_param_t *)node->template_param;
        node->template_param = NULL;
        return;
    }

    if (node->type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
    {
        delete (direct_atom_template_warp_compress_node_param_t *)node->template_param;
        node->template_param = NULL;
        return;
    }

    if (node->type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
    {
        delete (direct_atom_template_warp_block_compress_node_param_t *)node->template_param;
        node->template_param = NULL;
        return;
    }

    if (node->type == SHARED_MEMORY_TEMPLATE)
    {
        delete (shared_memory_template_node_param_t *)node->template_param;
        node->template_param = NULL;
        return;
    }

    if (node->type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
    {
        delete (shared_memory_template_warp_compress_node_param_t *)node->template_param;
        node->template_param = NULL;
        return;
    }

    if (node->type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
    {
        delete (shared_memory_long_row_template_node_param_t *)node->template_param;
        node->template_param = NULL;
        return;
    }

    if (node->type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        delete (shared_memory_total_warp_reduce_template_node_param_t *)node->template_param;
        node->template_param = NULL;
        return;
    }

    if (node->type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        delete (direct_atom_total_warp_reduce_template_node_param_t *)node->template_param;
        node->template_param = NULL;
        return;
    }

    if (node->type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
    {
        delete (unaligned_warp_reduce_same_TLB_size_template_node_param_t *)node->template_param;
        node->template_param = NULL;
        return;
    }

    if (node->type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
    {
        delete (unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_node_param_t *)node->template_param;
        node->template_param = NULL;
        return;
    }

    cout << "del_param_of_exe_node: exe_node_type is not supported" << endl;
    assert(false);
}

void print_template_node(template_node_t* node)
{
    assert(node != NULL && node->template_param != NULL);

    cout << convert_template_node_to_string(node);
}


string convert_template_node_to_string(template_node_t* node)
{
    assert(node != NULL && node->template_param != NULL);

    #if IDEAL_OUTPUT == 1
    // 返回值
    string return_str = "";
    #elif
    string return_str = "template_node:{";
    #endif


    if (node->type == DIRECT_ATOM_TEMPLATE)
    {
        direct_atom_template_node_param_t* param_ptr = (direct_atom_template_node_param_t *)node->template_param;
        #if IDEAL_OUTPUT == 1
        // 对于这个模板来说，使用的是THREAD_TOTAL_RED和GBL_ATOM_RED
        return_str = return_str + "kernel_node_type:THREAD_TOTAL_RED" + "\n";
        return_str = return_str + "kernel_node_type:GBL_ATOM_RED" + "\n";
        return_str = return_str + "kernel_node_type:SET_RESOURCE" + "\n";
        return_str = return_str + "{\n";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + "\n";
        return_str = return_str + "tblock_num:" + to_string(param_ptr->tblock_num) + "\n";
        return_str = return_str + "}\n";
        
        #elif

        return_str = return_str + convert_template_type_to_string(node->type) + " , ";
        // 打印这个模板的参数
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + ", tblock_num:" + to_string(param_ptr->tblock_num);

        #endif
    }
    else if (node->type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
    {
        direct_atom_template_warp_compress_node_param_t* param_ptr = (direct_atom_template_warp_compress_node_param_t *)node->template_param;

        #if IDEAL_OUTPUT == 1
        // 对于这个模板来说，使用的是THREAD_TOTAL_RED和GBL_ATOM_RED
        return_str = return_str + "kernel_node_type:THREAD_TOTAL_RED" + "\n";
        return_str = return_str + "kernel_node_type:GBL_ATOM_RED" + "\n";
        return_str = return_str + "kernel_node_type:SET_RESOURCE" + "\n";
        return_str = return_str + "{\n";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + "\n";
        return_str = return_str + "tblock_num:" + to_string(param_ptr->tblock_num) + "\n";
        return_str = return_str + "}\n";
        
        #elif
        return_str = return_str + convert_template_type_to_string(node->type) + " , ";
        // 打印模板参数
        return_str = return_str +  "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + ", tblock_num:" + to_string(param_ptr->tblock_num);
        #endif
        
    }
    else if (node->type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
    {
        direct_atom_template_warp_block_compress_node_param_t* param_ptr = (direct_atom_template_warp_block_compress_node_param_t *)node->template_param;

        #if IDEAL_OUTPUT == 1
        // 对于这个模板来说，使用的是THREAD_TOTAL_RED和GBL_ATOM_RED
        return_str = return_str + "kernel_node_type:THREAD_TOTAL_RED" + "\n";
        return_str = return_str + "kernel_node_type:GBL_ATOM_RED" + "\n";
        return_str = return_str + "kernel_node_type:SET_RESOURCE" + "\n";
        return_str = return_str + "{\n";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + "\n";
        return_str = return_str + "tblock_num:" + to_string(param_ptr->tblock_num) + "\n";
        return_str = return_str + "}\n";
        
        #elif

        return_str = return_str + convert_template_type_to_string(node->type) + " , ";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + ", tblock_num:" + to_string(param_ptr->tblock_num);
        #endif
    }
    else if (node->type == SHARED_MEMORY_TEMPLATE)
    {
        shared_memory_template_node_param_t* param_ptr = (shared_memory_template_node_param_t *)node->template_param;

        #if IDEAL_OUTPUT == 1

        return_str = return_str + "kernel_node_type:THREAD_TOTAL_RED" + "\n";
        return_str = return_str + "kernel_node_type:SHARED_MEM_OFFSET_RED" + "\n";
        return_str = return_str + "{\n";
        return_str = return_str + "thread_num_of_row_reduce:" + to_string(param_ptr->thread_num_of_row_reduce) + "\n";
        return_str = return_str + "hybrid_reduce:0\n";
        return_str = return_str + "}\n";
        return_str = return_str + "kernel_node_type:SET_RESOURCE" + "\n";
        return_str = return_str + "{\n";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + "\n";
        return_str = return_str + "tblock_num:" + to_string(param_ptr->tblock_num) + "\n";
        return_str = return_str + "}\n";

        #elif
        return_str = return_str + convert_template_type_to_string(node->type) + " , ";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + ", tblock_num:" + to_string(param_ptr->tblock_num) + ", thread_num_of_row_reduce:" + to_string(param_ptr->thread_num_of_row_reduce);
        #endif
    }
    else if (node->type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
    {
        shared_memory_template_warp_compress_node_param_t* param_ptr = (shared_memory_template_warp_compress_node_param_t *)node->template_param;

        #if IDEAL_OUTPUT == 1

        return_str = return_str + "kernel_node_type:THREAD_TOTAL_RED" + "\n";
        return_str = return_str + "kernel_node_type:SHARED_MEM_OFFSET_RED" + "\n";
        return_str = return_str + "{\n";
        return_str = return_str + "thread_num_of_row_reduce:" + to_string(param_ptr->thread_num_of_row_reduce) + "\n";
        return_str = return_str + "hybrid_reduce:0\n";
        return_str = return_str + "}\n";
        return_str = return_str + "kernel_node_type:SET_RESOURCE" + "\n";
        return_str = return_str + "{\n";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + "\n";
        return_str = return_str + "tblock_num:" + to_string(param_ptr->tblock_num) + "\n";
        return_str = return_str + "}\n";
        
        #elif
        return_str = return_str + convert_template_type_to_string(node->type) + " , ";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + ", tblock_num:" + to_string(param_ptr->tblock_num) + ", thread_num_of_row_reduce:" + to_string(param_ptr->thread_num_of_row_reduce);
        #endif
    }
    else if (node->type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
    {
        shared_memory_long_row_template_node_param_t* param_ptr = (shared_memory_long_row_template_node_param_t *)node->template_param;

        #if IDEAL_OUTPUT == 1

        return_str = return_str + "kernel_node_type:SHARED_MEM_TOTAL_RED" + "\n";
        return_str = return_str + "kernel_node_type:GBL_ATOM_RED" + "\n";
        return_str = return_str + "kernel_node_type:SET_RESOURCE" + "\n";
        return_str = return_str + "{\n";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + "\n";
        return_str = return_str + "tblock_num:" + to_string(param_ptr->tblock_num) + "\n";
        return_str = return_str + "}\n";
        
        #elif
        return_str = return_str + convert_template_type_to_string(node->type) + " , ";        
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + ", tblock_num:" + to_string(param_ptr->tblock_num);
        #endif
    }
    else if (node->type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        shared_memory_total_warp_reduce_template_node_param_t* param_ptr = (shared_memory_total_warp_reduce_template_node_param_t*)node->template_param;

        #if IDEAL_OUTPUT == 1

        return_str = return_str + "kernel_node_type:WARP_TOTAL_RED" + "\n";
        return_str = return_str + "kernel_node_type:SHARED_MEM_OFFSET_RED" + "\n";
        return_str = return_str + "{\n";
        return_str = return_str + "thread_num_of_row_reduce:" + to_string(param_ptr->thread_num_of_row_reduce) + "\n";
        return_str = return_str + "hybrid_reduce:0\n";
        return_str = return_str + "}\n";
        return_str = return_str + "kernel_node_type:SET_RESOURCE" + "\n";
        return_str = return_str + "{\n";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + "\n";
        return_str = return_str + "tblock_num:" + to_string(param_ptr->tblock_num) + "\n";
        return_str = return_str + "}\n";

        #elif
        return_str = return_str + convert_template_type_to_string(node->type) + " , ";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + ", tblock_num:" + to_string(param_ptr->tblock_num) + ", thread_num_of_row_reduce:" + to_string(param_ptr->thread_num_of_row_reduce);
        #endif
    }
    else if (node->type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        direct_atom_total_warp_reduce_template_node_param_t* param_ptr = (direct_atom_total_warp_reduce_template_node_param_t *)node->template_param;

        #if IDEAL_OUTPUT == 1

        return_str = return_str + "kernel_node_type:WARP_TOTAL_RED" + "\n";
        return_str = return_str + "kernel_node_type:GBL_ATOM_RED" + "\n";
        return_str = return_str + "kernel_node_type:SET_RESOURCE" + "\n";
        return_str = return_str + "{\n";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + "\n";
        return_str = return_str + "tblock_num:" + to_string(param_ptr->tblock_num) + "\n";
        return_str = return_str + "}\n";

        #elif
        return_str = return_str + convert_template_type_to_string(node->type) + " , ";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + ", tblock_num:" + to_string(param_ptr->tblock_num);
        #endif
    }
    else if (node->type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
    {
        unaligned_warp_reduce_same_TLB_size_template_node_param_t* param_ptr = (unaligned_warp_reduce_same_TLB_size_template_node_param_t *)node->template_param;

        #if IDEAL_OUTPUT == 1

        return_str = return_str + "kernel_node_type:THREAD_BITMAP_RED" + "\n";
        return_str = return_str + "kernel_node_type:WARP_SEG_ADD_RED" + "\n";
        return_str = return_str + "{\n";
        return_str = return_str + "hybrid_reduce:0\n";
        return_str = return_str + "}\n";
        return_str = return_str + "kernel_node_type:SET_RESOURCE" + "\n";
        return_str = return_str + "{\n";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + "\n";
        return_str = return_str + "tblock_num:" + to_string(param_ptr->tblock_num) + "\n";
        return_str = return_str + "}\n";
        
        #elif
        return_str = return_str + convert_template_type_to_string(node->type) + " , ";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + ", tblock_num:" + to_string(param_ptr->tblock_num);
        #endif
    }
    else if (node->type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
    {
        unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_node_param_t* param_ptr = (unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_node_param_t *)node->template_param;

        #if IDEAL_OUTPUT == 1

        return_str = return_str + "kernel_node_type:THREAD_BITMAP_RED" + "\n";
        return_str = return_str + "kernel_node_type:WARP_SEG_ADD_RED" + "\n";
        return_str = return_str + "{\n";
        return_str = return_str + "hybrid_reduce:1\n";
        return_str = return_str + "}\n";
        return_str = return_str + "kernel_node_type:SET_RESOURCE" + "\n";
        return_str = return_str + "{\n";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block) + "\n";
        return_str = return_str + "}\n";

        #elif
        return_str = return_str + convert_template_type_to_string(node->type) + " , ";
        return_str = return_str + "thread_num_in_block:" + to_string(param_ptr->thread_num_in_block);
        #endif
    }
    else
    {
        cout << "print_template_node: exe_node_type is not supported" << endl;
        assert(false);
    }

    return_str = return_str + "}\n";

    return return_str;
}

template_node_t val_copy_from_old_template_node(template_node_t old_template)
{
    assert(old_template.template_param != NULL);

    // 值拷贝，主要是new新的参数，并且拷贝一份新的
    template_node_t return_template_node;

    // 拷贝模板类型
    return_template_node.type = old_template.type;

    // 不用类型的拷贝方法不一样
    if (return_template_node.type == DIRECT_ATOM_TEMPLATE)
    {
        // 申请一个新的参数
        return_template_node.template_param = new direct_atom_template_node_param_t();

        direct_atom_template_node_param_t* old_template_param_ptr = (direct_atom_template_node_param_t *)old_template.template_param;
        direct_atom_template_node_param_t* new_template_param_ptr = (direct_atom_template_node_param_t *)return_template_node.template_param;

        // 执行值拷贝
        new_template_param_ptr->tblock_num = old_template_param_ptr->tblock_num;
        new_template_param_ptr->thread_num_in_block = old_template_param_ptr->thread_num_in_block;
    }
    else if (return_template_node.type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
    {
        // 申请一个参数
        return_template_node.template_param = new direct_atom_template_warp_compress_node_param_t();

        direct_atom_template_warp_compress_node_param_t* old_template_param_ptr = (direct_atom_template_warp_compress_node_param_t *)old_template.template_param;
        direct_atom_template_warp_compress_node_param_t* new_template_param_ptr = (direct_atom_template_warp_compress_node_param_t *)return_template_node.template_param;

        new_template_param_ptr->tblock_num = old_template_param_ptr->tblock_num;
        new_template_param_ptr->thread_num_in_block = old_template_param_ptr->thread_num_in_block;
    }
    else if (return_template_node.type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
    {
        // 申请一个参数
        return_template_node.template_param = new direct_atom_template_warp_block_compress_node_param_t();

        direct_atom_template_warp_block_compress_node_param_t* old_template_param_ptr = (direct_atom_template_warp_block_compress_node_param_t *)old_template.template_param;
        direct_atom_template_warp_block_compress_node_param_t* new_template_param_ptr = (direct_atom_template_warp_block_compress_node_param_t *)return_template_node.template_param;

        new_template_param_ptr->tblock_num = old_template_param_ptr->tblock_num;
        new_template_param_ptr->thread_num_in_block = old_template_param_ptr->thread_num_in_block;
    }
    else if (return_template_node.type == SHARED_MEMORY_TEMPLATE)
    {
        // 申请一个参数
        return_template_node.template_param = new shared_memory_template_node_param_t();

        shared_memory_template_node_param_t* old_template_param_ptr = (shared_memory_template_node_param_t *)old_template.template_param;
        shared_memory_template_node_param_t* new_template_param_ptr = (shared_memory_template_node_param_t *)return_template_node.template_param;

        new_template_param_ptr->tblock_num = old_template_param_ptr->tblock_num;
        new_template_param_ptr->thread_num_in_block = old_template_param_ptr->thread_num_in_block;
        new_template_param_ptr->thread_num_of_row_reduce = old_template_param_ptr->thread_num_of_row_reduce;
    }
    else if (return_template_node.type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
    {
        // 申请一个参数
        return_template_node.template_param = new shared_memory_template_warp_compress_node_param_t();
        
        shared_memory_template_warp_compress_node_param_t* old_template_param_ptr = (shared_memory_template_warp_compress_node_param_t *)old_template.template_param;
        shared_memory_template_warp_compress_node_param_t* new_template_param_ptr = (shared_memory_template_warp_compress_node_param_t *)return_template_node.template_param;

        new_template_param_ptr->tblock_num = old_template_param_ptr->tblock_num;
        new_template_param_ptr->thread_num_in_block = old_template_param_ptr->thread_num_in_block;
        new_template_param_ptr->thread_num_of_row_reduce = old_template_param_ptr->thread_num_of_row_reduce;
    }
    else if (return_template_node.type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
    {
        // 申请一个参数
        return_template_node.template_param = new shared_memory_long_row_template_node_param_t();

        shared_memory_long_row_template_node_param_t* old_template_param_ptr = (shared_memory_long_row_template_node_param_t *)old_template.template_param;
        shared_memory_long_row_template_node_param_t* new_template_param_ptr = (shared_memory_long_row_template_node_param_t *)return_template_node.template_param;

        new_template_param_ptr->tblock_num = old_template_param_ptr->tblock_num;
        new_template_param_ptr->thread_num_in_block = old_template_param_ptr->thread_num_in_block;
    }
    else if (return_template_node.type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        // 申请一个参数
        return_template_node.template_param = new shared_memory_total_warp_reduce_template_node_param_t();

        shared_memory_total_warp_reduce_template_node_param_t* old_template_param_ptr = (shared_memory_total_warp_reduce_template_node_param_t *)old_template.template_param;
        shared_memory_total_warp_reduce_template_node_param_t* new_template_param_ptr = (shared_memory_total_warp_reduce_template_node_param_t *)return_template_node.template_param;

        new_template_param_ptr->tblock_num = old_template_param_ptr->tblock_num;
        new_template_param_ptr->thread_num_in_block = old_template_param_ptr->thread_num_in_block;
        new_template_param_ptr->thread_num_of_row_reduce = old_template_param_ptr->thread_num_of_row_reduce;
    }
    else if (return_template_node.type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        // 申请一个参数
        return_template_node.template_param = new direct_atom_total_warp_reduce_template_node_param_t();

        direct_atom_total_warp_reduce_template_node_param_t* old_template_param_ptr = (direct_atom_total_warp_reduce_template_node_param_t *)old_template.template_param;
        direct_atom_total_warp_reduce_template_node_param_t* new_template_param_ptr = (direct_atom_total_warp_reduce_template_node_param_t *)return_template_node.template_param;

        new_template_param_ptr->tblock_num = old_template_param_ptr->tblock_num;
        new_template_param_ptr->thread_num_in_block = old_template_param_ptr->thread_num_in_block;
    }
    else if (return_template_node.type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
    {
        // 申请了一个参数
        return_template_node.template_param = new unaligned_warp_reduce_same_TLB_size_template_node_param_t();

        unaligned_warp_reduce_same_TLB_size_template_node_param_t* old_template_param_ptr = (unaligned_warp_reduce_same_TLB_size_template_node_param_t *)old_template.template_param;
        unaligned_warp_reduce_same_TLB_size_template_node_param_t* new_template_param_ptr = (unaligned_warp_reduce_same_TLB_size_template_node_param_t *)return_template_node.template_param;

        new_template_param_ptr->tblock_num = old_template_param_ptr->tblock_num;
        new_template_param_ptr->thread_num_in_block = old_template_param_ptr->thread_num_in_block;
    }
    else if (return_template_node.type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
    {
        // 申请一个参数
        return_template_node.template_param = new unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_node_param_t();
        
        unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_node_param_t* old_template_param_ptr = (unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_node_param_t *)old_template.template_param;
        unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_node_param_t* new_template_param_ptr = (unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_node_param_t *)return_template_node.template_param;

        new_template_param_ptr->thread_num_in_block = old_template_param_ptr->thread_num_in_block;
    }
    else
    {
        // 不支持这个模板类型
        cout << "val_copy_from_old_template_node: this template type is not supported" << endl;
        assert(false);
    }
    
    // 返回
    assert(return_template_node.template_param != NULL);

    return return_template_node;
}

void execute_template_node_and_update_code_builder(code_builder_t* builder, unsigned long sub_matrix_id, template_node_t node)
{
    assert(builder != NULL && builder->op_manager != NULL && builder->op_manager->matrix != NULL);
    assert(node.template_param != NULL);
    
    sparse_struct_t* matrix = builder->op_manager->matrix;

    // 子块满足要求
    assert(sub_matrix_id < matrix->block_coor_table.item_arr.size());

    // 对应的子矩阵已经完成压缩
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);
    // 现在的对应的builder的对应位置还是空的
    assert(builder->template_type_vec[sub_matrix_id] == NONE_TEMPLATE);
    assert(builder->template_vec[sub_matrix_id] == NULL);

    // 根据节点的类型，执行模板的生成、模板的添加、模板的压缩
    if (node.type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
    {
        direct_atom_template_warp_block_compress_t* new_template = init_direct_atom_template_warp_block_compress(builder, sub_matrix_id);
        assert(new_template != NULL);
        add_template_to_builder(builder, new_template, node.type, sub_matrix_id);

        // 参数
        direct_atom_template_warp_block_compress_node_param_t* node_param_ptr = (direct_atom_template_warp_block_compress_node_param_t*)node.template_param;
        new_template->thread_num_in_block = node_param_ptr->thread_num_in_block;
        new_template->tblock_num = node_param_ptr->tblock_num;

        try_all_compress(new_template);
    }
    else if (node.type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
    {
        direct_atom_template_warp_compress_t* new_template = init_direct_atom_template_warp_compress(builder, sub_matrix_id);
        assert(new_template != NULL);
        add_template_to_builder(builder, new_template, node.type, sub_matrix_id);

        // 参数
        direct_atom_template_warp_compress_node_param_t* node_param_ptr = (direct_atom_template_warp_compress_node_param_t*)node.template_param;
        new_template->thread_num_in_block = node_param_ptr->thread_num_in_block;
        new_template->tblock_num = node_param_ptr->tblock_num;

        try_all_compress(new_template);
    }
    else if (node.type == DIRECT_ATOM_TEMPLATE)
    {
        direct_atom_template_t* new_template = init_direct_atom_template(builder, sub_matrix_id);
        assert(new_template != NULL);
        add_template_to_builder(builder, new_template, node.type, sub_matrix_id);

        // 参数
        direct_atom_template_node_param_t* node_param_ptr = (direct_atom_template_node_param_t*)node.template_param;
        new_template->thread_num_in_block = node_param_ptr->thread_num_in_block;
        new_template->tblock_num = node_param_ptr->tblock_num;
        
        try_all_compress(new_template);
    }
    else if (node.type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        direct_atom_total_warp_reduce_template_t* new_template = init_direct_atom_total_warp_reduce_template(builder, sub_matrix_id);
        assert(new_template != NULL);
        add_template_to_builder(builder, new_template, node.type, sub_matrix_id);

        // 参数
        direct_atom_total_warp_reduce_template_node_param_t* node_param_ptr = (direct_atom_total_warp_reduce_template_node_param_t*)node.template_param;
        new_template->thread_num_in_block = node_param_ptr->thread_num_in_block;
        new_template->tblock_num = node_param_ptr->tblock_num;

        try_all_compress(new_template);
    }
    else if (node.type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
    {
        shared_memory_long_row_template_t* new_template = init_shared_memory_long_row_template(builder, sub_matrix_id);
        assert(new_template != NULL);
        add_template_to_builder(builder, new_template, node.type, sub_matrix_id);
        
        // 参数
        shared_memory_long_row_template_node_param_t* node_param_ptr = (shared_memory_long_row_template_node_param_t*)node.template_param;
        new_template->thread_num_in_block = node_param_ptr->thread_num_in_block;
        new_template->tblock_num = node_param_ptr->tblock_num;

        try_all_compress(new_template);
    }
    else if (node.type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
    {
        shared_memory_template_warp_compress_t* new_template = init_shared_memory_template_warp_compress(builder, sub_matrix_id);
        assert(new_template != NULL);
        add_template_to_builder(builder, new_template, node.type, sub_matrix_id);

        // 参数
        shared_memory_template_warp_compress_node_param_t* node_param_ptr = (shared_memory_template_warp_compress_node_param_t*)node.template_param;
        new_template->tblock_num = node_param_ptr->tblock_num;
        new_template->thread_num_in_block = node_param_ptr->thread_num_in_block;
        new_template->thread_num_of_row_reduce = node_param_ptr->thread_num_of_row_reduce;

        try_all_compress(new_template);
    }
    else if (node.type == SHARED_MEMORY_TEMPLATE)
    {
        shared_memory_template_t* new_template = init_shared_memory_template(builder, sub_matrix_id);
        assert(new_template != NULL);
        add_template_to_builder(builder, new_template, node.type, sub_matrix_id);

        // 参数
        shared_memory_template_node_param_t* node_param_ptr = (shared_memory_template_node_param_t*)node.template_param;
        new_template->thread_num_in_block = node_param_ptr->thread_num_in_block;
        new_template->tblock_num = node_param_ptr->tblock_num;
        new_template->thread_num_of_row_reduce = node_param_ptr->thread_num_of_row_reduce;

        try_all_compress(new_template);
    }
    else if (node.type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        shared_memory_total_warp_reduce_template_t* new_template = init_shared_memory_total_warp_reduce_template(builder, sub_matrix_id);
        assert(new_template != NULL);
        add_template_to_builder(builder, new_template, node.type, sub_matrix_id);

        // 参数
        shared_memory_total_warp_reduce_template_node_param_t* node_param_ptr = (shared_memory_total_warp_reduce_template_node_param_t*)node.template_param;
        new_template->tblock_num = node_param_ptr->tblock_num;
        new_template->thread_num_in_block = node_param_ptr->thread_num_in_block;
        new_template->thread_num_of_row_reduce = node_param_ptr->thread_num_of_row_reduce;

        try_all_compress(new_template);
    }
    else if (node.type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
    {
        unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t* new_template = init_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce(builder, sub_matrix_id);
        assert(new_template != NULL);
        add_template_to_builder(builder, new_template, node.type, sub_matrix_id);

        // 参数
        unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_node_param_t* node_param_ptr = (unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_node_param_t*)node.template_param;
        new_template->thread_num_in_block = node_param_ptr->thread_num_in_block;

        try_all_compress(new_template);
    }
    else if (node.type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
    {
        unaligned_warp_reduce_same_TLB_size_template_t* new_template = init_unaligned_warp_reduce_same_TLB_size_template(builder, sub_matrix_id);
        assert(new_template != NULL);
        add_template_to_builder(builder, new_template, node.type, sub_matrix_id);

        // 参数
        unaligned_warp_reduce_same_TLB_size_template_node_param_t* node_param_ptr = (unaligned_warp_reduce_same_TLB_size_template_node_param_t*)node.template_param;
        new_template->thread_num_in_block = node_param_ptr->thread_num_in_block;
        new_template->tblock_num = node_param_ptr->tblock_num;
        
        try_all_compress(new_template);
    }
    else
    {
        cout << "execute_template_node_and_update_code_builder: template type is not supported" << endl;
        assert(false);
    }

    assert(builder->template_vec[sub_matrix_id] != NULL);
    assert(builder->template_type_vec[sub_matrix_id] != NONE_TEMPLATE);
}

// 查看是不是分别自增
bool check_begin_memory_cache_input_file(exe_begin_memory_cache_input_file_param_t input_node)
{
    assert(input_node.col_index_cache.size() == input_node.row_index_cache.size());
    
    if (input_node.val_data_type == FLOAT)
    {
        assert(input_node.double_val_cache.size() == 0);
        assert(input_node.float_val_cache.size() == input_node.row_index_cache.size());
    }

    if (input_node.val_data_type == DOUBLE)
    {
        assert(input_node.float_val_cache.size() == 0);
        assert(input_node.double_val_cache.size() == input_node.row_index_cache.size());
    }

    assert(input_node.row_index_cache.size() > 0);

    // 用一个bool判断是不是整个矩阵的第一行
    bool is_first_row_index_of_matrix = true;

    // 用一个bool判断是不是一行的第一列
    bool is_first_col_index_of_row = true;

    // 遍历所有非零元
    for (unsigned long i = 0; i < input_node.row_index_cache.size(); i++)
    {
        unsigned long cur_row_index = input_node.row_index_cache[i];
        
        if (is_first_row_index_of_matrix == false)
        {
            // 和之前的比较，至少不能小于之前的
            if (cur_row_index < input_node.row_index_cache[i - 1])
            {
                return false;
            }

            // 如果当前的行号和之前一样，说明不是当前行第一个列索引
            if (cur_row_index == input_node.row_index_cache[i - 1])
            {
                is_first_col_index_of_row = false;
            }
            else
            {
                // 当前列索引就是第一个列索引
                is_first_col_index_of_row = true;
            }
        }

        // 列索引
        unsigned long cur_col_index = input_node.col_index_cache[i];

        if (is_first_col_index_of_row == false)
        {
            if (cur_col_index < input_node.col_index_cache[i - 1])
            {
                return false;
            }
        }

        is_first_row_index_of_matrix = false;
    }

    return true;
}

bool has_empty_line_in_begin_memory_cache_input_file(exe_begin_memory_cache_input_file_param_t input_node)
{
    assert(input_node.col_index_cache.size() == input_node.row_index_cache.size());
    
    if (input_node.val_data_type == FLOAT)
    {
        assert(input_node.double_val_cache.size() == 0);
        assert(input_node.float_val_cache.size() == input_node.row_index_cache.size());
    }

    if (input_node.val_data_type == DOUBLE)
    {
        assert(input_node.float_val_cache.size() == 0);
        assert(input_node.double_val_cache.size() == input_node.row_index_cache.size());
    }

    assert(input_node.row_index_cache.size() > 0);

    set<unsigned long> row_index_set;

    for (unsigned long i = 0; i < input_node.row_index_cache.size(); i++)
    {
        row_index_set.insert(input_node.row_index_cache[i]);
    }

    if (row_index_set.size() == input_node.row_index_max + 1)
    {
        return false;
    }

    return true;
}