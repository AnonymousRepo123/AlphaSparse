#include "user_pruning_strategy.hpp"

bool is_supported_by_direct_atom_template_warp_block_compress_with_user_strategy(sparse_struct_t *matrix, unsigned long dense_block_id)
{
    assert(matrix != NULL);
    
    assert(dense_block_id < matrix->block_coor_table.item_arr.size());
    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr != NULL);

    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index.size() == 7);

    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr;

    index_of_compress_block_t *block_level_index = compressed_block_view->read_index[2];
    index_of_compress_block_t *warp_level_index = compressed_block_view->read_index[3];
    index_of_compress_block_t *thread_level_index = compressed_block_view->read_index[4];
    assert(block_level_index->level_of_this_index == TBLOCK_LEVEL);
    assert(warp_level_index->level_of_this_index == WRAP_LEVEL);
    assert(thread_level_index->level_of_this_index == THREAD_LEVEL);

    // block级别的分块和warp级别的分块必须被放弃，
    if (block_level_index->block_num != 1)
    {
        return false;
    }

    if (warp_level_index->block_num != 1)
    {
        return false;
    }

    return true;
}

bool is_supported_by_direct_atom_template_warp_compress_with_user_strategy(sparse_struct_t *matrix, unsigned long dense_block_id)
{
    assert(matrix != NULL);
    
    assert(dense_block_id < matrix->block_coor_table.item_arr.size());
    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr != NULL);

    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index.size() == 7);

    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr;

    index_of_compress_block_t *block_level_index = compressed_block_view->read_index[2];
    index_of_compress_block_t *warp_level_index = compressed_block_view->read_index[3];
    index_of_compress_block_t *thread_level_index = compressed_block_view->read_index[4];
    assert(block_level_index->level_of_this_index == TBLOCK_LEVEL);
    assert(warp_level_index->level_of_this_index == WRAP_LEVEL);
    assert(thread_level_index->level_of_this_index == THREAD_LEVEL);

    // 必须放弃warp层次的分块
    // 如果有wrap级别的分块，就要放弃
    if (warp_level_index->block_num != block_level_index->block_num)
    {
        assert(warp_level_index->block_num > block_level_index->block_num);
        return false;
    }

    // 只有一个block也算了
    if (block_level_index->block_num == 1)
    {
        return false;
    }

    return true;
}

// 必须存在block、warp的分块
bool is_supported_by_direct_atom_template_with_user_strategy(sparse_struct_t *matrix, unsigned long dense_block_id)
{
    assert(matrix != NULL);

    // 如果WLB和BLB数量相同，那么说明没有执行WLB层次的分块，如果BLB只有一块，说明没有执行BLB层次的分块，这两种情况都需要排除，
    assert(dense_block_id < matrix->block_coor_table.item_arr.size());

    assert(matrix->block_coor_table.item_arr[dense_block_id] != NULL);
    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr != NULL);

    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index.size() == 7);

    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr;

    index_of_compress_block_t *block_level_index = compressed_block_view->read_index[2];
    index_of_compress_block_t *warp_level_index = compressed_block_view->read_index[3];
    index_of_compress_block_t *thread_level_index = compressed_block_view->read_index[4];
    assert(block_level_index->level_of_this_index == TBLOCK_LEVEL);
    assert(warp_level_index->level_of_this_index == WRAP_LEVEL);
    assert(thread_level_index->level_of_this_index == THREAD_LEVEL);

    // 没有warp级别的分块，不通过
    if (block_level_index->block_num == warp_level_index->block_num)
    {
        return false;
    }

    // 没有block级别的分块，放弃
    if (block_level_index->block_num == 1)
    {
        return false;
    }

    if (warp_level_index->block_num == 1)
    {
        return false;
    }

    return true;
}

// 保证BLB和TLB都用了默认的分块方式，BLB只有一个块，TLB只有一个非零元
bool is_supported_by_direct_atom_total_warp_reduce_template_with_user_strategy(sparse_struct_t *matrix, unsigned long dense_block_id)
{
    // warp级别的归约的检查，因为没有block级别的遍历，所以BLB放弃分块
    assert(dense_block_id < matrix->block_coor_table.item_arr.size());

    assert(matrix->block_coor_table.item_arr[dense_block_id] != NULL);
    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr != NULL);

    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index.size() == 7);

    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr;

    index_of_compress_block_t *block_level_index = compressed_block_view->read_index[2];
    index_of_compress_block_t *warp_level_index = compressed_block_view->read_index[3];
    index_of_compress_block_t *thread_level_index = compressed_block_view->read_index[4];
    assert(block_level_index->level_of_this_index == TBLOCK_LEVEL);
    assert(warp_level_index->level_of_this_index == WRAP_LEVEL);
    assert(thread_level_index->level_of_this_index == THREAD_LEVEL);

    // 因为BLB级别的划分被抛弃了，所以BLB的数量不能超过1
    if (block_level_index->block_num != 1)
    {
        return false;
    }

    // WLB的数量不能少于8，要不并行度严重不够
    if (warp_level_index->block_num <= 8)
    {
        return false;
    }

    // 查看TLB的非零元数量是不是1
    for (unsigned long WLB_id = 0; WLB_id < warp_level_index->block_num; WLB_id++)
    {
        unsigned long TLB_size = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, WLB_id);

        if (TLB_size != 1)
        {
            return false;
        }
    }

    return true;
}

// 保证WLB和TLB的分块都被放弃了
bool is_supported_by_shared_memory_long_row_template_with_user_strategy(sparse_struct_t *matrix, unsigned long dense_block_id)
{
    // 保证WLB和BLB的大小是一样的，并且线程TLB的大小为1
    assert(dense_block_id < matrix->block_coor_table.item_arr.size());

    assert(matrix->block_coor_table.item_arr[dense_block_id] != NULL);
    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr != NULL);

    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index.size() == 7);

    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr;

    index_of_compress_block_t *block_level_index = compressed_block_view->read_index[2];
    index_of_compress_block_t *warp_level_index = compressed_block_view->read_index[3];
    index_of_compress_block_t *thread_level_index = compressed_block_view->read_index[4];
    assert(block_level_index->level_of_this_index == TBLOCK_LEVEL);
    assert(warp_level_index->level_of_this_index == WRAP_LEVEL);
    assert(thread_level_index->level_of_this_index == THREAD_LEVEL);

    // 检查BLB和WLB是不是一样多，不是一样多就代表不适合使用这个使用这个模板
    unsigned long BLB_num = block_level_index->block_num;
    unsigned long WLB_num = warp_level_index->block_num;

    // WLB级别的分块没有放弃，那就不支持这个模板
    if (BLB_num != WLB_num)
    {
        return false;
    }

    // 查看TLB的非零元数量是不是1，TLB级别的分块必须被放弃，如果TLB执行了分块，那就放弃
    for (unsigned long WLB_id = 0; WLB_id < warp_level_index->block_num; WLB_id++)
    {
        unsigned long TLB_size = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, WLB_id);

        if (TLB_size != 1)
        {
            // cout << "TLB_size:" << TLB_size <<  << endl;
            return false;
        }
    }

    return true;
}

// 没有WLB级别的分块
bool is_supported_by_shared_memory_template_warp_compress_with_user_strategy(sparse_struct_t *matrix, unsigned long dense_block_id)
{
    // 保证WLB和BLB的大小是一样的，并且线程TLB的大小为1
    assert(dense_block_id < matrix->block_coor_table.item_arr.size());

    assert(matrix->block_coor_table.item_arr[dense_block_id] != NULL);
    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr != NULL);

    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index.size() == 7);

    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr;

    index_of_compress_block_t *block_level_index = compressed_block_view->read_index[2];
    index_of_compress_block_t *warp_level_index = compressed_block_view->read_index[3];
    index_of_compress_block_t *thread_level_index = compressed_block_view->read_index[4];
    assert(block_level_index->level_of_this_index == TBLOCK_LEVEL);
    assert(warp_level_index->level_of_this_index == WRAP_LEVEL);
    assert(thread_level_index->level_of_this_index == THREAD_LEVEL);

    if (warp_level_index->block_num != block_level_index->block_num)
    {
        return false;
    }

    // 如果只有一个BLB，就还是算了吧，BLB必须执行分块
    if (block_level_index->block_num == 1)
    {
        return false;
    }

    return true;
}

// 必须有warp和block级别的分块
bool is_supported_by_shared_memory_template_with_user_strategy(sparse_struct_t *matrix, unsigned long dense_block_id)
{
    // 用这个必须保证三个层次的分块都在
    // 如果WLB和BLB数量相同，那么说明没有执行WLB层次的分块，如果BLB只有一块，说明没有执行BLB层次的分块，这两种情况都需要排除，
    assert(dense_block_id < matrix->block_coor_table.item_arr.size());

    assert(matrix->block_coor_table.item_arr[dense_block_id] != NULL);
    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr != NULL);

    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index.size() == 7);

    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr;

    index_of_compress_block_t *block_level_index = compressed_block_view->read_index[2];
    index_of_compress_block_t *warp_level_index = compressed_block_view->read_index[3];
    index_of_compress_block_t *thread_level_index = compressed_block_view->read_index[4];
    assert(block_level_index->level_of_this_index == TBLOCK_LEVEL);
    assert(warp_level_index->level_of_this_index == WRAP_LEVEL);
    assert(thread_level_index->level_of_this_index == THREAD_LEVEL);

    // 没有WLB的分块，不通过
    if (block_level_index->block_num == warp_level_index->block_num)
    {
        return false;
    }

    // 没有BLB的分块，不通过
    if (block_level_index->block_num == 1)
    {
        return false;
    }

    return true;
}

// 不能有TLB的分块
bool is_supported_by_shared_memory_total_warp_reduce_template_with_user_strategy(sparse_struct_t *matrix, unsigned long dense_block_id)
{
    // 保证WLB和BLB的大小是一样的，并且线程TLB的大小为1
    assert(dense_block_id < matrix->block_coor_table.item_arr.size());

    assert(matrix->block_coor_table.item_arr[dense_block_id] != NULL);
    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr != NULL);

    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index.size() == 7);

    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr;

    index_of_compress_block_t *block_level_index = compressed_block_view->read_index[2];
    index_of_compress_block_t *warp_level_index = compressed_block_view->read_index[3];
    index_of_compress_block_t *thread_level_index = compressed_block_view->read_index[4];
    assert(block_level_index->level_of_this_index == TBLOCK_LEVEL);
    assert(warp_level_index->level_of_this_index == WRAP_LEVEL);
    assert(thread_level_index->level_of_this_index == THREAD_LEVEL);

    // 检查BLB和WLB是不是一样多，一样多代表没有WLB层次的分块不能通过
    unsigned long BLB_num = block_level_index->block_num;
    unsigned long WLB_num = warp_level_index->block_num;

    // BLB层次必须分块，没有分块就不能通过
    if (BLB_num == 1)
    {
        return false;
    }
    
    // 没有warp分块
    if (BLB_num == WLB_num)
    {
        return false;
    }

    // 查看TLB的非零元数量是不是1
    for (unsigned long TLB_id = 0; TLB_id < thread_level_index->block_num; TLB_id++)
    {
        unsigned long TLB_size = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, TLB_id);

        if (TLB_size != 1)
        {
            return false;
        }
    }

    return true;
}

// 类似于CSR5-like
bool is_supported_by_unaligned_warp_reduce_same_TLB_size_template_with_user_strategy(sparse_struct_t* matrix, unsigned long dense_block_id)
{
    // 如果排序过了，就不用这个模板了
    // 保证WLB和BLB的大小是一样的，并且线程TLB的大小为1
    assert(dense_block_id < matrix->block_coor_table.item_arr.size());

    assert(matrix->block_coor_table.item_arr[dense_block_id] != NULL);
    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr != NULL);

    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index.size() == 7);

    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr;

    if (compressed_block_view->y_write_index.size() > 0)
    {
        // 排序过了，不能用
        return false;
    }

    // 全局排序过了
    if (matrix->is_sorted == true)
    {
        assert(matrix->sorted_row_index != NULL);
        return false;
    }

    return true;
}

bool is_supported_be_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_with_user_strategy(sparse_struct_t* matrix, unsigned long dense_block_id)
{
    // 保证WLB和BLB的大小是一样的，并且线程TLB的大小为1
    assert(dense_block_id < matrix->block_coor_table.item_arr.size());

    assert(matrix->block_coor_table.item_arr[dense_block_id] != NULL);
    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr != NULL);

    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index.size() == 7);

    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr;

    if (compressed_block_view->y_write_index.size() > 0)
    {
        // 排序过了，不能用
        return false;
    }

    // 全局排序过了
    if (matrix->is_sorted == true)
    {
        assert(matrix->sorted_row_index != NULL);
        return false;
    }

    return true;
}

set<template_type> filter_from_existing_template_set(set<template_type> old_temp_set)
{
    assert(old_temp_set.size() > 0);
    // 轮训筛查，直到产生的模板集不再变化
    while (true)
    {
        set<template_type> return_set;

        // 遍历temp_set中的所有类型，通过找出其他类型的存在，来判断自己是不是被需要的
        for (set<template_type>::iterator cur_temp_type_ptr = old_temp_set.begin(); cur_temp_type_ptr != old_temp_set.end(); cur_temp_type_ptr++)
        {
            // 当前模板类型
            template_type cur_temp_type = *cur_temp_type_ptr;

            if (cur_temp_type == DIRECT_ATOM_TEMPLATE)
            {
                // 最基础的模板，会被除了基础shared memory之外的所有模板覆盖
                if (old_temp_set.count(DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS) == 1 || old_temp_set.count(DIRECT_ATOM_TEMPLATE_WARP_COMPRESS) == 1 ||
                    old_temp_set.count(DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE) == 1 || old_temp_set.count(SHARED_MEMORY_LONG_ROW_TEMPLATE) == 1 || old_temp_set.count(SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE) == 1)
                {
                }
                else
                {
                    // 只好前面的这些都没有，就保留在下一轮矩阵中
                    return_set.insert(cur_temp_type);
                }
            }

            if (cur_temp_type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
            {
                if (old_temp_set.count(DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS) == 1 || old_temp_set.count(DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE) == 1 || old_temp_set.count(SHARED_MEMORY_LONG_ROW_TEMPLATE) == 1 || old_temp_set.count(SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE) == 1)
                {
                }
                else
                {
                    return_set.insert(cur_temp_type);
                }
            }

            // 多级归约胜过单级归约
            if (cur_temp_type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
            {
                if (old_temp_set.count(DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE) == 1 || old_temp_set.count(SHARED_MEMORY_LONG_ROW_TEMPLATE) == 1 || old_temp_set.count(SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE) == 1)
                {
                }
                else
                {
                    return_set.insert(cur_temp_type);
                }
            }

            // 共享内存的版本，可以被带压缩的warp级别归约+shared mem的替代
            if (cur_temp_type == SHARED_MEMORY_TEMPLATE)
            {
                if (old_temp_set.count(SHARED_MEMORY_TEMPLATE_WARP_COMPRESS) == 1 || old_temp_set.count(SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE) == 1 || old_temp_set.count(SHARED_MEMORY_LONG_ROW_TEMPLATE) == 1)
                {
                }
                else
                {
                    return_set.insert(cur_temp_type);
                }
            }

            // 带压缩的共享内存，可以被同样用共享内存的更多层次替代
            if (cur_temp_type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
            {
                if (old_temp_set.count(SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE) == 1 || old_temp_set.count(SHARED_MEMORY_LONG_ROW_TEMPLATE) == 1)
                {
                }
                else
                {
                    return_set.insert(cur_temp_type);
                }
            }

            // 带warp归约的shared mem模板，被超长行的共享内存法替代
            if (cur_temp_type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
            {
                if (old_temp_set.count(SHARED_MEMORY_LONG_ROW_TEMPLATE) == 1)
                {
                }
                else
                {
                    return_set.insert(cur_temp_type);
                }
            }

            // 带warp归约的原子加，被超长行的共享内存法替代
            if (cur_temp_type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
            {
                if (old_temp_set.count(SHARED_MEMORY_LONG_ROW_TEMPLATE) == 1)
                {
                }
                else
                {
                    return_set.insert(cur_temp_type);
                }
            }

            if (cur_temp_type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
            {
                return_set.insert(cur_temp_type);
            }

            if (cur_temp_type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
            {
                return_set.insert(cur_temp_type);
            }

            if (cur_temp_type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
            {
                return_set.insert(cur_temp_type);
            }
        }

        // 如果把自己筛没了，就返回上一层结果
        if (return_set.size() == 0)
        {
            cout << "get empty template set after filter" << endl;
            return old_temp_set;
        }

        // 如果，筛选之后的集合数量和之前的一样，就不需要再筛选了，直接返回筛选的结果
        if (old_temp_set.size() == return_set.size())
        {
            return return_set;
        }

        old_temp_set = return_set;
    }

    assert(false);
    return old_temp_set;
}

bool dependence_of_exe_begin_artificial_input_node_with_user_strategy(exe_graph_t *graph, exe_begin_artificial_input_param_t param, int input_index)
{
    return true;
}

bool dependence_of_exe_compress_node_with_user_strategy(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_param_t param, int sub_graph, int input_index)
{
    return true;
}

bool dependence_of_exe_begin_input_file_node_with_user_strategy(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_begin_input_file_param_t param, int sub_graph, int input_index)
{
    return true;
}

bool dependence_of_exe_dense_row_div_node_with_user_strategy(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_dense_row_div_param_t param, int sub_graph, int input_index)
{
    assert(graph != NULL);
    // 如果之前已经出现过分块操作，那就不能再分块了，主要是还没有实现
    if (graph->dense_sub_graph.preorder_node_set.count(DENSE_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->dense_sub_graph.preorder_node_set.count(DENSE_FIXED_COL_DIV) != 0)
    {
        return false;
    }

    return true;
}

bool dependence_of_exe_dense_fixed_col_div_node_with_user_strategy(exe_graph_t* graph, exe_sub_graph_type graph_type, exe_dense_fixed_col_div_param_t param, int sub_graph, int input_index)
{
    return true;
}

bool dependence_of_exe_dense_row_coarse_sort_node_with_user_strategy(exe_graph_t* graph, exe_sub_graph_type graph_type, exe_dense_row_coarse_sort_param_t param, int sub_graph, int input_index)
{
    return true;
}

bool dependence_of_exe_compress_WLB_row_div_node_with_user_strategy(exe_graph_t *graph, exe_sub_graph_type graph_type, exe_compress_warp_level_row_div_param_t param, int sub_graph, int input_index)
{
    // 当已经出现了BLB的切块的时候，就不需要WLB分块了
    assert(graph != NULL);

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_ROW_DIV) != 0)
    {
        return false;
    }

    if (graph->total_compressed_sub_graph.compressed_sub_graph_vec[sub_graph].preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_COL_DIV) != 0)
    {
        return false;
    }

    return true;
}