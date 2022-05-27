#include "default_auto_tuner.hpp"
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

// template_node_t find_best_param_of_specific_template_node(sparse_struct_t* matrix, int sub_matrix_id, template_type type, float& best_time, float& best_gflops)
// {
//     assert(matrix != NULL);
//     assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
//     assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL);
//     // 已经被压缩过
//     assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
//     assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() >= 2);

//     compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

//     // 执行默认的一系列分块，将所有的分块补齐
//     execute_default_div_to_complete_each_level_blocking(matrix, sub_matrix_id);

//     // 查看分块之后的索引数量
//     // cout << "compressed_block_ptr->read_index.size():" << compressed_block_ptr->read_index.size() << endl;
//     assert(compressed_block_ptr->read_index.size() == 7);
//     template_node_t return_template_node;

//     // 获取可能的模板集合
//     set<template_type> template_set = supported_template_of_sub_matrix(matrix, sub_matrix_id);

//     // 遍历所有的模板类型，为每个模板找出对应的参数
    
    
    
    
//     // 如果搜不到对应的模板，这个时候对应的这里的template param是空的，之后在外面处理
//     return return_template_node;
// }

template_node_t find_best_template_node_of_specific_sub_matrix_from_template_set(sparse_struct_t* matrix, int sub_matrix_id, set<template_type> template_set, float& best_time, float& best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(matrix != NULL);
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL);
    // 已经被压缩过
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    // 自动补齐没有执行的对应层次的分块
    execute_default_div_to_complete_each_level_blocking(matrix, sub_matrix_id);

    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);

    // 要返回的最优模板节点类型和节点参数
    template_node_t return_template_node;
    return_template_node.template_param = NULL;

    vector<int> sub_matrix_id_vec;
    sub_matrix_id_vec.push_back(sub_matrix_id);

    best_gflops = 0;

    // 遍历候选的所有的模板，查看性能，并且不断更新最终的参数，在过程中注意析构最后模板的参数
    for (auto template_type : template_set)
    {
        float cur_template_gflops = 0;
        float cur_template_time = 99999999999;

        // if (data_set_collector != NULL)
        // {
        //     cout << "find_best_template_node_of_specific_sub_matrix_from_template_set: need to collect ml data" << endl;
        // }

        template_node_t cur_template_node = find_best_param_of_specific_template_node_of_sub_matrix(matrix, sub_matrix_id, template_type, cur_template_time, cur_template_gflops, search_strategy_ptr, data_set_collector);

        // 如果吞吐量更大，那就更新一下最优的模板和参数
        if (cur_template_gflops > best_gflops)
        {
            // 可能需要析构之前模板参数
            if (return_template_node.template_param != NULL)
            {
                del_param_of_template_node(&return_template_node);
            }
            else
            {
                assert(best_gflops == 0);
            }

            // 赋值为新的参数
            return_template_node.template_param = cur_template_node.template_param;
            return_template_node.type = cur_template_node.type;
            best_gflops = cur_template_gflops;
            best_time = cur_template_time;
        }

        // 这里查看是不是要直接退出，只有在有搜索策略的时候需要考虑提前终止的问题
        if (search_strategy_ptr != NULL)
        {
            if (continue_search(search_strategy_ptr) == false)
            {
                break;
            }
        }
    }
    
    assert(best_gflops >= 0);
    
    if (best_gflops != 0)
    {
        assert(return_template_node.template_param != NULL);    
    }
    
    return return_template_node;
}

template_node_t find_best_param_of_specific_template_node_of_sub_matrix(sparse_struct_t* matrix, int sub_matrix_id, template_type type, float& best_time, float& best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(matrix != NULL);
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL);
    // 已经被压缩过
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);

    // 要返回的最优模板节点类型和节点参数
    template_node_t return_template_node;

    vector<int> sub_matrix_id_vec;
    sub_matrix_id_vec.push_back(sub_matrix_id);

    // if (data_set_collector != NULL)
    // {
    //     cout << "find_best_param_of_specific_template_node_of_sub_matrix: need to collect ml data" << endl;
    // }

    // 首先创建一个临时的代码生成器
    // 然后生成对应的模板，找到对应的参数，最后执行对应析构将对应的代码生成器析构，包括对应的模板也会被析构
    if (type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
    {
        // 生成一个操作管理器
        operator_manager_t* op_manager = init_op_manager(matrix);
        // 生成一个代码生成器
        code_builder_t* builder = init_code_builder(op_manager, sub_matrix_id_vec);
        // 生成一个模板
        direct_atom_template_warp_block_compress_t* new_template = init_direct_atom_template_warp_block_compress(builder, sub_matrix_id);
        // 注册模板
        add_template_to_builder(builder, new_template, DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS, sub_matrix_id);
        // 压缩
        try_all_compress(new_template);

        // 生成最优的参数节点
        return_template_node = find_best_param_of_direct_atom_template_warp_block_compress(builder, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);
        assert(return_template_node.type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS);

        // 析构代码生成器
        memory_garbage_manager_t mem_manager;
        delete_code_builder_without_operator_manager(&mem_manager, builder);
        // 然后析构操作管理器
        delete op_manager;
    }
    else if (type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
    {
        // 生成一个操作管理器
        operator_manager_t* op_manager = init_op_manager(matrix);
        // 生成一个代码生成器
        code_builder_t* builder = init_code_builder(op_manager, sub_matrix_id_vec);
        // 生成一个模板
        direct_atom_template_warp_compress_t* new_template = init_direct_atom_template_warp_compress(builder, sub_matrix_id);
        // 将模板加到代码生成器中
        add_template_to_builder(builder, new_template, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, sub_matrix_id);
        // 压缩
        try_all_compress(new_template);

        // 生成最优的参数节点
        return_template_node = find_best_param_of_direct_atom_template_warp_compress(builder, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);
        assert(return_template_node.type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS);

        // 析构代码生成器
        memory_garbage_manager_t mem_manager;
        delete_code_builder_without_operator_manager(&mem_manager, builder);
        // 然后析构操作管理器
        delete op_manager;
    }
    else if (type == DIRECT_ATOM_TEMPLATE)
    {
        // 生成一个操作管理器
        operator_manager_t* op_manager = init_op_manager(matrix);
        // 生成一个代码生成器
        code_builder_t* builder = init_code_builder(op_manager, sub_matrix_id_vec);
        // 生成一个模板
        direct_atom_template_t* new_template = init_direct_atom_template(builder, sub_matrix_id);
        // 将模板加到代码生成器中
        add_template_to_builder(builder, new_template, DIRECT_ATOM_TEMPLATE, sub_matrix_id);
        // 压缩
        try_all_compress(new_template);

        // 生成最后参数节点
        return_template_node = find_best_param_of_direct_atom_template(builder, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);
        assert(return_template_node.type == DIRECT_ATOM_TEMPLATE);

        // 析构代码生成器
        memory_garbage_manager_t mem_manager;
        delete_code_builder_without_operator_manager(&mem_manager, builder);
        // 然后析构操作管理器
        delete op_manager;
    }
    else if (type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        // 操作管理器
        operator_manager_t* op_manager = init_op_manager(matrix);
        // 生成一个代码生成器
        code_builder_t* builder = init_code_builder(op_manager, sub_matrix_id_vec);
        // 生成一个模板
        direct_atom_total_warp_reduce_template_t* new_template = init_direct_atom_total_warp_reduce_template(builder, sub_matrix_id);
        // 将模板放到代码生成器中
        add_template_to_builder(builder, new_template, DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE, sub_matrix_id);
        // 压缩
        try_all_compress(new_template);

        // 生成最后一个参数节点
        return_template_node = find_best_param_of_direct_atom_total_warp_reduce_template(builder, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);
        assert(return_template_node.type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE);

        // 析构代码生成器
        memory_garbage_manager_t mem_manager;
        delete_code_builder_without_operator_manager(&mem_manager, builder);
        // 代码析构管理器
        delete op_manager;
    }
    else if (type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
    {
        // 操作管理器
        operator_manager_t* op_manager = init_op_manager(matrix);
        // 生成一个代码生成器
        code_builder_t* builder = init_code_builder(op_manager, sub_matrix_id_vec);
        // 生成一个模板
        shared_memory_long_row_template_t* new_template = init_shared_memory_long_row_template(builder, sub_matrix_id);
        // 将模板放到代码生成器中
        add_template_to_builder(builder, new_template, SHARED_MEMORY_LONG_ROW_TEMPLATE, sub_matrix_id);
        // 压缩
        try_all_compress(new_template);

        // 生成最后一个参数节点
        return_template_node = find_best_param_of_shared_memory_long_row_template(builder, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);
        assert(return_template_node.type == SHARED_MEMORY_LONG_ROW_TEMPLATE);

        // 析构代码生成器
        memory_garbage_manager_t mem_manager;
        delete_code_builder_without_operator_manager(&mem_manager, builder);
        // 析构动作管理器
        delete op_manager;
    }
    else if (type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
    {
        // 操作管理器
        operator_manager_t* op_manager = init_op_manager(matrix);
        // 生成代码生成器
        code_builder_t* builder = init_code_builder(op_manager, sub_matrix_id_vec);
        // 生成一个模板
        shared_memory_template_warp_compress_t* new_template = init_shared_memory_template_warp_compress(builder, sub_matrix_id);
        // 将模板放在代码生成器中
        add_template_to_builder(builder, new_template, SHARED_MEMORY_TEMPLATE_WARP_COMPRESS, sub_matrix_id);
        // 压缩
        try_all_compress(new_template);

        // 生成最优模板参数节点
        return_template_node = find_best_param_of_shared_memory_template_warp_compress(builder, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);
        assert(return_template_node.type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS);

        // 析构代码生成器
        memory_garbage_manager_t mem_manager;
        delete_code_builder_without_operator_manager(&mem_manager, builder);
        // 析构动作管理器
        delete op_manager;
    }
    else if (type == SHARED_MEMORY_TEMPLATE)
    {
        // 操作管理器
        operator_manager_t* op_manager = init_op_manager(matrix);
        // 生成代码生成器
        code_builder_t* builder = init_code_builder(op_manager, sub_matrix_id_vec);
        // 生成一个模板
        shared_memory_template_t* new_template = init_shared_memory_template(builder, sub_matrix_id);
        // 将模板放在生成器中
        add_template_to_builder(builder, new_template, SHARED_MEMORY_TEMPLATE, sub_matrix_id);
        // 压缩
        try_all_compress(new_template);

        // 生成最优模板参数节点
        return_template_node = find_best_param_of_shared_memory_template(builder, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);
        assert(return_template_node.type == SHARED_MEMORY_TEMPLATE);

        // 析构动作管理器
        memory_garbage_manager_t mem_manager;
        delete_code_builder_without_operator_manager(&mem_manager, builder);
        // 析构动作管理器
        delete op_manager;
    }
    else if (type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        // 操作管理器
        operator_manager_t* op_manager = init_op_manager(matrix);
        // 生成代码生成器
        code_builder_t* builder = init_code_builder(op_manager, sub_matrix_id_vec);
        // 生成一个模板
        shared_memory_total_warp_reduce_template_t* new_template = init_shared_memory_total_warp_reduce_template(builder, sub_matrix_id);
        // 将模板放到代码生成器中
        add_template_to_builder(builder, new_template, SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE, sub_matrix_id);
        // 压缩
        try_all_compress(new_template);

        // 生成最优模板参数
        return_template_node = find_best_param_of_shared_memory_total_warp_reduce_template(builder, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);
        assert(return_template_node.type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE);

        // 析构代码生成器
        memory_garbage_manager_t mem_manager;
        delete_code_builder_without_operator_manager(&mem_manager, builder);
        // 析构动作管理器
        delete op_manager;
    }
    else if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
    {
        // 操作管理器
        operator_manager_t* op_manager = init_op_manager(matrix);
        // 代码生成器
        code_builder_t* builder = init_code_builder(op_manager, sub_matrix_id_vec);
        // 模板
        unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t* new_template = init_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce(builder, sub_matrix_id);
        // 放到生成器中
        add_template_to_builder(builder, new_template, UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE, sub_matrix_id);
        // 压缩
        try_all_compress(new_template);

        // 生成最优模板参数
        return_template_node = find_best_param_of_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce(builder, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);
        assert(return_template_node.type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE);

        // 析构代码生成器
        memory_garbage_manager_t mem_manager;
        delete_code_builder_without_operator_manager(&mem_manager, builder);
        // 析构动作管理器
        delete op_manager;
    }
    else if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
    {
        // 操作管理器
        operator_manager_t* op_manager = init_op_manager(matrix);
        // 代码生成器
        code_builder_t* builder = init_code_builder(op_manager, sub_matrix_id_vec);
        // 模板
        unaligned_warp_reduce_same_TLB_size_template_t* new_template = init_unaligned_warp_reduce_same_TLB_size_template(builder, sub_matrix_id);
        // 生成器
        add_template_to_builder(builder, new_template, UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE, sub_matrix_id);
        // 压缩
        try_all_compress(new_template);

        
        
        // 生成模板参数
        return_template_node = find_best_param_of_unaligned_warp_reduce_same_TLB_size_template(builder, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);
        assert(return_template_node.type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE);

        // 析构代码生成器
        memory_garbage_manager_t mem_manager;
        delete_code_builder_without_operator_manager(&mem_manager, builder);
        // 析构动作管理器
        delete op_manager;
    }
    else
    {
        // 当前模板的类型不支持
        cout << "find_best_param_of_specific_template_node: template type is not supported" << endl;
        assert(false);
    }

    return return_template_node;
}

void execute_default_div_to_complete_each_level_blocking(sparse_struct_t* matrix, int sub_matrix_id)
{
    assert(matrix != NULL);
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL);
    // 已经被压缩过
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() >= 2);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    if (compressed_block_ptr->read_index.size() == 2)
    {
        // 如果当前索引数量是两个，就补一个block级别的分块
        // 根据行数量补一个默认的行分块
        // 行数量
        unsigned long row_num = compressed_block_ptr->read_index[0]->max_row_index - compressed_block_ptr->read_index[0]->min_row_index + 1;

        vector<unsigned int> BLB_row_num;
        BLB_row_num.push_back(row_num);

        sep_tblock_level_row_csr(compressed_block_ptr, BLB_row_num);

        // 已经完成，检查当前索引数量
        assert(compressed_block_ptr->read_index.size() == 3);
    }

    if (compressed_block_ptr->read_index.size() == 3)
    {
        // 需要一个WLB级别的分块
        // warp不处理
        vector<unsigned long> sep_BLB_id;
        vector<vector<unsigned int>> WLB_row_size_of_each_BLB;

        sep_warp_level_row_csr(compressed_block_ptr, sep_BLB_id, WLB_row_size_of_each_BLB);
        // 已经完成，检查当前索引数量
        assert(compressed_block_ptr->read_index.size() == 4);
    }

    if (compressed_block_ptr->read_index.size() == 4)
    {
        // 默认的行切分
        vector<unsigned long> sep_WLB_id;
        vector<unsigned long> thread_col_size_of_each_WLB;

        // 查看warp的数量
        unsigned long warp_num = compressed_block_ptr->read_index[3]->block_num;

        for (unsigned long i = 0; i < warp_num; i++)
        {
            sep_WLB_id.push_back(i);
            thread_col_size_of_each_WLB.push_back(1);
        }

        sep_thread_level_col_ell_with_padding(compressed_block_ptr, sep_WLB_id, thread_col_size_of_each_WLB);

        assert(compressed_block_ptr->read_index.size() == 7);
    }

    // cout << "compressed_block_ptr->read_index.size():" << compressed_block_ptr->read_index.size() << endl;
    assert(compressed_block_ptr->read_index.size() == 7);
}

// 这个模板的参数替换是不用，性能会输出出来，用来横向比较
template_node_t find_best_param_of_direct_atom_template_warp_block_compress(code_builder_t* builder, int sub_matrix_id, float& return_best_time, float& return_best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(builder != NULL && sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());
    assert(builder->op_manager != NULL && builder->op_manager->matrix != NULL && sub_matrix_id < builder->template_vec.size());
    assert(sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());

    assert(builder->template_type_vec[sub_matrix_id] == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS);
    assert(builder->template_vec[sub_matrix_id] != NULL && builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);

    // 有两个参数，一个是block数量和block内thread数量
    long tblock_num;
    long thread_num_in_block;

    // 创建一个参数枚举器，只有thread_num_in_block的参数是需要调整的
    param_enumerater_t param_setter;
    // 性能影响差距不是很大
    register_integer_independ_param_to_enumerater(&param_setter, &thread_num_in_block, 32, 256, 32);

    // 查看是不是第一个模板的枚举
    bool is_first_enumerate = true;

    // 记录最佳的时间和最佳的性能，以及对应的最佳参数
    float best_time = 0;
    float best_gflops = 0;
    long best_tblock_num = 0;
    long best_thread_num_in_block = 0;

    direct_atom_template_warp_block_compress_t* target_template = (direct_atom_template_warp_block_compress_t *)builder->template_vec[sub_matrix_id];

    // 用一个参数来判断在搜索策略下是不是需要
    bool search_finished_by_strategy = false;

    while (set_param_combination_to_next(&param_setter) == false)
    {
        // 查看TLB的数量
        unsigned long TLB_number = target_template->size_of_global_row_index_of_thread_level_block;

        // 计算tblock的数量
        tblock_num = TLB_number / thread_num_in_block;

        if (TLB_number % thread_num_in_block != 0)
        {
            // 不能乘除要多加一个线程块
            tblock_num = tblock_num + 1;
        }

        if (tblock_num > get_config()["MAX_TBLOCK_NUM"].as_integer() - 1)
        {
            tblock_num = get_config()["MAX_TBLOCK_NUM"].as_integer() - 1;
        }

        // 将参数写到模板中
        target_template->tblock_num = tblock_num;
        target_template->thread_num_in_block = thread_num_in_block;

        cout << "find_best_param_of_direct_atom_template_warp_block_compress: target_template->tblock_num:" << target_template->tblock_num << endl;
        cout << "find_best_param_of_direct_atom_template_warp_block_compress: target_template->thread_num_in_block:" << target_template->thread_num_in_block << endl;

        vector<int> sub_matrix_id_vec;
        sub_matrix_id_vec.push_back(sub_matrix_id);
        float exe_time = 0;
        float exe_gflops = 0;

        // 这里对模板的具体参数执行
        bool is_success_exe = part_execute_code_builder(builder, sub_matrix_id_vec, exe_time, exe_gflops, string(get_config()["ROOT_PATH_STR"].as_string()) + "/cuda_code", string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source", is_first_enumerate, true);

        // 如果不成功就跳过
        if (is_success_exe == false)
        {
            continue;
        }

        // 如果运行成功，将参数和最终性能插入到参数数组中
        vector<float> param_vec;
        
        if (data_set_collector != NULL)
        {
            // 已有的内容存在一些积累
            assert(data_set_collector->accu_dense_param_strategy_type_vec.size() > 0);
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
            assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() > 0);
            assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() == data_set_collector->accu_compressed_sub_graph_node_type_vec.size());

            // 将参数记录下来
            param_vec.push_back(target_template->tblock_num);
            param_vec.push_back(target_template->thread_num_in_block);
            param_vec.push_back(exe_gflops);
            
            // 将参数写到数据集收集器中
            data_set_collector->insert_template_node_and_param_to_cur_item_and_add_to_dataset(DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS, param_vec);
        }

        is_first_enumerate = false;

        // 当glops更大的时候，就替换
        if (exe_gflops > best_gflops)
        {
            // 找到更好的参数了
            best_gflops = exe_gflops;
            best_time = exe_time;

            best_tblock_num = tblock_num;
            best_thread_num_in_block = thread_num_in_block;
        }

        // 如果有搜索策略，可能需要看看提前退出的问题
        if (search_strategy_ptr != NULL)
        {
            if (continue_search(search_strategy_ptr, exe_gflops) == false)
            {
                search_finished_by_strategy = true;
            }
        }

        if (search_finished_by_strategy == true)
        {
            break;
        }
    }
    
    // 产生一个新的节点，来记录当前最优的参数组合
    template_node_t return_node;

    direct_atom_template_warp_block_compress_node_param_t* param_ptr = new direct_atom_template_warp_block_compress_node_param_t();

    param_ptr->tblock_num = best_tblock_num;
    param_ptr->thread_num_in_block = best_thread_num_in_block;

    return_node.type = DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS;
    return_node.template_param = param_ptr;

    // 将最优性能传到外面
    return_best_time = best_time;
    return_best_gflops = best_gflops;
    
    return return_node;
}

template_node_t find_best_param_of_direct_atom_template_warp_compress(code_builder_t* builder, int sub_matrix_id, float& return_best_time, float& return_best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(builder != NULL && sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());
    assert(builder->op_manager != NULL && builder->op_manager->matrix != NULL && sub_matrix_id < builder->template_vec.size());
    assert(sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());

    assert(builder->template_type_vec[sub_matrix_id] == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS);
    assert(builder->template_vec[sub_matrix_id] != NULL && builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);
    
    index_of_compress_block_t* BLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2];
    index_of_compress_block_t* WLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[3];

    unsigned long max_TLB_num = 0;
    // 最少的TLB数量
    unsigned long min_TLB_num = 9999999;

    // 唯一需要调的参数应该只有thread的数量，枚举的下界找小于最小TLB数量的32的倍数，从32的倍数开始调。上界是BLB中TLB数量的最大值最近的大于TLB数量的32的倍数
    // 首先找出最大的块TLB数量
    for (unsigned long BLB_id = 0; BLB_id < BLB_index->block_num; BLB_id++)
    {
        // 这个BLB第一个WLB的索引
        unsigned long cur_WLB_first_id = read_from_array_with_data_type(BLB_index->index_arr, BLB_index->index_data_type, BLB_id);
        // 下一个BLB的第一个WLB的索引
        unsigned long next_WLB_first_id = read_from_array_with_data_type(BLB_index->index_arr, BLB_index->index_data_type, BLB_id + 1);

        // 只要两个同时小于length，就算出他们的首个TLB索引
        if (cur_WLB_first_id < WLB_index->length && next_WLB_first_id < WLB_index->length)
        {
            unsigned long cur_BLB_first_TLB_id = read_from_array_with_data_type(WLB_index->index_arr, WLB_index->index_data_type, cur_WLB_first_id);
            unsigned long next_BLB_first_TLB_id = read_from_array_with_data_type(WLB_index->index_arr, WLB_index->index_data_type, next_WLB_first_id);

            unsigned long TLB_num = next_BLB_first_TLB_id - cur_BLB_first_TLB_id;

            if (TLB_num > max_TLB_num)
            {
                max_TLB_num = TLB_num;
            }

            if (TLB_num < min_TLB_num)
            {
                min_TLB_num = TLB_num;
            }
        }
    }

    // 找到大于TLB的最大32倍数的值
    unsigned long enum_up_bound = max_TLB_num / 32;
    enum_up_bound = enum_up_bound * 32;

    if (max_TLB_num % 32 != 0)
    {
        enum_up_bound = enum_up_bound + 32;
    }

    // 大于1024就按照1024来取
    if (enum_up_bound > 1024)
    {
        enum_up_bound = 1024;
    }

    // 枚举的下界找小于最小TLB数量的32的倍数
    unsigned long enum_low_bound = min_TLB_num / 32;
    enum_low_bound = enum_low_bound * 32;

    if (enum_low_bound < 32)
    {
        enum_low_bound = 32;
    }

    if (enum_low_bound > 1024)
    {
        enum_low_bound = 1024;
    }

    assert(enum_low_bound <= enum_up_bound);

    // 步长按照上界除8之后小于这个值的32的倍数计算。最小不能高过32
    unsigned long step_size = (enum_up_bound - enum_low_bound) / 8;

    if (step_size % 32 == 0)
    {
        // 找到大于step_size最小32的倍数
        step_size = (step_size / 32) * 32;
    }
    else
    {
        step_size = (step_size / 32 + 1) * 32;
    }
    
    // 如果这个值小于32， 那就至少32
    if (step_size < 32)
    {
        step_size = 32;
    }

    cout << "find_best_param_of_direct_atom_template_warp_compress: thread_num_of_tblock setter param:" << " enum_low_bound:" << enum_low_bound << " , enum_up_bound:" << enum_up_bound << " , step_size:" << step_size << endl;

    long tblock_num;
    long thread_num_in_tblock;

    // tblock数量和BLB数量相同
    tblock_num = BLB_index->block_num;
    
    if (tblock_num > get_config()["MAX_TBLOCK_NUM"].as_integer() - 1)
    {
        tblock_num = get_config()["MAX_TBLOCK_NUM"].as_integer() - 1; 
    }

    // 登记参数调优器
    param_enumerater_t param_setter;
    // 性能影响差距不是很大
    register_integer_independ_param_to_enumerater(&param_setter, &thread_num_in_tblock, enum_low_bound, enum_up_bound, step_size);
    
    // 模板的指针
    direct_atom_template_warp_compress_t* target_template = (direct_atom_template_warp_compress_t*)(builder->template_vec[sub_matrix_id]);

    long best_thread_num_in_block;
    long best_tblock_num;
    float best_time = 0;
    float best_gflops = 0;

    // 查看是不是第一个模板的枚举
    bool is_first_enumerate = true;

    // 是不是要提前退出
    bool search_finished_by_strategy = false;

    // 一个个枚举
    while (set_param_combination_to_next(&param_setter) == false)
    {
        // 将参数写到模板中
        target_template->tblock_num = tblock_num;
        target_template->thread_num_in_block = thread_num_in_tblock;

        cout << "find_best_param_of_direct_atom_template_warp_compress: target_template->tblock_num:" << target_template->tblock_num << endl;
        cout << "find_best_param_of_direct_atom_template_warp_compress: target_template->thread_num_in_block:" << target_template->thread_num_in_block << endl;

        vector<int> sub_matrix_id_vec;
        sub_matrix_id_vec.push_back(sub_matrix_id);
        float exe_time = 0;
        float exe_gflops = 0;

        // 这里对模板的具体参数执行
        bool is_success_exe = part_execute_code_builder(builder, sub_matrix_id_vec, exe_time, exe_gflops, string(get_config()["ROOT_PATH_STR"].as_string()) + "/cuda_code", string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source", is_first_enumerate, true);

        // 如果不成功就跳过
        if (is_success_exe == false)
        {
            continue;
        }

        vector<float> param_vec;

        // 如果有数据集收集
        if (data_set_collector != NULL)
        {
            // 已有的内容存在一些积累
            assert(data_set_collector->accu_dense_param_strategy_type_vec.size() > 0);
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
            assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() > 0);
            assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() == data_set_collector->accu_compressed_sub_graph_node_type_vec.size());

            param_vec.push_back(target_template->thread_num_in_block);
            param_vec.push_back(target_template->tblock_num);
            param_vec.push_back(exe_gflops);
            
            data_set_collector->insert_template_node_and_param_to_cur_item_and_add_to_dataset(DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, param_vec);
        }

        is_first_enumerate = false;

        // 当glops更大的时候，就替换
        if (exe_gflops > best_gflops)
        {
            // 找到更好的参数了
            best_gflops = exe_gflops;
            best_time = exe_time;

            best_tblock_num = tblock_num;
            best_thread_num_in_block = thread_num_in_tblock;
        }

        // 如果有搜索策略，可能需要看看提前退出的问题
        if (search_strategy_ptr != NULL)
        {
            if (continue_search(search_strategy_ptr, exe_gflops) == false)
            {
                search_finished_by_strategy = true;
            }
        }

        if (search_finished_by_strategy == true)
        {
            break;
        }
    }

    // 建立并返回一个模板节点
    template_node_t return_node;

    return_node.type = DIRECT_ATOM_TEMPLATE_WARP_COMPRESS;
    
    direct_atom_template_warp_compress_node_param_t* param_ptr = new direct_atom_template_warp_compress_node_param_t();

    param_ptr->tblock_num = best_tblock_num;
    param_ptr->thread_num_in_block = best_thread_num_in_block;

    return_node.template_param = param_ptr;

    return_best_gflops = best_gflops;
    return_best_time = best_time;

    return return_node;
}

// 不带任何压缩的
template_node_t find_best_param_of_direct_atom_template(code_builder_t* builder, int sub_matrix_id, float& return_best_time, float& return_best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(builder != NULL && sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());
    assert(builder->op_manager != NULL && builder->op_manager->matrix != NULL && sub_matrix_id < builder->template_vec.size());
    assert(sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());

    assert(builder->template_type_vec[sub_matrix_id] == DIRECT_ATOM_TEMPLATE);
    assert(builder->template_vec[sub_matrix_id] != NULL && builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);
    
    index_of_compress_block_t* BLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2];
    index_of_compress_block_t* WLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[3];

    // 这个不执行调参，没用
    return_best_time = 9999999999999;
    return_best_gflops = 0;

    template_node_t return_node;

    return_node.type = DIRECT_ATOM_TEMPLATE;

    direct_atom_template_node_param_t* param_ptr = new direct_atom_template_node_param_t();
    
    return_node.template_param = param_ptr;

    return return_node;
}

template_node_t find_best_param_of_direct_atom_total_warp_reduce_template(code_builder_t* builder, int sub_matrix_id, float& return_best_time, float& return_best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(builder != NULL && sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());
    assert(builder->op_manager != NULL && builder->op_manager->matrix != NULL && sub_matrix_id < builder->template_vec.size());
    assert(sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());

    assert(builder->template_type_vec[sub_matrix_id] == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE);
    assert(builder->template_vec[sub_matrix_id] != NULL && builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);
    
    index_of_compress_block_t* BLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2];
    index_of_compress_block_t* WLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[3];

    // 有两个参数，一个是block数量和block内thread数量
    long tblock_num;
    long thread_num_in_block;

    // 创建一个参数枚举器，只有thread_num_in_block的参数是需要调整的
    param_enumerater_t param_setter;
    // 性能影响差距不是很大
    register_integer_independ_param_to_enumerater(&param_setter, &thread_num_in_block, 32, 512, 64);

    // 查看是不是第一个模板的枚举
    bool is_first_enumerate = true;

    // 记录最佳的时间和最佳的性能，以及对应的最佳参数
    float best_time = 0;
    float best_gflops = 0;
    long best_tblock_num = 0;
    long best_thread_num_in_block = 0;

    direct_atom_total_warp_reduce_template_t* target_template = (direct_atom_total_warp_reduce_template_t *)builder->template_vec[sub_matrix_id];

    bool search_finished_by_strategy = false;

    while (set_param_combination_to_next(&param_setter) == false)
    {
        // 查看WLB的数量
        unsigned long WLB_num = WLB_index->block_num;

        // 查看最多需要的线程数量，每个WLB需要32个线程
        unsigned long TLB_num = WLB_num * 32;

        // 计算线程块的数量，原则上线程数量可以稍微多一点
        tblock_num = TLB_num / thread_num_in_block;

        if (TLB_num % thread_num_in_block != 0)
        {
            tblock_num = tblock_num + 1;
        }

        if (tblock_num > get_config()["MAX_TBLOCK_NUM"].as_integer() - 1)
        {
            tblock_num = get_config()["MAX_TBLOCK_NUM"].as_integer() - 1;
        }
        
        // 将参数写到模板中
        target_template->tblock_num = tblock_num;
        target_template->thread_num_in_block = thread_num_in_block;

        cout << "find_best_param_of_direct_atom_total_warp_reduce_template: target_template->tblock_num:" << target_template->tblock_num << endl;
        cout << "find_best_param_of_direct_atom_total_warp_reduce_template: target_template->thread_num_in_block:" << target_template->thread_num_in_block << endl;

        vector<int> sub_matrix_id_vec;
        sub_matrix_id_vec.push_back(sub_matrix_id);
        float exe_time = 0;
        float exe_gflops = 0;

        // 这里对模板的具体参数执行
        bool is_success_exe = part_execute_code_builder(builder, sub_matrix_id_vec, exe_time, exe_gflops, string(get_config()["ROOT_PATH_STR"].as_string()) + "/cuda_code", string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source", is_first_enumerate, true);

        // 如果不成功就跳过
        if (is_success_exe == false)
        {
            continue;
        }

        vector<float> param_vec;

        // 如果有数据集收集
        if (data_set_collector != NULL)
        {
            // 已有的内容存在一些积累
            assert(data_set_collector->accu_dense_param_strategy_type_vec.size() > 0);
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
            assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() > 0);
            assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() == data_set_collector->accu_compressed_sub_graph_node_type_vec.size());

            param_vec.push_back(target_template->thread_num_in_block);
            param_vec.push_back(target_template->tblock_num);
            param_vec.push_back(exe_gflops);
            
            data_set_collector->insert_template_node_and_param_to_cur_item_and_add_to_dataset(DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE, param_vec);
        }

        is_first_enumerate = false;

        // 当glops更大的时候，就替换
        if (exe_gflops > best_gflops)
        {
            // 找到更好的参数了
            best_gflops = exe_gflops;
            best_time = exe_time;

            best_tblock_num = tblock_num;
            best_thread_num_in_block = thread_num_in_block;
        }

        // 如果有搜索策略，可能需要看看提前退出的问题
        if (search_strategy_ptr != NULL)
        {
            if (continue_search(search_strategy_ptr, exe_gflops) == false)
            {
                search_finished_by_strategy = true;
            }
        }

        if (search_finished_by_strategy == true)
        {
            break;
        }
    }

    // 产生一个新的节点，来记录当前最优的参数组合
    template_node_t return_node;

    direct_atom_total_warp_reduce_template_node_param_t* param_ptr = new direct_atom_total_warp_reduce_template_node_param_t();

    param_ptr->tblock_num = best_tblock_num;
    param_ptr->thread_num_in_block = best_thread_num_in_block;

    return_node.type = DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE;
    return_node.template_param = param_ptr;

    // 将最优性能传到外面
    return_best_time = best_time;
    return_best_gflops = best_gflops;
    
    return return_node;
}

template_node_t find_best_param_of_shared_memory_long_row_template(code_builder_t* builder, int sub_matrix_id, float& return_best_time, float& return_best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(builder != NULL && sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());
    assert(builder->op_manager != NULL && builder->op_manager->matrix != NULL && sub_matrix_id < builder->template_vec.size());
    assert(sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());

    assert(builder->template_type_vec[sub_matrix_id] == SHARED_MEMORY_LONG_ROW_TEMPLATE);
    assert(builder->template_vec[sub_matrix_id] != NULL && builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);
    
    index_of_compress_block_t* BLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2];
    index_of_compress_block_t* WLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[3];

    // 有两个参数，一个是block数量和block内thread数量
    long tblock_num;
    long thread_num_in_block;

    // tblock的和BLB的数量一致
    tblock_num = BLB_index->block_num;

    if (tblock_num > get_config()["MAX_TBLOCK_NUM"].as_integer() - 1)
    {
        tblock_num = get_config()["MAX_TBLOCK_NUM"].as_integer() - 1;
    }

    // 查看是不是第一个模板的枚举
    bool is_first_enumerate = true;

    // 记录最佳的时间和最佳的性能，以及对应的最佳参数
    float best_time = 0;
    float best_gflops = 0;
    long best_tblock_num = 0;
    long best_thread_num_in_block = 0;

    // 计算线程块内线程数量枚举的上界
    // 结算线程块内线程枚举的下界，和BLB中非零元的最小值有关
    // 获得BLB中非零元的最大数量
    unsigned long max_BLB_nnz = 0;
    unsigned long min_BLB_nnz = 9999999999999999;
    

    assert(BLB_index->coo_begin_index_arr != NULL);

    for (unsigned long BLB_id = 0; BLB_id < BLB_index->block_num; BLB_id++)
    {
        unsigned long first_BLB_nz_index = read_from_array_with_data_type(BLB_index->coo_begin_index_arr, BLB_index->data_type_of_coo_begin_index_arr, BLB_id);
        unsigned long next_BLB_nz_index = read_from_array_with_data_type(BLB_index->coo_begin_index_arr, BLB_index->data_type_of_coo_begin_index_arr, BLB_id + 1);

        assert(next_BLB_nz_index >= first_BLB_nz_index);

        // nz的数量
        unsigned long BLB_nnz = next_BLB_nz_index - first_BLB_nz_index;

        if (BLB_nnz > max_BLB_nnz)
        {
            max_BLB_nnz = BLB_nnz;
        }
        
        if (BLB_nnz < min_BLB_nnz)
        {
            min_BLB_nnz = BLB_nnz;
        }
    }

    if (min_BLB_nnz > max_BLB_nnz)
    {
        cout << "find_best_param_of_shared_memory_long_row_template: min_BLB_nnz:" << min_BLB_nnz << endl;
        cout << "find_best_param_of_shared_memory_long_row_template: max_BLB_nnz:" << min_BLB_nnz << endl;
    }

    assert(max_BLB_nnz >= min_BLB_nnz);
    
    // 按照32的倍数，找出大于max_BLB_nnz的最小的32的倍数
    unsigned long enum_up_bound = max_BLB_nnz / 32;
    enum_up_bound = enum_up_bound * 32;

    if (enum_up_bound % 32 != 0)
    {
        enum_up_bound = enum_up_bound + 32;
    }

    // 超出1024就还是1024
    if (enum_up_bound > 1024)
    {
        enum_up_bound = 1024;
    }

    if (enum_up_bound < 32)
    {
        enum_up_bound = 32;
    }

    unsigned long enum_low_bound = min_BLB_nnz / 32;
    enum_low_bound = enum_low_bound * 32;

    // 如果小于32，那就还是32
    if (enum_low_bound < 32)
    {
        enum_low_bound = 32;
    }

    if (enum_low_bound > 1024)
    {
        enum_low_bound = 1024;
    }

    if (enum_up_bound < enum_low_bound)
    {
        cout << "find_best_param_of_shared_memory_long_row_template: enum_up_bound:" << enum_up_bound << ", enum_low_bound:" << enum_low_bound << endl;    
    }

    assert(enum_up_bound >= enum_low_bound);

    // 步长
    // 步长按照上界除8之后小于这个值的32的倍数计算。最小不能高过32
    unsigned long step_size = (enum_up_bound - enum_low_bound) / 8;

    // 找到大于step_size的最小32的倍数
    if (step_size % 32 == 0)
    {
        step_size = (step_size / 32) * 32;
    }
    else
    {
        step_size = (step_size / 32 + 1) * 32;
    }
    
    // 如果这个值小于32， 那就至少32
    if (step_size < 32)
    {
        step_size = 32;
    }

    cout << "find_best_param_of_shared_memory_long_row_template: thread_num_of_tblock setter param:" << "enum_low_bound:" << enum_low_bound << ", enum_up_bound:" << enum_up_bound << " , step_size:" << step_size << endl;
    
    // 登记参数调优器
    param_enumerater_t param_setter;
    // 性能影响差距不是很大
    register_integer_independ_param_to_enumerater(&param_setter, &thread_num_in_block, enum_low_bound, enum_up_bound, step_size);
    
    shared_memory_long_row_template_t* target_template = (shared_memory_long_row_template_t*)(builder->template_vec[sub_matrix_id]);

    bool search_finished_by_strategy = false;

    while (set_param_combination_to_next(&param_setter) == false)
    {
        // 将参数写到模板中
        target_template->tblock_num = tblock_num;
        target_template->thread_num_in_block = thread_num_in_block;

        cout << "find_best_param_of_shared_memory_long_row_template: target_template->tblock_num:" << target_template->tblock_num << endl;
        cout << "find_best_param_of_shared_memory_long_row_template: target_template->thread_num_in_block:" << target_template->thread_num_in_block << endl;

        vector<int> sub_matrix_id_vec;
        sub_matrix_id_vec.push_back(sub_matrix_id);
        float exe_time = 0;
        float exe_gflops = 0;

        // 这里对模板的具体参数执行
        bool is_success_exe = part_execute_code_builder(builder, sub_matrix_id_vec, exe_time, exe_gflops, string(get_config()["ROOT_PATH_STR"].as_string()) + "/cuda_code", string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source", is_first_enumerate, true);

        // 如果不成功就跳过
        if (is_success_exe == false)
        {
            continue;
        }

        vector<float> param_vec;

        // 如果有数据集收集
        if (data_set_collector != NULL)
        {
            // 已有的内容存在一些积累
            assert(data_set_collector->accu_dense_param_strategy_type_vec.size() > 0);
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
            assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() > 0);
            assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() == data_set_collector->accu_compressed_sub_graph_node_type_vec.size());

            // 添加参数
            param_vec.push_back(target_template->tblock_num);
            param_vec.push_back(target_template->thread_num_in_block);
            param_vec.push_back(exe_gflops);

            // 性能
            data_set_collector->insert_template_node_and_param_to_cur_item_and_add_to_dataset(SHARED_MEMORY_LONG_ROW_TEMPLATE, param_vec);
        }

        is_first_enumerate = false;

        // 当glops更大的时候，就替换
        if (exe_gflops > best_gflops)
        {
            // 找到更好的参数了
            best_gflops = exe_gflops;
            best_time = exe_time;

            best_tblock_num = tblock_num;
            best_thread_num_in_block = thread_num_in_block;
        }

        // 如果有搜索策略，可能需要看看提前退出的问题
        if (search_strategy_ptr != NULL)
        {
            if (continue_search(search_strategy_ptr, exe_gflops) == false)
            {
                search_finished_by_strategy = true;
            }
        }

        if (search_finished_by_strategy == true)
        {
            break;
        }
    }

    // 将数据输出
    template_node_t return_node;

    return_node.type = SHARED_MEMORY_LONG_ROW_TEMPLATE;

    shared_memory_long_row_template_node_param_t* param_ptr = new shared_memory_long_row_template_node_param_t();

    param_ptr->tblock_num = best_tblock_num;
    param_ptr->thread_num_in_block = best_thread_num_in_block;

    return_node.template_param = param_ptr;

    return_best_time = best_time;
    return_best_gflops = best_gflops;

    return return_node;
}

template_node_t find_best_param_of_shared_memory_template_warp_compress(code_builder_t* builder, int sub_matrix_id, float& return_best_time, float& return_best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(builder != NULL && sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());
    assert(builder->op_manager != NULL && builder->op_manager->matrix != NULL && sub_matrix_id < builder->template_vec.size());
    assert(sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());

    assert(builder->template_type_vec[sub_matrix_id] == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS);
    assert(builder->template_vec[sub_matrix_id] != NULL && builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);
    
    index_of_compress_block_t* BLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2];
    index_of_compress_block_t* WLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[3];

    // 模板的指针
    shared_memory_template_warp_compress_t* target_template = (shared_memory_template_warp_compress_t*)(builder->template_vec[sub_matrix_id]);

    // 有三个参数要调，线程块数量，线程块大小
    long thread_num_in_tblock;
    long tblock_num;
    long thread_num_of_each_row;

    // tblock数量和BLB数量尽可能一致
    tblock_num = BLB_index->block_num;

    if (tblock_num > get_config()["MAX_TBLOCK_NUM"].as_integer() - 1)
    {
        tblock_num = get_config()["MAX_TBLOCK_NUM"].as_integer() - 1;
    }

    // 查看是不是第一个模板的枚举
    bool is_first_enumerate = true;

    // 记录最佳的时间和最佳的性能，以及对应的最佳参数
    float best_time = 0;
    float best_gflops = 0;
    long best_tblock_num = 0;
    long best_thread_num_in_block = 0;
    long best_thread_num_of_row_reduce = 0;


    // 定义thread_num_in_tblock的上界，和BLB中TLB的最大数量有关
    // 定义thread_num_in_tblock的下界，和BLB中TLB的最小数量有关
    unsigned long min_TLB_num = 4096;
    unsigned long max_TLB_num = 0;

    // 查看最大行TLB数量
    unsigned long max_row_TLB_num = 0;
    assert(target_template->row_offset_in_thread_tmp_result != NULL);

    for (unsigned long i = 0; i < target_template->size_of_row_offset_in_thread_tmp_result - 1; i++)
    {
        unsigned long cur_offset = read_from_array_with_data_type(target_template->row_offset_in_thread_tmp_result, target_template->data_type_of_row_offset_in_thread_tmp_result, i);
        unsigned long next_offset = read_from_array_with_data_type(target_template->row_offset_in_thread_tmp_result, target_template->data_type_of_row_offset_in_thread_tmp_result, i + 1);

        assert(next_offset >= cur_offset);

        unsigned long row_result_num = next_offset - cur_offset;

        if (row_result_num > max_row_TLB_num)
        {
            max_row_TLB_num = row_result_num;
        }
    }

    // 唯一需要调的参数应该只有thread的数量，最少32，从32的倍数开始调。上界是BLB中TLB数量的最大值最近的大于TLB数量的32的倍数
    // 首先找出最大的块TLB数量
    for (unsigned long BLB_id = 0; BLB_id < BLB_index->block_num; BLB_id++)
    {
        // 这个BLB第一个WLB的索引
        unsigned long cur_WLB_first_id = read_from_array_with_data_type(BLB_index->index_arr, BLB_index->index_data_type, BLB_id);
        // 下一个BLB的第一个WLB的索引
        unsigned long next_WLB_first_id = read_from_array_with_data_type(BLB_index->index_arr, BLB_index->index_data_type, BLB_id + 1);

        // 只要两个同时小于length，就算出他们的首个TLB索引
        if (cur_WLB_first_id < WLB_index->length && next_WLB_first_id < WLB_index->length)
        {
            unsigned long cur_BLB_first_TLB_id = read_from_array_with_data_type(WLB_index->index_arr, WLB_index->index_data_type, cur_WLB_first_id);
            unsigned long next_BLB_first_TLB_id = read_from_array_with_data_type(WLB_index->index_arr, WLB_index->index_data_type, next_WLB_first_id);

            unsigned long TLB_num = next_BLB_first_TLB_id - cur_BLB_first_TLB_id;

            if (TLB_num > max_TLB_num)
            {
                max_TLB_num = TLB_num;
            }

            if (TLB_num < min_TLB_num)
            {
                min_TLB_num = TLB_num;
            }
        }
    }

    unsigned long thread_num_in_tblock_up_bound = max_TLB_num / 32;
    thread_num_in_tblock_up_bound = thread_num_in_tblock_up_bound * 32;
    
    if (max_TLB_num % 32 != 0)
    {
        thread_num_in_tblock_up_bound = thread_num_in_tblock_up_bound + 32;
    }

    if (thread_num_in_tblock_up_bound > 1024)
    {
        thread_num_in_tblock_up_bound = 1024;
    }

    // 计算下界，下界是BLB中TLB的最小数量，如果下界太小了，那么乘阶段的并行度不够，如果下界太大了，那么在加阶段可能空闲的线程太多了
    unsigned long thread_num_in_tblock_low_bound = min_TLB_num;
    thread_num_in_tblock_low_bound = thread_num_in_tblock_low_bound / 32;
    thread_num_in_tblock_low_bound = thread_num_in_tblock_low_bound * 32;
    
    // 计算下界
    if (thread_num_in_tblock_low_bound < 32)
    {
        thread_num_in_tblock_low_bound = 32;
    }

    if (thread_num_in_tblock_low_bound > 1024)
    {
        thread_num_in_tblock_low_bound = 1024;
    }

    // 下界是32，算出步长
    unsigned long step_size = (thread_num_in_tblock_up_bound - thread_num_in_tblock_low_bound) / 4;

    // 换成大于step_size的最小32的倍数
    if (step_size % 32 == 0)
    {
        step_size = step_size / 32;
        step_size = step_size * 32;
    }
    else
    {
        step_size = (step_size / 32 + 1) * 32;
    }

    // 如果这个值小于32， 那就至少32
    if (step_size < 32)
    {
        step_size = 32;
    }

    param_enumerater_t param_setter;

    // 注册一下
    register_integer_independ_param_to_enumerater(&param_setter, &thread_num_in_tblock, thread_num_in_tblock_low_bound, thread_num_in_tblock_up_bound, step_size);

    bool search_finished_by_strategy = false;

    // 调参
    while (set_param_combination_to_next(&param_setter) == false)
    {
        // 按照4倍的位置去变化
        for (unsigned long thread_num_of_row_reduce = 1; thread_num_of_row_reduce <= 32 && thread_num_of_row_reduce <= max_row_TLB_num; thread_num_of_row_reduce = thread_num_of_row_reduce * 4)
        {
            target_template->tblock_num = tblock_num;
            target_template->thread_num_in_block = thread_num_in_tblock;
            target_template->thread_num_of_row_reduce = thread_num_of_row_reduce;

            // 之后执行内核
            cout << "find_best_param_of_shared_memory_template_warp_compress: target_template->tblock_num:" << target_template->tblock_num << endl;
            cout << "find_best_param_of_shared_memory_template_warp_compress: target_template->thread_num_in_block:" << target_template->thread_num_in_block << endl;
            cout << "find_best_param_of_shared_memory_template_warp_compress: target_template->thread_num_of_row_reduce:" << target_template->thread_num_of_row_reduce << endl;

            vector<int> sub_matrix_id_vec;
            sub_matrix_id_vec.push_back(sub_matrix_id);
            float exe_time = 0;
            float exe_gflops = 0;

            // 这里对模板的具体参数执行
            bool is_success_exe = part_execute_code_builder(builder, sub_matrix_id_vec, exe_time, exe_gflops, string(get_config()["ROOT_PATH_STR"].as_string()) + "/cuda_code", string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source", is_first_enumerate, true);

            // 如果不成功就跳过
            if (is_success_exe == false)
            {
                continue;
            }
            else
            {
                is_first_enumerate = false;
            }

            vector<float> param_vec;

            // 如果有数据集收集
            if (data_set_collector != NULL)
            {
                // 已有的内容存在一些积累
                assert(data_set_collector->accu_dense_param_strategy_type_vec.size() > 0);
                assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
                assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() > 0);
                assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() == data_set_collector->accu_compressed_sub_graph_node_type_vec.size());
                
                param_vec.push_back(target_template->tblock_num);
                param_vec.push_back(target_template->thread_num_in_block);
                param_vec.push_back(target_template->thread_num_of_row_reduce);
                param_vec.push_back(exe_gflops);

                data_set_collector->insert_template_node_and_param_to_cur_item_and_add_to_dataset(SHARED_MEMORY_TEMPLATE_WARP_COMPRESS, param_vec);
            }


            // 当glops更大的时候，就替换
            if (exe_gflops > best_gflops)
            {
                // 找到更好的参数了
                best_gflops = exe_gflops;
                best_time = exe_time;
                best_thread_num_of_row_reduce = thread_num_of_row_reduce;

                best_tblock_num = tblock_num;
                best_thread_num_in_block = thread_num_in_tblock;
            }

            // 如果有搜索策略，可能需要看看提前退出的问题
            if (search_strategy_ptr != NULL)
            {
                if (continue_search(search_strategy_ptr, exe_gflops) == false)
                {
                    search_finished_by_strategy = true;
                }
            }

            if (search_finished_by_strategy == true)
            {
                break;
            }
        }

        if (search_finished_by_strategy == true)
        {
            break;
        }
    }
    
    // 将数据输出
    template_node_t return_node;

    return_node.type = SHARED_MEMORY_TEMPLATE_WARP_COMPRESS;

    shared_memory_template_warp_compress_node_param_t* param_ptr = new shared_memory_template_warp_compress_node_param_t();

    param_ptr->tblock_num = best_tblock_num;
    param_ptr->thread_num_in_block = best_thread_num_in_block;
    param_ptr->thread_num_of_row_reduce = best_thread_num_of_row_reduce;

    return_node.template_param = param_ptr;

    return_best_time = best_time;
    return_best_gflops = best_gflops;

    return return_node;
}

template_node_t find_best_param_of_shared_memory_template(code_builder_t* builder, int sub_matrix_id, float& return_best_time, float& return_best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(builder != NULL && sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());
    assert(builder->op_manager != NULL && builder->op_manager->matrix != NULL && sub_matrix_id < builder->template_vec.size());
    assert(sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());

    assert(builder->template_type_vec[sub_matrix_id] == SHARED_MEMORY_TEMPLATE);
    assert(builder->template_vec[sub_matrix_id] != NULL && builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);
    
    index_of_compress_block_t* BLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2];
    index_of_compress_block_t* WLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[3];

    // 放弃调优，直接输出
    // 这个不执行调参，没用
    return_best_time = 9999999999999;
    return_best_gflops = 0;

    template_node_t return_node;

    return_node.type = SHARED_MEMORY_TEMPLATE;

    shared_memory_template_t* param_ptr = new shared_memory_template_t();
    
    return_node.template_param = param_ptr;

    return return_node;
}

template_node_t find_best_param_of_shared_memory_total_warp_reduce_template(code_builder_t* builder, int sub_matrix_id, float& return_best_time, float& return_best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(builder != NULL && sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());
    assert(builder->op_manager != NULL && builder->op_manager->matrix != NULL && sub_matrix_id < builder->template_vec.size());
    assert(sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());

    assert(builder->template_type_vec[sub_matrix_id] == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE);
    assert(builder->template_vec[sub_matrix_id] != NULL && builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);
    
    index_of_compress_block_t* BLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2];
    index_of_compress_block_t* WLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[3];

    // 模板的指针
    shared_memory_total_warp_reduce_template_t* target_template = (shared_memory_total_warp_reduce_template_t*)(builder->template_vec[sub_matrix_id]);

    // 有三个参数要调，线程块数量，线程块大小
    long thread_num_in_tblock;
    long tblock_num;
    long thread_num_of_each_row;

    // tblock数量和BLB数量尽可能一致
    tblock_num = BLB_index->block_num;

    if (tblock_num > get_config()["MAX_TBLOCK_NUM"].as_integer() - 1)
    {
        tblock_num = get_config()["MAX_TBLOCK_NUM"].as_integer() - 1;
    }

    // 查看是不是第一个模板的枚举
    bool is_first_enumerate = true;

    // 记录最佳的时间和最佳的性能，以及对应的最佳参数
    float best_time = 0;
    float best_gflops = 0;
    long best_tblock_num = 0;
    long best_thread_num_in_block = 0;
    long best_thread_num_of_row_reduce = 0;

    // 定义thread_num_in_tblock的上界，和BLB中WLB的最大数量有关
    unsigned long max_WLB_num_in_BLB = 0;
    // 定义thread_num_in_tblock的下界，和BLB中WLB最小的数量有关
    unsigned long min_WLB_num_in_BLB = 1024;

    // 查看最大行WLB数量
    unsigned long max_row_WLB_num = 0;

    // 遍历结果偏移量，查看最大行非零元数量
    assert(target_template->row_offset_in_warp_tmp_result != NULL);

    for (unsigned long i = 0; i < target_template->size_of_row_offset_in_warp_tmp_result - 1; i++)
    {
        unsigned long cur_row_offset = read_from_array_with_data_type(target_template->row_offset_in_warp_tmp_result, target_template->data_type_of_row_offset_in_warp_tmp_result, i);
        unsigned long next_row_offset = read_from_array_with_data_type(target_template->row_offset_in_warp_tmp_result, target_template->data_type_of_row_offset_in_warp_tmp_result, i + 1);

        assert(next_row_offset >= cur_row_offset);

        // 结果的数量
        unsigned long row_result_num = next_row_offset - cur_row_offset;

        if (row_result_num > max_row_WLB_num)
        {
            max_row_WLB_num = row_result_num;
        }
    }

    assert(BLB_index->index_arr != NULL);

    for (unsigned long i = 0; i < BLB_index->block_num; i++)
    {
        unsigned long cur_first_WLB_index = read_from_array_with_data_type(BLB_index->index_arr, BLB_index->index_data_type, i);
        unsigned long next_first_WLB_index = read_from_array_with_data_type(BLB_index->index_arr, BLB_index->index_data_type, i + 1);

        assert(next_first_WLB_index >= cur_first_WLB_index);

        // BLB中WLB的数量
        unsigned long WLB_num_of_cur_BLB = next_first_WLB_index - cur_first_WLB_index;

        if (WLB_num_of_cur_BLB > max_WLB_num_in_BLB)
        {
            max_WLB_num_in_BLB = WLB_num_of_cur_BLB;
        }

        if (WLB_num_of_cur_BLB < min_WLB_num_in_BLB)
        {
            min_WLB_num_in_BLB = WLB_num_of_cur_BLB;
        }
    }

    // thread_num_in_tblock的上界
    unsigned long up_bound_of_thread_num_in_block = max_WLB_num_in_BLB * 32;

    if (up_bound_of_thread_num_in_block > 1024)
    {
        up_bound_of_thread_num_in_block = 1024;
    }

    // 下界，是WLB的最小值*0.9，保证一个适中的并行度
    unsigned long low_bound_of_thread_num_in_block = min_WLB_num_in_BLB * 32;

    if (low_bound_of_thread_num_in_block < 32)
    {
        low_bound_of_thread_num_in_block = 32;
    }

    if (low_bound_of_thread_num_in_block > 1024)
    {
        low_bound_of_thread_num_in_block = 1024;
    }
    
    // 下界是32，步长是除8
    unsigned long step_size = (up_bound_of_thread_num_in_block - low_bound_of_thread_num_in_block) / 8;

    // 用大于stepsize的最小32的倍数
    if (step_size % 32 != 0)
    {
        step_size = (step_size / 32 + 1) * 32;
    }

    // 如果这个值小于32， 那就至少32
    if (step_size < 32)
    {
        step_size = 32;
    }

    // 注册thread_num_in_block
    param_enumerater_t param_setter;

    // 注册一下
    register_integer_independ_param_to_enumerater(&param_setter, &thread_num_in_tblock, low_bound_of_thread_num_in_block, up_bound_of_thread_num_in_block, step_size);

    bool search_finished_by_strategy = false;
    
    // 执行枚举
    while (set_param_combination_to_next(&param_setter) == false)
    {
        for (unsigned long thread_num_of_row_reduce = 1; thread_num_of_row_reduce <= 32 && thread_num_of_row_reduce <= max_row_WLB_num; thread_num_of_row_reduce = thread_num_of_row_reduce * 2)
        {
            target_template->tblock_num = tblock_num;
            target_template->thread_num_in_block = thread_num_in_tblock;
            target_template->thread_num_of_row_reduce = thread_num_of_row_reduce;

            // 之后执行内核
            cout << "find_best_param_of_shared_memory_total_warp_reduce_template: target_template->tblock_num:" << target_template->tblock_num << endl;
            cout << "find_best_param_of_shared_memory_total_warp_reduce_template: target_template->thread_num_in_block:" << target_template->thread_num_in_block << endl;
            cout << "find_best_param_of_shared_memory_total_warp_reduce_template: target_template->thread_num_of_row_reduce:" << target_template->thread_num_of_row_reduce << endl;

            vector<int> sub_matrix_id_vec;
            sub_matrix_id_vec.push_back(sub_matrix_id);
            float exe_time = 0;
            float exe_gflops = 0;

            // 这里对模板的具体参数执行
            bool is_success_exe = part_execute_code_builder(builder, sub_matrix_id_vec, exe_time, exe_gflops, string(get_config()["ROOT_PATH_STR"].as_string()) + "/cuda_code", string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source", is_first_enumerate, true);

            // 如果不成功就跳过
            if (is_success_exe == false)
            {
                continue;
            }
            else
            {
                is_first_enumerate = false;
            }

            vector<float> param_vec;

            // 如果有数据集收集
            if (data_set_collector != NULL)
            {
                // 已有的内容存在一些积累
                assert(data_set_collector->accu_dense_param_strategy_type_vec.size() > 0);
                assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
                assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() > 0);
                assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() == data_set_collector->accu_compressed_sub_graph_node_type_vec.size());

                // 加入参数
                param_vec.push_back(target_template->tblock_num);
                param_vec.push_back(target_template->thread_num_in_block);
                param_vec.push_back(target_template->thread_num_of_row_reduce);
                param_vec.push_back(exe_gflops);

                data_set_collector->insert_template_node_and_param_to_cur_item_and_add_to_dataset(SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE, param_vec);
            }

            // 当glops更大的时候，就替换
            if (exe_gflops > best_gflops)
            {
                // 找到更好的参数了
                best_gflops = exe_gflops;
                best_time = exe_time;
                best_thread_num_of_row_reduce = thread_num_of_row_reduce;

                best_tblock_num = tblock_num;
                best_thread_num_in_block = thread_num_in_tblock;
            }

            // 如果有搜索策略，可能需要看看提前退出的问题
            if (search_strategy_ptr != NULL)
            {
                if (continue_search(search_strategy_ptr, exe_gflops) == false)
                {
                    search_finished_by_strategy = true;
                }
            }

            if (search_finished_by_strategy == true)
            {
                break;
            }
        }

        if (search_finished_by_strategy == true)
        {
            break;
        }
    }

    // 将数据输出
    template_node_t return_node;

    return_node.type = SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE;

    shared_memory_total_warp_reduce_template_node_param_t* param_ptr = new shared_memory_total_warp_reduce_template_node_param_t();

    param_ptr->tblock_num = best_tblock_num;
    param_ptr->thread_num_in_block = best_thread_num_in_block;
    param_ptr->thread_num_of_row_reduce = best_thread_num_of_row_reduce;

    return_node.template_param = param_ptr;

    return_best_time = best_time;
    return_best_gflops = best_gflops;

    return return_node;
}

template_node_t find_best_param_of_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce(code_builder_t* builder, int sub_matrix_id, float& return_best_time, float& return_best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(builder != NULL && sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());
    assert(builder->op_manager != NULL && builder->op_manager->matrix != NULL && sub_matrix_id < builder->template_vec.size());
    assert(sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());

    assert(builder->template_type_vec[sub_matrix_id] == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE);
    assert(builder->template_vec[sub_matrix_id] != NULL && builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);
    
    index_of_compress_block_t* BLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2];
    index_of_compress_block_t* WLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[3];
    index_of_compress_block_t* TLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[4];

    unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t* target_template = (unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t*)builder->template_vec[sub_matrix_id];

    // 记录最佳的时间和最佳的性能，以及对应的最佳参数
    float best_time = 0;
    float best_gflops = 0;
    long best_thread_num_in_block = 0;

    bool is_first_enumerate = true;

    // 只有一个参数要调节，就是BLB的数量
    long thread_num_of_tblock;

    // 带warp reduce的CSR5-like模板不存在一个线程负责多个TLB的情况，所以线程数量一定要够用，故线程线程块中的线程数量有一个下界，保证可以覆盖所有TLB
    unsigned long TLB_num = TLB_index->block_num;

    unsigned long low_bound_thread_num_in_tblock;

    low_bound_thread_num_in_tblock = TLB_num / (get_config()["MAX_TBLOCK_NUM"].as_integer() - 1);

    // 如果小于32，则赋值为32
    if (low_bound_thread_num_in_tblock < 32)
    {
        low_bound_thread_num_in_tblock = 32;
    }

    // 下界必须是32的倍数，向上取整
    if (low_bound_thread_num_in_tblock % 32 != 0)
    {
        low_bound_thread_num_in_tblock = (low_bound_thread_num_in_tblock / 32 + 1) * 32;
    }

    // 如果大于512，那就直接返回
    if (low_bound_thread_num_in_tblock > 512)
    {
        return_best_time = 9999999999999;
        return_best_gflops = 0;

        template_node_t return_node;

        return_node.type = UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE;

        unaligned_warp_reduce_same_TLB_size_template_node_param_t* param_ptr = new unaligned_warp_reduce_same_TLB_size_template_node_param_t();
        
        return_node.template_param = param_ptr;

        return return_node;
    }

    // 步长
    unsigned long step_size = (512 - low_bound_thread_num_in_tblock) / 4;

    // 步长是大于step_size的最小32的倍数
    if (step_size % 32 != 0)
    {
        step_size = (step_size / 32 + 1) * 32;
    }

    // 如果这个值小于32， 那就至少32
    if (step_size < 32)
    {
        step_size = 32;
    }

    param_enumerater_t param_setter;

    // 注册对应的参数
    register_integer_independ_param_to_enumerater(&param_setter, &thread_num_of_tblock, low_bound_thread_num_in_tblock, 512, step_size);

    bool search_finished_by_strategy = false;

    // 执行参数的枚举
    while (set_param_combination_to_next(&param_setter) == false)
    {
        // 写对应的参数
        target_template->thread_num_in_block = thread_num_of_tblock;

        // 执行对应模板
        cout << "find_best_param_of_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce: target_template->thread_num_in_block:" << target_template->thread_num_in_block << endl;

        vector<int> sub_matrix_id_vec;
        sub_matrix_id_vec.push_back(sub_matrix_id);
        float exe_time = 0;
        float exe_gflops = 0;

        // 这里对模板的具体参数执行
        bool is_success_exe = part_execute_code_builder(builder, sub_matrix_id_vec, exe_time, exe_gflops, string(get_config()["ROOT_PATH_STR"].as_string()) + "/cuda_code", string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source", is_first_enumerate, true);

        // 如果不成功就跳过
        if (is_success_exe == false)
        {
            continue;
        }
        else
        {
            is_first_enumerate = false;
        }

        vector<float> param_vec;

        // 如果有数据集收集
        if (data_set_collector != NULL)
        {
            // 已有的内容存在一些积累
            assert(data_set_collector->accu_dense_param_strategy_type_vec.size() > 0);
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
            assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() > 0);
            assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() == data_set_collector->accu_compressed_sub_graph_node_type_vec.size());

            param_vec.push_back(target_template->thread_num_in_block);
            param_vec.push_back(exe_gflops);

            data_set_collector->insert_template_node_and_param_to_cur_item_and_add_to_dataset(UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE, param_vec);
        }

        // 当glops更大的时候，就替换
        if (exe_gflops > best_gflops)
        {
            // 找到更好的参数了
            best_gflops = exe_gflops;
            best_time = exe_time;
            best_thread_num_in_block = thread_num_of_tblock;
        }

        // 如果有搜索策略，可能需要看看提前退出的问题
        if (search_strategy_ptr != NULL)
        {
            if (continue_search(search_strategy_ptr, exe_gflops) == false)
            {
                search_finished_by_strategy = true;
            }
        }

        if (search_finished_by_strategy == true)
        {
            break;
        }
    }

    // 将数据输出
    template_node_t return_node;

    return_node.type = UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE;

    unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_node_param_t* param_ptr = new unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_node_param_t();

    param_ptr->thread_num_in_block = best_thread_num_in_block;

    return_node.template_param = param_ptr;

    return_best_time = best_time;
    return_best_gflops = best_gflops;

    return return_node;
}

template_node_t find_best_param_of_unaligned_warp_reduce_same_TLB_size_template(code_builder_t* builder, int sub_matrix_id, float& return_best_time, float& return_best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(builder != NULL && sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());
    assert(builder->op_manager != NULL && builder->op_manager->matrix != NULL && sub_matrix_id < builder->template_vec.size());
    assert(sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());

    assert(builder->template_type_vec[sub_matrix_id] == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE);
    assert(builder->template_vec[sub_matrix_id] != NULL && builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);
    
    index_of_compress_block_t* BLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[2];
    index_of_compress_block_t* WLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[3];
    index_of_compress_block_t* TLB_index = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[4];

    unaligned_warp_reduce_same_TLB_size_template_t* target_template = (unaligned_warp_reduce_same_TLB_size_template_t*)builder->template_vec[sub_matrix_id];

    unsigned long TLB_num = TLB_index->block_num;

    // 有两个参数，一个是block数量和block内thread数量
    long tblock_num;
    long thread_num_in_block;

    // 创建一个参数枚举器，只有thread_num_in_block的参数是需要调整的
    param_enumerater_t param_setter;
    // 性能影响差距不是很大
    register_integer_independ_param_to_enumerater(&param_setter, &thread_num_in_block, 0, 256, 64);

    // 查看是不是第一个模板的枚举
    bool is_first_enumerate = true;

    // 记录最佳的时间和最佳的性能，以及对应的最佳参数
    float best_time = 0;
    float best_gflops = 0;
    long best_tblock_num = 0;
    long best_thread_num_in_block = 0;

    // 查看是不是要提前退出
    bool search_finished_by_strategy = false;

    while (set_param_combination_to_next(&param_setter) == false)
    {
        if (thread_num_in_block == 0)
        {
            continue;
        }

        // 让tblock稍微多一点，正好可以让所有线程覆盖TLB
        tblock_num = TLB_num / thread_num_in_block;

        if (TLB_num % thread_num_in_block != 0)
        {
            tblock_num = tblock_num + 1;
        }

        if (tblock_num > get_config()["MAX_TBLOCK_NUM"].as_integer() - 1)
        {
            tblock_num = get_config()["MAX_TBLOCK_NUM"].as_integer() - 1;
        }

        target_template->thread_num_in_block = thread_num_in_block;
        target_template->tblock_num = tblock_num;

        // 执行对应的内核
        // 之后执行内核
        cout << "find_best_param_of_unaligned_warp_reduce_same_TLB_size_template: target_template->tblock_num:" << target_template->tblock_num << endl;
        cout << "find_best_param_of_unaligned_warp_reduce_same_TLB_size_template: target_template->thread_num_in_block:" << target_template->thread_num_in_block << endl;

        vector<int> sub_matrix_id_vec;
        sub_matrix_id_vec.push_back(sub_matrix_id);
        float exe_time = 0;
        float exe_gflops = 0;

        // 这里对模板的具体参数执行
        bool is_success_exe = part_execute_code_builder(builder, sub_matrix_id_vec, exe_time, exe_gflops, string(get_config()["ROOT_PATH_STR"].as_string()) + "/cuda_code", string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source", is_first_enumerate, true);

        // 如果不成功就跳过
        if (is_success_exe == false)
        {
            continue;
        }
        else
        {
            is_first_enumerate = false;
        }

        vector<float> param_vec;

        // 如果有数据集收集
        if (data_set_collector != NULL)
        {
            // cout << "find_best_param_of_unaligned_warp_reduce_same_TLB_size_template: need to collect ml data" << endl;
            // 已有的内容存在一些积累
            assert(data_set_collector->accu_dense_param_strategy_type_vec.size() > 0);
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
            assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() > 0);
            assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() == data_set_collector->accu_compressed_sub_graph_node_type_vec.size());

            param_vec.push_back(target_template->thread_num_in_block);
            param_vec.push_back(target_template->tblock_num);
            param_vec.push_back(exe_gflops);

            data_set_collector->insert_template_node_and_param_to_cur_item_and_add_to_dataset(UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE, param_vec);
        }

        // 当glops更大的时候，就替换
        if (exe_gflops > best_gflops)
        {
            // 找到更好的参数了
            best_gflops = exe_gflops;
            best_time = exe_time;

            best_tblock_num = tblock_num;
            best_thread_num_in_block = thread_num_in_block;
        }

        // 如果有搜索策略，可能需要看看提前退出的问题
        if (search_strategy_ptr != NULL)
        {
            if (continue_search(search_strategy_ptr, exe_gflops) == false)
            {
                search_finished_by_strategy = true;
            }
        }

        if (search_finished_by_strategy == true)
        {
            break;
        }
    }

    // 将数据输出
    template_node_t return_node;

    return_node.type = UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE;

    unaligned_warp_reduce_same_TLB_size_template_node_param_t* param_ptr = new unaligned_warp_reduce_same_TLB_size_template_node_param_t();

    param_ptr->thread_num_in_block = best_thread_num_in_block;
    param_ptr->tblock_num = best_tblock_num;

    return_node.template_param = param_ptr;

    return_best_time = best_time;
    return_best_gflops = best_gflops;

    return return_node;
}

void execute_sub_matrix_exe_graph_with_param_strategy(sparse_struct_t* matrix, unsigned long sub_matrix_id, exe_compressed_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy)
{
    assert(sub_graph != NULL && sub_graph_param_strategy != NULL);
    assert(matrix != NULL && sub_graph->exe_node_vec.size() > 0 && sub_graph_param_strategy->param_strategy_vec.size() > 0);
    assert(sub_graph_param_strategy->param_strategy_vec.size() == sub_graph->exe_node_vec.size());

    // 这个时候matrix仅仅被压缩过
    assert(sub_matrix_id < matrix->block_coor_table.item_arr.size() && matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    // 索引只有两个
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    // cout << "row_padding_direct_param_strategy->padding_row_length:" << ((compressed_row_padding_direct_param_strategy_t*)sub_graph_param_strategy.param_strategy_vec[0].param_strategy)->padding_row_length << endl;

    // sub_graph_param_strategy和exe_node_vec中的内容交错执行
    for (unsigned long i = 0; i < sub_graph_param_strategy->param_strategy_vec.size(); i++)
    {
        // 执行对应的参数设定
        param_strategy_node_t strategy_node = sub_graph_param_strategy->param_strategy_vec[i];
        assert(strategy_node.param_strategy != NULL && strategy_node.param != NULL);
        exe_node_t exe_node = sub_graph->exe_node_vec[i];
        assert(exe_node.param != NULL);
        assert(exe_node.param == strategy_node.param);

        // 查看内容
        // cout << "row_padding_direct_param_strategy->padding_row_length:" << ((compressed_row_padding_direct_param_strategy_t*)strategy_node.param)->padding_row_length << endl;

        execute_param_strategy_node_of_sub_compressed_matrix(&strategy_node, matrix, sub_matrix_id);
        
        // 执行对应的图节点
        execute_exe_node_in_compressed_sub_matrix(matrix, sub_matrix_id, exe_node);        
    }

    // 对应子块的所有索引的is_sort_arr是NULL
    for (unsigned long i = 0; i < matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size(); i++)
    {
        assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[i]->is_sort_arr == NULL);
    }
}

sparse_struct_t* execute_dense_matrix_exe_graph_with_param_strategy(exe_dense_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy)
{
    assert(sub_graph != NULL && sub_graph_param_strategy != NULL);
    
    // 两个图的大小是一样的
    assert(sub_graph->exe_node_vec.size() == sub_graph_param_strategy->param_strategy_vec.size());
    assert(sub_graph->exe_node_vec.size() > 0);

    // 策略和执行节点的参数指针是对应的
    for (unsigned long i = 0; i < sub_graph->exe_node_vec.size(); i++)
    {
        assert(sub_graph->exe_node_vec[i].param == sub_graph_param_strategy->param_strategy_vec[i].param);
        assert(sub_graph_param_strategy->param_strategy_vec[i].param_strategy != NULL);
    }

    // 创造一个临时的图
    // 让这张图中的内容一步步执行execute_node_of_dense_sub_graph
    exe_graph_t graph;
    graph.dense_sub_graph = *sub_graph;
    graph.builder = NULL;
    graph.op_manager = NULL;
    
    // 执行对应的策略和对应的优化节点
    for (unsigned long i = 0; i < sub_graph->exe_node_vec.size(); i++)
    {
        // 执行对应的参数设定
        param_strategy_node_t strategy_node = sub_graph_param_strategy->param_strategy_vec[i];
        assert(strategy_node.param_strategy != NULL && strategy_node.param != NULL);
        exe_node_t exe_node = sub_graph->exe_node_vec[i];
        assert(exe_node.param != NULL);
        assert(exe_node.param == strategy_node.param);

        // 执行对应的对应的策略，如果还没有op_manager指针，那就直接传一个空指针进去
        if (graph.op_manager == NULL)
        {
            execute_param_strategy_node_of_dense_matrix(&strategy_node, NULL);
        }
        else
        {
            assert(graph.op_manager->matrix != NULL);
            execute_param_strategy_node_of_dense_matrix(&strategy_node, graph.op_manager->matrix);
        }

        // 执行对应执行图中的节点
        execute_node_of_dense_sub_graph(&graph, i);
    }

    // 执行完之后矩阵和操作器是一定存在的
    assert(graph.op_manager != NULL && graph.op_manager->matrix != NULL);

    // 析构操作管理器
    sparse_struct_t *return_matrix_ptr = graph.op_manager->matrix;

    delete graph.op_manager;

    return return_matrix_ptr;
}

void add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(exe_dense_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy, exe_node_type node_type, exe_node_param_set_strategy strategy_type, void* strategy_param_ptr)
{
    assert(sub_graph != NULL && sub_graph_param_strategy != NULL && strategy_param_ptr != NULL);

    // 根据子块类型，为稠密视图加入不同的节点
    if (node_type == BEGIN_MEMORY_CACHE_INPUT_FILE)
    {
        // 之前不能出现所有的稠密视图操作
        assert(sub_graph->preorder_node_set.count(BEGIN_INPUT_FILE) == 0);
        assert(sub_graph->preorder_node_set.count(BEGIN_ARTIFICIAL_INPUT) == 0);
        assert(sub_graph->preorder_node_set.count(BEGIN_MEMORY_CACHE_INPUT_FILE) == 0);
        assert(sub_graph->preorder_node_set.count(DENSE_ROW_COARSE_SORT) == 0);
        assert(sub_graph->preorder_node_set.count(DENSE_FINE_SORT) == 0);
        assert(sub_graph->preorder_node_set.count(DENSE_TOTAL_ROW_LEVEL_PADDING) == 0);
        assert(sub_graph->preorder_node_set.count(DENSE_BLOCK_SORT) == 0);
        assert(sub_graph->preorder_node_set.count(DENSE_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(DENSE_FIXED_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESS) == 0);

        // 初始化一个空白的节点
        exe_node_t node;
        node.type = BEGIN_MEMORY_CACHE_INPUT_FILE;
        node.param = new exe_begin_memory_cache_input_file_param_t();

        sub_graph->exe_node_vec.push_back(node);
        sub_graph->preorder_node_set.insert(node.type);
    }
    else if (node_type == DENSE_ROW_COARSE_SORT)
    {
        // 直接不能出现除了输入节点之外的所有参数
        assert(sub_graph->preorder_node_set.count(DENSE_ROW_COARSE_SORT) == 0);
        assert(sub_graph->preorder_node_set.count(DENSE_FINE_SORT) == 0);
        assert(sub_graph->preorder_node_set.count(DENSE_TOTAL_ROW_LEVEL_PADDING) == 0);
        assert(sub_graph->preorder_node_set.count(DENSE_BLOCK_SORT) == 0);
        assert(sub_graph->preorder_node_set.count(DENSE_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(DENSE_FIXED_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESS) == 0);

        // 增加一个执行节点
        exe_node_t node;
        node.type = DENSE_ROW_COARSE_SORT;
        node.param = new exe_dense_row_coarse_sort_param_t();
        
        sub_graph->exe_node_vec.push_back(node);
        sub_graph->preorder_node_set.insert(node.type);
    }
    else if (node_type == COMPRESS)
    {
        // 之前不能出现压缩
        assert(sub_graph->preorder_node_set.count(COMPRESS) == 0);

        // 增加一个压缩执行节点
        exe_node_t node;
        node.type = COMPRESS;
        node.param = new exe_compress_param_t();
        
        sub_graph->exe_node_vec.push_back(node);
        sub_graph->preorder_node_set.insert(node.type);
    }
    else if (node_type == DENSE_ROW_DIV)
    {
        // 之前不能出现压缩
        assert(sub_graph->preorder_node_set.count(COMPRESS) == 0);

        exe_node_t node;
        node.type = DENSE_ROW_DIV;
        node.param = new exe_dense_row_div_param_t();

        sub_graph->exe_node_vec.push_back(node);
        sub_graph->preorder_node_set.insert(node.type);
    }
    else
    {
        cout << "add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph: exe node type is not supported" << endl;
        assert(false);
    }

    // 将策略节点加到对应的视图中
    if (strategy_type == DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY)
    {
        // 最后一个节点是输入节点
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].type == BEGIN_MEMORY_CACHE_INPUT_FILE);
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param != NULL);

        param_strategy_node_t node = init_dense_begin_memory_cache_input_file_direct_param_strategy(*((dense_begin_memory_cache_input_file_direct_param_strategy_t*)strategy_param_ptr), (exe_begin_memory_cache_input_file_param_t *)(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param));

        assert(node.param_strategy != NULL);
        assert(node.param != NULL && node.param == sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param);
        assert(node.node_type == BEGIN_MEMORY_CACHE_INPUT_FILE);
        assert(node.strategy_type == DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY);

        sub_graph_param_strategy->param_strategy_vec.push_back(node);
    }
    else if (strategy_type == DENSE_ROW_COARSE_SORT_FIXED_PARAM_STRATEGY)
    {
        // 最后一个节点是排序节点
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].type == DENSE_ROW_COARSE_SORT);
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param != NULL);

        param_strategy_node_t node = init_dense_row_coarse_sort_fixed_param_strategy(*((dense_row_coarse_sort_fixed_param_strategy_t*)strategy_param_ptr), (exe_dense_row_coarse_sort_param_t *)(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param));

        assert(node.param_strategy != NULL);
        assert(node.param != NULL && node.param == sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param);
        assert(node.node_type == DENSE_ROW_COARSE_SORT);
        assert(node.strategy_type == DENSE_ROW_COARSE_SORT_FIXED_PARAM_STRATEGY);

        sub_graph_param_strategy->param_strategy_vec.push_back(node);
    }
    else if (strategy_type == COMPRESS_NONE_PARAM_STRATEGY)
    {
        // 最后一个节点是压缩节点
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].type == COMPRESS);
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param != NULL);

        param_strategy_node_t node = init_compress_none_param_strategy(*((compress_none_param_strategy_t*)strategy_param_ptr), (exe_compress_param_t *)(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param));

        assert(node.param_strategy != NULL);
        assert(node.param != NULL && node.param == sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param);
        assert(node.node_type == COMPRESS);
        assert(node.strategy_type == COMPRESS_NONE_PARAM_STRATEGY);

        sub_graph_param_strategy->param_strategy_vec.push_back(node);
    }
    else if (strategy_type == DENSE_ROW_DIV_ACC_TO_EXPONENTIAL_INCREASE_ROW_NNZ_PARAM_STRATEGY)
    {
        // 最后一个节点是行分块节点
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].type == DENSE_ROW_DIV);
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param != NULL);

        param_strategy_node_t node = init_dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy(*((dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t*)strategy_param_ptr), (exe_dense_row_div_param_t *)(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param));
        
        assert(node.param_strategy != NULL);
        assert(node.param != NULL && node.param == sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param);
        assert(node.node_type == DENSE_ROW_DIV);
        assert(node.strategy_type == DENSE_ROW_DIV_ACC_TO_EXPONENTIAL_INCREASE_ROW_NNZ_PARAM_STRATEGY);

        sub_graph_param_strategy->param_strategy_vec.push_back(node);
    }
    else
    {
        cout << "add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph: startegy type is not supported" << endl;
        assert(false);
    }

    // 最后检查
    assert(sub_graph_param_strategy->param_strategy_vec.size() == sub_graph->exe_node_vec.size());

    assert(sub_graph_param_strategy->param_strategy_vec[sub_graph_param_strategy->param_strategy_vec.size() - 1].param == sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param);
}

void add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(exe_compressed_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy, exe_node_type node_type, exe_node_param_set_strategy strategy_type, void* strategy_param_ptr)
{
    assert(sub_graph != NULL && sub_graph_param_strategy != NULL && strategy_param_ptr != NULL);
    
    // 根据子块类型为子图加入不同的节点
    if (node_type == COMPRESSED_ROW_PADDING)
    {
        // 之前不能出现所有的padding和分块操作
        assert(sub_graph->preorder_node_set.count(COMPRESSED_ROW_PADDING) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_WARP_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_WARP_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_NNZ_DIV) == 0);

        // 初始化一个空白的节点
        exe_node_t node;
        node.type = COMPRESSED_ROW_PADDING;
        node.param = new exe_compress_row_padding_param_t();
        
        // 将这个节点插入到子图中
        sub_graph->exe_node_vec.push_back(node);

        // 将前序准备好
        sub_graph->preorder_node_set.insert(node_type);
    }
    else if (node_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV)
    {
        // 不能出现分块操作
        assert(sub_graph->preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_WARP_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_WARP_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_NNZ_DIV) == 0);

        // 初始化一个空白的节点
        exe_node_t node;
        node.type = COMPRESSED_TBLOCK_LEVEL_ROW_DIV;
        node.param = new exe_compress_tblock_level_row_div_param_t();

        sub_graph->exe_node_vec.push_back(node);
        sub_graph->preorder_node_set.insert(node_type);
    }
    else if (node_type == COMPRESSED_TBLOCK_LEVEL_COL_DIV)
    {
        // 不能出现分块操作
        assert(sub_graph->preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_WARP_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_WARP_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_NNZ_DIV) == 0);
        
        // 创建一个空白的节点
        exe_node_t node;
        node.type = COMPRESSED_TBLOCK_LEVEL_COL_DIV;
        node.param = new exe_compress_tblock_level_col_div_param_t();

        sub_graph->exe_node_vec.push_back(node);
        sub_graph->preorder_node_set.insert(node_type);
    }
    else if (node_type == COMPRESSED_WARP_LEVEL_ROW_DIV)
    {
        assert(sub_graph->preorder_node_set.count(COMPRESSED_WARP_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_WARP_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_NNZ_DIV) == 0);

        // 创建一个空白的节点
        exe_node_t node;
        node.type = COMPRESSED_WARP_LEVEL_ROW_DIV;
        node.param = new exe_compress_warp_level_row_div_param_t();
        
        sub_graph->exe_node_vec.push_back(node);
        sub_graph->preorder_node_set.insert(node_type);
    }
    else if (node_type == COMPRESSED_WARP_LEVEL_COL_DIV)
    {
        assert(sub_graph->preorder_node_set.count(COMPRESSED_WARP_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_WARP_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_NNZ_DIV) == 0);

        // 创建一个空白的节点
        exe_node_t node;
        node.type = COMPRESSED_WARP_LEVEL_COL_DIV;
        node.param = new exe_compress_warp_level_col_div_param_t();

        sub_graph->exe_node_vec.push_back(node);
        sub_graph->preorder_node_set.insert(node_type);
    }
    else if (node_type == COMPRESSED_THREAD_LEVEL_ROW_DIV)
    {
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_NNZ_DIV) == 0);

        // 创建一个空白节点
        exe_node_t node;
        node.type = COMPRESSED_THREAD_LEVEL_ROW_DIV;
        node.param = new exe_compress_thread_level_row_div_param_t();

        sub_graph->exe_node_vec.push_back(node);
        sub_graph->preorder_node_set.insert(node_type);
    }
    else if (node_type == COMPRESSED_THREAD_LEVEL_COL_DIV)
    {
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_NNZ_DIV) == 0);

        // 创建一个空白的节点
        exe_node_t node;
        node.type = COMPRESSED_THREAD_LEVEL_COL_DIV;
        node.param = new exe_compress_thread_level_col_div_param_t();
        
        sub_graph->exe_node_vec.push_back(node);
        sub_graph->preorder_node_set.insert(node_type);
    }
    else if (node_type == COMPRESSED_THREAD_LEVEL_NNZ_DIV)
    {
        assert(sub_graph->preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_TBLOCK_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_WARP_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_WARP_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_ROW_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_COL_DIV) == 0);
        assert(sub_graph->preorder_node_set.count(COMPRESSED_THREAD_LEVEL_NNZ_DIV) == 0);

        exe_node_t node;
        node.type = COMPRESSED_THREAD_LEVEL_NNZ_DIV;
        node.param = new exe_compress_thread_level_nnz_div_param_t();

        sub_graph->exe_node_vec.push_back(node);
        sub_graph->preorder_node_set.insert(node_type);
    }
    else
    {
        cout << "add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph: exe node type is not supported" << endl;
        assert(false);
    }

    // 加入一个参数类型
    if (strategy_type == COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY)
    {
        // 子图的最后一个节点是row padding节点
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].type == COMPRESSED_ROW_PADDING);
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param != NULL);

        // cout << ((compressed_row_padding_direct_param_strategy_t *)strategy_param_ptr)->padding_row_length << endl;

        // assert(((compressed_row_padding_direct_param_strategy_t *)strategy_param_ptr)->padding_row_length == 1);
        // 创造一个参数策略节点
        param_strategy_node_t node = init_compressed_row_padding_direct_param_strategy(*((compressed_row_padding_direct_param_strategy_t *)strategy_param_ptr), (exe_compress_row_padding_param_t *)(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param));
        assert(node.param_strategy != NULL);
        assert(node.param != NULL && node.param == sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param);
        assert(node.node_type == COMPRESSED_ROW_PADDING);
        assert(node.strategy_type == COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY);
        
        // 将参数策略放到矩阵中
        sub_graph_param_strategy->param_strategy_vec.push_back(node);
    }
    else if (strategy_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY)
    {
        // 最后一个节点是BLB的行分块
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param != NULL);

        // 策略节点
        param_strategy_node_t node = init_compressed_tblock_level_row_div_evenly_param_strategy(*((compressed_tblock_level_row_div_evenly_param_strategy_t *)strategy_param_ptr), (exe_compress_tblock_level_row_div_param_t *)(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param));
        assert(node.param_strategy != NULL);
        assert(node.param != NULL && node.param == sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param);
        assert(node.node_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
        assert(node.strategy_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY);

        // 将参数策略放到矩阵中
        sub_graph_param_strategy->param_strategy_vec.push_back(node);
    }
    else if (strategy_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY)
    {
        // 最后一个节点是行分块
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param != NULL);

        // 策略节点
        param_strategy_node_t node = init_compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy(*((compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t *)strategy_param_ptr), (exe_compress_tblock_level_row_div_param_t *)(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param));
        assert(node.param_strategy != NULL);
        assert(node.param != NULL && node.param == sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param);
        assert(node.node_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
        assert(node.strategy_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY);
        
        // 将参数策略放到矩阵中
        sub_graph_param_strategy->param_strategy_vec.push_back(node);
    }
    else if (strategy_type == COMPRESSED_TBLOCK_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
    {
        // 最后一个节点的是列分块
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].type == COMPRESSED_TBLOCK_LEVEL_COL_DIV);
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param != NULL);

        // 策略节点
        param_strategy_node_t node = init_compressed_tblock_level_col_div_fixed_param_strategy(*((compressed_tblock_level_col_div_fixed_param_strategy_t *)strategy_param_ptr), (exe_compress_tblock_level_col_div_param_t *)(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param));
        assert(node.param_strategy != NULL);
        assert(node.param != NULL && node.param == sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param);
        assert(node.node_type == COMPRESSED_TBLOCK_LEVEL_COL_DIV);
        assert(node.strategy_type == COMPRESSED_TBLOCK_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY);

        // 将参数策略放到矩阵中
        sub_graph_param_strategy->param_strategy_vec.push_back(node);
    }
    else if (strategy_type == COMPRESSED_WARP_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY)
    {
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].type == COMPRESSED_WARP_LEVEL_ROW_DIV);
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param != NULL);

        // 策略节点
        param_strategy_node_t node = init_compressed_warp_level_row_div_evenly_param_strategy(*((compressed_warp_level_row_div_evenly_param_strategy_t *)strategy_param_ptr), (exe_compress_warp_level_row_div_param_t *)(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param));
        assert(node.param_strategy != NULL);
        assert(node.param != NULL && node.param == sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param);
        assert(node.node_type == COMPRESSED_WARP_LEVEL_ROW_DIV);
        assert(node.strategy_type == COMPRESSED_WARP_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY);

        // 将参数策略放到矩阵中
        sub_graph_param_strategy->param_strategy_vec.push_back(node);
    }
    else if (strategy_type == COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
    {
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].type == COMPRESSED_WARP_LEVEL_COL_DIV);
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param != NULL);

        // 策略节点
        param_strategy_node_t node = init_compressed_warp_level_col_div_fixed_param_strategy(*((compressed_warp_level_col_div_fixed_param_strategy_t *)strategy_param_ptr), (exe_compress_warp_level_col_div_param_t *)(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param));
        assert(node.param_strategy != NULL);
        assert(node.param != NULL && node.param == sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param);
        assert(node.node_type == COMPRESSED_WARP_LEVEL_COL_DIV);
        assert(node.strategy_type == COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY);

        // 将参数策略放到矩阵中
        sub_graph_param_strategy->param_strategy_vec.push_back(node);
    }
    else if (strategy_type == COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY)
    {
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].type == COMPRESSED_THREAD_LEVEL_ROW_DIV);
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param != NULL);

        // 策略算法
        param_strategy_node_t node = init_compressed_thread_level_row_div_none_param_strategy(*((compressed_thread_level_row_div_none_param_strategy_t *)strategy_param_ptr), (exe_compress_thread_level_row_div_param_t *)(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param));
        assert(node.param_strategy != NULL);
        assert(node.param != NULL && node.param == sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param);
        assert(node.node_type == COMPRESSED_THREAD_LEVEL_ROW_DIV);
        assert(node.strategy_type == COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY);

        // 将参数策略放到矩阵中
        sub_graph_param_strategy->param_strategy_vec.push_back(node);
    }
    else if (strategy_type == COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
    {
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].type == COMPRESSED_THREAD_LEVEL_COL_DIV);
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param != NULL);

        // 策略算法
        param_strategy_node_t node = init_compressed_thread_level_col_div_fixed_param_strategy(*((compressed_thread_level_col_div_fixed_param_strategy_t *)strategy_param_ptr), (exe_compress_thread_level_col_div_param_t *)(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param));
        assert(node.param_strategy != NULL);
        assert(node.param != NULL && node.param == sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param);
        assert(node.node_type == COMPRESSED_THREAD_LEVEL_COL_DIV);
        assert(node.strategy_type == COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY);

        // 将参数策略放到矩阵中
        sub_graph_param_strategy->param_strategy_vec.push_back(node);
    }
    else if (strategy_type == COMPRESSED_THREAD_LEVEL_NNZ_DIV_DIRECT_PARAM_STRATEGY)
    {
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].type == COMPRESSED_THREAD_LEVEL_NNZ_DIV);
        assert(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param != NULL);

        param_strategy_node_t node = init_compressed_thread_level_nnz_div_direct_param_strategy(*((compressed_thread_level_nnz_div_direct_param_strategy_t *)strategy_param_ptr), (exe_compress_thread_level_nnz_div_param_t *)(sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param));
        assert(node.param_strategy != NULL);
        assert(node.param != NULL && node.param == sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param);
        assert(node.node_type == COMPRESSED_THREAD_LEVEL_NNZ_DIV);
        assert(node.strategy_type == COMPRESSED_THREAD_LEVEL_NNZ_DIV_DIRECT_PARAM_STRATEGY);

        // 将参数策略放到矩阵中
        sub_graph_param_strategy->param_strategy_vec.push_back(node);
    }
    else
    {
        cout << "add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph: param strategy type is not supported" << endl;
        assert(false);
    }

    // 最后检查
    assert(sub_graph_param_strategy->param_strategy_vec.size() == sub_graph->exe_node_vec.size());

    assert(sub_graph_param_strategy->param_strategy_vec[sub_graph_param_strategy->param_strategy_vec.size() - 1].param == sub_graph->exe_node_vec[sub_graph->exe_node_vec.size() - 1].param);
}

// 将两个节点分别放到数组中
void add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(sparse_struct_t* matrix, unsigned long sub_matrix_id, exe_compressed_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy, exe_node_type node_type, exe_node_param_set_strategy strategy_type, void* strategy_param_ptr)
{
    assert(matrix != NULL && strategy_param_ptr != NULL && sub_graph_param_strategy != NULL && sub_graph != NULL);

    // 子块索引满足要求，并且子块存在且压缩
    assert(sub_matrix_id < matrix->block_coor_table.item_arr.size());
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    // 已有的子图大小是一样的
    assert(sub_graph_param_strategy->param_strategy_vec.size() == sub_graph->exe_node_vec.size() && sub_graph->exe_node_vec.size() == sub_graph->preorder_node_set.size());

    // 执行对应子块的
    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(sub_graph, sub_graph_param_strategy, node_type, strategy_type, strategy_param_ptr);
}


void del_param_of_compressed_sub_block_exe_graph_and_template(compressed_sub_block_exe_graph_and_template_t* sub_graph_and_template)
{
    del_strategy_of_param_strategy_node_in_sub_matrix(&(sub_graph_and_template->sub_graph_param_strategy));
    del_exe_node_param_of_compress_sub_matrix(&(sub_graph_and_template->sub_graph));

    // 删完之后是空的
    for (unsigned long i = 0; i < sub_graph_and_template->sub_graph.exe_node_vec.size(); i++)
    {
        assert(sub_graph_and_template->sub_graph.exe_node_vec[i].param == NULL);
    }

    for (unsigned long i = 0; i < sub_graph_and_template->sub_graph_param_strategy.param_strategy_vec.size(); i++)
    {
        assert(sub_graph_and_template->sub_graph_param_strategy.param_strategy_vec[i].param_strategy == NULL);
    }

    // 如果模板节点的参数不存在，那就不需要析构
    if (sub_graph_and_template->temp_node.template_param != NULL)
    {
        del_param_of_template_node(&(sub_graph_and_template->temp_node));
        assert(sub_graph_and_template->temp_node.template_param == NULL);
    }
}

// compressed_sub_block_exe_graph_and_template_t find_best_sub_matrix_optimization_path(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id)
// {
//     // 用一个指针来指向最优的模板的子图优化路径
//     exe_compressed_sub_graph_t* best_compressed_sub_graph_ptr = NULL;
//     // 用一个指针来指向最优的模板
//     template_node_t* best_compressed_sub_graph_ptr = NULL;
    
//     // 最佳的性能
//     float gflops = 0;
//     float time = 99999999;

    
// }

// exe_compressed_sub_graph_t find_best_path_of_white_list_strategy1(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id)
// {
//     // 创建一个临时图来执行压缩子图部分
    
// }