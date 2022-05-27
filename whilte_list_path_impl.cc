#include "default_auto_tuner.hpp"
using namespace std;

void reset_exe_node_param_and_param_strategy_of_sub_graph(exe_compressed_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy)
{
    // cout << "reset_exe_node_param_and_param_strategy_of_sub_graph" << endl;
    assert(sub_graph != NULL && sub_graph_param_strategy != NULL);
    assert(sub_graph->exe_node_vec.size() == sub_graph_param_strategy->param_strategy_vec.size());
    assert(sub_graph->exe_node_vec.size() > 0 && sub_graph_param_strategy->param_strategy_vec.size() > 0);

    // 检查输入
    for (unsigned long i = 0; i < sub_graph->exe_node_vec.size(); i++)
    {
        assert(sub_graph->exe_node_vec[i].param != NULL);
        assert(sub_graph_param_strategy->param_strategy_vec[i].param == sub_graph->exe_node_vec[i].param);
        assert(sub_graph_param_strategy->param_strategy_vec[i].param_strategy != NULL);
    }

    // 执行exe node的reset
    reset_param_of_all_sub_compressed_graph(sub_graph);

    // 重新连接参数策略和参数
    bind_exe_node_param_param_strategy_of_sub_graph(sub_graph, sub_graph_param_strategy);
}

void reset_exe_node_param_and_param_strategy_of_sub_graph(exe_dense_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy)
{
    // cout << "reset_exe_node_param_and_param_strategy_of_sub_graph" << endl;
    assert(sub_graph != NULL && sub_graph_param_strategy != NULL);
    assert(sub_graph->exe_node_vec.size() == sub_graph_param_strategy->param_strategy_vec.size());
    assert(sub_graph->exe_node_vec.size() > 0 && sub_graph_param_strategy->param_strategy_vec.size() > 0);

    // 检查输入
    for (unsigned long i = 0; i < sub_graph->exe_node_vec.size(); i++)
    {
        assert(sub_graph->exe_node_vec[i].param != NULL);
        assert(sub_graph_param_strategy->param_strategy_vec[i].param == sub_graph->exe_node_vec[i].param);
        assert(sub_graph_param_strategy->param_strategy_vec[i].param_strategy != NULL);
    }
    
    // 执行exe node的reset
    reset_param_of_all_sub_dense_graph(sub_graph);

    // 重新连接参数策略和参数
    bind_exe_node_param_param_strategy_of_sub_graph(sub_graph, sub_graph_param_strategy);
}

void malloc_exe_node_param_and_param_strategy_of_sub_graph(exe_compressed_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy)
{
    assert(sub_graph != NULL && sub_graph_param_strategy != NULL);
    assert(sub_graph->exe_node_vec.size() == sub_graph->preorder_node_set.size() && sub_graph->preorder_node_set.size() == sub_graph_param_strategy->param_strategy_vec.size());

    // 检查输入
    for (unsigned long i = 0; i < sub_graph->exe_node_vec.size(); i++)
    {
        assert(sub_graph->exe_node_vec[i].param != NULL);
        assert(sub_graph_param_strategy->param_strategy_vec[i].param == sub_graph->exe_node_vec[i].param);
        assert(sub_graph_param_strategy->param_strategy_vec[i].param_strategy != NULL);
    }

    // 为所有的节点重新申请新的参数，但是不析构已有的参数
    malloc_param_of_all_sub_compressed_graph(sub_graph);

    bind_exe_node_param_param_strategy_of_sub_graph(sub_graph, sub_graph_param_strategy);
}

void bind_exe_node_param_param_strategy_of_sub_graph(exe_compressed_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy)
{
    assert(sub_graph != NULL && sub_graph_param_strategy != NULL);
    assert(sub_graph->exe_node_vec.size() == sub_graph_param_strategy->param_strategy_vec.size());
    assert(sub_graph->exe_node_vec.size() > 0 && sub_graph->preorder_node_set.size() > 0);

    // 重新绑定参数策略和参数
    for (unsigned long i = 0; i < sub_graph->exe_node_vec.size(); i++)
    {
        // 检查类型是不是正确
        assert(sub_graph_param_strategy->param_strategy_vec[i].node_type == sub_graph->exe_node_vec[i].type);
        assert(sub_graph_param_strategy->param_strategy_vec[i].param_strategy != NULL);
        sub_graph_param_strategy->param_strategy_vec[i].param = sub_graph->exe_node_vec[i].param;
    }
}

void bind_exe_node_param_param_strategy_of_sub_graph(exe_dense_sub_graph_t* sub_graph, param_strategy_of_sub_graph_t* sub_graph_param_strategy)
{
    assert(sub_graph != NULL && sub_graph_param_strategy != NULL);
    assert(sub_graph->exe_node_vec.size() == sub_graph_param_strategy->param_strategy_vec.size());
    assert(sub_graph->exe_node_vec.size() > 0 && sub_graph->preorder_node_set.size() > 0);

    // 重新绑定参数策略和参数
    for (unsigned long i = 0; i < sub_graph->exe_node_vec.size(); i++)
    {
        assert(sub_graph_param_strategy->param_strategy_vec[i].node_type == sub_graph->exe_node_vec[i].type);
        assert(sub_graph_param_strategy->param_strategy_vec[i].param_strategy != NULL);
        sub_graph_param_strategy->param_strategy_vec[i].param = sub_graph->exe_node_vec[i].param;
    }
}

void del_param_of_total_exe_graph_and_strategy_graph_safely(dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t* total_graph)
{
    assert(total_graph != NULL);
    // 首先析构稠密视图的路径
    del_exe_node_param_of_dense_view_matrix(&(total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph));
    // 然后析构稠密视图的策略路径
    del_strategy_of_param_strategy_node_in_sub_matrix(&(total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy));

    // 析构玩之后都变成空的
    for (auto sub_graph_item : total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph.exe_node_vec)
    {
        assert(sub_graph_item.param == NULL);
    }

    for (auto strategy_item : total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy.param_strategy_vec)
    {
        assert(strategy_item.param_strategy == NULL);
    }

    // 遍历所有的子图
    for (unsigned long i = 0; i < total_graph->compressed_sub_block_exe_graph_and_template_vec.size(); i++)
    {
        del_exe_node_param_of_compress_sub_matrix(&(total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph));
        del_strategy_of_param_strategy_node_in_sub_matrix(&(total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy));

        // 析构玩之后参数都变成NULL
        for (auto sub_graph_item : total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph.exe_node_vec)
        {
            assert(sub_graph_item.param == NULL);   
        }

        for (auto strategy_item : total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy.param_strategy_vec)
        {
            assert(strategy_item.param_strategy == NULL);
        }
        
        if (total_graph->compressed_sub_block_exe_graph_and_template_vec[i].temp_node.template_param != NULL)
        {
            del_param_of_template_node(&(total_graph->compressed_sub_block_exe_graph_and_template_vec[i].temp_node));
            assert(total_graph->compressed_sub_block_exe_graph_and_template_vec[i].temp_node.template_param == NULL);
        }
    }
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy1(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 一上来先执行对应的密集子图
    sparse_struct_t* matrix = get_matrix_dense_view_graph(&dense_graph);
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    // 获得最优的子图优化路径
    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_white_list_strategy1(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    // 析构当前的矩阵
    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);

    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy1(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    sparse_struct_t* matrix = execute_dense_matrix_exe_graph_with_param_strategy(&(dense_graph.dense_sub_graph), &(dense_graph.dense_sub_graph_param_strategy));

    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    // 获得最优的子图优化路径
    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_white_list_strategy1(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    // 析构当前的矩阵
    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);

    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy1(sparse_struct_t* input_matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 检查从外部传入的矩阵
    assert(input_matrix != NULL && input_matrix->block_coor_table.item_arr.size() > sub_matrix_id && input_matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    // 执行拷贝
    sparse_struct_t* matrix = val_copy_from_old_matrix_struct(input_matrix);

    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 找出最优的子图、最优的参数策略、最优模板类型及其参数
    template_node_t best_temp_node;
    exe_compressed_sub_graph_t best_sub_graph;
    param_strategy_of_sub_graph_t best_sub_graph_param_strategy;

    best_time = 99999999999999;
    best_gflops = 0;

    // 定一个优化路径骨架
    exe_compressed_sub_graph_t sub_graph_skeleton;
    // 定一个参数设定的骨架
    param_strategy_of_sub_graph_t param_strategy_skeleton;

    // 定义骨架，首先是一个ROW_padding和BLB_row
    compressed_row_padding_direct_param_strategy_t row_padding_param_strategy;
    compressed_tblock_level_row_div_evenly_param_strategy_t BLB_row_div_evenly_param_strategy;
    
    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_ROW_PADDING, COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY, &row_padding_param_strategy);
    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_TBLOCK_LEVEL_ROW_DIV, COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY, &BLB_row_div_evenly_param_strategy);

    // 增加对于参数的调节
    param_enumerater_t param_setter;

    assert(sub_graph_skeleton.exe_node_vec.size() == 2);
    
    // 对应参数的指针
    compressed_row_padding_direct_param_strategy_t* row_padding_direct_param_strategy = (compressed_row_padding_direct_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param_strategy;
    compressed_tblock_level_row_div_evenly_param_strategy_t* tblock_level_row_div_evenly_param_strategy = (compressed_tblock_level_row_div_evenly_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[1].param_strategy;
    assert(row_padding_direct_param_strategy != NULL && tblock_level_row_div_evenly_param_strategy != NULL);
    
    // 对于BLB row padding的长度
    register_integer_independ_param_to_enumerater(&param_setter, &(row_padding_direct_param_strategy->multiply), 0, 256, 64);
    row_padding_direct_param_strategy->padding_row_length = 1;

    // cout << "row_padding_direct_param_strategy->padding_row_length:" << ((compressed_row_padding_direct_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param)->padding_row_length << endl;
    // BLB行分块的宽度依赖于rowpadding的大小，在实际处理的时候要执行赋值

    // 添加一行一个TLB的切分
    compressed_thread_level_row_div_none_param_strategy_t thread_row_param_strategy;
    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_THREAD_LEVEL_ROW_DIV, COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY, &thread_row_param_strategy);

    assert(sub_graph_skeleton.exe_node_vec.size() == 3 && param_strategy_skeleton.param_strategy_vec.size() == 3);

    // 是不是要停止搜索
    bool search_finished_by_strategy = false;

    // 枚举参数，分别得出不同的矩阵
    while (set_param_combination_to_next(&param_setter) == false)
    {
        if (row_padding_direct_param_strategy->multiply == 0)
        {
            continue;
        }

        // 再定义一些参数
        tblock_level_row_div_evenly_param_strategy->block_row_num = row_padding_direct_param_strategy->multiply;
        
        // 如果没有稠密视图产生的矩阵
        if (matrix == NULL)
        {
            // 通过稠密视图的块得出需要的子块
            matrix = val_copy_from_old_matrix_struct(input_matrix);
            assert(matrix != NULL);
            // 矩阵的一系列检查
            assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
            assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

            compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;
        }

        // 根据BLB窗口的大小获得padding率
        // 获取子块行非零元数量
        vector<unsigned long> row_nnz_of_compressed_block = get_nnz_of_each_row_in_compressed_sub_matrix(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr);

        // 打印现在一些参数
        cout << "row_padding_direct_param_strategy->multiply:" << row_padding_direct_param_strategy->multiply << ",row_padding_direct_param_strategy->padding_row_length:" << row_padding_direct_param_strategy->padding_row_length << endl;

        // padding之前的非零元数量
        unsigned long old_nnz = compressed_block_ptr->read_index[0]->length;
        unsigned long new_nnz = 0;

        // 遍历每一个BLB行条带，计算每个条带的累计的新的nnz数量，不考虑WLB的padding
        for (unsigned long i = 0; i < row_nnz_of_compressed_block.size() / tblock_level_row_div_evenly_param_strategy->block_row_num; i++)
        {
            // 找出行条带的最大值
            unsigned long largest_row_nnz = 0;

            for (unsigned long global_row_id = tblock_level_row_div_evenly_param_strategy->block_row_num * i; global_row_id < tblock_level_row_div_evenly_param_strategy->block_row_num * (i + 1); global_row_id++)
            {
                assert(global_row_id < row_nnz_of_compressed_block.size());

                if (largest_row_nnz < row_nnz_of_compressed_block[global_row_id])
                {
                    largest_row_nnz = row_nnz_of_compressed_block[global_row_id];
                }
            }

            new_nnz = new_nnz + largest_row_nnz * tblock_level_row_div_evenly_param_strategy->block_row_num;
        }

        // 看看有没有剩下的部分
        if (row_nnz_of_compressed_block.size() % tblock_level_row_div_evenly_param_strategy->block_row_num != 0)
        {
            unsigned long remain_row_num = row_nnz_of_compressed_block.size() % tblock_level_row_div_evenly_param_strategy->block_row_num;

            // 行的最大长度
            unsigned long largest_row_nnz = 0;
            
            // 找出行条带的最大值
            for (unsigned long global_row_id = row_nnz_of_compressed_block.size() - remain_row_num; global_row_id < row_nnz_of_compressed_block.size(); global_row_id++)
            {
                assert(global_row_id < row_nnz_of_compressed_block.size());

                if (largest_row_nnz < row_nnz_of_compressed_block[global_row_id])
                {
                    largest_row_nnz = row_nnz_of_compressed_block[global_row_id];
                }
            }

            // 计算新条带的nnz数量，但是因为经过了padding，所以最后一块的实际行号是BLB行条带的宽度
            new_nnz = new_nnz + largest_row_nnz * tblock_level_row_div_evenly_param_strategy->block_row_num;
        }

        assert(new_nnz >= old_nnz);

        cout << "new_nnz:" << new_nnz << ",old_nnz:" << old_nnz << ",padding rate:" << (float)new_nnz/(float)old_nnz << endl;

        if ((float)new_nnz/(float)old_nnz > get_config()["PADDING_RATE_UP_BOUND"].as_integer())
        {
            cout << "padding rate is larger than " << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
            
            // 析构matrix
            memory_garbage_manager_t mem_manager;

            delete_sparse_struct_t(&mem_manager, matrix);

            matrix = NULL;

            // 析构完之后置0
            continue;
        }

        if (data_set_collector != NULL)
        {
            // 按照当前的逻辑，稠密视图已经加入了对应的数据
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() > 0);

            vector<exe_node_type> compressed_node_type_vec;
            vector<exe_node_param_set_strategy> compressed_param_strategy_vec;
            vector<float> compressed_param_vec;

            // 检查一下插入的数据的数量是不是满足要求，类型是不是满足要求
            assert(sub_graph_skeleton.exe_node_vec.size() == 3);
            
            // 分别处理三个节点，第一个节点直接row padding，第二个节点行分块，第三个节点一行一个线程
            
            // 第一个节点的类型，参数策略的类型，以及策略的参数
            // COMPRESSED_ROW_PADDING, COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY, compressed_row_padding_direct_param_strategy_t
            assert(sub_graph_skeleton.exe_node_vec[0].type == COMPRESSED_ROW_PADDING);
            assert(param_strategy_skeleton.param_strategy_vec[0].strategy_type == COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY);
            compressed_node_type_vec.push_back(COMPRESSED_ROW_PADDING);
            compressed_param_strategy_vec.push_back(COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY);
            compressed_row_padding_direct_param_strategy_t* strategy1_ptr = (compressed_row_padding_direct_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param_strategy;
            assert(strategy1_ptr != NULL);
            compressed_param_vec.push_back(strategy1_ptr->multiply);
            compressed_param_vec.push_back(strategy1_ptr->padding_row_length);
            
            // 第二个节点的类型，参数策略的类型，以及策略的参数
            // COMPRESSED_TBLOCK_LEVEL_ROW_DIV, COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY, compressed_tblock_level_row_div_evenly_param_strategy_t
            assert(sub_graph_skeleton.exe_node_vec[1].type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
            assert(param_strategy_skeleton.param_strategy_vec[1].strategy_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY);
            compressed_node_type_vec.push_back(COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
            compressed_param_strategy_vec.push_back(COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY);
            compressed_tblock_level_row_div_evenly_param_strategy_t* strategy2_ptr = (compressed_tblock_level_row_div_evenly_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[1].param_strategy;
            assert(strategy2_ptr != NULL);
            compressed_param_vec.push_back(strategy2_ptr->block_row_num);
            
            // 第三个节点类型，参数策略已经策略参数
            // COMPRESSED_THREAD_LEVEL_ROW_DIV, COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY
            assert(sub_graph_skeleton.exe_node_vec[2].type == COMPRESSED_THREAD_LEVEL_ROW_DIV);
            assert(param_strategy_skeleton.param_strategy_vec[2].strategy_type == COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY);
            compressed_node_type_vec.push_back(COMPRESSED_THREAD_LEVEL_ROW_DIV);
            compressed_param_strategy_vec.push_back(COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY);

            // 清除当前compressed阶段的所有积累值
            data_set_collector->clear_compressed_accu_info();
            data_set_collector->insert_compressed_stage_node_and_param_to_cur_item(compressed_node_type_vec, compressed_param_strategy_vec, compressed_param_vec);
        }

        // 执行对应的子块
        execute_sub_matrix_exe_graph_with_param_strategy(matrix, sub_matrix_id, &sub_graph_skeleton, &param_strategy_skeleton);

        // 候选的模板类型
        set<template_type> candi_template_type_set;
        candi_template_type_set.insert(DIRECT_ATOM_TEMPLATE_WARP_COMPRESS);

        float time;
        float gflops;

        // 寻找对应的最优模板，这里先不处理
        template_node_t temp_node = find_best_template_node_of_specific_sub_matrix_from_template_set(matrix, sub_matrix_id, candi_template_type_set, time, gflops, search_strategy_ptr, data_set_collector);

        // 根据当前性能是不是超过最佳性能来执行不同的处理
        if (gflops > best_gflops)
        {
            if (best_gflops == 0)
            {
                // 直接赋值，并且修改最佳优化路径
                // 这个时候还没有最优的优化路径
                best_gflops = gflops;
                best_time = time;

                // 从best拷贝出来
                best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
                best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
                best_temp_node = val_copy_from_old_template_node(temp_node);

                // 重新绑定优化骨架和策略骨架
                bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
            }
            else
            {
                // 直接赋值，析构已有的最优路径
                best_gflops = gflops;
                best_time = time;

                // 析构已有的最优参数
                del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                // 析构已有的最优策略
                del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                // 已有的最优模板
                del_param_of_template_node(&best_temp_node);
                
                // 执行新的拷贝
                // 从best拷贝出来
                best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
                best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
                best_temp_node = val_copy_from_old_template_node(temp_node);

                // 重新绑定优化骨架和策略骨架
                bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);   
            }
        }

        // 析构matrix
        // 现在matrix肯定存在
        assert(matrix != NULL);
        memory_garbage_manager_t mem_manager;
        delete_sparse_struct_t(&mem_manager, matrix);
        matrix = NULL;
        
        // 重置所有参数，并且重置所有参数指针，这一步用来处理一些参数是数组的节点，数组中的内容应该重新清空
        reset_exe_node_param_and_param_strategy_of_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton);

        // 如果所有的模板参数执行后都发生错误，那么这里的temp_node中可能是没有参数的
        if (temp_node.template_param != NULL)
        {
            del_param_of_template_node(&temp_node);
        }
        else
        {
            // 当前可能没有出现对应的
            assert(gflops == 0);
        }

        // 加入提前结束的相关内容
        if (search_strategy_ptr != NULL)
        {
            if (continue_search(search_strategy_ptr) == false)
            {
                search_finished_by_strategy = true;
            }
        }

        if (search_finished_by_strategy == true)
        {
            break;
        }
    }

    // 如果没有进上面的循环，那就可能需要在这里析构矩阵
    if (matrix != NULL)
    {
        assert(matrix != NULL);
        memory_garbage_manager_t mem_manager;
        delete_sparse_struct_t(&mem_manager, matrix);
        matrix = NULL;
    }

    // 析构用以遍历各种优化路径的两个骨架的参数
    del_strategy_of_param_strategy_node_in_sub_matrix(&param_strategy_skeleton);
    del_exe_node_param_of_compress_sub_matrix(&sub_graph_skeleton);

    compressed_sub_block_exe_graph_and_template_t return_sub_graph_exe_node_and_template;
    return_sub_graph_exe_node_and_template.sub_graph = best_sub_graph;
    return_sub_graph_exe_node_and_template.sub_graph_param_strategy = best_sub_graph_param_strategy;
    return_sub_graph_exe_node_and_template.temp_node = best_temp_node;

    return return_sub_graph_exe_node_and_template;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy2(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 首先执行对应的稠密子图优化
    sparse_struct_t* matrix = get_matrix_dense_view_graph(&dense_graph);
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    // 查找最优子图
    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_white_list_strategy2(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    // 析构原来的矩阵
    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);

    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy2(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 首先执行对应的稠密子图优化
    sparse_struct_t* matrix = execute_dense_matrix_exe_graph_with_param_strategy(&(dense_graph.dense_sub_graph), &(dense_graph.dense_sub_graph_param_strategy));
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    // 查找最优子图
    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_white_list_strategy2(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    // 析构原来的矩阵
    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);

    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy2(sparse_struct_t* input_matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(input_matrix != NULL);
    // 矩阵的一系列检查
    assert(input_matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(input_matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    // 执行拷贝
    sparse_struct_t* matrix = val_copy_from_old_matrix_struct(input_matrix);

    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 获取每一行的行非零元数量，并且获得最大和最小行非零元数量
    vector<unsigned long> row_nnz_of_compressed_block = get_nnz_of_each_row_in_compressed_sub_matrix(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr);
    
    assert(row_nnz_of_compressed_block.size() > 0);

    unsigned long max_row_nnz = row_nnz_of_compressed_block[0];
    unsigned long min_row_nnz = row_nnz_of_compressed_block[0];

    // 遍历获得最大的和最小的行非零元数量
    for (unsigned long i = 1; i < row_nnz_of_compressed_block.size(); i++)
    {
        unsigned long cur_row_nnz = row_nnz_of_compressed_block[i];

        if (max_row_nnz < cur_row_nnz)
        {
            max_row_nnz = cur_row_nnz;
        }

        if (min_row_nnz > cur_row_nnz)
        {
            min_row_nnz = cur_row_nnz;
        }
    }

    // 找出最优的子图、最优的参数策略、最优模板类型及其参数
    template_node_t best_temp_node;
    exe_compressed_sub_graph_t best_sub_graph;
    param_strategy_of_sub_graph_t best_sub_graph_param_strategy;

    best_time = 99999999999999;
    best_gflops = 0;

    // 定一个优化路径骨架
    exe_compressed_sub_graph_t sub_graph_skeleton;
    // 定一个参数设定的骨架
    param_strategy_of_sub_graph_t param_strategy_skeleton;

    // 定义骨架，一个是row padding
    // 定义骨架，首先是一个ROW_padding和BLB_row
    compressed_row_padding_direct_param_strategy_t row_padding_param_strategy;
    compressed_tblock_level_row_div_evenly_param_strategy_t BLB_row_div_evenly_param_strategy;
    
    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_ROW_PADDING, COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY, &row_padding_param_strategy);
    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_TBLOCK_LEVEL_ROW_DIV, COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY, &BLB_row_div_evenly_param_strategy);

    // 增加对于参数的调节
    param_enumerater_t param_setter;

    assert(sub_graph_skeleton.exe_node_vec.size() == 2);

    // 执行一个列分块
    // 对应参数的指针
    compressed_row_padding_direct_param_strategy_t* row_padding_direct_param_strategy = (compressed_row_padding_direct_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param_strategy;
    compressed_tblock_level_row_div_evenly_param_strategy_t* tblock_level_row_div_evenly_param_strategy = (compressed_tblock_level_row_div_evenly_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[1].param_strategy;
    assert(row_padding_direct_param_strategy != NULL && tblock_level_row_div_evenly_param_strategy != NULL);
    
    // 对于BLB row padding的长度，找一个非常小的值
    register_integer_independ_param_to_enumerater(&param_setter, &(row_padding_direct_param_strategy->multiply), 32, 64, 32);
    // padding的数量是当前子块的最小行号
    row_padding_direct_param_strategy->padding_row_length = min_row_nnz;

    // 添加一个TLB的纵分块
    compressed_thread_level_col_div_fixed_param_strategy_t thread_col_param_strategy;
    
    // 将策略加到优化骨架中
    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_THREAD_LEVEL_COL_DIV, COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY, &thread_col_param_strategy);

    // 将参数的指针提取出来
    compressed_thread_level_col_div_fixed_param_strategy_t* thread_level_col_div_fixed_param_strategy = (compressed_thread_level_col_div_fixed_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[2].param_strategy;

    // 列分块的参数调试范围，从0到16，高过这个范围的就使用warp的模板
    // col最大值为最大行非零元数量的一半，向上取整，在遍历的过程中需要进一步剪枝，现在只能设定一个固定的值范围
    unsigned long fixed_max_col_size = 16;

    unsigned long max_col_size_acc_to_nnz = max_row_nnz / 2;

    if (max_row_nnz % 2 != 0)
    {
        max_col_size_acc_to_nnz = max_col_size_acc_to_nnz + 1;
    }

    if (max_col_size_acc_to_nnz < fixed_max_col_size)
    {
        fixed_max_col_size = max_col_size_acc_to_nnz;
    }

    assert(thread_level_col_div_fixed_param_strategy != NULL);
    register_integer_independ_param_to_enumerater(&param_setter, &(thread_level_col_div_fixed_param_strategy->col_block_nnz_num), 1, fixed_max_col_size, 3);

    assert(sub_graph_skeleton.exe_node_vec.size() == 3 && param_strategy_skeleton.param_strategy_vec.size() == 3);
    
    bool search_finished_by_strategy = false;

    // 遍历所有参数
    while (set_param_combination_to_next(&param_setter) == false)
    {
        // 矩阵已经不存在，那就执行稠密子图的优化路径，产生新的矩阵
        if (matrix == NULL)
        {
            // 通过稠密视图的块得出需要的子块
            matrix = val_copy_from_old_matrix_struct(input_matrix);
            assert(matrix != NULL);
            // 矩阵的一系列检查
            assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
            assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

            compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;
        }

        // 再定义一些参数
        tblock_level_row_div_evenly_param_strategy->block_row_num = row_padding_direct_param_strategy->multiply;

        // 打印一些参数
        cout << "row_padding_direct_param_strategy->multiply:" << row_padding_direct_param_strategy->multiply << ",row_padding_direct_param_strategy->padding_row_length:" << row_padding_direct_param_strategy->padding_row_length << "thread_level_col_div_fixed_param_strategy->col_block_nnz_num:" << thread_level_col_div_fixed_param_strategy->col_block_nnz_num << endl;

        // 查看当前的padding率以及每个BLB中的TLB的数量。超过1008就放弃分块的条带，这里也考虑padding产生的新条带
        // 并且需要计算padding率
        unsigned long old_nnz = compressed_block_ptr->read_index[0]->length;
        unsigned long new_nnz = 0;

        // 最大的TLB数量
        unsigned long max_TLB_of_BLB = 0;

        // 遍历每个BLB条，计算累计的nnz数量，不考虑WLB的padding，并且计算每个条带的TLB的数量
        for (unsigned long i = 0; i < row_nnz_of_compressed_block.size() / tblock_level_row_div_evenly_param_strategy->block_row_num; i++)
        {
            // 累计非零元数量和TLB数量
            unsigned long nnz_of_cur_BLB = 0;
            unsigned long TLB_num_of_cur_BLB = 0;

            for (unsigned long global_row_id = tblock_level_row_div_evenly_param_strategy->block_row_num * i; global_row_id < tblock_level_row_div_evenly_param_strategy->block_row_num * (i + 1); global_row_id++)
            {
                assert(global_row_id < row_nnz_of_compressed_block.size());
                
                unsigned long cur_row_nnz = row_nnz_of_compressed_block[global_row_id];

                // TLB分块之后的行长度初始化为和当前行长度相等的样子
                unsigned long new_cur_row_nnz = cur_row_nnz;

                // 如果当前行非零元数量不能整除TLB的列长度，那就要补成列的长度
                if (cur_row_nnz % thread_level_col_div_fixed_param_strategy->col_block_nnz_num != 0)
                {
                    new_cur_row_nnz = (cur_row_nnz / thread_level_col_div_fixed_param_strategy->col_block_nnz_num + 1) * thread_level_col_div_fixed_param_strategy->col_block_nnz_num;
                    assert(new_cur_row_nnz > cur_row_nnz);
                    assert(new_cur_row_nnz % thread_level_col_div_fixed_param_strategy->col_block_nnz_num == 0);
                }

                // 计算当前TLB数量
                unsigned long new_row_TLB_num = new_cur_row_nnz / thread_level_col_div_fixed_param_strategy->col_block_nnz_num;

                nnz_of_cur_BLB = nnz_of_cur_BLB + new_cur_row_nnz;
                TLB_num_of_cur_BLB = TLB_num_of_cur_BLB + new_row_TLB_num;
            }

            // 总非零元数量
            new_nnz = new_nnz + nnz_of_cur_BLB;
            
            if (max_TLB_of_BLB < TLB_num_of_cur_BLB)
            {
                max_TLB_of_BLB = TLB_num_of_cur_BLB;
            }
        }

        // 可能还剩最后一个块
        if (row_nnz_of_compressed_block.size() % tblock_level_row_div_evenly_param_strategy->block_row_num != 0)
        {
            // 累计非零元数量和TLB数量
            unsigned long nnz_of_cur_BLB = 0;
            unsigned long TLB_num_of_cur_BLB = 0;

            unsigned long remain_row_num = row_nnz_of_compressed_block.size() % tblock_level_row_div_evenly_param_strategy->block_row_num;

            for (unsigned long global_row_id = row_nnz_of_compressed_block.size() - remain_row_num; global_row_id < row_nnz_of_compressed_block.size(); global_row_id++)
            {
                assert(global_row_id < row_nnz_of_compressed_block.size());

                unsigned long cur_row_nnz = row_nnz_of_compressed_block[global_row_id];

                // TLB分块之后的行长度初始化为和当前行长度相等的样子
                unsigned long new_cur_row_nnz = cur_row_nnz;

                // 如果当前行非零元数量不能整除TLB的列长度，那就要补成列的长度
                if (cur_row_nnz % thread_level_col_div_fixed_param_strategy->col_block_nnz_num != 0)
                {
                    new_cur_row_nnz = (cur_row_nnz / thread_level_col_div_fixed_param_strategy->col_block_nnz_num + 1) * thread_level_col_div_fixed_param_strategy->col_block_nnz_num;
                    assert(new_cur_row_nnz > cur_row_nnz);
                    assert(new_cur_row_nnz % thread_level_col_div_fixed_param_strategy->col_block_nnz_num == 0);
                }

                // 计算当前TLB数量
                unsigned long new_row_TLB_num = new_cur_row_nnz / thread_level_col_div_fixed_param_strategy->col_block_nnz_num;

                nnz_of_cur_BLB = nnz_of_cur_BLB + new_cur_row_nnz;
                TLB_num_of_cur_BLB = TLB_num_of_cur_BLB + new_row_TLB_num;
            }

            // 加上被row padding的数量
            assert(tblock_level_row_div_evenly_param_strategy->block_row_num > remain_row_num);
            unsigned long padding_row_num = tblock_level_row_div_evenly_param_strategy->block_row_num - remain_row_num;

            // 每一行非零元的数量
            unsigned long new_padding_row_nnz = min_row_nnz;
            
            if (new_padding_row_nnz % thread_level_col_div_fixed_param_strategy->col_block_nnz_num != 0)
            {
                new_padding_row_nnz = (new_padding_row_nnz / thread_level_col_div_fixed_param_strategy->col_block_nnz_num + 1) * thread_level_col_div_fixed_param_strategy->col_block_nnz_num;
                assert(new_padding_row_nnz > min_row_nnz);
                assert(new_padding_row_nnz % thread_level_col_div_fixed_param_strategy->col_block_nnz_num == 0);
            }

            nnz_of_cur_BLB = nnz_of_cur_BLB + padding_row_num * new_padding_row_nnz;
            TLB_num_of_cur_BLB = TLB_num_of_cur_BLB + new_padding_row_nnz / thread_level_col_div_fixed_param_strategy->col_block_nnz_num * padding_row_num;

            // 总非零元数量
            new_nnz = new_nnz + nnz_of_cur_BLB;
            
            if (max_TLB_of_BLB < TLB_num_of_cur_BLB)
            {
                max_TLB_of_BLB = TLB_num_of_cur_BLB;
            }
        }

        assert(new_nnz >= old_nnz);

        cout << "new_nnz:" << new_nnz << ",old_nnz:" << old_nnz << ",padding rate:" << (float)new_nnz/(float)old_nnz << endl;

        if ((float)new_nnz/(float)old_nnz > get_config()["PADDING_RATE_UP_BOUND"].as_integer())
        {
            cout << "padding rate is larger than " << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
            
            // 析构matrix
            memory_garbage_manager_t mem_manager;

            delete_sparse_struct_t(&mem_manager, matrix);

            matrix = NULL;

            // 放弃这一轮调参
            continue;
        }

        // 如果BLB中的TLB数量大于一个阈值，就直接放弃这轮调参
        if (max_TLB_of_BLB > 1024)
        {
            cout << "too many (> 1024) TLB in BLB, cause low performance" << endl;

            // 析构matrix
            memory_garbage_manager_t mem_manager;

            delete_sparse_struct_t(&mem_manager, matrix);

            matrix = NULL;

            continue;
        }

        if (data_set_collector != NULL)
        {
            // 如果存在一个数据集收集器，那么就需要收集节点的类型和参数
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() > 0);
            
            vector<exe_node_type> compressed_node_type_vec;
            vector<exe_node_param_set_strategy> compressed_param_strategy_vec;
            vector<float> compressed_param_vec;

            // 一共三个节点
            assert(sub_graph_skeleton.exe_node_vec.size() == 3);

            // 第一个节点的类型，参数策略的类型，以及策略的参数
            // COMPRESSED_ROW_PADDING, COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY, compressed_row_padding_direct_param_strategy_t
            assert(sub_graph_skeleton.exe_node_vec[0].type == COMPRESSED_ROW_PADDING);
            assert(param_strategy_skeleton.param_strategy_vec[0].strategy_type == COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY);
            compressed_node_type_vec.push_back(COMPRESSED_ROW_PADDING);
            compressed_param_strategy_vec.push_back(COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY);
            compressed_row_padding_direct_param_strategy_t* strategy1_ptr = (compressed_row_padding_direct_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param_strategy;
            assert(strategy1_ptr != NULL);
            compressed_param_vec.push_back(strategy1_ptr->multiply);
            compressed_param_vec.push_back(strategy1_ptr->padding_row_length);

            // 第二个节点的类型
            // COMPRESSED_TBLOCK_LEVEL_ROW_DIV, COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY, compressed_tblock_level_row_div_evenly_param_strategy_t
            assert(sub_graph_skeleton.exe_node_vec[1].type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
            assert(param_strategy_skeleton.param_strategy_vec[1].strategy_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY);
            compressed_node_type_vec.push_back(COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
            compressed_param_strategy_vec.push_back(COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY);
            compressed_tblock_level_row_div_evenly_param_strategy_t* strategy2_ptr = (compressed_tblock_level_row_div_evenly_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[1].param_strategy;
            assert(strategy2_ptr != NULL);
            compressed_param_vec.push_back(strategy2_ptr->block_row_num);

            // 第三个节点的类型
            // COMPRESSED_THREAD_LEVEL_COL_DIV, COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY, compressed_thread_level_col_div_fixed_param_strategy_t
            assert(sub_graph_skeleton.exe_node_vec[2].type == COMPRESSED_THREAD_LEVEL_COL_DIV);
            assert(param_strategy_skeleton.param_strategy_vec[2].strategy_type == COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY);
            compressed_node_type_vec.push_back(COMPRESSED_THREAD_LEVEL_COL_DIV);
            compressed_param_strategy_vec.push_back(COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY);
            compressed_thread_level_col_div_fixed_param_strategy_t* strategy3_ptr = (compressed_thread_level_col_div_fixed_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[2].param_strategy;
            assert(strategy3_ptr != NULL);
            compressed_param_vec.push_back(strategy3_ptr->col_block_nnz_num);

            // 清除当前compressed阶段的所有积累值
            data_set_collector->clear_compressed_accu_info();
            data_set_collector->insert_compressed_stage_node_and_param_to_cur_item(compressed_node_type_vec, compressed_param_strategy_vec, compressed_param_vec);
        }

        // 执行对应的的子块
        execute_sub_matrix_exe_graph_with_param_strategy(matrix, sub_matrix_id, &sub_graph_skeleton, &param_strategy_skeleton);

        // 候选模板两个
        set<template_type> candi_template_type_set;
        // candi_template_type_set.insert(DIRECT_ATOM_TEMPLATE_WARP_COMPRESS);
        candi_template_type_set.insert(SHARED_MEMORY_TEMPLATE_WARP_COMPRESS);

        // 性能
        float time;
        float gflops;

        // 寻找对应的最优模板，这里先不处理
        template_node_t temp_node = find_best_template_node_of_specific_sub_matrix_from_template_set(matrix, sub_matrix_id, candi_template_type_set, time, gflops, search_strategy_ptr, data_set_collector);

        if (gflops > best_gflops)
        {
            if (best_gflops == 0)
            {
                // 直接赋值，并且修改最佳优化路径
                // 这个时候还没有最优的优化路径
                best_gflops = gflops;
                best_time = time;

                // 从best拷贝出来
                best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
                best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
                best_temp_node = val_copy_from_old_template_node(temp_node);

                // 重新绑定优化骨架和策略骨架
                bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
            }
            else
            {
                // 直接赋值，析构已有的最优路径
                best_gflops = gflops;
                best_time = time;

                // 析构已有的最优参数
                del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                // 析构已有的最优策略
                del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                // 已有的最优模板
                del_param_of_template_node(&best_temp_node);
                
                // 执行新的拷贝
                // 从best拷贝出来
                best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
                best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
                best_temp_node = val_copy_from_old_template_node(temp_node);

                // 重新绑定优化骨架和策略骨架
                bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);   
            }
        }

        // 析构matrix
        // 现在matrix肯定存在
        assert(matrix != NULL);
        memory_garbage_manager_t mem_manager;
        delete_sparse_struct_t(&mem_manager, matrix);
        matrix = NULL;
        
        // 重置所有参数，并且重置所有参数指针，这一步用来处理一些参数是数组的节点，数组中的内容应该重新清空
        reset_exe_node_param_and_param_strategy_of_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton);
        
        // 如果所有的模板参数执行后都发生错误，那么这里的temp_node中可能是没有参数的
        if (temp_node.template_param != NULL)
        {
            del_param_of_template_node(&temp_node);
        }
        else
        {
            // 当前可能没有出现对应的
            assert(gflops == 0);
        }

        // 加入提前结束的相关内容
        if (search_strategy_ptr != NULL)
        {
            if (continue_search(search_strategy_ptr) == false)
            {
                search_finished_by_strategy = true;
            }
        }

        if (search_finished_by_strategy == true)
        {
            break;
        }
    }

    // 如果之前没有析构矩阵，现在就析构矩阵
    if (matrix != NULL)
    {
        assert(matrix != NULL);
        memory_garbage_manager_t mem_manager;
        delete_sparse_struct_t(&mem_manager, matrix);
        matrix = NULL;
    }

    // 析构用以遍历各种优化路径的两个骨架的参数
    del_strategy_of_param_strategy_node_in_sub_matrix(&param_strategy_skeleton);
    del_exe_node_param_of_compress_sub_matrix(&sub_graph_skeleton);

    compressed_sub_block_exe_graph_and_template_t return_sub_graph_exe_node_and_template;
    return_sub_graph_exe_node_and_template.sub_graph = best_sub_graph;
    return_sub_graph_exe_node_and_template.sub_graph_param_strategy = best_sub_graph_param_strategy;
    return_sub_graph_exe_node_and_template.temp_node = best_temp_node;

    return return_sub_graph_exe_node_and_template;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy3(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 首先执行对应的稠密子图优化
    sparse_struct_t* matrix = get_matrix_dense_view_graph(&dense_graph);
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_white_list_strategy3(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);
    matrix = NULL;

    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy3(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    sparse_struct_t* matrix = execute_dense_matrix_exe_graph_with_param_strategy(&(dense_graph.dense_sub_graph), &(dense_graph.dense_sub_graph_param_strategy));
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_white_list_strategy3(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);
    matrix = NULL;

    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy3(sparse_struct_t* input_matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(input_matrix != NULL);
    // 矩阵的一系列检查
    assert(input_matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(input_matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    // 执行复制
    sparse_struct_t* matrix = val_copy_from_old_matrix_struct(input_matrix);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 获得行非零元数量
    vector<unsigned long> row_nnz_of_compressed_block = get_nnz_of_each_row_in_compressed_sub_matrix(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr);

    unsigned long max_row_nnz = row_nnz_of_compressed_block[0];
    unsigned long min_row_nnz = row_nnz_of_compressed_block[0];

    // 遍历获得最大的和最小的行非零元数量
    for (unsigned long i = 1; i < row_nnz_of_compressed_block.size(); i++)
    {
        unsigned long cur_row_nnz = row_nnz_of_compressed_block[i];

        if (max_row_nnz < cur_row_nnz)
        {
            max_row_nnz = cur_row_nnz;
        }

        if (min_row_nnz > cur_row_nnz)
        {
            min_row_nnz = cur_row_nnz;
        }
    }

    // 找出最优的子图、最优的参数策略、最优模板类型及其参数
    template_node_t best_temp_node;
    exe_compressed_sub_graph_t best_sub_graph;
    param_strategy_of_sub_graph_t best_sub_graph_param_strategy;

    best_time = 99999999999999;
    best_gflops = 0;

    // 定一个优化路径骨架
    exe_compressed_sub_graph_t sub_graph_skeleton;
    // 定一个参数设定的骨架
    param_strategy_of_sub_graph_t param_strategy_skeleton;

    // 按照非零元数量执行行分块
    compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t BLB_row_div_nnz_param_strategy;
    
    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_TBLOCK_LEVEL_ROW_DIV, COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY, &BLB_row_div_nnz_param_strategy);

    // 增加对于参数的调节
    param_enumerater_t param_setter;

    assert(sub_graph_skeleton.exe_node_vec.size() == 1);

    compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t* tblock_level_row_div_acc_to_least_nnz_param_strategy_ptr = (compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param_strategy;

    assert(tblock_level_row_div_acc_to_least_nnz_param_strategy_ptr != NULL);

    // 行分块最少的非零元数量
    register_integer_independ_param_to_enumerater(&param_setter, &(tblock_level_row_div_acc_to_least_nnz_param_strategy_ptr->nnz_low_bound), 128, 1024, 384);

    // 根据行条带非零元的数量控制列分块的上界，上界是nnz / 64，步长为nnz / 192，下界为1
    // 增加一个thread的列分块
    // 添加一个TLB的纵分块
    compressed_thread_level_col_div_fixed_param_strategy_t thread_col_param_strategy;
    
    // 将策略加到优化骨架中
    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_THREAD_LEVEL_COL_DIV, COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY, &thread_col_param_strategy);

    // 将参数的指针提取出来
    compressed_thread_level_col_div_fixed_param_strategy_t* thread_level_col_div_fixed_param_strategy = (compressed_thread_level_col_div_fixed_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[1].param_strategy;
    
    // 计算最大的TLB列分块大小
    unsigned long max_col_size_acc_to_nnz = max_row_nnz / 2;

    if (max_row_nnz % 2 == 0)
    {
        max_col_size_acc_to_nnz = max_col_size_acc_to_nnz + 1;
    }
    
    assert(thread_level_col_div_fixed_param_strategy != NULL);
    assert(sub_graph_skeleton.exe_node_vec.size() == 2 && param_strategy_skeleton.param_strategy_vec.size() == 2);

    bool search_finished_by_strategy = false;

    while (set_param_combination_to_next(&param_setter) == false)
    {
        // 先规定一个上界
        unsigned long TLB_up_bound_acc_to_row_block_nnz = tblock_level_row_div_acc_to_least_nnz_param_strategy_ptr->nnz_low_bound / 64;

        if (TLB_up_bound_acc_to_row_block_nnz < 1)
        {
            TLB_up_bound_acc_to_row_block_nnz = 1;
        }

        unsigned long TLB_step_size = tblock_level_row_div_acc_to_least_nnz_param_strategy_ptr->nnz_low_bound / 192;

        if (TLB_step_size < 1)
        {
            TLB_step_size = 1;
        }

        // 分块之后每个行条带的行数量
        // 遍历每一行，找出行条带的分块位置，然后找出两个分块位置之间经过padding的nnz数量和每个块的TLB数量
        vector<unsigned long> row_num_of_each_BLB = row_block_size_of_a_sub_matrix_by_nnz_low_bound(row_nnz_of_compressed_block, tblock_level_row_div_acc_to_least_nnz_param_strategy_ptr->nnz_low_bound);

        // 用一个循环来遍历，处理不同的TLB非零元数量
        for (unsigned long TLB_col_size = 1; TLB_col_size <= TLB_up_bound_acc_to_row_block_nnz && TLB_col_size <= max_col_size_acc_to_nnz; TLB_col_size = TLB_col_size + TLB_step_size)
        {
            thread_level_col_div_fixed_param_strategy->col_block_nnz_num = TLB_col_size;

            // 如果矩阵已经不存在，那就执行稠密子图的优化路径，产生新的矩阵
            if (matrix == NULL)
            {
                matrix = val_copy_from_old_matrix_struct(input_matrix);
                assert(matrix != NULL);
                // 矩阵的一系列检查
                assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
                assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

                compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;
            }

            // 打印参数，行分块非零元数量和TLB分块的列大小
            cout << "thread_level_col_div_fixed_param_strategy->col_block_nnz_num:" << thread_level_col_div_fixed_param_strategy->col_block_nnz_num << ",tblock_level_row_div_acc_to_least_nnz_param_strategy_ptr->nnz_low_bound:" << tblock_level_row_div_acc_to_least_nnz_param_strategy_ptr->nnz_low_bound << endl;

            // // 查看当前的padding率以及每个BLB中的TLB的数量。超过1024就放弃分块的条带，这里也考虑padding产生的新条带
            // 并且需要计算padding率
            unsigned long old_nnz = compressed_block_ptr->read_index[0]->length;
            unsigned long new_nnz = 0;

            // 最大的TLB数量
            unsigned long max_TLB_of_BLB = 0;

            // 当前BLB的首行行索引
            unsigned long BLB_first_row_index = 0;

            // 遍历每一个BLB，查看padding的nnz数量和
            for (unsigned long i = 0; i < row_num_of_each_BLB.size(); i++)
            {
                unsigned long nnz_of_cur_BLB = 0;
                unsigned long TLB_num_of_cur_BLB = 0;

                // BLB行
                for (unsigned long local_row_id = 0; local_row_id < row_num_of_each_BLB[i]; local_row_id++)
                {
                    unsigned long global_row_id = BLB_first_row_index + local_row_id;
                    
                    assert(global_row_id < row_nnz_of_compressed_block.size());

                    unsigned long cur_row_nnz = row_nnz_of_compressed_block[global_row_id];

                    unsigned long new_cur_row_nnz = cur_row_nnz;

                    // 如果当前行非零元数量不能整除TLB的列长度，那就要补成列的长度
                    if (cur_row_nnz % thread_level_col_div_fixed_param_strategy->col_block_nnz_num != 0)
                    {
                        new_cur_row_nnz = (cur_row_nnz / thread_level_col_div_fixed_param_strategy->col_block_nnz_num + 1) * thread_level_col_div_fixed_param_strategy->col_block_nnz_num;
                        assert(new_cur_row_nnz > cur_row_nnz);
                        assert(new_cur_row_nnz % thread_level_col_div_fixed_param_strategy->col_block_nnz_num == 0);
                    }

                    // 计算当前TLB的数量
                    unsigned long new_row_TLB_num = new_cur_row_nnz / thread_level_col_div_fixed_param_strategy->col_block_nnz_num;

                    nnz_of_cur_BLB = nnz_of_cur_BLB + new_cur_row_nnz;
                    TLB_num_of_cur_BLB = TLB_num_of_cur_BLB + new_row_TLB_num;
                }

                // 总非零元数量
                new_nnz = new_nnz + nnz_of_cur_BLB;
            
                if (max_TLB_of_BLB < TLB_num_of_cur_BLB)
                {
                    max_TLB_of_BLB = TLB_num_of_cur_BLB;
                }

                BLB_first_row_index = BLB_first_row_index + row_num_of_each_BLB[i];
            }

            assert(BLB_first_row_index == row_nnz_of_compressed_block.size());

            // 因为new_nnz的计算并不完全精确，所以这里放弃
            if (new_nnz < old_nnz)
            {
                cout << "new_nnz:" << new_nnz << endl;
                cout << "old_nnz:" << old_nnz << endl;
                assert(false);
            }

            cout << "new_nnz:" << new_nnz << ",old_nnz:" << old_nnz << ",padding rate:" << (float)new_nnz/(float)old_nnz << endl;

            if ((float)new_nnz/(float)old_nnz > get_config()["PADDING_RATE_UP_BOUND"].as_integer())
            {
                cout << "padding rate is larger than " << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                
                // 析构matrix
                memory_garbage_manager_t mem_manager;

                delete_sparse_struct_t(&mem_manager, matrix);

                matrix = NULL;

                // 放弃这一轮调参
                continue;
            }

            if (max_TLB_of_BLB > 1024)
            {
                cout << "too many (> 1024) TLB in BLB, cause low performance" << endl;

                // 析构matrix
                memory_garbage_manager_t mem_manager;

                delete_sparse_struct_t(&mem_manager, matrix);

                matrix = NULL;

                continue;
            }

            if (data_set_collector != NULL)
            {
                // 按照当前的逻辑，稠密视图已经加入了对应的数据
                assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
                assert(data_set_collector->accu_dense_graph_node_type_vec.size() > 0);

                vector<exe_node_type> compressed_node_type_vec;
                vector<exe_node_param_set_strategy> compressed_param_strategy_vec;
                vector<float> compressed_param_vec;

                // 检查一下插入的数据的数量是不是满足要求，类型是不是满足要求
                assert(sub_graph_skeleton.exe_node_vec.size() == 2);

                // 第一个节点的类型，参数策略类型，参数策略参数
                // COMPRESSED_TBLOCK_LEVEL_ROW_DIV, COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY, compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t
                assert(sub_graph_skeleton.exe_node_vec[0].type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
                assert(param_strategy_skeleton.param_strategy_vec[0].strategy_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY);
                compressed_node_type_vec.push_back(COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
                compressed_param_strategy_vec.push_back(COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY);
                compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t* strategy1_ptr = (compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param_strategy;
                assert(strategy1_ptr != NULL);
                compressed_param_vec.push_back(strategy1_ptr->nnz_low_bound);

                // 第二个节点的类型，
                // COMPRESSED_THREAD_LEVEL_COL_DIV, COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY, compressed_thread_level_col_div_fixed_param_strategy_t
                assert(sub_graph_skeleton.exe_node_vec[1].type == COMPRESSED_THREAD_LEVEL_COL_DIV);
                assert(param_strategy_skeleton.param_strategy_vec[1].strategy_type == COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY);
                compressed_node_type_vec.push_back(COMPRESSED_THREAD_LEVEL_COL_DIV);
                compressed_param_strategy_vec.push_back(COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY);
                compressed_thread_level_col_div_fixed_param_strategy_t* strategy2_ptr = (compressed_thread_level_col_div_fixed_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[1].param_strategy;
                assert(strategy2_ptr != NULL);
                compressed_param_vec.push_back(strategy2_ptr->col_block_nnz_num);

                // 清除当前compressed阶段的所有积累值
                data_set_collector->clear_compressed_accu_info();
                data_set_collector->insert_compressed_stage_node_and_param_to_cur_item(compressed_node_type_vec, compressed_param_strategy_vec, compressed_param_vec);
            }

            execute_sub_matrix_exe_graph_with_param_strategy(matrix, sub_matrix_id, &sub_graph_skeleton, &param_strategy_skeleton);

            // 有一个候选模板
            set<template_type> candi_template_type_set;
            candi_template_type_set.insert(SHARED_MEMORY_TEMPLATE_WARP_COMPRESS);

            // 性能
            float time;
            float gflops;

            template_node_t temp_node = find_best_template_node_of_specific_sub_matrix_from_template_set(matrix, sub_matrix_id, candi_template_type_set, time, gflops, search_strategy_ptr, data_set_collector);

            if (gflops > best_gflops)
            {
                if (best_gflops == 0)
                {
                    best_gflops = gflops;
                    best_time = time;

                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
                    best_temp_node = val_copy_from_old_template_node(temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
                else
                {
                    // 直接赋值，析构已有的最优路径
                    best_gflops = gflops;
                    best_time = time;

                    // 析构已有的最优参数
                    del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                    // 析构已有的最优策略
                    del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                    // 已有的最优模板
                    del_param_of_template_node(&best_temp_node);
                    
                    // 执行新的拷贝
                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
                    best_temp_node = val_copy_from_old_template_node(temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);   
                }
            }
            
            // 析构matrix
            // 现在matrix肯定存在
            assert(matrix != NULL);
            memory_garbage_manager_t mem_manager;
            delete_sparse_struct_t(&mem_manager, matrix);
            matrix = NULL;
            
            // 重置所有参数，并且重置所有参数指针，这一步用来处理一些参数是数组的节点，数组中的内容应该重新清空
            reset_exe_node_param_and_param_strategy_of_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton);
            
            // 如果所有的模板参数执行后都发生错误，那么这里的temp_node中可能是没有参数的
            if (temp_node.template_param != NULL)
            {
                del_param_of_template_node(&temp_node);
            }
            else
            {
                // 当前可能没有出现对应的
                assert(gflops == 0);
            }

            // 加入提前结束的相关内容
            if (search_strategy_ptr != NULL)
            {
                if (continue_search(search_strategy_ptr) == false)
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
    
    // 如果之前没有析构矩阵，现在就析构矩阵
    if (matrix != NULL)
    {
        assert(matrix != NULL);
        memory_garbage_manager_t mem_manager;
        delete_sparse_struct_t(&mem_manager, matrix);
        matrix = NULL;
    }

    // 析构用以遍历各种优化路径的两个骨架的参数
    del_strategy_of_param_strategy_node_in_sub_matrix(&param_strategy_skeleton);
    del_exe_node_param_of_compress_sub_matrix(&sub_graph_skeleton);

    compressed_sub_block_exe_graph_and_template_t return_sub_graph_exe_node_and_template;
    return_sub_graph_exe_node_and_template.sub_graph = best_sub_graph;
    return_sub_graph_exe_node_and_template.sub_graph_param_strategy = best_sub_graph_param_strategy;
    return_sub_graph_exe_node_and_template.temp_node = best_temp_node;

    return return_sub_graph_exe_node_and_template;
}

// WLB的列分块，列分块的宽度从最大的行非零元数量到最小的行非零元数量
compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy4(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 首先执行对应的稠密子图优化
    sparse_struct_t* matrix = get_matrix_dense_view_graph(&dense_graph);
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    compressed_sub_block_exe_graph_and_template_t sub_graph_best_path = find_best_path_of_white_list_strategy4(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);
    matrix = NULL;

    return sub_graph_best_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy4(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 首先执行对应的稠密子图优化
    sparse_struct_t* matrix = execute_dense_matrix_exe_graph_with_param_strategy(&(dense_graph.dense_sub_graph), &(dense_graph.dense_sub_graph_param_strategy));
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    compressed_sub_block_exe_graph_and_template_t sub_graph_best_path = find_best_path_of_white_list_strategy4(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);
    matrix = NULL;

    return sub_graph_best_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy4(sparse_struct_t* input_matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(input_matrix != NULL);
    // 矩阵的一系列检查
    assert(input_matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(input_matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    sparse_struct_t* matrix = val_copy_from_old_matrix_struct(input_matrix);

    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 获得行非零元数量
    vector<unsigned long> row_nnz_of_compressed_block = get_nnz_of_each_row_in_compressed_sub_matrix(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr);

    unsigned long max_row_nnz = row_nnz_of_compressed_block[0];
    unsigned long min_row_nnz = row_nnz_of_compressed_block[0];

    // 遍历获得最大的和最小的行非零元数量
    for (unsigned long i = 1; i < row_nnz_of_compressed_block.size(); i++)
    {
        unsigned long cur_row_nnz = row_nnz_of_compressed_block[i];

        if (max_row_nnz < cur_row_nnz)
        {
            max_row_nnz = cur_row_nnz;
        }

        if (min_row_nnz > cur_row_nnz)
        {
            min_row_nnz = cur_row_nnz;
        }
    }

    // 找出最优的子图、最优的参数策略、最优模板类型及其参数
    template_node_t best_temp_node;
    exe_compressed_sub_graph_t best_sub_graph;
    param_strategy_of_sub_graph_t best_sub_graph_param_strategy;

    best_time = 99999999999999;
    best_gflops = 0;

    // 定一个优化路径骨架
    exe_compressed_sub_graph_t sub_graph_skeleton;
    // 定一个参数设定的骨架
    param_strategy_of_sub_graph_t param_strategy_skeleton;

    // 增加一个WLB列分块
    compressed_warp_level_col_div_fixed_param_strategy_t WLB_col_div_strategy;
    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_WARP_LEVEL_COL_DIV, COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY, &WLB_col_div_strategy);

    param_enumerater_t param_setter;

    assert(sub_graph_skeleton.exe_node_vec.size() == 1);

    compressed_warp_level_col_div_fixed_param_strategy_t* warp_level_col_div_fixed_param_strategy_ptr = (compressed_warp_level_col_div_fixed_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param_strategy;

    assert(warp_level_col_div_fixed_param_strategy_ptr != NULL);

    // 增加一个参数变化器，warp列分块大小从最小行号到最大行号，中间加一个步长，为了增加一个行分块的效果，上界可以稍稍高过最大非零元数量，从而产生WLB级别行分块的效果
    // 下界是比最小行长度大的最小32的倍数
    unsigned long warp_low_bound = 32;

    // if (warp_low_bound % 32 != 0)
    // {
    //     warp_low_bound = (warp_low_bound / 32 + 1) * 32;
    // }

    // 计算步长，尽可能保证走完所有参数只有4格
    unsigned long step_size = (max_row_nnz - min_row_nnz) / 3;
    // 必须保证，是32的倍数，按上取整
    
    if (step_size % 32 != 0)
    {
        step_size = (step_size / 32 + 1) * 32;
    }

    if (step_size < 32)
    {
        step_size = 32;
    }

    // 从下界不断加stepsize，上界是刚好大于等于行最大非零元的值
    unsigned long warp_up_bound = warp_low_bound;

    while (warp_up_bound < max_row_nnz)
    {
        warp_up_bound = warp_up_bound + step_size;
    }

    // 注册warp的参数调节
    register_integer_independ_param_to_enumerater(&param_setter, &(warp_level_col_div_fixed_param_strategy_ptr->col_block_nnz_num), warp_low_bound, warp_up_bound, step_size);

    bool search_finished_by_strategy = false;

    while (set_param_combination_to_next(&param_setter) == false)
    {
        // warp列分块的大小为32的倍数
        assert(warp_level_col_div_fixed_param_strategy_ptr->col_block_nnz_num % 32 == 0);

        // 如果matrix不存在了，那就重新创造一个matrix
        if (matrix == NULL)
        {
            matrix = val_copy_from_old_matrix_struct(input_matrix);
            assert(matrix != NULL);
            // 矩阵的一系列检查
            assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
            assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

            compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;
        }

        // 查看padding率，不管切分的宽度是多少，最终都padding到32的倍数
        unsigned long old_nnz = compressed_block_ptr->read_index[0]->length;
        unsigned long new_nnz = 0;

        for (unsigned long i = 0; i < row_nnz_of_compressed_block.size(); i++)
        {
            unsigned long cur_row_nnz = row_nnz_of_compressed_block[i];

            unsigned long new_row_nnz = cur_row_nnz;

            if (new_row_nnz % 32 != 0)
            {
                new_row_nnz = (new_row_nnz / 32 + 1) * 32;
            }

            new_nnz = new_row_nnz + new_nnz;
        }

        // 如果padding率过高就放弃
        // 这个时候
        assert(new_nnz >= old_nnz);

        cout << "new_nnz:" << new_nnz << ",old_nnz:" << old_nnz << ",padding rate:" << (float)new_nnz/(float)old_nnz << endl;

        if ((float)new_nnz/(float)old_nnz > get_config()["PADDING_RATE_UP_BOUND"].as_integer())
        {
            cout << "padding rate is larger than " << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
            
            // 析构matrix
            memory_garbage_manager_t mem_manager;

            delete_sparse_struct_t(&mem_manager, matrix);

            matrix = NULL;

            // 放弃这一轮调参
            continue;
        }

        if (data_set_collector != NULL)
        {
            // 收集节点的类型和参数
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() > 0);
            
            vector<exe_node_type> compressed_node_type_vec;
            vector<exe_node_param_set_strategy> compressed_param_strategy_vec;
            vector<float> compressed_param_vec;

            // 只有一个节点
            assert(sub_graph_skeleton.exe_node_vec.size() == 1);

            // 唯一的一个节点
            // COMPRESSED_WARP_LEVEL_COL_DIV, COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY, compressed_warp_level_col_div_fixed_param_strategy_t
            assert(sub_graph_skeleton.exe_node_vec[0].type == COMPRESSED_WARP_LEVEL_COL_DIV);
            assert(param_strategy_skeleton.param_strategy_vec[0].strategy_type == COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY);
            compressed_node_type_vec.push_back(COMPRESSED_WARP_LEVEL_COL_DIV);
            compressed_param_strategy_vec.push_back(COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY);
            compressed_warp_level_col_div_fixed_param_strategy_t* strategy1_ptr = (compressed_warp_level_col_div_fixed_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param_strategy;
            assert(strategy1_ptr != NULL);
            compressed_param_vec.push_back(strategy1_ptr->col_block_nnz_num);
            
            // 清除当前compressed阶段的所有积累值
            data_set_collector->clear_compressed_accu_info();
            data_set_collector->insert_compressed_stage_node_and_param_to_cur_item(compressed_node_type_vec, compressed_param_strategy_vec, compressed_param_vec);
        }

        execute_sub_matrix_exe_graph_with_param_strategy(matrix, sub_matrix_id, &sub_graph_skeleton, &param_strategy_skeleton);

        // 执行找到对应的模板
        // 有一个候选模板
        set<template_type> candi_template_type_set;
        candi_template_type_set.insert(DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE);

        // 性能
        float time;
        float gflops;

        template_node_t temp_node = find_best_template_node_of_specific_sub_matrix_from_template_set(matrix, sub_matrix_id, candi_template_type_set, time, gflops, search_strategy_ptr, data_set_collector);

        if (gflops > best_gflops)
        {
            if (best_gflops == 0)
            {
                best_gflops = gflops;
                best_time = time;

                // 从best拷贝出来
                best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
                best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
                best_temp_node = val_copy_from_old_template_node(temp_node);

                // 重新绑定优化骨架和策略骨架
                bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
            }
            else
            {
                // 直接赋值，析构已有的最优路径
                best_gflops = gflops;
                best_time = time;

                // 析构已有的最优参数
                del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                // 析构已有的最优策略
                del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                // 已有的最优模板
                del_param_of_template_node(&best_temp_node);
                
                // 执行新的拷贝
                // 从best拷贝出来
                best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
                best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
                best_temp_node = val_copy_from_old_template_node(temp_node);

                // 重新绑定优化骨架和策略骨架
                bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);   
            }
        }
        
        // 析构matrix
        // 现在matrix肯定存在
        assert(matrix != NULL);
        memory_garbage_manager_t mem_manager;
        delete_sparse_struct_t(&mem_manager, matrix);
        matrix = NULL;

        // 重置所有参数，并且重置所有参数指针，这一步用来处理一些参数是数组的节点，数组中的内容应该重新清空
        reset_exe_node_param_and_param_strategy_of_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton);
        
        // 如果所有的模板参数执行后都发生错误，那么这里的temp_node中可能是没有参数的
        // 如果所有的模板参数执行后都发生错误，那么这里的temp_node中可能是没有参数的
        if (temp_node.template_param != NULL)
        {
            del_param_of_template_node(&temp_node);
        }
        else
        {
            // 当前可能没有出现对应的
            assert(gflops == 0);
        }

        // 加入提前结束的相关内容
        if (search_strategy_ptr != NULL)
        {
            if (continue_search(search_strategy_ptr) == false)
            {
                search_finished_by_strategy = true;
            }
        }

        if (search_finished_by_strategy == true)
        {
            break;
        }
    }

    // 如果之前没有析构矩阵，现在就析构矩阵
    if (matrix != NULL)
    {
        assert(matrix != NULL);
        memory_garbage_manager_t mem_manager;
        delete_sparse_struct_t(&mem_manager, matrix);
        matrix = NULL;
    }

    // 析构用以遍历各种优化路径的两个骨架的参数
    del_strategy_of_param_strategy_node_in_sub_matrix(&param_strategy_skeleton);
    del_exe_node_param_of_compress_sub_matrix(&sub_graph_skeleton);

    compressed_sub_block_exe_graph_and_template_t return_sub_graph_exe_node_and_template;
    return_sub_graph_exe_node_and_template.sub_graph = best_sub_graph;
    return_sub_graph_exe_node_and_template.sub_graph_param_strategy = best_sub_graph_param_strategy;
    return_sub_graph_exe_node_and_template.temp_node = best_temp_node;

    return return_sub_graph_exe_node_and_template;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy5(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 首先执行对应的稠密子图优化
    sparse_struct_t* matrix = get_matrix_dense_view_graph(&dense_graph);
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_white_list_strategy5(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);
    matrix = NULL;

    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy5(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    sparse_struct_t* matrix = execute_dense_matrix_exe_graph_with_param_strategy(&(dense_graph.dense_sub_graph), &(dense_graph.dense_sub_graph_param_strategy));
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_white_list_strategy5(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);
    matrix = NULL;

    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy5(sparse_struct_t* input_matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(input_matrix != NULL);
    // 矩阵的一系列检查
    assert(input_matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(input_matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    sparse_struct_t* matrix = val_copy_from_old_matrix_struct(input_matrix);

    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 获得行非零元数量
    vector<unsigned long> row_nnz_of_compressed_block = get_nnz_of_each_row_in_compressed_sub_matrix(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr);

    unsigned long max_row_nnz = row_nnz_of_compressed_block[0];
    unsigned long min_row_nnz = row_nnz_of_compressed_block[0];

    // 遍历获得最大的和最小的行非零元数量
    for (unsigned long i = 1; i < row_nnz_of_compressed_block.size(); i++)
    {
        unsigned long cur_row_nnz = row_nnz_of_compressed_block[i];

        if (max_row_nnz < cur_row_nnz)
        {
            max_row_nnz = cur_row_nnz;
        }

        if (min_row_nnz > cur_row_nnz)
        {
            min_row_nnz = cur_row_nnz;
        }
    }

    // 找出最优的子图、最优的参数策略、最优模板类型及其参数
    template_node_t best_temp_node;
    exe_compressed_sub_graph_t best_sub_graph;
    param_strategy_of_sub_graph_t best_sub_graph_param_strategy;

    best_time = 99999999999999;
    best_gflops = 0;

    // 定一个优化路径骨架
    exe_compressed_sub_graph_t sub_graph_skeleton;
    // 定一个参数设定的骨架
    param_strategy_of_sub_graph_t param_strategy_skeleton;

    // 执行一个32的row padding
    compressed_row_padding_direct_param_strategy_t row_padding_param_strategy;
    row_padding_param_strategy.multiply = 32;
    row_padding_param_strategy.padding_row_length = 1;

    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_ROW_PADDING, COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY, &row_padding_param_strategy);

    // 增加一个TLB行分块
    compressed_thread_level_row_div_none_param_strategy_t TLB_row_div_strategy;
    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_THREAD_LEVEL_ROW_DIV, COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY, &TLB_row_div_strategy);

    assert(sub_graph_skeleton.exe_node_vec.size() == 2);

    // 没有要遍历的参数，直接查看padding率
    unsigned long old_nnz = compressed_block_ptr->read_index[0]->length;
    
    unsigned long row_num_after_padding = row_nnz_of_compressed_block.size();
    // padding之后的行数量
    if (row_num_after_padding % 32 != 0)
    {
        row_num_after_padding = (row_num_after_padding / 32 + 1) * 32;
    }

    // 新的非零元数量
    unsigned long new_nnz = max_row_nnz * row_num_after_padding;

    assert(new_nnz >= old_nnz);

    cout << "new_nnz:" << new_nnz << ",old_nnz:" << old_nnz << ",padding rate:" << (float)new_nnz/(float)old_nnz << endl;

    if ((float)new_nnz/(float)old_nnz > get_config()["PADDING_RATE_UP_BOUND"].as_integer())
    {
        cout << "padding rate is larger than " << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
        
        // 析构matrix
        memory_garbage_manager_t mem_manager;

        delete_sparse_struct_t(&mem_manager, matrix);

        matrix = NULL;

        // 直接返回
        // 析构用以遍历各种优化路径的两个骨架的参数
        del_strategy_of_param_strategy_node_in_sub_matrix(&param_strategy_skeleton);
        del_exe_node_param_of_compress_sub_matrix(&sub_graph_skeleton);

        compressed_sub_block_exe_graph_and_template_t return_sub_graph_exe_node_and_template;
        return_sub_graph_exe_node_and_template.sub_graph = best_sub_graph;
        return_sub_graph_exe_node_and_template.sub_graph_param_strategy = best_sub_graph_param_strategy;
        return_sub_graph_exe_node_and_template.temp_node = best_temp_node;

        return return_sub_graph_exe_node_and_template;
    }

    if (data_set_collector != NULL)
    {
        // 如果存在一个数据集收集器，那么就需要收集节点的类型和参数
        assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
        assert(data_set_collector->accu_dense_graph_node_type_vec.size() > 0);
        
        vector<exe_node_type> compressed_node_type_vec;
        vector<exe_node_param_set_strategy> compressed_param_strategy_vec;
        vector<float> compressed_param_vec;

        // 一共两个节点
        assert(sub_graph_skeleton.exe_node_vec.size() == 2);

        // 第一个节点
        // COMPRESSED_ROW_PADDING, COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY, compressed_row_padding_direct_param_strategy_t
        assert(sub_graph_skeleton.exe_node_vec[0].type == COMPRESSED_ROW_PADDING);
        assert(param_strategy_skeleton.param_strategy_vec[0].strategy_type == COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY);
        compressed_node_type_vec.push_back(COMPRESSED_ROW_PADDING);
        compressed_param_strategy_vec.push_back(COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY);
        compressed_row_padding_direct_param_strategy_t* strategy1_ptr = (compressed_row_padding_direct_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param_strategy;
        assert(strategy1_ptr != NULL);
        compressed_param_vec.push_back(strategy1_ptr->multiply);
        compressed_param_vec.push_back(strategy1_ptr->padding_row_length);

        // 第二个节点的类型
        // COMPRESSED_THREAD_LEVEL_ROW_DIV, COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY, compressed_thread_level_row_div_none_param_strategy_t
        assert(sub_graph_skeleton.exe_node_vec[1].type == COMPRESSED_THREAD_LEVEL_ROW_DIV);
        assert(param_strategy_skeleton.param_strategy_vec[1].strategy_type == COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY);
        compressed_node_type_vec.push_back(COMPRESSED_THREAD_LEVEL_ROW_DIV);
        compressed_param_strategy_vec.push_back(COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY);
        
        // 清除当前compressed阶段的所有积累值
        data_set_collector->clear_compressed_accu_info();
        data_set_collector->insert_compressed_stage_node_and_param_to_cur_item(compressed_node_type_vec, compressed_param_strategy_vec, compressed_param_vec);
    }

    // padding率达标
    execute_sub_matrix_exe_graph_with_param_strategy(matrix, sub_matrix_id, &sub_graph_skeleton, &param_strategy_skeleton);

    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[4]->block_num % 32 == 0);

    // 候选模板两个
    set<template_type> candi_template_type_set;
    candi_template_type_set.insert(DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS);

    // 性能
    float time;
    float gflops;

    // 寻找对应的最优模板，这里先不处理
    template_node_t temp_node = find_best_template_node_of_specific_sub_matrix_from_template_set(matrix, sub_matrix_id, candi_template_type_set, time, gflops, search_strategy_ptr, data_set_collector);

    if (gflops > best_gflops)
    {
        if (best_gflops == 0)
        {
            // 直接赋值，并且修改最佳优化路径
            // 这个时候还没有最优的优化路径
            best_gflops = gflops;
            best_time = time;

            // 从best拷贝出来
            best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
            best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
            best_temp_node = val_copy_from_old_template_node(temp_node);

            // 重新绑定优化骨架和策略骨架
            bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
        }
        else
        {
            // 直接赋值，析构已有的最优路径
            best_gflops = gflops;
            best_time = time;

            // 析构已有的最优参数
            del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
            // 析构已有的最优策略
            del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
            // 已有的最优模板
            del_param_of_template_node(&best_temp_node);
            
            // 执行新的拷贝
            // 从best拷贝出来
            best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
            best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
            best_temp_node = val_copy_from_old_template_node(temp_node);

            // 重新绑定优化骨架和策略骨架
            bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);   
            
            // 如果所有的模板参数执行后都发生错误，那么这里的temp_node中可能是没有参数的
            // 如果所有的模板参数执行后都发生错误，那么这里的temp_node中可能是没有参数的
            if (temp_node.template_param != NULL)
            {
                del_param_of_template_node(&temp_node);
            }
            else
            {
                // 当前可能没有出现对应的
                assert(gflops == 0);
            }
        }
    }
    
    // 析构matrix
    // 现在matrix肯定存在
    assert(matrix != NULL);
    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);
    matrix = NULL;
    
    // 析构用以遍历各种优化路径的两个骨架的参数
    del_strategy_of_param_strategy_node_in_sub_matrix(&param_strategy_skeleton);
    del_exe_node_param_of_compress_sub_matrix(&sub_graph_skeleton);

    compressed_sub_block_exe_graph_and_template_t return_sub_graph_exe_node_and_template;
    return_sub_graph_exe_node_and_template.sub_graph = best_sub_graph;
    return_sub_graph_exe_node_and_template.sub_graph_param_strategy = best_sub_graph_param_strategy;
    return_sub_graph_exe_node_and_template.temp_node = best_temp_node;

    return return_sub_graph_exe_node_and_template;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy6(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 首先执行对应的稠密子图优化
    sparse_struct_t* matrix = get_matrix_dense_view_graph(&dense_graph);
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_white_list_strategy6(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);
    matrix = NULL;
    
    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy6(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 首先执行对应的稠密子图优化
    sparse_struct_t* matrix = execute_dense_matrix_exe_graph_with_param_strategy(&(dense_graph.dense_sub_graph), &(dense_graph.dense_sub_graph_param_strategy));
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_white_list_strategy6(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);
    matrix = NULL;
    
    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy6(sparse_struct_t* input_matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(input_matrix != NULL);
    // 矩阵的一系列检查
    assert(input_matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(input_matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    sparse_struct_t* matrix = val_copy_from_old_matrix_struct(input_matrix);

    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 获取每一行的行非零元数量，并且获得最大和最小行非零元数量
    vector<unsigned long> row_nnz_of_compressed_block = get_nnz_of_each_row_in_compressed_sub_matrix(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr);
    
    assert(row_nnz_of_compressed_block.size() > 0);

    unsigned long max_row_nnz = row_nnz_of_compressed_block[0];
    unsigned long min_row_nnz = row_nnz_of_compressed_block[0];

    // 遍历获得最大的和最小的行非零元数量
    for (unsigned long i = 1; i < row_nnz_of_compressed_block.size(); i++)
    {
        unsigned long cur_row_nnz = row_nnz_of_compressed_block[i];

        if (max_row_nnz < cur_row_nnz)
        {
            max_row_nnz = cur_row_nnz;
        }

        if (min_row_nnz > cur_row_nnz)
        {
            min_row_nnz = cur_row_nnz;
        }
    }

    // 找出最优的子图、最优的参数策略、最优模板类型及其参数
    template_node_t best_temp_node;
    exe_compressed_sub_graph_t best_sub_graph;
    param_strategy_of_sub_graph_t best_sub_graph_param_strategy;

    best_time = 99999999999999;
    best_gflops = 0;

    // 定一个优化路径骨架
    exe_compressed_sub_graph_t sub_graph_skeleton;
    // 定一个参数设定的骨架
    param_strategy_of_sub_graph_t param_strategy_skeleton;

    // 定义骨架，一个是row padding
    // 定义骨架，首先是一个ROW_padding和BLB_row
    compressed_row_padding_direct_param_strategy_t row_padding_param_strategy;
    compressed_tblock_level_row_div_evenly_param_strategy_t BLB_row_div_evenly_param_strategy;
    
    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_ROW_PADDING, COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY, &row_padding_param_strategy);
    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_TBLOCK_LEVEL_ROW_DIV, COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY, &BLB_row_div_evenly_param_strategy);

    // 增加对于参数的调节
    param_enumerater_t param_setter;

    assert(sub_graph_skeleton.exe_node_vec.size() == 2);

    // 执行一个列分块
    // 对应参数的指针
    compressed_row_padding_direct_param_strategy_t* row_padding_direct_param_strategy = (compressed_row_padding_direct_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param_strategy;
    compressed_tblock_level_row_div_evenly_param_strategy_t* tblock_level_row_div_evenly_param_strategy = (compressed_tblock_level_row_div_evenly_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[1].param_strategy;
    assert(row_padding_direct_param_strategy != NULL && tblock_level_row_div_evenly_param_strategy != NULL);
    
    // 对于BLB row padding的长度，找一个非常小的值
    register_integer_independ_param_to_enumerater(&param_setter, &(row_padding_direct_param_strategy->multiply), 2, 4, 2);
    // padding的数量是当前子块的最小行号
    row_padding_direct_param_strategy->padding_row_length = min_row_nnz;

    // 添加一个warp级别的列分块
    compressed_warp_level_col_div_fixed_param_strategy_t warp_col_param_strategy;

    // 将策略添加到优化骨架中
    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_WARP_LEVEL_COL_DIV, COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY, &warp_col_param_strategy);

    // 将warp级别的指针取出
    compressed_warp_level_col_div_fixed_param_strategy_t* warp_level_col_div_fixed_param_strategy_ptr = (compressed_warp_level_col_div_fixed_param_strategy_t*) param_strategy_skeleton.param_strategy_vec[2].param_strategy;

    // 列分块最大128和最大行非零元/2的32倍数向上取整的最小值
    unsigned long fixed_max_col_size = 128;

    unsigned long max_col_size_acc_to_nnz = max_row_nnz / 2;

    if (max_col_size_acc_to_nnz % 32 != 0)
    {
        max_col_size_acc_to_nnz = (max_col_size_acc_to_nnz / 32 + 1) * 32;
    }

    if (max_col_size_acc_to_nnz < fixed_max_col_size)
    {
        fixed_max_col_size = max_col_size_acc_to_nnz;
    }

    if (fixed_max_col_size < 32)
    {
        fixed_max_col_size = 32;
    }

    assert(warp_level_col_div_fixed_param_strategy_ptr != NULL);
    register_integer_independ_param_to_enumerater(&param_setter, &(warp_level_col_div_fixed_param_strategy_ptr->col_block_nnz_num), 32, fixed_max_col_size, 32);

    assert(sub_graph_skeleton.exe_node_vec.size() == 3 && param_strategy_skeleton.param_strategy_vec.size() == 3);

    bool search_finished_by_strategy = false;

    // 遍历所有参数
    while (set_param_combination_to_next(&param_setter) == false)
    {
        // 矩阵已经不存在，那就执行稠密子图的优化路径，产生新的矩阵
        if (matrix == NULL)
        {
            // 通过稠密视图的块得出需要的子块
            matrix = val_copy_from_old_matrix_struct(input_matrix);
            assert(matrix != NULL);
            // 矩阵的一系列检查
            assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
            assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

            compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;
        }

        // 在定义一些参数
        tblock_level_row_div_evenly_param_strategy->block_row_num = row_padding_direct_param_strategy->multiply;

        // 打印一些参数
        cout << "row_padding_direct_param_strategy->multiply:" << row_padding_direct_param_strategy->multiply << ",row_padding_direct_param_strategy->padding_row_length:" << row_padding_direct_param_strategy->padding_row_length << "warp_level_col_div_fixed_param_strategy_ptr->col_block_nnz_num:" << warp_level_col_div_fixed_param_strategy_ptr->col_block_nnz_num << endl;

        unsigned long old_nnz = compressed_block_ptr->read_index[0]->length;
        unsigned long new_nnz = 0;

        // 在WLB的分块中不需要计较共享内存的溢出，不是很重要
        for (unsigned long i = 0; i < row_nnz_of_compressed_block.size(); i++)
        {
            // 每一行都会补齐为32的倍数
            unsigned long cur_row_nnz = row_nnz_of_compressed_block[i];
            unsigned long new_row_nnz = cur_row_nnz;

            if (new_row_nnz % 32 != 0)
            {
                new_row_nnz = (new_row_nnz / 32 + 1) * 32;
            }

            new_nnz = new_nnz + new_row_nnz;
        }

        // padding之后的行数量
        unsigned long row_num_after_padding = row_nnz_of_compressed_block.size();

        if (row_num_after_padding % tblock_level_row_div_evenly_param_strategy->block_row_num != 0)
        {
            row_num_after_padding = (row_num_after_padding / tblock_level_row_div_evenly_param_strategy->block_row_num + 1) * tblock_level_row_div_evenly_param_strategy->block_row_num;
        }

        // 对于剩下的部分，计算整行的padding量
        unsigned long remain_row_nnz = min_row_nnz;

        if (remain_row_nnz % 32 != 0)
        {
            remain_row_nnz = (remain_row_nnz / 32 + 1) * 32;
        }

        new_nnz = new_nnz + (row_num_after_padding - row_nnz_of_compressed_block.size()) * remain_row_nnz;

        // 查看padding量，大于一个阈值就放弃
        // 如果padding率过高就放弃
        // 这个时候
        assert(new_nnz >= old_nnz);

        cout << "new_nnz:" << new_nnz << ",old_nnz:" << old_nnz << ",padding rate:" << (float)new_nnz/(float)old_nnz << endl;

        if ((float)new_nnz/(float)old_nnz > get_config()["PADDING_RATE_UP_BOUND"].as_integer())
        {
            cout << "padding rate is larger than " << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
            
            // 析构matrix
            memory_garbage_manager_t mem_manager;

            delete_sparse_struct_t(&mem_manager, matrix);

            matrix = NULL;

            // 放弃这一轮调参
            continue;
        }

        if (data_set_collector != NULL)
        {
            // 如果存在一个数据集收集器，那么就需要收集节点的类型和参数
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() > 0);
            
            vector<exe_node_type> compressed_node_type_vec;
            vector<exe_node_param_set_strategy> compressed_param_strategy_vec;
            vector<float> compressed_param_vec;

            // 一共三个节点
            assert(sub_graph_skeleton.exe_node_vec.size() == 3);

            // 第一个节点
            // COMPRESSED_ROW_PADDING, COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY, compressed_row_padding_direct_param_strategy_t
            assert(sub_graph_skeleton.exe_node_vec[0].type == COMPRESSED_ROW_PADDING);
            assert(param_strategy_skeleton.param_strategy_vec[0].strategy_type == COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY);
            compressed_node_type_vec.push_back(COMPRESSED_ROW_PADDING);
            compressed_param_strategy_vec.push_back(COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY);
            compressed_row_padding_direct_param_strategy_t* strategy1_ptr = (compressed_row_padding_direct_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param_strategy;
            assert(strategy1_ptr != NULL);
            compressed_param_vec.push_back(strategy1_ptr->multiply);
            compressed_param_vec.push_back(strategy1_ptr->padding_row_length);

            // 第二个节点
            // COMPRESSED_TBLOCK_LEVEL_ROW_DIV, COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY, compressed_tblock_level_row_div_evenly_param_strategy_t
            assert(sub_graph_skeleton.exe_node_vec[1].type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
            assert(param_strategy_skeleton.param_strategy_vec[1].strategy_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY);
            compressed_node_type_vec.push_back(COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
            compressed_param_strategy_vec.push_back(COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY);
            compressed_tblock_level_row_div_evenly_param_strategy_t* strategy2_ptr = (compressed_tblock_level_row_div_evenly_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[1].param_strategy;
            assert(strategy2_ptr != NULL);
            compressed_param_vec.push_back(strategy2_ptr->block_row_num);

            // 第三个节点
            // COMPRESSED_WARP_LEVEL_COL_DIV, COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY, compressed_warp_level_col_div_fixed_param_strategy_t
            assert(sub_graph_skeleton.exe_node_vec[2].type == COMPRESSED_WARP_LEVEL_COL_DIV);
            assert(param_strategy_skeleton.param_strategy_vec[2].strategy_type == COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY);
            compressed_node_type_vec.push_back(COMPRESSED_WARP_LEVEL_COL_DIV);
            compressed_param_strategy_vec.push_back(COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY);
            compressed_warp_level_col_div_fixed_param_strategy_t* strategy3_ptr = (compressed_warp_level_col_div_fixed_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[2].param_strategy;
            assert(strategy3_ptr != NULL);
            compressed_param_vec.push_back(strategy3_ptr->col_block_nnz_num);

            // 清除当前compressed阶段的所有积累值
            data_set_collector->clear_compressed_accu_info();
            data_set_collector->insert_compressed_stage_node_and_param_to_cur_item(compressed_node_type_vec, compressed_param_strategy_vec, compressed_param_vec);
        }
        
        // 执行对应的的子块
        execute_sub_matrix_exe_graph_with_param_strategy(matrix, sub_matrix_id, &sub_graph_skeleton, &param_strategy_skeleton);

        // 候选模板是shared_mem + warp_reduce
        set<template_type> candi_template_type_set;
        candi_template_type_set.insert(SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE);

        // 性能
        float time;
        float gflops;

        // 寻找对应的最优模板，这里先不处理
        template_node_t temp_node = find_best_template_node_of_specific_sub_matrix_from_template_set(matrix, sub_matrix_id, candi_template_type_set, time, gflops, search_strategy_ptr, data_set_collector);

        if (gflops > best_gflops)
        {
            if (best_gflops == 0)
            {
                // 直接赋值，并且修改最佳优化路径
                // 这个时候还没有最优的优化路径
                best_gflops = gflops;
                best_time = time;

                // 从best拷贝出来
                best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
                best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
                best_temp_node = val_copy_from_old_template_node(temp_node);

                // 重新绑定优化骨架和策略骨架
                bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
            }
            else
            {
                // 直接赋值，析构已有的最优路径
                best_gflops = gflops;
                best_time = time;

                // 析构已有的最优参数
                del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                // 析构已有的最优策略
                del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                // 已有的最优模板
                del_param_of_template_node(&best_temp_node);
                
                // 执行新的拷贝
                // 从best拷贝出来
                best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
                best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
                best_temp_node = val_copy_from_old_template_node(temp_node);

                // 重新绑定优化骨架和策略骨架
                bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);   
            }
        }

        // 析构matrix
        // 现在matrix肯定存在
        assert(matrix != NULL);
        memory_garbage_manager_t mem_manager;
        delete_sparse_struct_t(&mem_manager, matrix);
        matrix = NULL;
        
        // 重置所有参数，并且重置所有参数指针，这一步用来处理一些参数是数组的节点，数组中的内容应该重新清空
        reset_exe_node_param_and_param_strategy_of_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton);
        
        // 如果所有的模板参数执行后都发生错误，那么这里的temp_node中可能是没有参数的
        // 如果所有的模板参数执行后都发生错误，那么这里的temp_node中可能是没有参数的
        if (temp_node.template_param != NULL)
        {
            del_param_of_template_node(&temp_node);
        }
        else
        {
            // 当前可能没有出现对应的
            assert(gflops == 0);
        }

        // 加入提前结束的相关内容
        if (search_strategy_ptr != NULL)
        {
            if (continue_search(search_strategy_ptr) == false)
            {
                search_finished_by_strategy = true;
            }
        }

        if (search_finished_by_strategy == true)
        {
            break;
        }
    }

    // 如果之前没有析构矩阵，现在就析构矩阵
    if (matrix != NULL)
    {
        assert(matrix != NULL);
        memory_garbage_manager_t mem_manager;
        delete_sparse_struct_t(&mem_manager, matrix);
        matrix = NULL;
    }

    // 析构用以遍历各种优化路径的两个骨架的参数
    del_strategy_of_param_strategy_node_in_sub_matrix(&param_strategy_skeleton);
    del_exe_node_param_of_compress_sub_matrix(&sub_graph_skeleton);

    compressed_sub_block_exe_graph_and_template_t return_sub_graph_exe_node_and_template;
    return_sub_graph_exe_node_and_template.sub_graph = best_sub_graph;
    return_sub_graph_exe_node_and_template.sub_graph_param_strategy = best_sub_graph_param_strategy;
    return_sub_graph_exe_node_and_template.temp_node = best_temp_node;

    return return_sub_graph_exe_node_and_template;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy7(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 首先执行对应的稠密子图优化
    sparse_struct_t* matrix = get_matrix_dense_view_graph(&dense_graph);
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_white_list_strategy7(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);
    matrix = NULL;

    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy7(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 首先执行对应的稠密子图优化
    sparse_struct_t* matrix = execute_dense_matrix_exe_graph_with_param_strategy(&(dense_graph.dense_sub_graph), &(dense_graph.dense_sub_graph_param_strategy));
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_white_list_strategy7(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);
    matrix = NULL;

    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy7(sparse_struct_t* input_matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(input_matrix != NULL);
    // 矩阵的一系列检查
    assert(input_matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(input_matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    sparse_struct_t* matrix = val_copy_from_old_matrix_struct(input_matrix);

    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 获取每一行的行非零元数量，并且获得最大和最小行非零元数量
    vector<unsigned long> row_nnz_of_compressed_block = get_nnz_of_each_row_in_compressed_sub_matrix(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr);
    
    assert(row_nnz_of_compressed_block.size() > 0);

    unsigned long max_row_nnz = row_nnz_of_compressed_block[0];
    unsigned long min_row_nnz = row_nnz_of_compressed_block[0];

    // 遍历获得最大的和最小的行非零元数量
    for (unsigned long i = 1; i < row_nnz_of_compressed_block.size(); i++)
    {
        unsigned long cur_row_nnz = row_nnz_of_compressed_block[i];

        if (max_row_nnz < cur_row_nnz)
        {
            max_row_nnz = cur_row_nnz;
        }

        if (min_row_nnz > cur_row_nnz)
        {
            min_row_nnz = cur_row_nnz;
        }
    }

    // 找出最优的子图、最优的参数策略、最优模板类型及其参数
    template_node_t best_temp_node;
    exe_compressed_sub_graph_t best_sub_graph;
    param_strategy_of_sub_graph_t best_sub_graph_param_strategy;

    best_time = 99999999999999;
    best_gflops = 0;

    // 定一个优化路径骨架
    exe_compressed_sub_graph_t sub_graph_skeleton;
    // 定一个参数设定的骨架
    param_strategy_of_sub_graph_t param_strategy_skeleton;

    // 按照非零元数量执行行分块
    compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t BLB_row_div_nnz_param_strategy;
    
    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_TBLOCK_LEVEL_ROW_DIV, COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY, &BLB_row_div_nnz_param_strategy);

    // 增加对于参数的调节
    param_enumerater_t param_setter;

    assert(sub_graph_skeleton.exe_node_vec.size() == 1);

    compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t* tblock_level_row_div_acc_to_least_nnz_param_strategy_ptr = (compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param_strategy;

    assert(tblock_level_row_div_acc_to_least_nnz_param_strategy_ptr != NULL);

    // 行分块最少的非零元数量
    register_integer_independ_param_to_enumerater(&param_setter, &(tblock_level_row_div_acc_to_least_nnz_param_strategy_ptr->nnz_low_bound), 128, 1024, 384);

    // 增加一个WLB列分块
    compressed_warp_level_col_div_fixed_param_strategy_t warp_col_param_strategy;

    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_WARP_LEVEL_COL_DIV, COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY, &warp_col_param_strategy);

    compressed_warp_level_col_div_fixed_param_strategy_t* warp_level_col_div_fixed_param_strategy_ptr = (compressed_warp_level_col_div_fixed_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[1].param_strategy;

    assert(warp_level_col_div_fixed_param_strategy_ptr != NULL);

    unsigned long warp_low_bound = 32;

    assert(sub_graph_skeleton.exe_node_vec.size() == 2);

    bool search_finished_by_strategy = false;

    while (set_param_combination_to_next(&param_setter) == false)
    {
        // 上界是BLB的非零元数量/2，步长是32，最大warp列分块是128
        unsigned long warp_up_bound_acc_to_row_block_nnz = tblock_level_row_div_acc_to_least_nnz_param_strategy_ptr->nnz_low_bound / 2;
        
        // 上界不超过最大行非零元数量 / 2，
        if (warp_up_bound_acc_to_row_block_nnz > max_row_nnz / 2)
        {
            warp_up_bound_acc_to_row_block_nnz = max_row_nnz / 2;
        }

        // 上界不能超过128
        if (warp_up_bound_acc_to_row_block_nnz > 128)
        {
            warp_up_bound_acc_to_row_block_nnz = 128;
        }

        // 上界向上取整到32的倍数
        if (warp_up_bound_acc_to_row_block_nnz % 32 != 0)
        {
            warp_up_bound_acc_to_row_block_nnz = (warp_up_bound_acc_to_row_block_nnz / 32 + 1) * 32;
        }

        assert(warp_up_bound_acc_to_row_block_nnz % 32 == 0 && warp_up_bound_acc_to_row_block_nnz <= 128);

        // 以32为步长尝试不同的
        for (unsigned long warp_col_block_nnz = 32; warp_col_block_nnz <= warp_up_bound_acc_to_row_block_nnz; warp_col_block_nnz = warp_col_block_nnz + 32)
        {
            warp_level_col_div_fixed_param_strategy_ptr->col_block_nnz_num = warp_col_block_nnz;

            if (matrix == NULL)
            {
                matrix = val_copy_from_old_matrix_struct(input_matrix);
                assert(matrix != NULL);
                // 矩阵的一系列检查
                assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
                assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

                compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;
            }

            // 打印参数，行分块非零元数量和TLB分块的列大小
            cout << "warp_level_col_div_fixed_param_strategy_ptr->col_block_nnz_num:" << warp_level_col_div_fixed_param_strategy_ptr->col_block_nnz_num << ",tblock_level_row_div_acc_to_least_nnz_param_strategy_ptr->nnz_low_bound:" << tblock_level_row_div_acc_to_least_nnz_param_strategy_ptr->nnz_low_bound << endl;

            // 查看padding率
            unsigned long old_nnz = compressed_block_ptr->read_index[0]->length;
            unsigned long new_nnz = 0;

            // 在WLB的分块中不需要计较共享内存的溢出，不是很重要
            for (unsigned long i = 0; i < row_nnz_of_compressed_block.size(); i++)
            {
                // 每一行都会补齐为32的倍数
                unsigned long cur_row_nnz = row_nnz_of_compressed_block[i];
                unsigned long new_row_nnz = cur_row_nnz;

                if (new_row_nnz % 32 != 0)
                {
                    new_row_nnz = (new_row_nnz / 32 + 1) * 32;
                }

                new_nnz = new_nnz + new_row_nnz;
            }

            assert(new_nnz >= old_nnz);

            cout << "new_nnz:" << new_nnz << ",old_nnz:" << old_nnz << ",padding rate:" << (float)new_nnz/(float)old_nnz << endl;

            if ((float)new_nnz/(float)old_nnz > get_config()["PADDING_RATE_UP_BOUND"].as_integer())
            {
                cout << "padding rate is larger than " << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                
                // 析构matrix
                memory_garbage_manager_t mem_manager;

                delete_sparse_struct_t(&mem_manager, matrix);

                matrix = NULL;

                // 放弃这一轮调参
                continue;
            }

            if (data_set_collector != NULL)
            {
                assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
                assert(data_set_collector->accu_dense_graph_node_type_vec.size() > 0);
                
                vector<exe_node_type> compressed_node_type_vec;
                vector<exe_node_param_set_strategy> compressed_param_strategy_vec;
                vector<float> compressed_param_vec;

                // 一共2个节点
                assert(sub_graph_skeleton.exe_node_vec.size() == 2);

                // 第一个节点
                // COMPRESSED_TBLOCK_LEVEL_ROW_DIV, COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY, compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t
                assert(sub_graph_skeleton.exe_node_vec[0].type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
                assert(param_strategy_skeleton.param_strategy_vec[0].strategy_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY);
                compressed_node_type_vec.push_back(COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
                compressed_param_strategy_vec.push_back(COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY);
                compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t* strategy1_ptr = (compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param_strategy;
                assert(strategy1_ptr != NULL);
                compressed_param_vec.push_back(strategy1_ptr->nnz_low_bound);

                // 第二个节点
                // COMPRESSED_WARP_LEVEL_COL_DIV, COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY, compressed_warp_level_col_div_fixed_param_strategy_t
                assert(sub_graph_skeleton.exe_node_vec[1].type == COMPRESSED_WARP_LEVEL_COL_DIV);
                assert(param_strategy_skeleton.param_strategy_vec[1].strategy_type == COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY);
                compressed_node_type_vec.push_back(COMPRESSED_WARP_LEVEL_COL_DIV);
                compressed_param_strategy_vec.push_back(COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY);
                compressed_warp_level_col_div_fixed_param_strategy_t* strategy2_ptr = (compressed_warp_level_col_div_fixed_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[1].param_strategy;
                assert(strategy2_ptr != NULL);
                compressed_param_vec.push_back(strategy2_ptr->col_block_nnz_num);

                // 清除当前compressed阶段的所有积累值
                data_set_collector->clear_compressed_accu_info();
                data_set_collector->insert_compressed_stage_node_and_param_to_cur_item(compressed_node_type_vec, compressed_param_strategy_vec, compressed_param_vec);
            }

            // 执行对应的的子块
            execute_sub_matrix_exe_graph_with_param_strategy(matrix, sub_matrix_id, &sub_graph_skeleton, &param_strategy_skeleton);

            // 候选模板是shared_mem + warp_reduce
            set<template_type> candi_template_type_set;
            candi_template_type_set.insert(SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE);

            // 性能
            float time;
            float gflops;

            // 寻找对应的最优模板，这里先不处理
            template_node_t temp_node = find_best_template_node_of_specific_sub_matrix_from_template_set(matrix, sub_matrix_id, candi_template_type_set, time, gflops, search_strategy_ptr, data_set_collector);

            if (gflops > best_gflops)
            {
                if (best_gflops == 0)
                {
                    // 直接赋值，并且修改最佳优化路径
                    // 这个时候还没有最优的优化路径
                    best_gflops = gflops;
                    best_time = time;

                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
                    best_temp_node = val_copy_from_old_template_node(temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
                else
                {
                    // 直接赋值，析构已有的最优路径
                    best_gflops = gflops;
                    best_time = time;

                    // 析构已有的最优参数
                    del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                    // 析构已有的最优策略
                    del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                    // 已有的最优模板
                    del_param_of_template_node(&best_temp_node);
                    
                    // 执行新的拷贝
                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
                    best_temp_node = val_copy_from_old_template_node(temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);   
                }
            }

            // 析构matrix
            // 现在matrix肯定存在
            assert(matrix != NULL);
            memory_garbage_manager_t mem_manager;
            delete_sparse_struct_t(&mem_manager, matrix);
            matrix = NULL;
            
            // 重置所有参数，并且重置所有参数指针，这一步用来处理一些参数是数组的节点，数组中的内容应该重新清空
            reset_exe_node_param_and_param_strategy_of_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton);
            
            // 如果所有的模板参数执行后都发生错误，那么这里的temp_node中可能是没有参数的
            if (temp_node.template_param != NULL)
            {
                del_param_of_template_node(&temp_node);
            }
            else
            {
                // 当前可能没有出现对应的
                assert(gflops == 0);
            }

            // 加入提前结束的相关内容
            if (search_strategy_ptr != NULL)
            {
                if (continue_search(search_strategy_ptr) == false)
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

    // 如果之前没有析构矩阵，现在就析构矩阵
    if (matrix != NULL)
    {
        assert(matrix != NULL);
        memory_garbage_manager_t mem_manager;
        delete_sparse_struct_t(&mem_manager, matrix);
        matrix = NULL;
    }

    // 析构用以遍历各种优化路径的两个骨架的参数
    del_strategy_of_param_strategy_node_in_sub_matrix(&param_strategy_skeleton);
    del_exe_node_param_of_compress_sub_matrix(&sub_graph_skeleton);

    compressed_sub_block_exe_graph_and_template_t return_sub_graph_exe_node_and_template;
    return_sub_graph_exe_node_and_template.sub_graph = best_sub_graph;
    return_sub_graph_exe_node_and_template.sub_graph_param_strategy = best_sub_graph_param_strategy;
    return_sub_graph_exe_node_and_template.temp_node = best_temp_node;

    return return_sub_graph_exe_node_and_template;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy8(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 首先执行对应的稠密子图优化
    sparse_struct_t* matrix = get_matrix_dense_view_graph(&dense_graph);
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_white_list_strategy8(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);
    matrix = NULL;

    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy8(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 首先执行对应的稠密子图优化
    sparse_struct_t* matrix = execute_dense_matrix_exe_graph_with_param_strategy(&(dense_graph.dense_sub_graph), &(dense_graph.dense_sub_graph_param_strategy));
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_white_list_strategy8(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);
    matrix = NULL;

    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy8(sparse_struct_t* input_matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(input_matrix != NULL);
    // 矩阵的一系列检查
    assert(input_matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(input_matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    sparse_struct_t* matrix = val_copy_from_old_matrix_struct(input_matrix);

    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 获取每一行的行非零元数量，并且获得最大和最小行非零元数量
    vector<unsigned long> row_nnz_of_compressed_block = get_nnz_of_each_row_in_compressed_sub_matrix(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr);

    assert(row_nnz_of_compressed_block.size() > 0);

    unsigned long max_row_nnz = row_nnz_of_compressed_block[0];
    unsigned long min_row_nnz = row_nnz_of_compressed_block[0];

    // 遍历获得最大的和最小的行非零元数量
    for (unsigned long i = 1; i < row_nnz_of_compressed_block.size(); i++)
    {
        unsigned long cur_row_nnz = row_nnz_of_compressed_block[i];

        if (max_row_nnz < cur_row_nnz)
        {
            max_row_nnz = cur_row_nnz;
        }

        if (min_row_nnz > cur_row_nnz)
        {
            min_row_nnz = cur_row_nnz;
        }
    }

    // 找出最优的子图、最优的参数策略、最优模板类型及其参数
    template_node_t best_temp_node;
    exe_compressed_sub_graph_t best_sub_graph;
    param_strategy_of_sub_graph_t best_sub_graph_param_strategy;

    best_time = 99999999999999;
    best_gflops = 0;

    // 定一个优化路径骨架
    exe_compressed_sub_graph_t sub_graph_skeleton;
    // 定一个参数设定的骨架
    param_strategy_of_sub_graph_t param_strategy_skeleton;

    compressed_tblock_level_col_div_fixed_param_strategy_t BLB_col_div_param_strategy;
    
    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_TBLOCK_LEVEL_COL_DIV, COMPRESSED_TBLOCK_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY, &BLB_col_div_param_strategy);

    compressed_tblock_level_col_div_fixed_param_strategy_t* tblock_level_col_div_fixed_param_strategy_ptr = (compressed_tblock_level_col_div_fixed_param_strategy_t *)param_strategy_skeleton.param_strategy_vec[0].param_strategy;

    // 增加对于参数的调节
    param_enumerater_t param_setter;

    assert(sub_graph_skeleton.exe_node_vec.size() == 1);

    // BLB列分块和最大以及最小的行非零元数量有关
    unsigned long min_BLB_col_size = 128;

    if (min_BLB_col_size % 32 != 0)
    {
        min_BLB_col_size = (min_BLB_col_size / 32 + 1) * 32;
    }

    unsigned long max_BLB_col_size = max_row_nnz;

    if (max_BLB_col_size % 32 != 0)
    {
        max_BLB_col_size = (max_BLB_col_size / 32 + 1) * 32;
    }

    // 最大纵分块的长度为4096
    if (max_BLB_col_size > 4096)
    {
        max_BLB_col_size = 4096;
    }

    // 如果纵切分小于下界，那就让其等于下界
    if (max_BLB_col_size < min_BLB_col_size)
    {
        max_BLB_col_size = min_BLB_col_size;
    }

    bool search_finished_by_strategy = false;

    // 整个参数不能使用调参器，按照*2的倍率不断增加
    for (unsigned long BLB_col_size = min_BLB_col_size; BLB_col_size <= max_BLB_col_size; BLB_col_size = BLB_col_size * 2)
    {
        tblock_level_col_div_fixed_param_strategy_ptr->col_block_nnz_num = BLB_col_size;

        if (matrix == NULL)
        {
            matrix = val_copy_from_old_matrix_struct(input_matrix);
            assert(matrix != NULL);
            // 矩阵的一系列检查
            assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
            assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

            compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;
        }

        cout << "tblock_level_col_div_fixed_param_strategy_ptr->col_block_nnz_num:" << tblock_level_col_div_fixed_param_strategy_ptr->col_block_nnz_num << endl;

        // 查看padding率
        unsigned long old_nnz = compressed_block_ptr->read_index[0]->length;
        unsigned long new_nnz = 0;

        // 在WLB的分块中不需要计较共享内存的溢出，不是很重要
        for (unsigned long i = 0; i < row_nnz_of_compressed_block.size(); i++)
        {
            // 每一行都会补齐为32的倍数
            unsigned long cur_row_nnz = row_nnz_of_compressed_block[i];
            unsigned long new_row_nnz = cur_row_nnz;

            if (new_row_nnz % 32 != 0)
            {
                new_row_nnz = (new_row_nnz / 32 + 1) * 32;
            }

            new_nnz = new_nnz + new_row_nnz;
        }

        assert(new_nnz >= old_nnz);

        cout << "new_nnz:" << new_nnz << ",old_nnz:" << old_nnz << ",padding rate:" << (float)new_nnz/(float)old_nnz << endl;

        if ((float)new_nnz/(float)old_nnz > get_config()["PADDING_RATE_UP_BOUND"].as_integer())
        {
            cout << "padding rate is larger than " << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
            
            // 析构matrix
            memory_garbage_manager_t mem_manager;

            delete_sparse_struct_t(&mem_manager, matrix);

            matrix = NULL;

            // 放弃这一轮调参
            continue;
        }

        if (data_set_collector != NULL)
        {
            // 按照当前的逻辑，稠密视图已经加入了对应的数据
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() > 0);

            vector<exe_node_type> compressed_node_type_vec;
            vector<exe_node_param_set_strategy> compressed_param_strategy_vec;
            vector<float> compressed_param_vec;

            // 只有一个节点
            assert(sub_graph_skeleton.exe_node_vec.size() == 1);

            // 第一个节点
            // COMPRESSED_TBLOCK_LEVEL_COL_DIV, COMPRESSED_TBLOCK_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY, compressed_tblock_level_col_div_fixed_param_strategy_t
            assert(sub_graph_skeleton.exe_node_vec[0].type == COMPRESSED_TBLOCK_LEVEL_COL_DIV);
            assert(param_strategy_skeleton.param_strategy_vec[0].strategy_type == COMPRESSED_TBLOCK_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY);
            compressed_node_type_vec.push_back(COMPRESSED_TBLOCK_LEVEL_COL_DIV);
            compressed_param_strategy_vec.push_back(COMPRESSED_TBLOCK_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY);
            compressed_tblock_level_col_div_fixed_param_strategy_t* strategy1_ptr = (compressed_tblock_level_col_div_fixed_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param_strategy;
            assert(strategy1_ptr != NULL);
            compressed_param_vec.push_back(strategy1_ptr->col_block_nnz_num);

            // 清除当前compressed阶段的所有积累值
            data_set_collector->clear_compressed_accu_info();
            data_set_collector->insert_compressed_stage_node_and_param_to_cur_item(compressed_node_type_vec, compressed_param_strategy_vec, compressed_param_vec);
        }

        // 执行对应的子块
        execute_sub_matrix_exe_graph_with_param_strategy(matrix, sub_matrix_id, &sub_graph_skeleton, &param_strategy_skeleton);

        set<template_type> candi_template_type_set;
        candi_template_type_set.insert(SHARED_MEMORY_LONG_ROW_TEMPLATE);

        // 性能
        float time;
        float gflops;

        template_node_t temp_node = find_best_template_node_of_specific_sub_matrix_from_template_set(matrix, sub_matrix_id, candi_template_type_set, time, gflops, search_strategy_ptr, data_set_collector);

        if (gflops > best_gflops)
        {
            if (best_gflops == 0)
            {
                // 直接赋值，并且修改最佳优化路径
                // 这个时候还没有最优的优化路径
                best_gflops = gflops;
                best_time = time;

                // 从best拷贝出来
                best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
                best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
                best_temp_node = val_copy_from_old_template_node(temp_node);

                // 重新绑定优化骨架和策略骨架
                bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
            }
            else
            {
                // 直接赋值，析构已有的最优路径
                best_gflops = gflops;
                best_time = time;

                // 析构已有的最优参数
                del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                // 析构已有的最优策略
                del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                // 已有的最优模板
                del_param_of_template_node(&best_temp_node);
                
                // 执行新的拷贝
                // 从best拷贝出来
                best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
                best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
                best_temp_node = val_copy_from_old_template_node(temp_node);

                // 重新绑定优化骨架和策略骨架
                bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);   
            }
        }

        // 析构matrix
        // 现在matrix肯定存在
        assert(matrix != NULL);
        memory_garbage_manager_t mem_manager;
        delete_sparse_struct_t(&mem_manager, matrix);
        matrix = NULL;
        
        // 重置所有参数，并且重置所有参数指针，这一步用来处理一些参数是数组的节点，数组中的内容应该重新清空
        reset_exe_node_param_and_param_strategy_of_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton);
        
        // 如果所有的模板参数执行后都发生错误，那么这里的temp_node中可能是没有参数的
        if (temp_node.template_param != NULL)
        {
            del_param_of_template_node(&temp_node);
        }
        else
        {
            // 当前可能没有出现对应的
            assert(gflops == 0);
        }

        // 加入提前结束的相关内容
        if (search_strategy_ptr != NULL)
        {
            if (continue_search(search_strategy_ptr) == false)
            {
                search_finished_by_strategy = true;
            }
        }

        if (search_finished_by_strategy == true)
        {
            break;
        }
    }
    
    // 如果之前没有析构矩阵，现在就析构矩阵
    if (matrix != NULL)
    {
        memory_garbage_manager_t mem_manager;
        delete_sparse_struct_t(&mem_manager, matrix);
        matrix = NULL;
    }

    // 析构用以遍历各种优化路径的两个骨架的参数
    del_strategy_of_param_strategy_node_in_sub_matrix(&param_strategy_skeleton);
    del_exe_node_param_of_compress_sub_matrix(&sub_graph_skeleton);

    compressed_sub_block_exe_graph_and_template_t return_sub_graph_exe_node_and_template;
    return_sub_graph_exe_node_and_template.sub_graph = best_sub_graph;
    return_sub_graph_exe_node_and_template.sub_graph_param_strategy = best_sub_graph_param_strategy;
    return_sub_graph_exe_node_and_template.temp_node = best_temp_node;

    return return_sub_graph_exe_node_and_template;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy9(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 首先执行对应的稠密子图优化
    sparse_struct_t* matrix = get_matrix_dense_view_graph(&dense_graph);
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_white_list_strategy9(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);
    matrix = NULL;

    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy9(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 首先执行对应的稠密子图优化
    sparse_struct_t* matrix = execute_dense_matrix_exe_graph_with_param_strategy(&(dense_graph.dense_sub_graph), &(dense_graph.dense_sub_graph_param_strategy));
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_white_list_strategy9(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);
    matrix = NULL;

    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_white_list_strategy9(sparse_struct_t* input_matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // cout << "find_best_path_of_white_list_strategy9: begin" << endl;
    assert(input_matrix != NULL);
    // 矩阵的一系列检查
    assert(input_matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(input_matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(input_matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    sparse_struct_t* matrix = val_copy_from_old_matrix_struct(input_matrix);

    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 获取每一行的行非零元数量，并且获得最大和最小行非零元数量
    vector<unsigned long> row_nnz_of_compressed_block = get_nnz_of_each_row_in_compressed_sub_matrix(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr);

    assert(row_nnz_of_compressed_block.size() > 0);

    unsigned long TLB_nnz_low_bound = 2;

    // 算计行非零元平均值，从2到min(32,行非零元平均值)之间找几个梯度
    // 计算行非零元平均值
    unsigned long avg_nnz_row = 0;

    for (unsigned long i = 0; i < row_nnz_of_compressed_block.size(); i++)
    {
        avg_nnz_row = row_nnz_of_compressed_block[i] + avg_nnz_row;
    }

    avg_nnz_row = avg_nnz_row / row_nnz_of_compressed_block.size();

    if (avg_nnz_row < TLB_nnz_low_bound)
    {
        avg_nnz_row = TLB_nnz_low_bound;
    }

    // 如果太大了，上界就缩小一点
    if (avg_nnz_row > 32)
    {
        avg_nnz_row = 32;
    }

    // 计算步长争取走三步，所以只在中间走一步
    unsigned long step_size = (avg_nnz_row - TLB_nnz_low_bound) / 4;

    if (step_size < 1)
    {
        step_size = 1;
    }

    // 增加分块操作
    // 找出最优的子图、最优的参数策略、最优模板类型及其参数
    template_node_t best_temp_node;
    exe_compressed_sub_graph_t best_sub_graph;
    param_strategy_of_sub_graph_t best_sub_graph_param_strategy;

    best_time = 99999999999999;
    best_gflops = 0;

    // 定一个优化路径骨架
    exe_compressed_sub_graph_t sub_graph_skeleton;
    // 定一个参数设定的骨架
    param_strategy_of_sub_graph_t param_strategy_skeleton;

    compressed_thread_level_nnz_div_direct_param_strategy_t TLB_nnz_div_param_strategy;

    add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_THREAD_LEVEL_NNZ_DIV, COMPRESSED_THREAD_LEVEL_NNZ_DIV_DIRECT_PARAM_STRATEGY, &TLB_nnz_div_param_strategy);

    compressed_thread_level_nnz_div_direct_param_strategy_t* TLB_nnz_div_param_strategy_ptr = (compressed_thread_level_nnz_div_direct_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param_strategy;

    // 增加对于参数的调节
    param_enumerater_t param_setter;

    assert(sub_graph_skeleton.exe_node_vec.size() == 1 && param_strategy_skeleton.param_strategy_vec.size() == 1);

    register_integer_independ_param_to_enumerater(&param_setter, &(TLB_nnz_div_param_strategy_ptr->block_nnz_num), TLB_nnz_low_bound, avg_nnz_row, step_size);

    bool search_finished_by_strategy = false;

    while (set_param_combination_to_next(&param_setter) == false)
    {
        if (matrix == NULL)
        {
            // 通过稠密视图的块得出需要的子块
            matrix = val_copy_from_old_matrix_struct(input_matrix);
            assert(matrix != NULL);
            // 矩阵的一系列检查
            assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
            assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

            compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;
        }

        // 打印参数
        cout << "find_best_path_of_white_list_strategy9: TLB_nnz_div_param_strategy_ptr->block_nnz_num:" << TLB_nnz_div_param_strategy_ptr->block_nnz_num << endl;

        if (data_set_collector != NULL)
        {
            // cout << "find_best_path_of_white_list_strategy9: need to collect ml data" << endl;
             // 按照当前的逻辑，稠密视图已经加入了对应的数据
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() == data_set_collector->accu_dense_param_strategy_type_vec.size());
            assert(data_set_collector->accu_dense_graph_node_type_vec.size() > 0);

            vector<exe_node_type> compressed_node_type_vec;
            vector<exe_node_param_set_strategy> compressed_param_strategy_vec;
            vector<float> compressed_param_vec;

            // 检查一下， 只有一个节点
            assert(sub_graph_skeleton.exe_node_vec.size() == 1);

            // 第一个节点
            // COMPRESSED_THREAD_LEVEL_NNZ_DIV, COMPRESSED_THREAD_LEVEL_NNZ_DIV_DIRECT_PARAM_STRATEGY, compressed_thread_level_nnz_div_direct_param_strategy_t
            assert(sub_graph_skeleton.exe_node_vec[0].type == COMPRESSED_THREAD_LEVEL_NNZ_DIV);
            assert(param_strategy_skeleton.param_strategy_vec[0].strategy_type == COMPRESSED_THREAD_LEVEL_NNZ_DIV_DIRECT_PARAM_STRATEGY);
            compressed_node_type_vec.push_back(COMPRESSED_THREAD_LEVEL_NNZ_DIV);
            compressed_param_strategy_vec.push_back(COMPRESSED_THREAD_LEVEL_NNZ_DIV_DIRECT_PARAM_STRATEGY);
            compressed_thread_level_nnz_div_direct_param_strategy_t* strategy1_ptr = (compressed_thread_level_nnz_div_direct_param_strategy_t*)param_strategy_skeleton.param_strategy_vec[0].param_strategy;
            assert(strategy1_ptr != NULL);
            compressed_param_vec.push_back(strategy1_ptr->block_nnz_num);

            // 清除当前compressed阶段的所有积累值
            data_set_collector->clear_compressed_accu_info();
            data_set_collector->insert_compressed_stage_node_and_param_to_cur_item(compressed_node_type_vec, compressed_param_strategy_vec, compressed_param_vec);
        }

        // 执行对应的子块
        // 执行对应的的子块
        execute_sub_matrix_exe_graph_with_param_strategy(matrix, sub_matrix_id, &sub_graph_skeleton, &param_strategy_skeleton);

        // 对应子块的所有索引的is_sort_arr是NULL
        for (unsigned long i = 0; i < matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size(); i++)
        {
            assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[i]->is_sort_arr == NULL);
        }

        // 候选模板是shared_mem + warp_reduce
        set<template_type> candi_template_type_set;
        candi_template_type_set.insert(UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE);
        candi_template_type_set.insert(UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE);

        // 性能
        float time;
        float gflops;

        // 寻找对应的最优模板，这里先不处理
        template_node_t temp_node = find_best_template_node_of_specific_sub_matrix_from_template_set(matrix, sub_matrix_id, candi_template_type_set, time, gflops, search_strategy_ptr, data_set_collector);

        for (unsigned long i = 0; i < matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size(); i++)
        {
            assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[i]->is_sort_arr == NULL);
        }

        if (gflops > best_gflops)
        {
            if (best_gflops == 0)
            {
                // 直接赋值，并且修改最佳优化路径
                // 这个时候还没有最优的优化路径
                best_gflops = gflops;
                best_time = time;

                // 从best拷贝出来
                best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
                best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
                best_temp_node = val_copy_from_old_template_node(temp_node);

                // 重新绑定优化骨架和策略骨架
                bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
            }
            else
            {
                // 直接赋值，析构已有的最优路径
                best_gflops = gflops;
                best_time = time;

                // 析构已有的最优参数
                del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                // 析构已有的最优策略
                del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                // 已有的最优模板
                del_param_of_template_node(&best_temp_node);
                
                // 执行新的拷贝
                // 从best拷贝出来
                best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_graph_skeleton);
                best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(param_strategy_skeleton);
                best_temp_node = val_copy_from_old_template_node(temp_node);

                // 重新绑定优化骨架和策略骨架
                bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);   
            }
        }

        for (unsigned long i = 0; i < matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size(); i++)
        {
            assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index[i]->is_sort_arr == NULL);
        }

        // 析构matrix
        // 现在matrix肯定存在
        assert(matrix != NULL);
        memory_garbage_manager_t mem_manager;
        // cout << "find_best_path_of_white_list_strategy9: begin del matrix 1" << endl;
        delete_sparse_struct_t(&mem_manager, matrix);
        // cout << "find_best_path_of_white_list_strategy9: finish del matrix 1" << endl;
        matrix = NULL;
        
        // 重置所有参数，并且重置所有参数指针，这一步用来处理一些参数是数组的节点，数组中的内容应该重新清空
        reset_exe_node_param_and_param_strategy_of_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton);
        
        // 如果所有的模板参数执行后都发生错误，那么这里的temp_node中可能是没有参数的
        if (temp_node.template_param != NULL)
        {
            del_param_of_template_node(&temp_node);
        }
        else
        {
            // 当前可能没有出现对应的
            assert(gflops == 0);
        }

        // 加入提前结束的相关内容
        if (search_strategy_ptr != NULL)
        {
            if (continue_search(search_strategy_ptr) == false)
            {
                search_finished_by_strategy = true;
            }
        }

        if (search_finished_by_strategy == true)
        {
            break;
        }
    }
    
    // 如果之前没有析构矩阵，现在就析构矩阵
    if (matrix != NULL)
    {
        assert(matrix != NULL);
        memory_garbage_manager_t mem_manager;
        // cout << "find_best_path_of_white_list_strategy9: begin del matrix 2" << endl;
        delete_sparse_struct_t(&mem_manager, matrix);
        // cout << "find_best_path_of_white_list_strategy9: end del matrix 2" << endl;
        matrix = NULL;
    }

    // 析构用以遍历各种优化路径的两个骨架的参数
    del_strategy_of_param_strategy_node_in_sub_matrix(&param_strategy_skeleton);
    del_exe_node_param_of_compress_sub_matrix(&sub_graph_skeleton);

    compressed_sub_block_exe_graph_and_template_t return_sub_graph_exe_node_and_template;
    return_sub_graph_exe_node_and_template.sub_graph = best_sub_graph;
    return_sub_graph_exe_node_and_template.sub_graph_param_strategy = best_sub_graph_param_strategy;
    return_sub_graph_exe_node_and_template.temp_node = best_temp_node;

    // cout << "find_best_path_of_white_list_strategy9: end" << endl;

    return return_sub_graph_exe_node_and_template;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_compressed_sub_matrix(exe_dense_sub_graph_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(dense_graph.exe_node_vec.size() > 0 && dense_graph.preorder_node_set.size() > 0);

    // 压缩视图的所有分块操作
    // 首先执行对应的稠密子图优化
    sparse_struct_t* matrix = get_matrix_dense_view_graph(&dense_graph);
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_compressed_sub_matrix(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    // 析构矩阵
    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);

    // 返回对应的路径
    return best_sub_graph_path;
}

compressed_sub_block_exe_graph_and_template_t find_best_path_of_compressed_sub_matrix(dense_view_matrix_exe_graph_and_param_strategy_t dense_graph, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    assert(dense_graph.dense_sub_graph.exe_node_vec.size() > 0 && dense_graph.dense_sub_graph_param_strategy.param_strategy_vec.size() > 0);
    assert(dense_graph.dense_sub_graph.exe_node_vec.size() == dense_graph.dense_sub_graph_param_strategy.param_strategy_vec.size());

    // 压缩视图的所有分块操作
    // 首先执行对应的稠密子图优化
    sparse_struct_t* matrix = execute_dense_matrix_exe_graph_with_param_strategy(&(dense_graph.dense_sub_graph), &(dense_graph.dense_sub_graph_param_strategy));
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    compressed_sub_block_exe_graph_and_template_t best_sub_graph_path = find_best_path_of_compressed_sub_matrix(matrix, sub_matrix_id, best_time, best_gflops, search_strategy_ptr, data_set_collector);

    // 析构矩阵
    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);

    // 返回对应的路径
    return best_sub_graph_path;
}


// 为某一个压缩子图找出最优的路径
compressed_sub_block_exe_graph_and_template_t find_best_path_of_compressed_sub_matrix(sparse_struct_t* matrix, unsigned long sub_matrix_id, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    cout << "find_best_path_of_compressed_sub_matrix: begin" << endl;
    // 最外层的总策略，不用执行矩阵内容的值拷贝，因为值拷贝会在具体的内容中执行
    assert(matrix != NULL);
    // 矩阵的一系列检查
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 获得压缩子图的一些基本信息
    matrix_info_t matrix_information = get_sub_matrix_info_from_compressed_matrix_block(matrix->block_coor_table.item_arr[sub_matrix_id]);

    bool is_global_sorted = false;
    // 查看当前子图是不是已经排序过了
    if (matrix->sorted_row_index != NULL)
    {
        assert(matrix->is_sorted == true);
        is_global_sorted = true;
    }
    
    // 矩阵在外面析构
    // memory_garbage_manager_t mem_manager;

    // delete_sparse_struct_t(&mem_manager, matrix);
    // matrix = 0;

    // 最优的时间和gflops
    best_time = 9999999999999999;
    best_gflops = 0;

    // 最优的骨架
    template_node_t best_temp_node;
    exe_compressed_sub_graph_t best_sub_graph;
    param_strategy_of_sub_graph_t best_sub_graph_param_strategy;

    // 用一个变量查看是不是已经提前停止了
    bool search_finished_by_strategy = false;

    // 当非零元数量大于128时，采用超长行的方式处理
    if (matrix_information.min_row_nnz >= 128)
    {
        float time;
        float gflops;

        if (search_finished_by_strategy == false)
        {
            compressed_sub_block_exe_graph_and_template_t sub_matrix_opt_path_of_strategy8 = find_best_path_of_white_list_strategy8(matrix, sub_matrix_id, time, gflops, search_strategy_ptr, data_set_collector);

            // 如果当前性能比最优性能还好，那就要记录下来
            if (gflops > best_gflops)
            {
                if (best_gflops == 0)
                {
                    // 直接赋值，并且修改最佳优化路径
                    // 这个时候还没有最优的优化路径
                    best_gflops = gflops;
                    best_time = time;

                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy8.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy8.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy8.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
                else
                {
                    // 直接赋值，析构已有的最优路径
                    best_gflops = gflops;
                    best_time = time;

                    // 析构已有的最优参数
                    del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                    // 析构已有的最优策略
                    del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                    // 已有的最优模板
                    del_param_of_template_node(&best_temp_node);

                    // 执行新的拷贝
                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy8.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy8.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy8.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
            }

            // 然后析构找出的路径，最优的路径已经被取出来了，所以先析构掉
            if (sub_matrix_opt_path_of_strategy8.temp_node.template_param != NULL)
            {
                del_param_of_template_node(&(sub_matrix_opt_path_of_strategy8.temp_node));    
            }
            else
            {
                assert(gflops == 0);
            }
            
            del_strategy_of_param_strategy_node_in_sub_matrix(&(sub_matrix_opt_path_of_strategy8.sub_graph_param_strategy));
            del_exe_node_param_of_compress_sub_matrix(&(sub_matrix_opt_path_of_strategy8.sub_graph));

            if (search_strategy_ptr != NULL)
            {
                search_finished_by_strategy = !continue_search(search_strategy_ptr);
            }
        }
    }

    // 为每一个路径分别增加一个条件
    // 当行非零元小于512时，可以采用行归约+原子性加的方式
    if (matrix_information.max_row_nnz <= 512)
    {
        assert(matrix_information.min_row_nnz > 0);

        float time;
        float gflops;

        // 查看是不是被提前停止
        if (search_finished_by_strategy == false)
        {
            // 首先从策略1中查找
            // row_padding => evenly_BLB_row => one_TLB_row => direct_atom_template_warp_compress，会执行值拷贝
            compressed_sub_block_exe_graph_and_template_t sub_matrix_opt_path_of_strategy1 = find_best_path_of_white_list_strategy1(matrix, sub_matrix_id, time, gflops, search_strategy_ptr, data_set_collector);

            // 如果当前性能比最优性能还好，那就要记录下来
            if (gflops > best_gflops)
            {
                if (best_gflops == 0)
                {
                    // 直接赋值，并且修改最佳优化路径
                    // 这个时候还没有最优的优化路径
                    best_gflops = gflops;
                    best_time = time;

                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy1.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy1.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy1.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
                else
                {
                    // 直接赋值，析构已有的最优路径
                    best_gflops = gflops;
                    best_time = time;

                    // 析构已有的最优参数
                    del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                    // 析构已有的最优策略
                    del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                    // 已有的最优模板
                    del_param_of_template_node(&best_temp_node);
                    
                    // 执行新的拷贝
                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy1.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy1.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy1.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);   
                }
            }

            // 然后析构找出的路径，最优的路径已经被取出来了，所以先析构掉
            if (sub_matrix_opt_path_of_strategy1.temp_node.template_param != NULL)
            {
                del_param_of_template_node(&(sub_matrix_opt_path_of_strategy1.temp_node));    
            }
            else
            {
                assert(gflops == 0);
            }
            
            del_strategy_of_param_strategy_node_in_sub_matrix(&(sub_matrix_opt_path_of_strategy1.sub_graph_param_strategy));
            del_exe_node_param_of_compress_sub_matrix(&(sub_matrix_opt_path_of_strategy1.sub_graph));

            // 查看是不是已经提前停止的
            if (search_strategy_ptr != NULL)
            {
                search_finished_by_strategy = !continue_search(search_strategy_ptr);
            }
        }

        // 查看是不是要提前结束
        if (search_finished_by_strategy == false)
        {
            // row_padding32 => TLB_row => direct_atom_template_warp_block_compress
            compressed_sub_block_exe_graph_and_template_t sub_matrix_opt_path_of_strategy5 = find_best_path_of_white_list_strategy5(matrix, sub_matrix_id, time, gflops, search_strategy_ptr, data_set_collector);

            // 如果当前性能比最优性能还好，那就要记录下来
            if (gflops > best_gflops)
            {
                if (best_gflops == 0)
                {
                    // 直接赋值，并且修改最佳优化路径
                    // 这个时候还没有最优的优化路径
                    best_gflops = gflops;
                    best_time = time;

                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy5.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy5.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy5.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
                else
                {
                    // 直接赋值，析构已有的最优路径
                    best_gflops = gflops;
                    best_time = time;

                    // 析构已有的最优参数
                    del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                    // 析构已有的最优策略
                    del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                    // 已有的最优模板
                    del_param_of_template_node(&best_temp_node);

                    // 执行新的拷贝
                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy5.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy5.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy5.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
            }

            // 然后析构找出的路径，最优的路径已经被取出来了，所以先析构掉
            if (sub_matrix_opt_path_of_strategy5.temp_node.template_param != NULL)
            {
                del_param_of_template_node(&(sub_matrix_opt_path_of_strategy5.temp_node));    
            }
            else
            {
                assert(gflops == 0);
            }
            
            del_strategy_of_param_strategy_node_in_sub_matrix(&(sub_matrix_opt_path_of_strategy5.sub_graph_param_strategy));
            del_exe_node_param_of_compress_sub_matrix(&(sub_matrix_opt_path_of_strategy5.sub_graph));

            // 查看是不是要继续
            if (search_strategy_ptr != NULL)
            {
                search_finished_by_strategy = !continue_search(search_strategy_ptr);
            }
        }
    }

    // 对于没有排序过的情况尝试CSR5
    if (is_global_sorted == false)
    {
        float time;
        float gflops;

        if (search_finished_by_strategy == false)
        {
            // TLB_nnz => CSR5_like

            compressed_sub_block_exe_graph_and_template_t sub_matrix_opt_path_of_strategy9 = find_best_path_of_white_list_strategy9(matrix, sub_matrix_id, time, gflops, search_strategy_ptr, data_set_collector);

            // 如果当前性能比最优性能还好，那就要记录下来
            if (gflops > best_gflops)
            {
                if (best_gflops == 0)
                {
                    // 直接赋值，并且修改最佳优化路径
                    // 这个时候还没有最优的优化路径
                    best_gflops = gflops;
                    best_time = time;

                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy9.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy9.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy9.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
                else
                {
                    // 直接赋值，析构已有的最优路径
                    best_gflops = gflops;
                    best_time = time;

                    // 析构已有的最优参数
                    del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                    // 析构已有的最优策略
                    del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                    // 已有的最优模板
                    del_param_of_template_node(&best_temp_node);

                    // 执行新的拷贝
                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy9.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy9.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy9.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
            }

            // 然后析构找出的路径，最优的路径已经被取出来了，所以先析构掉
            if (sub_matrix_opt_path_of_strategy9.temp_node.template_param != NULL)
            {
                del_param_of_template_node(&(sub_matrix_opt_path_of_strategy9.temp_node));    
            }
            else
            {
                assert(gflops == 0);
            }
            
            del_strategy_of_param_strategy_node_in_sub_matrix(&(sub_matrix_opt_path_of_strategy9.sub_graph_param_strategy));
            del_exe_node_param_of_compress_sub_matrix(&(sub_matrix_opt_path_of_strategy9.sub_graph));

            if (search_strategy_ptr != NULL)
            {
                search_finished_by_strategy = !continue_search(search_strategy_ptr);
            }
        }
    }

    // 当行非零元的最小值大于30的时候，行非零元最大值小于512的时候，采用warp的原子加归约
    if (matrix_information.min_row_nnz >= 30 && matrix_information.max_row_nnz <= 512)
    {
        float time;
        float gflops;

        if (search_finished_by_strategy == false)
        {
            // row_padding => evenly_BLB_row => one_TLB_row => direct_atom_template_warp_compress
            compressed_sub_block_exe_graph_and_template_t sub_matrix_opt_path_of_strategy4 = find_best_path_of_white_list_strategy4(matrix, sub_matrix_id, time, gflops, search_strategy_ptr, data_set_collector);

            // 如果当前性能比最优性能还好，那就要记录下来
            if (gflops > best_gflops)
            {
                if (best_gflops == 0)
                {
                    // 直接赋值，并且修改最佳优化路径
                    // 这个时候还没有最优的优化路径
                    best_gflops = gflops;
                    best_time = time;

                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy4.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy4.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy4.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
                else
                {
                    // 直接赋值，析构已有的最优路径
                    best_gflops = gflops;
                    best_time = time;

                    // 析构已有的最优参数
                    del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                    // 析构已有的最优策略
                    del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                    // 已有的最优模板
                    del_param_of_template_node(&best_temp_node);

                    // 执行新的拷贝
                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy4.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy4.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy4.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
            }

            // 然后析构找出的路径，最优的路径已经被取出来了，所以先析构掉
            if (sub_matrix_opt_path_of_strategy4.temp_node.template_param != NULL)
            {
                del_param_of_template_node(&(sub_matrix_opt_path_of_strategy4.temp_node));    
            }
            else
            {
                assert(gflops == 0);
            }
            
            del_strategy_of_param_strategy_node_in_sub_matrix(&(sub_matrix_opt_path_of_strategy4.sub_graph_param_strategy));
            del_exe_node_param_of_compress_sub_matrix(&(sub_matrix_opt_path_of_strategy4.sub_graph));

            if (search_strategy_ptr != NULL)
            {
                search_finished_by_strategy = !continue_search(search_strategy_ptr);
            }
        }
    }

    // 如果行非零元数量在32和1024之间，可以采用带共享内存的warp reduce
    if (32 <= matrix_information.min_row_nnz && matrix_information.max_row_nnz <= 1024)
    {
        float time;
        float gflops;

        if (search_finished_by_strategy == false)
        {
            // 等宽行条带
            compressed_sub_block_exe_graph_and_template_t sub_matrix_opt_path_of_strategy6 = find_best_path_of_white_list_strategy6(matrix, sub_matrix_id, time, gflops, search_strategy_ptr, data_set_collector);

            // 如果当前性能比最优性能还好，那就要记录下来
            if (gflops > best_gflops)
            {
                if (best_gflops == 0)
                {
                    // 直接赋值，并且修改最佳优化路径
                    // 这个时候还没有最优的优化路径
                    best_gflops = gflops;
                    best_time = time;

                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy6.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy6.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy6.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
                else
                {
                    // 直接赋值，析构已有的最优路径
                    best_gflops = gflops;
                    best_time = time;

                    // 析构已有的最优参数
                    del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                    // 析构已有的最优策略
                    del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                    // 已有的最优模板
                    del_param_of_template_node(&best_temp_node);

                    // 执行新的拷贝
                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy6.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy6.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy6.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
            }

            // 然后析构找出的路径，最优的路径已经被取出来了，所以先析构掉
            if (sub_matrix_opt_path_of_strategy6.temp_node.template_param != NULL)
            {
                del_param_of_template_node(&(sub_matrix_opt_path_of_strategy6.temp_node));    
            }
            else
            {
                assert(gflops == 0);
            }
            
            del_strategy_of_param_strategy_node_in_sub_matrix(&(sub_matrix_opt_path_of_strategy6.sub_graph_param_strategy));
            del_exe_node_param_of_compress_sub_matrix(&(sub_matrix_opt_path_of_strategy6.sub_graph));

            if (search_strategy_ptr != NULL)
            {
                search_finished_by_strategy = !continue_search(search_strategy_ptr);
            }
        }

        if (search_finished_by_strategy == false)
        {
            // 根据nnz的行切分处理
            compressed_sub_block_exe_graph_and_template_t sub_matrix_opt_path_of_strategy7 = find_best_path_of_white_list_strategy7(matrix, sub_matrix_id, time, gflops, search_strategy_ptr, data_set_collector);

            // 如果当前性能比最优性能还好，那就要记录下来
            if (gflops > best_gflops)
            {
                if (best_gflops == 0)
                {
                    // 直接赋值，并且修改最佳优化路径
                    // 这个时候还没有最优的优化路径
                    best_gflops = gflops;
                    best_time = time;

                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy7.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy7.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy7.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
                else
                {
                    // 直接赋值，析构已有的最优路径
                    best_gflops = gflops;
                    best_time = time;

                    // 析构已有的最优参数
                    del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                    // 析构已有的最优策略
                    del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                    // 已有的最优模板
                    del_param_of_template_node(&best_temp_node);

                    // 执行新的拷贝
                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy7.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy7.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy7.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
            }

            // 然后析构找出的路径，最优的路径已经被取出来了，所以先析构掉
            if (sub_matrix_opt_path_of_strategy7.temp_node.template_param != NULL)
            {
                del_param_of_template_node(&(sub_matrix_opt_path_of_strategy7.temp_node));    
            }
            else
            {
                assert(gflops == 0);
            }

            del_strategy_of_param_strategy_node_in_sub_matrix(&(sub_matrix_opt_path_of_strategy7.sub_graph_param_strategy));
            del_exe_node_param_of_compress_sub_matrix(&(sub_matrix_opt_path_of_strategy7.sub_graph));

            if (search_strategy_ptr != NULL)
            {
                search_finished_by_strategy = !continue_search(search_strategy_ptr);
            }
        }
    }

    // 当最大非零元数量小于1024时，尝试使用sharedmemory的方式
    if (matrix_information.max_row_nnz <= 1024)
    {
        assert(matrix_information.min_row_nnz > 0);

        float time;
        float gflops;

        if (search_finished_by_strategy == false)
        {
            // row_padding => evenly_BLB_row => TLB_col => shared_memory_template_warp_compress
            compressed_sub_block_exe_graph_and_template_t sub_matrix_opt_path_of_strategy2 = find_best_path_of_white_list_strategy2(matrix, sub_matrix_id, time, gflops, search_strategy_ptr, data_set_collector);

            // 如果当前性能比最优性能还好，那就要记录下来
            if (gflops > best_gflops)
            {
                if (best_gflops == 0)
                {
                    // 直接赋值，并且修改最佳优化路径
                    // 这个时候还没有最优的优化路径
                    best_gflops = gflops;
                    best_time = time;

                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy2.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy2.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy2.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
                else
                {
                    // 直接赋值，析构已有的最优路径
                    best_gflops = gflops;
                    best_time = time;

                    // 析构已有的最优参数
                    del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                    // 析构已有的最优策略
                    del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                    // 已有的最优模板
                    del_param_of_template_node(&best_temp_node);
                    
                    // 执行新的拷贝
                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy2.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy2.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy2.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);   
                }
            }

            // 然后析构找出的路径，最优的路径已经被取出来了，所以先析构掉
            if (sub_matrix_opt_path_of_strategy2.temp_node.template_param != NULL)
            {
                del_param_of_template_node(&(sub_matrix_opt_path_of_strategy2.temp_node));    
            }
            else
            {
                assert(gflops == 0);
            }

            del_strategy_of_param_strategy_node_in_sub_matrix(&(sub_matrix_opt_path_of_strategy2.sub_graph_param_strategy));
            del_exe_node_param_of_compress_sub_matrix(&(sub_matrix_opt_path_of_strategy2.sub_graph));

            if (search_strategy_ptr != NULL)
            {
                search_finished_by_strategy = !continue_search(search_strategy_ptr);
            }
        }

        if (search_finished_by_strategy == false)
        {
            // nnz_BLB_row => nnz_BLB_row => TLB_col => shared_memory_template_warp_compress
            compressed_sub_block_exe_graph_and_template_t sub_matrix_opt_path_of_strategy3 = find_best_path_of_white_list_strategy3(matrix, sub_matrix_id, time, gflops, search_strategy_ptr, data_set_collector);

            // 如果当前性能比最优性能还好，那就要记录下来
            if (gflops > best_gflops)
            {
                if (best_gflops == 0)
                {
                    // 直接赋值，并且修改最佳优化路径
                    // 这个时候还没有最优的优化路径
                    best_gflops = gflops;
                    best_time = time;

                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy3.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy3.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy3.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);
                }
                else
                {
                    // 直接赋值，析构已有的最优路径
                    best_gflops = gflops;
                    best_time = time;

                    // 析构已有的最优参数
                    del_exe_node_param_of_compress_sub_matrix(&best_sub_graph);
                    // 析构已有的最优策略
                    del_strategy_of_param_strategy_node_in_sub_matrix(&best_sub_graph_param_strategy);
                    // 已有的最优模板
                    del_param_of_template_node(&best_temp_node);
                    
                    // 执行新的拷贝
                    // 从best拷贝出来
                    best_sub_graph = val_copy_from_old_compressed_sub_matrix(sub_matrix_opt_path_of_strategy3.sub_graph);
                    best_sub_graph_param_strategy = val_copy_from_old_param_strategy_of_sub_graph(sub_matrix_opt_path_of_strategy3.sub_graph_param_strategy);
                    best_temp_node = val_copy_from_old_template_node(sub_matrix_opt_path_of_strategy3.temp_node);

                    // 重新绑定优化骨架和策略骨架
                    bind_exe_node_param_param_strategy_of_sub_graph(&best_sub_graph, &best_sub_graph_param_strategy);   
                }
            }

            // 然后析构找出的路径，最优的路径已经被取出来了，所以先析构掉
            if (sub_matrix_opt_path_of_strategy3.temp_node.template_param != NULL)
            {
                del_param_of_template_node(&(sub_matrix_opt_path_of_strategy3.temp_node));    
            }
            else
            {
                assert(gflops == 0);
            }

            del_strategy_of_param_strategy_node_in_sub_matrix(&(sub_matrix_opt_path_of_strategy3.sub_graph_param_strategy));
            del_exe_node_param_of_compress_sub_matrix(&(sub_matrix_opt_path_of_strategy3.sub_graph));

            if (search_strategy_ptr != NULL)
            {
                search_finished_by_strategy = !continue_search(search_strategy_ptr);
            }
        }
    }
    
    compressed_sub_block_exe_graph_and_template_t return_sub_graph_exe_node_and_template;
    return_sub_graph_exe_node_and_template.sub_graph = best_sub_graph;
    return_sub_graph_exe_node_and_template.sub_graph_param_strategy = best_sub_graph_param_strategy;
    return_sub_graph_exe_node_and_template.temp_node = best_temp_node;

    return return_sub_graph_exe_node_and_template;
}

// 将整个图转化为一个字符串
string convert_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_to_string(dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t graph)
{
    assert(graph.dense_sub_graph_and_param_strategy.dense_sub_graph.exe_node_vec.size() == graph.dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy.param_strategy_vec.size());

    // 首先打印稠密子块的大小
    string return_str = "";

    // 保证稠密视图的参数都是存在的
    for (unsigned long i = 0; i < graph.dense_sub_graph_and_param_strategy.dense_sub_graph.exe_node_vec.size(); i++)
    {
        assert(graph.dense_sub_graph_and_param_strategy.dense_sub_graph.exe_node_vec[i].param != NULL);
        assert(graph.dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy.param_strategy_vec[i].param == graph.dense_sub_graph_and_param_strategy.dense_sub_graph.exe_node_vec[i].param);
        assert(graph.dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy.param_strategy_vec[i].param_strategy != NULL);
    }

    return_str = return_str + convert_all_stategy_node_of_sub_matrix_to_string(graph.dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy);
    
    for (unsigned long i = 0; i < graph.compressed_sub_block_exe_graph_and_template_vec.size(); i++)
    {
        assert(graph.compressed_sub_block_exe_graph_and_template_vec[i].sub_graph.exe_node_vec.size() == graph.compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy.param_strategy_vec.size());

        // 每一个子图的参数都是存在的
        for (unsigned long j = 0; j < graph.compressed_sub_block_exe_graph_and_template_vec[i].sub_graph.exe_node_vec.size(); j++)
        {
            // 参数是存在的
            assert(graph.compressed_sub_block_exe_graph_and_template_vec[i].sub_graph.exe_node_vec[j].param != NULL);
            assert(graph.compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy.param_strategy_vec[j].param == graph.compressed_sub_block_exe_graph_and_template_vec[i].sub_graph.exe_node_vec[j].param);
            assert(graph.compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy.param_strategy_vec[j].param_strategy != NULL);
            
        }

        assert(graph.compressed_sub_block_exe_graph_and_template_vec[i].temp_node.template_param != NULL);

        return_str = return_str + "sub_matrix_id:" + to_string(i) + "\n";

        return_str = return_str + convert_all_stategy_node_of_sub_matrix_to_string(graph.compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy);

        return_str = return_str + convert_template_node_to_string(&(graph.compressed_sub_block_exe_graph_and_template_vec[i].temp_node));
    }

    return return_str;
}

string convert_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_to_string_safety(dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t graph)
{
    assert(graph.dense_sub_graph_and_param_strategy.dense_sub_graph.exe_node_vec.size() == graph.dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy.param_strategy_vec.size());
    
    string return_str = "";

    for (unsigned long i = 0; i < graph.dense_sub_graph_and_param_strategy.dense_sub_graph.exe_node_vec.size(); i++)
    {
        if (graph.dense_sub_graph_and_param_strategy.dense_sub_graph.exe_node_vec[i].param == NULL)
        {
            cout << "convert_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_to_string_safety: param of exe_node is NULL, i:" << i << endl;
            return return_str;
        }

        if (graph.dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy.param_strategy_vec[i].param_strategy == NULL)
        {
            cout << "convert_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_to_string_safety: param of param_strategy_node is NULL, i:" << i << endl;
            return return_str;
        }

        assert(graph.dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy.param_strategy_vec[i].param == graph.dense_sub_graph_and_param_strategy.dense_sub_graph.exe_node_vec[i].param);
    }

    return_str = return_str + convert_all_stategy_node_of_sub_matrix_to_string(graph.dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy);

    for (unsigned long i = 0; i < graph.compressed_sub_block_exe_graph_and_template_vec.size(); i++)
    {
        assert(graph.compressed_sub_block_exe_graph_and_template_vec[i].sub_graph.exe_node_vec.size() == graph.compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy.param_strategy_vec.size());

        // 查看子图是不是需要打印
        bool need_print = true;
        
        // 每一个子图的参数都是存在的
        for (unsigned long j = 0; j < graph.compressed_sub_block_exe_graph_and_template_vec[i].sub_graph.exe_node_vec.size(); j++)
        {
            if (graph.compressed_sub_block_exe_graph_and_template_vec[i].sub_graph.exe_node_vec[j].param == NULL)
            {
                cout << "convert_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_to_string_safety: param of exe_node is NULL, sub_matrix_id:" << i << ", node_id:" << j << endl;
                need_print = false;
            }

            if (graph.compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy.param_strategy_vec[j].param != graph.compressed_sub_block_exe_graph_and_template_vec[i].sub_graph.exe_node_vec[j].param)
            {
                cout << "convert_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_to_string_safety: param of exe_node and strategy_node is not equal, sub_matrix_id:" << i << ", node_id:" << j << endl;
                need_print = false;
            }

            if (graph.compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy.param_strategy_vec[j].param_strategy == NULL)
            {
                cout << "convert_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_to_string_safety: param of strategy_node is NULL, sub_matrix_id:" << i << ", node_id:" << j << endl;
                need_print = false;    
            }
        }

        assert(graph.compressed_sub_block_exe_graph_and_template_vec[i].temp_node.template_param != NULL);
        if (graph.compressed_sub_block_exe_graph_and_template_vec[i].temp_node.template_param == NULL)
        {
            cout << "convert_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_to_string_safety: param of template_node is NULL" << endl;
            need_print = false;
        }

        return_str = return_str + "sub_matrix_id:" + to_string(i) + "\n";

        if (need_print == true)
        {
            return_str = return_str + convert_all_stategy_node_of_sub_matrix_to_string(graph.compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy);    
        }
        else
        {
            return_str = return_str + "something mistake, cannot print sub matrix opt path\n";
        }
        

        return_str = return_str + convert_template_node_to_string(&(graph.compressed_sub_block_exe_graph_and_template_vec[i].temp_node));
    }

    return return_str;
}

// 增加两个节点
dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t find_best_graph_of_white_list_strategy1(exe_begin_memory_cache_input_file_param_t input_matrix_node, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 检查一下输入的节点是不是正确
    assert(input_matrix_node.col_index_cache.size() > 0);
    assert(input_matrix_node.col_index_cache.size() == input_matrix_node.row_index_cache.size());

    assert(input_matrix_node.val_data_type == DOUBLE || input_matrix_node.val_data_type == FLOAT);

    if (input_matrix_node.val_data_type == DOUBLE)
    {
        assert(input_matrix_node.col_index_cache.size() == input_matrix_node.double_val_cache.size());
        assert(input_matrix_node.float_val_cache.size() == 0);
    }
    else
    {
        assert(input_matrix_node.col_index_cache.size() == input_matrix_node.float_val_cache.size());
        assert(input_matrix_node.double_val_cache.size() == 0);
    }

    // 如果存在数据集，那数据集即将插入表项应该不存在积累的数据
    if (data_set_collector != NULL)
    {
        // cout << "find_best_graph_of_white_list_strategy1: need to collect ml data" << endl;
        assert((data_set_collector->accu_dense_graph_node_type_vec.size()) == 0);
        assert(data_set_collector->accu_dense_param_strategy_type_vec.size() == 0);
        assert(data_set_collector->accu_compressed_sub_graph_node_type_vec.size() == 0);
        assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() == 0);
        assert(data_set_collector->accu_dense_param_vec.size() == 0);
        assert(data_set_collector->accu_compressed_param_vec.size() == 0);
    }

    // 申请三个数组，分别存dense阶段的节点类型，策略类型和参数
    vector<exe_node_type> dense_node_type_vec;
    vector<exe_node_param_set_strategy> dense_param_strategy_type_vec;
    vector<float> dense_param_vec;

    // 稠密子矩阵和对应的调参策略
    exe_dense_sub_graph_t sub_graph;
    param_strategy_of_sub_graph_t sub_graph_param_strategy;

    // 创造输入的节点，用内存中初始化
    dense_begin_memory_cache_input_file_direct_param_strategy_t input_node_param_strategy;
    input_node_param_strategy.col_index_cache = input_matrix_node.col_index_cache;
    input_node_param_strategy.col_index_max = input_matrix_node.col_index_max;
    input_node_param_strategy.double_val_cache = input_matrix_node.double_val_cache;
    input_node_param_strategy.float_val_cache = input_matrix_node.float_val_cache;
    input_node_param_strategy.row_index_cache = input_matrix_node.row_index_cache;
    input_node_param_strategy.row_index_max = input_matrix_node.row_index_max;
    input_node_param_strategy.val_data_type = input_matrix_node.val_data_type;

    add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&sub_graph, &sub_graph_param_strategy, BEGIN_MEMORY_CACHE_INPUT_FILE, DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY, &input_node_param_strategy);
    // 增加节点的类型和参数策略的类型
    dense_node_type_vec.push_back(BEGIN_MEMORY_CACHE_INPUT_FILE);
    dense_param_strategy_type_vec.push_back(DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY);

    // 创造一个新的节点，执行一次压缩
    compress_none_param_strategy_t compress_param_strategy;
    add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&sub_graph, &sub_graph_param_strategy, COMPRESS, COMPRESS_NONE_PARAM_STRATEGY, &compress_param_strategy);
    // 增加节点类型和参数策略类型
    dense_node_type_vec.push_back(COMPRESS);
    dense_param_strategy_type_vec.push_back(COMPRESS_NONE_PARAM_STRATEGY);
    
    // 执行对应的稠密子块调优
    sparse_struct_t* matrix = execute_dense_matrix_exe_graph_with_param_strategy(&sub_graph, &sub_graph_param_strategy);

    // 没有需要被调节的参数，直接执行对应的子块的调优
    float time = 99999999999;
    float gflops = 0;

    // 如果数据收集器是存在的，那就给下一个表项dense阶段的累计值赋值
    if (data_set_collector != NULL)
    {
        // 消除所有的内容
        data_set_collector->clear_all_accu_info();
        data_set_collector->insert_dense_stage_node_and_param_to_cur_item(dense_node_type_vec, dense_param_strategy_type_vec, dense_param_vec);
    }

    compressed_sub_block_exe_graph_and_template_t sub_graph_matrix = find_best_path_of_compressed_sub_matrix(matrix, 0, time, gflops, search_strategy_ptr, data_set_collector);

    // 将最优值拷贝出来
    best_gflops = gflops;
    best_time = time;

    // 将子图的优化路径拷贝出来
    dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t return_best_graph;

    return_best_graph.dense_sub_graph_and_param_strategy.dense_sub_graph = sub_graph;
    return_best_graph.dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy = sub_graph_param_strategy;
    return_best_graph.compressed_sub_block_exe_graph_and_template_vec.push_back(sub_graph_matrix);

    // 将当前的最优结果输出
    if (best_gflops > 0)
    {
        write_graph_structure_and_performance_to_file(best_gflops, best_time, return_best_graph, string(string(get_config()["ROOT_PATH_STR"].as_string())) + "/data_source/best_result_of_strategy1");
    }

    return return_best_graph;
}

// 直接行分块之后执行压缩
dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t find_best_graph_of_white_list_strategy2(exe_begin_memory_cache_input_file_param_t input_matrix_node, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 检查一下输入的节点是不是正确
    assert(input_matrix_node.col_index_cache.size() > 0);
    assert(input_matrix_node.col_index_cache.size() == input_matrix_node.row_index_cache.size());

    assert(input_matrix_node.val_data_type == DOUBLE || input_matrix_node.val_data_type == FLOAT);

    if (input_matrix_node.val_data_type == DOUBLE)
    {
        assert(input_matrix_node.col_index_cache.size() == input_matrix_node.double_val_cache.size());
        assert(input_matrix_node.float_val_cache.size() == 0);
    }
    else
    {
        assert(input_matrix_node.col_index_cache.size() == input_matrix_node.float_val_cache.size());
        assert(input_matrix_node.double_val_cache.size() == 0);
    }

    // 如果存在数据集，那数据集即将插入表项应该不存在积累的数据
    if (data_set_collector != NULL)
    {
        assert((data_set_collector->accu_dense_graph_node_type_vec.size()) == 0);
        assert(data_set_collector->accu_dense_param_strategy_type_vec.size() == 0);
        assert(data_set_collector->accu_compressed_sub_graph_node_type_vec.size() == 0);
        assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() == 0);
        assert(data_set_collector->accu_dense_param_vec.size() == 0);
        assert(data_set_collector->accu_compressed_param_vec.size() == 0);
    }

    // 三个数组来分别存储节点类型，策略类型和积累的参数
    vector<exe_node_type> dense_node_type_vec;
    vector<exe_node_param_set_strategy> dense_param_strategy_type_vec;
    vector<float> dense_param_vec;
    
    // 首先仅仅执行一个输入，然后找出行非零元数量
    matrix_info_t matrix_info = get_global_matrix_info_from_input_node(input_matrix_node);

    assert(matrix_info.row_nnz.size() > 0);

    // 稠密子矩阵和对应的调参策略
    exe_dense_sub_graph_t sub_graph;
    param_strategy_of_sub_graph_t sub_graph_param_strategy;

    // 创造输入的节点，用内存中初始化
    dense_begin_memory_cache_input_file_direct_param_strategy_t input_node_param_strategy;
    input_node_param_strategy.col_index_cache = input_matrix_node.col_index_cache;
    input_node_param_strategy.col_index_max = input_matrix_node.col_index_max;
    input_node_param_strategy.double_val_cache = input_matrix_node.double_val_cache;
    input_node_param_strategy.float_val_cache = input_matrix_node.float_val_cache;
    input_node_param_strategy.row_index_cache = input_matrix_node.row_index_cache;
    input_node_param_strategy.row_index_max = input_matrix_node.row_index_max;
    input_node_param_strategy.val_data_type = input_matrix_node.val_data_type;

    add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&sub_graph, &sub_graph_param_strategy, BEGIN_MEMORY_CACHE_INPUT_FILE, DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY, &input_node_param_strategy);

    // 增加一个节点
    dense_node_type_vec.push_back(BEGIN_MEMORY_CACHE_INPUT_FILE);
    dense_param_strategy_type_vec.push_back(DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY);

    // 加入一个行分块节点
    dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t row_div_param_strategy;
    row_div_param_strategy.expansion_rate = 4;
    row_div_param_strategy.lowest_nnz_bound_of_row = 32;
    row_div_param_strategy.highest_nnz_bound_of_row = 4096;
    row_div_param_strategy.sub_dense_block_id = 0;

    // 查看当前分块之后的块数量
    vector<unsigned long> row_div_position = row_div_position_acc_to_exponential_increase_row_nnz_range(matrix_info.row_nnz, row_div_param_strategy.lowest_nnz_bound_of_row, row_div_param_strategy.expansion_rate);

    assert(row_div_position[0] == 0 && row_div_position[row_div_position.size() - 1] == matrix_info.row_num);

    // 如果分不出块，那也直接退出
    if (row_div_position.size() > 5 || row_div_position.size() == 2)
    {
        // 直接返回
        best_gflops = 0;
        best_time = 9999999999;

        // del_exe_node_param_of_dense_view_matrix(sub_graph);
        // del_strategy_of_param_strategy_node_in_sub_matrix(sub_graph_param_strategy);


        dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t return_best_graph;

        return return_best_graph;
    }

    // 只有分块位置高于2的时候才需要分块
    // if (row_div_position.size() > 2)
    // {
    add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&sub_graph, &sub_graph_param_strategy, DENSE_ROW_DIV, DENSE_ROW_DIV_ACC_TO_EXPONENTIAL_INCREASE_ROW_NNZ_PARAM_STRATEGY, &row_div_param_strategy);
    // 执行分块，将节点类型和参数策略类型已经参数记录起来
    dense_node_type_vec.push_back(DENSE_ROW_DIV);
    dense_param_strategy_type_vec.push_back(DENSE_ROW_DIV_ACC_TO_EXPONENTIAL_INCREASE_ROW_NNZ_PARAM_STRATEGY);
    dense_param_vec.push_back(row_div_param_strategy.expansion_rate);
    dense_param_vec.push_back(row_div_param_strategy.highest_nnz_bound_of_row);
    dense_param_vec.push_back(row_div_param_strategy.lowest_nnz_bound_of_row);
    dense_param_vec.push_back(row_div_param_strategy.sub_dense_block_id);
    // }

    // 创造一个新的节点，执行一次压缩
    compress_none_param_strategy_t compress_param_strategy;
    
    add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&sub_graph, &sub_graph_param_strategy, COMPRESS, COMPRESS_NONE_PARAM_STRATEGY, &compress_param_strategy);

    dense_node_type_vec.push_back(COMPRESS);
    dense_param_strategy_type_vec.push_back(COMPRESS_NONE_PARAM_STRATEGY);

    // 执行对应的稠密子块调优
    sparse_struct_t* matrix = execute_dense_matrix_exe_graph_with_param_strategy(&sub_graph, &sub_graph_param_strategy);

    print_dense_block_table(&(matrix->block_coor_table));

    assert(matrix != NULL);
    assert(matrix->block_coor_table.item_arr.size() > 0);
    assert(matrix->block_coor_table.item_arr.size() < 10);

    // 将子图的优化路径拷贝出来
    dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t return_best_graph;

    // 每一个子块的gflops
    vector<float> gflops_of_each_sub_matrix;
    vector<float> time_of_each_sub_matrix;

    return_best_graph.dense_sub_graph_and_param_strategy.dense_sub_graph = sub_graph;
    return_best_graph.dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy = sub_graph_param_strategy;

    // 对每一个子块执行对应的搜索
    for (unsigned long i = 0; i < matrix->block_coor_table.item_arr.size(); i++)
    {
        assert(matrix->block_coor_table.item_arr[i] != NULL);
        assert(matrix->block_coor_table.item_arr[i]->compressed_block_ptr != NULL);

        // 没有需要被调节的参数，直接执行对应的子块的调优
        float time = 99999999999;
        float gflops = 0;

        compressed_sub_block_exe_graph_and_template_t sub_graph_matrix;

        // 如果收集器是存在的，那就初始化dense阶段表项的累计内容
        if (data_set_collector != NULL)
        {
            // 消除当前的积累的当前表项
            data_set_collector->clear_all_accu_info();
            data_set_collector->insert_dense_stage_node_and_param_to_cur_item(dense_node_type_vec, dense_param_strategy_type_vec, dense_param_vec);
        }
        
        // 如果存在搜索策略，那就创建一个新的搜索策略
        if (search_strategy_ptr != NULL)
        {
            // 当前子块的非零元数量
            unsigned long nnz_of_this_sub_block = matrix->block_coor_table.item_arr[i]->end_coo_index - matrix->block_coor_table.item_arr[i]->begin_coo_index + 1;            

            float cur_sub_block_time_limit = ((float)nnz_of_this_sub_block / (float)matrix->origin_nnz) * (float)search_strategy_ptr->total_allow_search_time;

            // 保证一个最低限度的搜索时间
            if (cur_sub_block_time_limit < (float)search_strategy_ptr->total_allow_search_time / matrix->block_coor_table.item_arr.size() / 2)
            {
                cur_sub_block_time_limit = (float)search_strategy_ptr->total_allow_search_time / matrix->block_coor_table.item_arr.size() / 2;
            }

            // 保证一个最高限度的搜索时间，
            if (cur_sub_block_time_limit > (float)search_strategy_ptr->total_allow_search_time / matrix->block_coor_table.item_arr.size() * 3)
            {
                cur_sub_block_time_limit = (float)search_strategy_ptr->total_allow_search_time / matrix->block_coor_table.item_arr.size() * 3;
            }

            // cout << "matrix->block_coor_table.item_arr.size():" << matrix->block_coor_table.item_arr.size() << endl;

            cout << "find_best_graph_of_white_list_strategy2: time limit of sub_matrix " << i << " : " << cur_sub_block_time_limit << endl;

            // cout << "(float)search_strategy_ptr->total_allow_search_time:" << (float)search_strategy_ptr->total_allow_search_time << endl;

            // exit(-1);

            // 按照nnz的数量来决定每个子块的搜索时间分成
            search_strategy_t sub_block_search_strategy = init_search_strategy(search_strategy_ptr->struggle_step_num, cur_sub_block_time_limit);

            sub_graph_matrix = find_best_path_of_compressed_sub_matrix(matrix, i, time, gflops, &sub_block_search_strategy, data_set_collector);
        }
        else
        {
            // 不存在搜索策略，那就执行全局搜索
            sub_graph_matrix = find_best_path_of_compressed_sub_matrix(matrix, i, time, gflops, NULL, data_set_collector);
        }

        gflops_of_each_sub_matrix.push_back(gflops);
        time_of_each_sub_matrix.push_back(time);

        return_best_graph.compressed_sub_block_exe_graph_and_template_vec.push_back(sub_graph_matrix);
    }

    // cout << "find_best_graph_of_white_list_strategy2: finish search" << endl;

    // 查看对应最优解是不是应该被执行的
    bool can_be_execute = true;

    for (unsigned long i = 0; i < gflops_of_each_sub_matrix.size(); i++)
    {
        if (gflops_of_each_sub_matrix[i] == 0)
        {
            can_be_execute = false;
        }
    }

    // 如果允许运行就全局执行
    if (can_be_execute == true)
    {
        execute_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template(&return_best_graph, best_gflops, best_time, 2);
    }

    // for (unsigned long i = 0; i < gflops_of_each_sub_matrix.size(); i++)
    // {
    //     // 当前子块的非零元数量
    //     unsigned long sub_block_nnz_num = matrix->block_coor_table.item_arr[i]->end_coo_index - matrix->block_coor_table.item_arr[i]->begin_coo_index + 1;
    //     assert(sub_block_nnz_num > 0);

    //     total_gflops = total_gflops + ((float)sub_block_nnz_num / (float)matrix->origin_nnz) * gflops_of_each_sub_matrix[i];
    //     total_time = total_time + time_of_each_sub_matrix[i];
    // }

    // // 多流导致的问题
    // if (gflops_of_each_sub_matrix.size() > 1)
    // {
    //     total_gflops = total_gflops * 1.03;
    //     total_time = total_time / (float)gflops_of_each_sub_matrix.size();
    // }

    // assert(total_gflops > 0);

    // 析构矩阵
    assert(matrix != NULL);
    memory_garbage_manager_t mem_manager;
    delete_sparse_struct_t(&mem_manager, matrix);

    // best_gflops = total_gflops;
    // best_time = total_time;

    if (best_gflops > 0)
    {
        write_graph_structure_and_performance_to_file(best_gflops, best_time, return_best_graph, string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source/best_result_of_strategy2");    
    }

    return return_best_graph;
}

// 执行排序，而后执行分块，最后对每个子块执行对应的子块分块
// 这里的搜索参数是重头开始的，时间是从头开始算的
dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t find_best_graph_of_white_list_strategy3(exe_begin_memory_cache_input_file_param_t input_matrix_node, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 检查一下输入的节点是不是正确
    assert(input_matrix_node.col_index_cache.size() > 0);
    assert(input_matrix_node.col_index_cache.size() == input_matrix_node.row_index_cache.size());

    assert(input_matrix_node.val_data_type == DOUBLE || input_matrix_node.val_data_type == FLOAT);

    if (input_matrix_node.val_data_type == DOUBLE)
    {
        assert(input_matrix_node.col_index_cache.size() == input_matrix_node.double_val_cache.size());
        assert(input_matrix_node.float_val_cache.size() == 0);
    }
    else
    {
        assert(input_matrix_node.col_index_cache.size() == input_matrix_node.float_val_cache.size());
        assert(input_matrix_node.double_val_cache.size() == 0);
    }

    // 如果存在数据集，那数据集即将插入表项应该不存在积累的数据
    if (data_set_collector != NULL)
    {
        assert((data_set_collector->accu_dense_graph_node_type_vec.size()) == 0);
        assert(data_set_collector->accu_dense_param_strategy_type_vec.size() == 0);
        assert(data_set_collector->accu_compressed_sub_graph_node_type_vec.size() == 0);
        assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() == 0);
        assert(data_set_collector->accu_dense_param_vec.size() == 0);
        assert(data_set_collector->accu_compressed_param_vec.size() == 0);
    }

    // 潜在的行分块枚举数量
    unsigned long num_of_row_div_enumerate = 0;

    // 将排序之后矩阵基本信息取出
    matrix_info_t matrix_info_after_sort;
    
    // 设定在处理之后子块的信息
    {
        // 先执行一个排序之后，获取一个矩阵的基本信息
        exe_dense_sub_graph_t temp_sub_dense_graph;
        param_strategy_of_sub_graph_t temp_sub_dense_strategy_graph;

        dense_begin_memory_cache_input_file_direct_param_strategy_t input_node_param_strategy;
        input_node_param_strategy.col_index_cache = input_matrix_node.col_index_cache;
        input_node_param_strategy.col_index_max = input_matrix_node.col_index_max;
        input_node_param_strategy.double_val_cache = input_matrix_node.double_val_cache;
        input_node_param_strategy.float_val_cache = input_matrix_node.float_val_cache;
        input_node_param_strategy.row_index_cache = input_matrix_node.row_index_cache;
        input_node_param_strategy.row_index_max = input_matrix_node.row_index_max;
        input_node_param_strategy.val_data_type = input_matrix_node.val_data_type;

        // 增加一个输入节点和一个排序节点
        add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&temp_sub_dense_graph, &temp_sub_dense_strategy_graph, BEGIN_MEMORY_CACHE_INPUT_FILE, DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY, &input_node_param_strategy);

        // 加入一个排序节点
        dense_row_coarse_sort_fixed_param_strategy_t sort_param_strategy;
        sort_param_strategy.row_nnz_low_bound_step_size = 1;

        add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&temp_sub_dense_graph, &temp_sub_dense_strategy_graph, DENSE_ROW_COARSE_SORT, DENSE_ROW_COARSE_SORT_FIXED_PARAM_STRATEGY, &sort_param_strategy);

        // 执行临时的图
        sparse_struct_t* temp_matrix = execute_dense_matrix_exe_graph_with_param_strategy(&temp_sub_dense_graph, &temp_sub_dense_strategy_graph);

        assert(temp_matrix != NULL);

        // 一定是经过排序的
        assert(temp_matrix->sorted_row_index != NULL);

        matrix_info_t temp_info = get_matrix_info_from_sparse_matrix_ptr(temp_matrix);

        assert(temp_info.is_sorted == true);

        matrix_info_after_sort = temp_info;
        
        // 使用一个集合来存储所有的分块结果
        set<vector<unsigned long>> row_div_pos_set;        

        // 分块的起始位置，先粗分，然后细分
        for (long min_row_nnz_bound = 512; min_row_nnz_bound >= 32; min_row_nnz_bound = min_row_nnz_bound / 4)
        {
            // 分块行非零元数量的膨胀系数
            for (long expansion_rate = 4; expansion_rate <= 16; expansion_rate = expansion_rate * 2)
            {
                // 上界永远都是2048
                // 这里注意去重
                vector<unsigned long> row_div_position = row_div_position_acc_to_exponential_increase_row_nnz_range(temp_info.row_nnz, min_row_nnz_bound, 2048, expansion_rate);
                
                // 分块数量不能超过4个
                if (row_div_position.size() <= 5)
                {
                    row_div_pos_set.insert(row_div_position);
                    // cout << "min_row_nnz_bound:" << min_row_nnz_bound << ", expansion_rate:" << expansion_rate << endl;

                    // // 打印分块的位置
                    // cout << "[";

                    // for (unsigned long pos_id = 0; pos_id < row_div_position.size(); pos_id++)
                    // {
                    //     cout << row_div_position[pos_id] << ",";
                    // }

                    // cout << "]" << endl;
                }
            }
        }

        num_of_row_div_enumerate = row_div_pos_set.size();

        // 临时图的内容压缩
        del_strategy_of_param_strategy_node_in_sub_matrix(&temp_sub_dense_strategy_graph);
        del_exe_node_param_of_dense_view_matrix(&temp_sub_dense_graph);

        // 将整个矩阵析构
        memory_garbage_manager_t mem_manager;
        delete_sparse_struct_t(&mem_manager, temp_matrix);
    }

    // 打印枚举的数量
    cout << "find_best_graph_of_white_list_strategy3: num_of_row_div_enumerate: " << num_of_row_div_enumerate << endl;

    // 最优全局图
    dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t best_graph;

    // 最优的性能和时间
    best_time = 9999999999;
    best_gflops = 0;

    // 如果已经有单个分块的，那么如果又出现只有一个分块的情况就不用处理了，用来剪枝一下
    bool have_handled_one_block_situation = false;

    // 查看已经经过的分块
    set<vector<unsigned long>> existing_div_pos_set;

    // 查看是不是要接着搜索
    bool is_continue_search = true;

    // 查看是不是自带排序
    bool is_origin_sorted = get_global_matrix_info_from_input_node(input_matrix_node).is_sorted;

    // 当前最优的稠密子图
    // 两个循环，分别是分块的起始行非零元数量和其实行非零元的膨胀倍数
    for (long min_row_nnz_bound = 512; min_row_nnz_bound >= 32; min_row_nnz_bound = min_row_nnz_bound / 4)
    {
        // 膨胀的系数
        for (long expansion_rate = 4; expansion_rate <= 16; expansion_rate = expansion_rate * 2)
        {
            // 插入三个数组，记录当前dense阶段的节点和参数
            vector<exe_node_type> dense_node_type_vec;
            vector<exe_node_param_set_strategy> dense_param_strategy_type_vec;
            vector<float> dense_param_vec;

            cout << "find_best_graph_of_white_list_strategy3: try min_row_nnz_bound:" << min_row_nnz_bound << ", expansion_rate:" << expansion_rate << endl;
            // 当前枚举的最大性能
            float cur_gflops = 0;
            float cur_time = 9999999999;

            // 执行稠密子图
            // 稠密子矩阵和对应的调参策略
            exe_dense_sub_graph_t sub_graph;
            param_strategy_of_sub_graph_t sub_graph_param_strategy;

            // 创造输入的节点，用内存中初始化
            dense_begin_memory_cache_input_file_direct_param_strategy_t input_node_param_strategy;
            input_node_param_strategy.col_index_cache = input_matrix_node.col_index_cache;
            input_node_param_strategy.col_index_max = input_matrix_node.col_index_max;
            input_node_param_strategy.double_val_cache = input_matrix_node.double_val_cache;
            input_node_param_strategy.float_val_cache = input_matrix_node.float_val_cache;
            input_node_param_strategy.row_index_cache = input_matrix_node.row_index_cache;
            input_node_param_strategy.row_index_max = input_matrix_node.row_index_max;
            input_node_param_strategy.val_data_type = input_matrix_node.val_data_type;

            add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&sub_graph, &sub_graph_param_strategy, BEGIN_MEMORY_CACHE_INPUT_FILE, DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY, &input_node_param_strategy);
            // 记录节点的类型
            dense_node_type_vec.push_back(BEGIN_MEMORY_CACHE_INPUT_FILE);
            dense_param_strategy_type_vec.push_back(DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY);
            
            // 如果已经有自带的排序，就不需要排序了
            if (is_origin_sorted == false)
            {
                // 插入一个排序
                dense_row_coarse_sort_fixed_param_strategy_t sort_param_strategy;
                sort_param_strategy.row_nnz_low_bound_step_size = 1;

                add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&sub_graph, &sub_graph_param_strategy, DENSE_ROW_COARSE_SORT, DENSE_ROW_COARSE_SORT_FIXED_PARAM_STRATEGY, &sort_param_strategy);
                // 增加参数
                dense_node_type_vec.push_back(DENSE_ROW_COARSE_SORT);
                dense_param_strategy_type_vec.push_back(DENSE_ROW_COARSE_SORT_FIXED_PARAM_STRATEGY);
                dense_param_vec.push_back(sort_param_strategy.row_nnz_low_bound_step_size);
            }

            // 执行分块啊
            // 加入一个行分块节点
            dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t row_div_param_strategy;
            row_div_param_strategy.expansion_rate = expansion_rate;
            row_div_param_strategy.lowest_nnz_bound_of_row = min_row_nnz_bound;
            assert(min_row_nnz_bound <= 2048);
            row_div_param_strategy.highest_nnz_bound_of_row = 2048;
            row_div_param_strategy.sub_dense_block_id = 0;
            
            vector<unsigned long> row_div_position = row_div_position_acc_to_exponential_increase_row_nnz_range(matrix_info_after_sort.row_nnz, row_div_param_strategy.lowest_nnz_bound_of_row, row_div_param_strategy.highest_nnz_bound_of_row, row_div_param_strategy.expansion_rate);

            if (row_div_position.size() > 5)
            {
                // 这里代表不需要进一步尝试了，分块太多了
                cout << "find_best_graph_of_white_list_strategy3: num of sub matrix is more than 5, skip" << endl;

                del_exe_node_param_of_dense_view_matrix(&sub_graph);
                del_strategy_of_param_strategy_node_in_sub_matrix(&sub_graph_param_strategy);

                // 继续
                continue;
            }

            // 进一步执行分块
            add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&sub_graph, &sub_graph_param_strategy, DENSE_ROW_DIV, DENSE_ROW_DIV_ACC_TO_EXPONENTIAL_INCREASE_ROW_NNZ_PARAM_STRATEGY, &row_div_param_strategy);

            dense_node_type_vec.push_back(DENSE_ROW_DIV);
            dense_param_strategy_type_vec.push_back(DENSE_ROW_DIV_ACC_TO_EXPONENTIAL_INCREASE_ROW_NNZ_PARAM_STRATEGY);
            dense_param_vec.push_back(row_div_param_strategy.expansion_rate);
            dense_param_vec.push_back(row_div_param_strategy.highest_nnz_bound_of_row);
            dense_param_vec.push_back(row_div_param_strategy.lowest_nnz_bound_of_row);
            dense_param_vec.push_back(row_div_param_strategy.sub_dense_block_id);

            // 创造一个新的节点，执行一次压缩
            compress_none_param_strategy_t compress_param_strategy;

            add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&sub_graph, &sub_graph_param_strategy, COMPRESS, COMPRESS_NONE_PARAM_STRATEGY, &compress_param_strategy);

            dense_node_type_vec.push_back(COMPRESS);
            dense_param_strategy_type_vec.push_back(COMPRESS_NONE_PARAM_STRATEGY);

            // 执行对应的稠密子块调优
            sparse_struct_t *matrix = execute_dense_matrix_exe_graph_with_param_strategy(&sub_graph, &sub_graph_param_strategy);

            // 看看有没有好好执行排序
            assert(matrix->is_sorted == true || is_origin_sorted == true);

            // 查看子块的数量
            assert(matrix->block_coor_table.item_arr.size() <= 4 && matrix->block_coor_table.item_arr.size() > 0);

            print_dense_block_table(&(matrix->block_coor_table));

            // 每一个子块的gflops
            vector<float> gflops_of_each_sub_matrix;
            vector<float> time_of_each_sub_matrix;

            dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t cur_best_graph;

            // 处理稠密子图的部分，稠密子图的部分没有比较的问题
            cur_best_graph.dense_sub_graph_and_param_strategy.dense_sub_graph = sub_graph;
            cur_best_graph.dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy = sub_graph_param_strategy;

            // 根据是不是有搜索策略来执行不同的搜索
            if (search_strategy_ptr != NULL)
            {
                // 当前时间
                struct timeval cur_timeval;

                // 已经经过的时间，因为策略从头开始算时间，所以这里可以算出当前策略已经经过的时间
                gettimeofday(&(cur_timeval), NULL);

                // 已经经过的时间，用秒来计时
                float time_has_spend = (float)(cur_timeval.tv_sec - search_strategy_ptr->begin_time.tv_sec) + (float)(cur_timeval.tv_usec - search_strategy_ptr->begin_time.tv_usec) / 1000000.0;

                // 如果已经超时了就直接退出
                if (time_has_spend >= search_strategy_ptr->total_allow_search_time)
                {
                    cout << "find_best_graph_of_white_list_strategy3: time is not enough to try a new enumeration" << endl;

                    // 时间不够了，析构所有的数据，并且退出循环
                    del_exe_node_param_of_dense_view_matrix(&sub_graph);
                    del_strategy_of_param_strategy_node_in_sub_matrix(&sub_graph_param_strategy);

                    // 析构稠密视图的矩阵
                    memory_garbage_manager_t mem_manager;
                    assert(matrix != NULL);
                    delete_sparse_struct_t(&mem_manager, matrix);

                    is_continue_search = false;

                    break;
                }

                // 剩余时间
                float remain_time = search_strategy_ptr->total_allow_search_time - time_has_spend;

                // 如果剩余时间多于5400s，也就是一个小时多一点，那么就给这个枚举5400s的时间
                search_strategy_t search_strategy_of_this_enumeration;
                
                if (remain_time < 5400)
                {
                    search_strategy_of_this_enumeration = init_search_strategy(search_strategy_ptr->struggle_step_num, remain_time);
                }
                else
                {
                    search_strategy_of_this_enumeration = init_search_strategy(search_strategy_ptr->struggle_step_num, 4000);
                }

                cout << "find_best_graph_of_white_list_strategy3: time limit of min_row_nnz_bound:" << min_row_nnz_bound << ", expansion_rate:" << expansion_rate << ", time limit:" << search_strategy_of_this_enumeration.total_allow_search_time << endl;

                // 对每一个子块分别找到最优的路径
                // 对每一个子块执行对应的搜索
                for (unsigned long i = 0; i < matrix->block_coor_table.item_arr.size(); i++)
                {
                    assert(matrix->block_coor_table.item_arr[i] != NULL);
                    assert(matrix->block_coor_table.item_arr[i]->compressed_block_ptr != NULL);

                    // 没有需要被调节的参数，直接执行对应的子块的调优
                    float time = 99999999999;
                    float gflops = 0;

                    unsigned long nnz_of_this_sub_block = matrix->block_coor_table.item_arr[i]->end_coo_index - matrix->block_coor_table.item_arr[i]->begin_coo_index + 1;

                    // 按照非零元占比来决定搜索时间，并且为搜索时间设定一个上界和下界
                    float cur_sub_block_time_limit = ((float)nnz_of_this_sub_block / (float)matrix->origin_nnz) * (float)search_strategy_ptr->total_allow_search_time;

                    // 保证一个最低限度的搜索时间
                    if (cur_sub_block_time_limit < (float)search_strategy_ptr->total_allow_search_time / matrix->block_coor_table.item_arr.size() / 2)
                    {
                        cur_sub_block_time_limit = (float)search_strategy_ptr->total_allow_search_time / matrix->block_coor_table.item_arr.size() / 2;
                    }

                    // 保证一个最高限度的搜索时间，
                    if (cur_sub_block_time_limit > (float)search_strategy_ptr->total_allow_search_time / matrix->block_coor_table.item_arr.size() * 3)
                    {
                        cur_sub_block_time_limit = (float)search_strategy_ptr->total_allow_search_time / matrix->block_coor_table.item_arr.size() * 3;
                    }

                    cout << "find_best_graph_of_white_list_strategy3: time limit of sub_matrix " << i << " : " << cur_sub_block_time_limit << endl;

                    // 申请一个新的搜索策略
                    search_strategy_t sub_block_search_strategy = init_search_strategy(search_strategy_ptr->struggle_step_num, cur_sub_block_time_limit);

                    // 如果要收集数据集，就初始化当前表项的累计值
                    if (data_set_collector != NULL)
                    {
                        data_set_collector->clear_all_accu_info();
                        data_set_collector->insert_dense_stage_node_and_param_to_cur_item(dense_node_type_vec, dense_param_strategy_type_vec, dense_param_vec);
                    }

                    compressed_sub_block_exe_graph_and_template_t sub_graph_matrix;

                    // 这里设定策略，只要时间足够，让整个矩阵的搜索时间至少90分钟，落实到每一个子块也有至少超过20分钟
                    // 这里需要修改，将dense阶段的图结构放到数据集收集器中
                    sub_graph_matrix = find_best_path_of_compressed_sub_matrix(matrix, i, time, gflops, &sub_block_search_strategy, data_set_collector);

                    assert(gflops >= 0);

                    gflops_of_each_sub_matrix.push_back(gflops);
                    time_of_each_sub_matrix.push_back(time);

                    cur_best_graph.compressed_sub_block_exe_graph_and_template_vec.push_back(sub_graph_matrix);
                }
            }
            else
            {
                // 不带搜索策略的搜索过程
                for (unsigned long i = 0; i < matrix->block_coor_table.item_arr.size(); i++)
                {
                    assert(matrix->block_coor_table.item_arr[i] != NULL);
                    assert(matrix->block_coor_table.item_arr[i]->compressed_block_ptr != NULL);

                    // 没有需要被调节的参数，直接执行对应的子块的调优
                    float time = 99999999999;
                    float gflops = 0;

                    // 如果要收集数据集，就初始化当前表项的累计值
                    if (data_set_collector != NULL)
                    {
                        data_set_collector->clear_all_accu_info();
                        data_set_collector->insert_dense_stage_node_and_param_to_cur_item(dense_node_type_vec, dense_param_strategy_type_vec, dense_param_vec);
                    }

                    compressed_sub_block_exe_graph_and_template_t sub_graph_matrix;

                    // 这里设定策略，只要时间足够，让整个矩阵的搜索时间至少90分钟，落实到每一个子块也有至少超过20分钟
                    sub_graph_matrix = find_best_path_of_compressed_sub_matrix(matrix, i, time, gflops, NULL, data_set_collector);

                    assert(gflops >= 0);

                    gflops_of_each_sub_matrix.push_back(gflops);
                    time_of_each_sub_matrix.push_back(time);

                    cur_best_graph.compressed_sub_block_exe_graph_and_template_vec.push_back(sub_graph_matrix);
                }
            }

            assert(gflops_of_each_sub_matrix.size() == matrix->block_coor_table.item_arr.size());

            // 检查cur_best_graph，执行得到组合之后性能
            // 只要每一个子块都有数据，那就是每一个子块都是可以成功执行的
            bool can_be_execute = true;

            for (unsigned long sub_matrix_id = 0; sub_matrix_id < gflops_of_each_sub_matrix.size(); sub_matrix_id++)
            {
                if (gflops_of_each_sub_matrix[sub_matrix_id] == 0)
                {
                    // 有子块是执行失败的
                    can_be_execute = false;
                    break;
                }
            }

            // 如果执行才执行
            if (can_be_execute == true)
            {
                float cur_iter_gflops = 0;
                float cur_iter_time = 999999999;

                execute_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template(&cur_best_graph, cur_gflops, cur_time, 2);
            }

            cout << "find_best_graph_of_white_list_strategy3: local best gflops:" << cur_gflops << ", local best time:" << cur_time << endl;

            // 和全局的大小比较
            if (cur_gflops > best_gflops)
            {
                // 析构全局图，然后替换全局图
                // 这个析构是可以忽略空指针和控图的
                del_param_of_total_exe_graph_and_strategy_graph_safely(&best_graph);
                
                // 给最优子图重新赋值
                best_graph = cur_best_graph;
                
                best_gflops = cur_gflops;
                best_time = cur_time;

                write_graph_structure_and_performance_to_file(best_gflops, best_time, best_graph, string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source/best_result_of_strategy3");
            }
            else
            {
                // 析构和当前的图
                del_param_of_total_exe_graph_and_strategy_graph_safely(&cur_best_graph);
            }

            // 析构矩阵，
            assert(matrix != NULL);

            memory_garbage_manager_t mem_manager;
            delete_sparse_struct_t(&mem_manager, matrix);

            // 如果时间不够用了，但是标记还是继续
            if (search_strategy_ptr != NULL && is_continue_search == true)
            {
                is_continue_search = continue_search(search_strategy_ptr);
            }

            if (is_continue_search == false)
            {
                break;
            }
        }

        if (is_continue_search == false)
        {
            break;
        }
    }

    return best_graph;
}

dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t find_best_graph_of_white_list_strategy(exe_begin_memory_cache_input_file_param_t input_matrix_node, float &best_time, float &best_gflops, search_strategy_t* search_strategy_ptr, shared_ptr<machine_learning_data_set_collector> data_set_collector)
{
    // 检查一下输入的节点是不是正确
    assert(input_matrix_node.col_index_cache.size() > 0);
    assert(input_matrix_node.col_index_cache.size() == input_matrix_node.row_index_cache.size());

    assert(input_matrix_node.val_data_type == DOUBLE || input_matrix_node.val_data_type == FLOAT);

    if (input_matrix_node.val_data_type == DOUBLE)
    {
        assert(input_matrix_node.col_index_cache.size() == input_matrix_node.double_val_cache.size());
        assert(input_matrix_node.float_val_cache.size() == 0);
    }
    else
    {
        assert(input_matrix_node.col_index_cache.size() == input_matrix_node.float_val_cache.size());
        assert(input_matrix_node.double_val_cache.size() == 0);
    }

    // 如果存在数据集，那数据集即将插入表项应该不存在积累的数据
    if (data_set_collector != NULL)
    {
        // cout << "find_best_graph_of_white_list_strategy: need to collect ml data" << endl;
        assert((data_set_collector->accu_dense_graph_node_type_vec.size()) == 0);
        assert(data_set_collector->accu_dense_param_strategy_type_vec.size() == 0);
        assert(data_set_collector->accu_compressed_sub_graph_node_type_vec.size() == 0);
        assert(data_set_collector->accu_compressed_sub_param_strategy_type_vec.size() == 0);
        assert(data_set_collector->accu_dense_param_vec.size() == 0);
        assert(data_set_collector->accu_compressed_param_vec.size() == 0);
    }

    // 当前的最优时间
    float global_best_gflops = 0;
    float global_best_time = 99999999999;

    dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t total_best_graph;

    // 第一个策略用八分之一的时间，剩下的两个策略按照1：4的时间分配
    float cur_best_gflops1 = 0;
    float cur_best_time1 = 99999999999;
    dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t best_graph1;

    // 如果是需要数据集的情况，那就需要清除所有下一个要插入的表项已经积累的数据
    if (data_set_collector != NULL)
    {
        data_set_collector->clear_all_accu_info();
    }

    if (search_strategy_ptr != NULL)
    {
        // 第一个策略，设定时间
        float search_time_limit1 = search_strategy_ptr->total_allow_search_time / 10;

        search_strategy_t search_strategy1 = init_search_strategy(search_strategy_ptr->struggle_step_num, search_time_limit1);
        
        // 根据要不要收集数据，来使用不同的方法
        // 不管数据集指针是不是NULL，都传入处理
        best_graph1 = find_best_graph_of_white_list_strategy1(input_matrix_node, cur_best_time1, cur_best_gflops1, &search_strategy1, data_set_collector);
    }
    else
    {
        best_graph1 = find_best_graph_of_white_list_strategy1(input_matrix_node, cur_best_time1, cur_best_gflops1, NULL, data_set_collector);
    }


    // 如果已经搜到了
    if (cur_best_gflops1 > 0 && cur_best_gflops1 > global_best_gflops)
    {
        global_best_gflops = cur_best_gflops1;
        global_best_time = cur_best_time1;
        
        // 析构老的最优子图
        del_param_of_total_exe_graph_and_strategy_graph_safely(&total_best_graph);

        total_best_graph = best_graph1;
    }
    else 
    {
        del_param_of_total_exe_graph_and_strategy_graph_safely(&best_graph1);
    }

    float cur_best_gflops2 = 0;
    float cur_best_time2 = 99999999999;
    dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t best_graph2;

    // 清空当前表项的累积数据
    if (data_set_collector != NULL)
    {
        data_set_collector->clear_all_accu_info();
    }
    
    if (search_strategy_ptr != NULL)
    {
        // 第二个策略
        // 用剩余的时间的一半来处理
        // 计算当前的时间
        struct timeval end_time;

        gettimeofday(&(end_time), NULL);

        float time_use = (float)(end_time.tv_sec - search_strategy_ptr->begin_time.tv_sec) + (float)(end_time.tv_usec - search_strategy_ptr->begin_time.tv_usec)/1000000.0;

        time_use = time_use * 0.95;

        if (time_use >= search_strategy_ptr->total_allow_search_time)
        {
            // 这里代表时间用完了，直接返回
            // 将最优结果拷贝出来
            best_time = global_best_time;
            best_gflops = global_best_gflops;
            
            // 返回
            return total_best_graph;
        }

        float search_time_limit2 = (search_strategy_ptr->total_allow_search_time - time_use) / 4;

        cout << "find_best_graph_of_white_list_strategy: search_time_limit2: " << search_time_limit2 << endl;

        search_strategy_t search_strategy2 = init_search_strategy(search_strategy_ptr->struggle_step_num, search_time_limit2);
        
        // 开始搜索，通过判断有没有数据集指针来判断是不是要收集数据集
        best_graph2 = find_best_graph_of_white_list_strategy2(input_matrix_node, cur_best_time2, cur_best_gflops2, &search_strategy2, data_set_collector);
    }
    else
    {
        best_graph2 = find_best_graph_of_white_list_strategy2(input_matrix_node, cur_best_time2, cur_best_gflops2, NULL, data_set_collector);
    }
    
    // 如果已经搜到了
    if (cur_best_gflops2 > global_best_gflops)
    {
        global_best_gflops = cur_best_gflops2;
        global_best_time = cur_best_time2;
        
        // 析构老的最优子图
        del_param_of_total_exe_graph_and_strategy_graph_safely(&total_best_graph);

        total_best_graph = best_graph2;
    }
    else 
    {
        del_param_of_total_exe_graph_and_strategy_graph_safely(&best_graph2);
    }

    float cur_best_gflops3 = 0;
    float cur_best_time3 = 99999999999;
    dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t best_graph3;

    // 清空当前表项的累积数据
    if (data_set_collector != NULL)
    {
        data_set_collector->clear_all_accu_info();
    }

    if (search_strategy_ptr != NULL)
    {
        // 用剩余的时间的一半来处理
        // 计算当前的时间
        struct timeval end_time;

        gettimeofday(&(end_time), NULL);

        float time_use = (double)(end_time.tv_sec - search_strategy_ptr->begin_time.tv_sec) + (float)(end_time.tv_usec - search_strategy_ptr->begin_time.tv_usec)/1000000.0;

        time_use = time_use * 0.95;

        if (time_use >= search_strategy_ptr->total_allow_search_time)
        {
            // 这里代表时间用完了，直接返回
            // 将最优结果拷贝出来
            best_time = global_best_time;
            best_gflops = global_best_gflops;
            
            // 返回
            return total_best_graph;
        }

        float search_time_limit3 = search_strategy_ptr->total_allow_search_time - time_use;

        cout << "find_best_graph_of_white_list_strategy: search_time_limit3: " << search_time_limit3 << endl;

        search_strategy_t search_strategy3 = init_search_strategy(search_strategy_ptr->struggle_step_num, search_time_limit3);

        // 开始搜索
        best_graph3 = find_best_graph_of_white_list_strategy3(input_matrix_node, cur_best_time3, cur_best_gflops3, &search_strategy3, data_set_collector);
    }
    else
    {
        best_graph3 = find_best_graph_of_white_list_strategy3(input_matrix_node, cur_best_time3, cur_best_gflops3, NULL, data_set_collector);
    }    

    // 如果已经搜到了
    // 搜到的性能肯定是大于0的
    assert(cur_best_gflops3 > 0);
    
    if (cur_best_gflops3 > global_best_gflops)
    {
        global_best_gflops = cur_best_gflops3;
        global_best_time = cur_best_time3;
        
        // 析构老的最优子图
        del_param_of_total_exe_graph_and_strategy_graph_safely(&total_best_graph);

        total_best_graph = best_graph3;
    }
    else 
    {
        del_param_of_total_exe_graph_and_strategy_graph_safely(&best_graph3);
    }

    // 将最优结果拷贝出来
    best_time = global_best_time;
    best_gflops = global_best_gflops;
    
    // 返回
    return total_best_graph;
}

sparse_struct_t* get_matrix_from_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template(dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t* total_graph)
{
    assert(total_graph != NULL);
    // cout << "get_matrix_from_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template" << endl;
    // 稠密视图的所有参数都是存在的
    assert(total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph.exe_node_vec.size() == total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy.param_strategy_vec.size());

    for (unsigned long i = 0; i < total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph.exe_node_vec.size(); i++)
    {
        assert(total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph.exe_node_vec[i].param != NULL);
        assert(total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph.exe_node_vec[i].param == total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy.param_strategy_vec[i].param);
        assert(total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy.param_strategy_vec[i].param_strategy != NULL);
        assert(total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph.exe_node_vec[i].type == total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy.param_strategy_vec[i].node_type);
    }

    // 重置稠密视图的子图
    reset_exe_node_param_and_param_strategy_of_sub_graph(&(total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph), &(total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy));

    // 从稠密视图中执行获得矩阵
    sparse_struct_t* matrix = execute_dense_matrix_exe_graph_with_param_strategy(&(total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph), &(total_graph->dense_sub_graph_and_param_strategy.dense_sub_graph_param_strategy));

    assert(matrix != NULL);

    // 子块的数量和子图的数量相一致
    assert(matrix->block_coor_table.item_arr.size() == total_graph->compressed_sub_block_exe_graph_and_template_vec.size());
    assert(matrix->coo_col_index_cache != NULL && matrix->coo_row_index_cache != NULL  && matrix->coo_value_cache != NULL);

    // 如果子图需要被执行，必须保证矩阵是刚刚执行过的
    for (unsigned long i = 0; i < total_graph->compressed_sub_block_exe_graph_and_template_vec.size(); i++)
    {
        // 对应子块存在并且已经被压缩
        assert(matrix->block_coor_table.item_arr[i] != NULL);
        assert(matrix->block_coor_table.item_arr[i]->compressed_block_ptr != NULL);

        // 检查对应的子图指针是不是都在
        assert(total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph.exe_node_vec.size() == total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy.param_strategy_vec.size());

        for (unsigned long j = 0; j < total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph.exe_node_vec.size(); j++)
        {
            assert(total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph.exe_node_vec[j].param != NULL);
            assert(total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy.param_strategy_vec[j].param != NULL);
            assert(total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph.exe_node_vec[j].param == total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy.param_strategy_vec[j].param);
            assert(total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy.param_strategy_vec[j].param_strategy != NULL);
            assert(total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph.exe_node_vec[j].type == total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy.param_strategy_vec[j].node_type);
        }

        // 在执行之前进行参数的重置
        reset_exe_node_param_and_param_strategy_of_sub_graph(&(total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph), &(total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy));

        // 执行对应的子块
        cout << "get_matrix_from_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template: execute sub matrix:" << i << endl;
        execute_sub_matrix_exe_graph_with_param_strategy(matrix, i, &(total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph), &(total_graph->compressed_sub_block_exe_graph_and_template_vec[i].sub_graph_param_strategy));
    }

    // 返回
    return matrix;
}

void execute_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template(dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t* total_graph, float& gflops, float& time, int repeat_time)
{
    // cout << "execute_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template" << endl;
    assert(repeat_time > 0);
    // 执行之后获得对应的矩阵的指针，这里的total_graph是拷贝传值，这会导致在函数外，图的参数指针永远都是那个最早的版本，从而导致对同一个参数指针的多次reset。
    sparse_struct_t* matrix = get_matrix_from_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template(total_graph);

    assert(matrix != NULL);

    // 所有的子块被都有压缩，并且都已经进行了充分地分块
    assert(matrix->block_coor_table.item_arr.size() == total_graph->compressed_sub_block_exe_graph_and_template_vec.size());

    for (unsigned long i = 0; i < matrix->block_coor_table.item_arr.size(); i++)
    {
        assert(matrix->block_coor_table.item_arr[i] != NULL);
        assert(matrix->block_coor_table.item_arr[i]->compressed_block_ptr != NULL);

        // 执行缺省的分块，保证全部分到TLB层次
        execute_default_div_to_complete_each_level_blocking(matrix, i);

        assert(matrix->block_coor_table.item_arr[i]->compressed_block_ptr->read_index.size() == 7);

        // 模板是确实存在的
        assert(total_graph->compressed_sub_block_exe_graph_and_template_vec[i].temp_node.template_param != NULL);
    }

    // 然后为每一个子块增加一个模板
    // 创造一个操作管理器
    operator_manager_t* op_manager = init_op_manager(matrix);
    code_builder_t* builder = init_code_builder(op_manager);

    // cout << "execute_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template: finish init code_builder" << endl;

    // 执行模板节点，构造code_builder
    for (unsigned long i = 0; i < matrix->block_coor_table.item_arr.size(); i++)
    {
        cout << "execute_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template: execute template node" << endl;
        // 执行对应模板节点
        execute_template_node_and_update_code_builder(builder, i, total_graph->compressed_sub_block_exe_graph_and_template_vec[i].temp_node);

        cout << "execute_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template: finish execute template node" << endl;

        assert(builder->template_vec[i] != NULL);
    }

    assert(matrix->block_coor_table.item_arr.size() == builder->template_vec.size());

    float best_gflops = 0;
    float best_time = 99999999999;

    // 执行多次，取最大值
    for (unsigned long i = 0; i < repeat_time; i++)
    {
        // 执行对应的代码
        bool is_success = execute_code_builder(builder, time, gflops, string(get_config()["ROOT_PATH_STR"].as_string()) + "/cuda_code", string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source", true);
        
        if (is_success == false)
        {
            gflops = 0;
            time = 999999999;
        }

        if (gflops > best_gflops)
        {
            best_gflops = gflops;
            best_time = time;
        }
    }

    gflops = best_gflops;
    time = best_time;

    // 析构整个code_builder
    memory_garbage_manager_t mem_manager;
    delete_code_builder(&mem_manager, builder);
}

void write_graph_structure_and_performance_to_file(float gflops, float time, dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t total_graph, string file_name)
{
    ofstream outfile(file_name, ios::out | ios::trunc);

    if (!outfile)
    {
        cout << "write_graph_structure_and_performance_to_file: fail to open the file" <<endl;
        assert(false);
    }

    // 先输出结构
    outfile << convert_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_to_string_safety(total_graph) << endl;

    // 然后输出性能
    outfile << "gflops:" << gflops << endl;
    outfile << "time:" << time << endl;

    outfile.close();
}