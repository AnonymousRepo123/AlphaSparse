#include "parameter_set_strategy.hpp"

dense_begin_memory_cache_input_file_direct_param_strategy_t get_begin_memory_cache_input_file_direct_param_strategy_from_coo_file(string file_name, data_type type)
{
    assert(type == DOUBLE || type == FLOAT);

    vector<float> float_val_vec;
    vector<double> double_val_vec;
    unsigned long max_col_index;
    unsigned long max_row_index;
    vector<unsigned long> col_index_vec;
    vector<unsigned long> row_index_vec;

    get_matrix_index_and_val_from_file(file_name, row_index_vec, col_index_vec, float_val_vec, double_val_vec, type, max_row_index, max_col_index);

    dense_begin_memory_cache_input_file_direct_param_strategy_t param;
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

param_strategy_of_sub_graph_t val_copy_from_old_param_strategy_of_sub_graph(param_strategy_of_sub_graph_t old_param_strategy_of_compressed_sub_graph)
{
    // 检查输入是不是正确
    assert(old_param_strategy_of_compressed_sub_graph.param_strategy_vec.size() > 0);

    for (unsigned long i = 0; i < old_param_strategy_of_compressed_sub_graph.param_strategy_vec.size(); i++)
    {
        assert(old_param_strategy_of_compressed_sub_graph.param_strategy_vec[i].param_strategy != NULL);
        assert(old_param_strategy_of_compressed_sub_graph.param_strategy_vec[i].param != NULL);
    }

    // 遍历所有的策略，新建策略参数，然后拷贝出来，最后返回
    param_strategy_of_sub_graph_t return_param_strategy_of_compressed_sub_graph;

    for (unsigned long i = 0; i < old_param_strategy_of_compressed_sub_graph.param_strategy_vec.size(); i++)
    {
        // 申请新的节点
        param_strategy_node_t new_node;
        param_strategy_node_t old_node = old_param_strategy_of_compressed_sub_graph.param_strategy_vec[i];

        // 执行一系列的拷贝，其中优化路径的参数在这个函数外面可能需要重新绑定
        new_node.node_type = old_node.node_type;
        new_node.param = old_node.param;
        new_node.strategy_type = old_node.strategy_type;

        if (new_node.strategy_type == COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY)
        {
            new_node.param_strategy = new compressed_row_padding_direct_param_strategy_t();

            compressed_row_padding_direct_param_strategy_t* new_param_strategy_ptr = (compressed_row_padding_direct_param_strategy_t *)new_node.param_strategy;
            compressed_row_padding_direct_param_strategy_t* old_param_strategy_ptr = (compressed_row_padding_direct_param_strategy_t *)old_node.param_strategy;
            
            new_param_strategy_ptr->multiply = old_param_strategy_ptr->multiply;
            new_param_strategy_ptr->padding_row_length = old_param_strategy_ptr->padding_row_length;
        }
        else if (new_node.strategy_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY)
        {
            new_node.param_strategy = new compressed_tblock_level_row_div_evenly_param_strategy_t();

            compressed_tblock_level_row_div_evenly_param_strategy_t* new_param_strategy_ptr = (compressed_tblock_level_row_div_evenly_param_strategy_t *)new_node.param_strategy;
            compressed_tblock_level_row_div_evenly_param_strategy_t* old_param_strategy_ptr = (compressed_tblock_level_row_div_evenly_param_strategy_t *)old_node.param_strategy;

            new_param_strategy_ptr->block_row_num = old_param_strategy_ptr->block_row_num;
        }
        else if (new_node.strategy_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY)
        {
            new_node.param_strategy = new compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t();

            compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t* new_param_strategy_ptr = (compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t *)new_node.param_strategy;
            compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t* old_param_strategy_ptr = (compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t *)old_node.param_strategy;

            new_param_strategy_ptr->nnz_low_bound = old_param_strategy_ptr->nnz_low_bound;
        }
        else if (new_node.strategy_type == COMPRESSED_TBLOCK_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
        {
            new_node.param_strategy = new compressed_tblock_level_col_div_fixed_param_strategy_t();

            compressed_tblock_level_col_div_fixed_param_strategy_t* new_param_strategy_ptr = (compressed_tblock_level_col_div_fixed_param_strategy_t *)new_node.param_strategy;
            compressed_tblock_level_col_div_fixed_param_strategy_t* old_param_strategy_ptr = (compressed_tblock_level_col_div_fixed_param_strategy_t *)old_node.param_strategy;

            new_param_strategy_ptr->col_block_nnz_num = old_param_strategy_ptr->col_block_nnz_num;
        }
        else if (new_node.strategy_type == COMPRESSED_WARP_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY)
        {
            new_node.param_strategy = new compressed_warp_level_row_div_evenly_param_strategy_t();

            compressed_warp_level_row_div_evenly_param_strategy_t* new_param_strategy_ptr = (compressed_warp_level_row_div_evenly_param_strategy_t *)new_node.param_strategy;
            compressed_warp_level_row_div_evenly_param_strategy_t* old_param_strategy_ptr = (compressed_warp_level_row_div_evenly_param_strategy_t *)old_node.param_strategy;

            new_param_strategy_ptr->warp_row_num_of_each_BLB = old_param_strategy_ptr->warp_row_num_of_each_BLB;
        }
        else if (new_node.strategy_type == COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
        {
            new_node.param_strategy = new compressed_warp_level_col_div_fixed_param_strategy_t();

            compressed_warp_level_col_div_fixed_param_strategy_t* new_param_strategy_ptr = (compressed_warp_level_col_div_fixed_param_strategy_t *)new_node.param_strategy;
            compressed_warp_level_col_div_fixed_param_strategy_t* old_param_strategy_ptr = (compressed_warp_level_col_div_fixed_param_strategy_t *)old_node.param_strategy;

            new_param_strategy_ptr->col_block_nnz_num = old_param_strategy_ptr->col_block_nnz_num;
        }
        else if (new_node.strategy_type == COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY)
        {
            new_node.param_strategy = new compressed_thread_level_row_div_none_param_strategy_t();

            compressed_thread_level_row_div_none_param_strategy_t* new_param_strategy_ptr = (compressed_thread_level_row_div_none_param_strategy_t *)new_node.param_strategy;
            compressed_thread_level_row_div_none_param_strategy_t* old_param_strategy_ptr = (compressed_thread_level_row_div_none_param_strategy_t *)old_node.param_strategy;
        }
        else if (new_node.strategy_type == COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
        {
            new_node.param_strategy = new compressed_thread_level_col_div_fixed_param_strategy_t();
            
            compressed_thread_level_col_div_fixed_param_strategy_t* new_param_strategy_ptr = (compressed_thread_level_col_div_fixed_param_strategy_t *)new_node.param_strategy;
            compressed_thread_level_col_div_fixed_param_strategy_t* old_param_strategy_ptr = (compressed_thread_level_col_div_fixed_param_strategy_t *)old_node.param_strategy;

            new_param_strategy_ptr->col_block_nnz_num = old_param_strategy_ptr->col_block_nnz_num;
        }
        else if (new_node.strategy_type == COMPRESSED_THREAD_LEVEL_NNZ_DIV_DIRECT_PARAM_STRATEGY)
        {
            new_node.param_strategy = new compressed_thread_level_nnz_div_direct_param_strategy_t();
            
            compressed_thread_level_nnz_div_direct_param_strategy_t* new_param_strategy_ptr = (compressed_thread_level_nnz_div_direct_param_strategy_t *)new_node.param_strategy;
            compressed_thread_level_nnz_div_direct_param_strategy_t* old_param_strategy_ptr = (compressed_thread_level_nnz_div_direct_param_strategy_t *)old_node.param_strategy;

            new_param_strategy_ptr->block_nnz_num = old_param_strategy_ptr->block_nnz_num;
        }
        else if (new_node.strategy_type == DENSE_ROW_COARSE_SORT_FIXED_PARAM_STRATEGY)
        {
            new_node.param_strategy = new dense_row_coarse_sort_fixed_param_strategy_t();
            
            dense_row_coarse_sort_fixed_param_strategy_t* new_param_strategy_ptr = (dense_row_coarse_sort_fixed_param_strategy_t*)new_node.param_strategy;
            dense_row_coarse_sort_fixed_param_strategy_t* old_param_strategy_ptr = (dense_row_coarse_sort_fixed_param_strategy_t*)old_node.param_strategy;

            new_param_strategy_ptr->row_nnz_low_bound_step_size = old_param_strategy_ptr->row_nnz_low_bound_step_size;
        }
        else if (new_node.strategy_type == DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY)
        {
            new_node.param_strategy = new dense_begin_memory_cache_input_file_direct_param_strategy_t();

            dense_begin_memory_cache_input_file_direct_param_strategy_t* new_param_strategy_ptr = (dense_begin_memory_cache_input_file_direct_param_strategy_t *)new_node.param_strategy;
            dense_begin_memory_cache_input_file_direct_param_strategy_t* old_param_strategy_ptr = (dense_begin_memory_cache_input_file_direct_param_strategy_t *)old_node.param_strategy;

            new_param_strategy_ptr->col_index_cache = old_param_strategy_ptr->col_index_cache;
            new_param_strategy_ptr->col_index_max = old_param_strategy_ptr->col_index_max;
            new_param_strategy_ptr->double_val_cache = old_param_strategy_ptr->double_val_cache;
            new_param_strategy_ptr->float_val_cache = old_param_strategy_ptr->float_val_cache;
            new_param_strategy_ptr->row_index_cache = old_param_strategy_ptr->row_index_cache;
            new_param_strategy_ptr->row_index_max = old_param_strategy_ptr->row_index_max;
            new_param_strategy_ptr->val_data_type = old_param_strategy_ptr->val_data_type;
        }
        else if (new_node.strategy_type == COMPRESS_NONE_PARAM_STRATEGY)
        {
            new_node.param_strategy = new compress_none_param_strategy_t();

            // 但是实际上啥都不用做
            compress_none_param_strategy_t* new_param_strategy_ptr = (compress_none_param_strategy_t*)new_node.param_strategy;
            compress_none_param_strategy_t* old_param_strategy_ptr = (compress_none_param_strategy_t*)old_node.param_strategy;
        }
        else if (new_node.strategy_type == DENSE_ROW_DIV_ACC_TO_EXPONENTIAL_INCREASE_ROW_NNZ_PARAM_STRATEGY)
        {
            new_node.param_strategy = new dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t();
            
            // 新的和旧的参数
            dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t* new_param_strategy_ptr = (dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t*)new_node.param_strategy;
            dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t* old_param_strategy_ptr = (dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t*)old_node.param_strategy;

            new_param_strategy_ptr->expansion_rate = old_param_strategy_ptr->expansion_rate;
            new_param_strategy_ptr->lowest_nnz_bound_of_row = old_param_strategy_ptr->lowest_nnz_bound_of_row;
            new_param_strategy_ptr->highest_nnz_bound_of_row = old_param_strategy_ptr->highest_nnz_bound_of_row;
            new_param_strategy_ptr->sub_dense_block_id = old_param_strategy_ptr->sub_dense_block_id;
        }
        else
        {
            // 当前策略不被支持
            cout << "val_copy_from_old_param_strategy_of_sub_graph: param strategy type is not supported" << endl;
            assert(false);
        }

        // 将新的策略节点放到子图的策略骨架中
        return_param_strategy_of_compressed_sub_graph.param_strategy_vec.push_back(new_node);
    }
    
    return return_param_strategy_of_compressed_sub_graph;
}

void execute_compressed_row_padding_direct_param_strategy(compressed_row_padding_direct_param_strategy_t* param_strategy, exe_compress_row_padding_param_t* param, sparse_struct_t* matrix, unsigned long sub_matrix_id)
{
    // 当前应该仅仅完成了压缩
    assert(matrix != NULL);
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);

    // cout << "param_strategy->multiply:" << param_strategy->multiply << ",param_strategy->padding_row_length:" << param_strategy->padding_row_length << endl;

    assert(param_strategy->multiply > 0 && param_strategy->padding_row_length > 0);

    // 仅仅完成了压缩
    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;
    
    assert(param_strategy != NULL && param != NULL);

    // 给对应的节点赋值
    param->multiply = param_strategy->multiply;
    param->padding_row_length = param_strategy->padding_row_length;
}

// 执行等长的行条带分块
void execute_compressed_tblock_level_row_div_evenly_param_strategy(compressed_tblock_level_row_div_evenly_param_strategy_t* param_strategy, exe_compress_tblock_level_row_div_param_t* param, sparse_struct_t* matrix, unsigned long sub_matrix_id)
{
    assert(matrix != NULL);
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);
    // cout << "param_strategy->block_row_num:" << param_strategy->block_row_num << ",param->row_num_of_each_BLB.size():" << param->row_num_of_each_BLB.size() << endl;
    assert(param_strategy != NULL && param_strategy->block_row_num > 0 && param->row_num_of_each_BLB.size() == 0);

    // 仅仅完成了压缩
    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    assert(param_strategy != NULL && param != NULL);

    // 计算当前压缩子块的行数量
    assert(compressed_block_ptr->read_index[0]->max_row_index >= compressed_block_ptr->read_index[0]->min_row_index);

    unsigned long sub_matrix_row_num = compressed_block_ptr->read_index[0]->max_row_index - compressed_block_ptr->read_index[0]->min_row_index + 1;

    vector<unsigned long> block_row_num_vec = row_block_size_of_a_sub_matrix_by_fixed_div(sub_matrix_row_num, param_strategy->block_row_num);

    // 将参数放到对应的节点中
    for (auto row_num : block_row_num_vec)
    {
        param->row_num_of_each_BLB.push_back(row_num);
    }
}

void execute_compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy(compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t* param_strategy, exe_compress_tblock_level_row_div_param_t* param, sparse_struct_t* matrix, unsigned long sub_matrix_id)
{
    assert(matrix != NULL);
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);
    assert(param_strategy != NULL && param_strategy->nnz_low_bound > 0 && param->row_num_of_each_BLB.size() == 0);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 获取当前子块的非零元数量
    vector<unsigned long> row_nnz_vec = get_nnz_of_each_row_in_compressed_sub_matrix(compressed_block_ptr);

    // 获取分块结果
    vector<unsigned long> block_row_num_vec = row_block_size_of_a_sub_matrix_by_nnz_low_bound(row_nnz_vec, param_strategy->nnz_low_bound);

    // 将分块结果赋值给对应的节点
    for (auto row_num : block_row_num_vec)
    {
        param->row_num_of_each_BLB.push_back(row_num);
    }
}

// 按照纵切分
void execute_compressed_tblock_level_col_div_fixed_param_strategy(compressed_tblock_level_col_div_fixed_param_strategy_t* param_strategy, exe_compress_tblock_level_col_div_param_t* param, sparse_struct_t* matrix, unsigned long sub_matrix_id)
{
    assert(matrix != NULL);
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 2);
    assert(param_strategy != NULL && param_strategy->col_block_nnz_num > 0 && param->col_block_nnz_num_of_each_BLB.size() == 0);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 获取当前子块的非零元数量
    vector<unsigned long> row_nnz_vec = get_nnz_of_each_row_in_compressed_sub_matrix(compressed_block_ptr);

    // 获得每一行的纵分块的非零元数量
    vector<vector<unsigned int>> col_block_size_vec = col_block_size_of_each_row(row_nnz_vec, param_strategy->col_block_nnz_num);

    assert(row_nnz_vec.size() >= col_block_size_vec.size());

    param->col_block_nnz_num_of_each_BLB = col_block_size_vec;
}

void execute_compressed_warp_level_row_div_evenly_param_strategy(compressed_warp_level_row_div_evenly_param_strategy_t* param_strategy, exe_compress_warp_level_row_div_param_t* param, sparse_struct_t* matrix, unsigned long sub_matrix_id)
{
    assert(matrix != NULL);
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(param_strategy != NULL && param_strategy->warp_row_num_of_each_BLB > 0 && param->row_num_of_each_WLB_in_BLB.size() == 0);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() <= 3);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;
    
    // 如果没有行分块就执行一个默认的行分块
    if (compressed_block_ptr->read_index.size() == 2)
    {
        // 执行默认的行分块
        default_sep_tblock_level_row_csr(compressed_block_ptr);
    }
    
    assert(matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 3);
    index_of_compress_block_t* block_level_index = compressed_block_ptr->read_index[2];

    // 获取每一个BLB子块的行数量，BLB块中的行在索引上可能并不是连续的
    vector<unsigned long> row_num_of_each_BLB;

    for (unsigned long BLB_id = 0; BLB_id < block_level_index->block_num; BLB_id++)
    {
        row_num_of_each_BLB.push_back(read_from_array_with_data_type(block_level_index->row_number_of_block_arr, block_level_index->data_type_of_row_number_of_block_arr, BLB_id));
    }

    assert(row_num_of_each_BLB.size() > 0);

    // 获取进一步分块的数组
    vector<vector<unsigned int>> row_num_of_each_sub_block = row_block_size_of_each_sub_block_by_fixed_div(row_num_of_each_BLB, param_strategy->warp_row_num_of_each_BLB);
    
    assert(row_num_of_each_sub_block.size() == row_num_of_each_BLB.size());

    param->row_num_of_each_WLB_in_BLB = row_num_of_each_sub_block;
}


void execute_compressed_warp_level_col_div_fixed_param_strategy(compressed_warp_level_col_div_fixed_param_strategy_t* param_strategy, exe_compress_warp_level_col_div_param_t* param, sparse_struct_t* matrix, unsigned long sub_matrix_id)
{
    assert(matrix != NULL);
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(param_strategy != NULL && param != NULL && param_strategy->col_block_nnz_num > 0 && param->col_num_of_WLB_in_each_parent_row_block_or_BLB.size() == 0);
    
    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    assert(compressed_block_ptr->read_index.size() <= 3);

    // 如果没有执行过BLB切分，那就执行一个默认的BLB分块
    if (compressed_block_ptr->read_index.size() == 2)
    {
        // 默认的block分块
        default_sep_tblock_level_row_csr(compressed_block_ptr);
    }

    // 仅仅经过了BLB切分
    assert(compressed_block_ptr->read_index.size() == 3);
    index_of_compress_block_t* block_level_index = compressed_block_ptr->read_index[2];

    // 做一个检查，禁止在BLB层次进行纵分块，查看BLB的首行行号，
    for (unsigned long i = 0; i < block_level_index->block_num - 1; i++)
    {
        unsigned long cur_BLB_first_row_index = read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, i);
        unsigned long next_BLB_first_row_index = read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, i + 1);
        
        // BLB的首行行号不能相等，并且递增
        assert(cur_BLB_first_row_index < next_BLB_first_row_index);
    }

    // 查看整个子块行非零元的数量
    vector<unsigned long> row_nnz_vec = get_nnz_of_each_row_in_compressed_sub_matrix(compressed_block_ptr);

    // 为每一个非空行执行列分块
    vector<vector<unsigned int>> col_block_size_vec = col_block_size_of_each_row_without_block_merge(row_nnz_vec, param_strategy->col_block_nnz_num);

    param->col_num_of_WLB_in_each_parent_row_block_or_BLB = col_block_size_vec;
}

void execute_compressed_thread_level_row_div_none_param_strategy(compressed_thread_level_row_div_none_param_strategy_t* param_strategy, exe_compress_thread_level_row_div_param_t* param, sparse_struct_t* matrix, unsigned long sub_matrix_id)
{
    // 必须执行完对应的
    assert(matrix != NULL);
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(param != NULL && param_strategy != NULL);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    // 最多只能执行到行分块，
    assert(compressed_block_ptr->read_index.size() <= 4);

    // 什么都不用做
}

void execute_compressed_thread_level_col_div_fixed_param_strategy(compressed_thread_level_col_div_fixed_param_strategy_t* param_strategy, exe_compress_thread_level_col_div_param_t* param, sparse_struct_t* matrix, unsigned long sub_matrix_id)
{
    assert(matrix != NULL);
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
    assert(param != NULL && param_strategy != NULL);
    assert(param->col_num_of_TLB_in_each_parent_block.size() == 0 && param_strategy->col_block_nnz_num > 0);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    assert(compressed_block_ptr->read_index.size() <= 4);

    // 因为需要获得WLB的块数量，如果缺少对应的分块就需要执行默认分块
    if (compressed_block_ptr->read_index.size() == 2)
    {
        // 执行BLB的默认分块
        default_sep_tblock_level_row_csr(compressed_block_ptr);
        assert(compressed_block_ptr->read_index.size() == 3);
    }

    // 默认的WLB分块
    if (compressed_block_ptr->read_index.size() == 3)
    {
        default_sep_warp_level_row_csr(compressed_block_ptr);
    }

    assert(compressed_block_ptr->read_index.size() == 4);

    unsigned long WLB_num = compressed_block_ptr->read_index[3]->block_num;

    assert(WLB_num > 0);

    // 设定TLB纵分块的参数
    vector<unsigned long> col_block_size_vec = col_div_of_TLB_global_fixed_col_size(WLB_num, param_strategy->col_block_nnz_num);
    
    assert(col_block_size_vec.size() == WLB_num);

    param->col_num_of_TLB_in_each_parent_block = col_block_size_vec;
}

void execute_compressed_thread_level_nnz_div_direct_param_strategy(compressed_thread_level_nnz_div_direct_param_strategy_t* param_strategy, exe_compress_thread_level_nnz_div_param_t* param, sparse_struct_t* matrix, unsigned long sub_matrix_id)
{
    // 之前不能执行任何的分块
    assert(param_strategy != NULL && param != NULL && matrix != NULL);
    assert(param_strategy->block_nnz_num > 0);
    assert(matrix->block_coor_table.item_arr.size() > sub_matrix_id);
    assert(matrix->block_coor_table.item_arr[sub_matrix_id] != NULL && matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

    compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

    assert(compressed_block_ptr->read_index.size() == 2);

    // 执行对应的参数设定
    param->TLB_nnz_num = param_strategy->block_nnz_num;
}

void execute_dense_row_coarse_sort_fixed_param_strategy(dense_row_coarse_sort_fixed_param_strategy_t* param_strategy, exe_dense_row_coarse_sort_param_t* param, sparse_struct_t *matrix)
{
    assert(param_strategy != NULL && param != NULL && matrix != NULL);

    // 之前不能排序，不能分块
    assert(matrix->is_sorted == false && matrix->is_blocked == false && matrix->is_compressed == false);
    assert(matrix->sorted_row_index == NULL && matrix->coo_value_cache != NULL && matrix->coo_col_index_cache != NULL && matrix->coo_row_index_cache != NULL);

    // 根据矩阵的最长非零元来处理
    // 最大非零元数量
    vector<unsigned long> row_nnz_vec = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);

    assert(row_nnz_vec.size() == matrix->dense_row_number);
    assert(row_nnz_vec.size() > 0);

    // 查看最大的行非零元数量
    unsigned long max_row_nnz = row_nnz_vec[0];
    // 查看最小的行非零元数量
    unsigned long min_row_nnz = row_nnz_vec[0];

    for (unsigned long i = 0; i < row_nnz_vec.size(); i++)
    {
        if (max_row_nnz < row_nnz_vec[i])
        {
            max_row_nnz = row_nnz_vec[i];
        }
        
        if (min_row_nnz > row_nnz_vec[i])
        {
            min_row_nnz = row_nnz_vec[i];
        }
    }

    // 要执行的粗粒度排序的行非零元下界
    vector<unsigned long> bin_nnz_range;

    // cout << "param_strategy->row_nnz_low_bound_step_size:" << param_strategy->row_nnz_low_bound_step_size << endl;

    // 如果可以容纳的行非零元宽度为仅仅为1，这个操作就退化成了排序操作，引入排序相关的操作，提高性能
    if (param_strategy->row_nnz_low_bound_step_size == 1)
    {
        bin_nnz_range = bin_row_nnz_low_bound_of_fixed_granularity_coar_sort(row_nnz_vec, 1);
    }
    else
    {
        for (unsigned long i = min_row_nnz; i <= max_row_nnz; i = i + param_strategy->row_nnz_low_bound_step_size)
        {
            if (i == min_row_nnz)
            {
                bin_nnz_range.push_back(0);
            }
            else
            {
                bin_nnz_range.push_back(i);
            }
        }
    }

    // 将对应的排序算好，给对应的图节点赋值
    param->bin_row_nnz_low_bound = bin_nnz_range;
}

void execute_dense_begin_memory_cache_input_file_direct_param_strategy(dense_begin_memory_cache_input_file_direct_param_strategy_t* param_strategy, exe_begin_memory_cache_input_file_param_t* param, sparse_struct_t* matrix)
{
    // 保证参数都是有的，但是矩阵是没的
    assert(param_strategy != NULL && param != NULL && matrix == NULL);

    // 检查一下参数，将策略的参数拷贝到节点的参数中
    assert(param_strategy->row_index_cache.size() > 0 && param_strategy->col_index_max > 0 && param_strategy->row_index_max > 0);
    assert(param_strategy->row_index_cache.size() == param_strategy->col_index_cache.size());

    assert(param_strategy->val_data_type == DOUBLE || param_strategy->val_data_type == FLOAT);

    // 检查值数组的缓存
    if (param_strategy->val_data_type == DOUBLE)
    {
        assert(param_strategy->double_val_cache.size() == param_strategy->row_index_cache.size());
        assert(param_strategy->float_val_cache.size() == 0);
    }

    if (param_strategy->val_data_type == FLOAT)
    {
        assert(param_strategy->float_val_cache.size() == param_strategy->row_index_cache.size());
        assert(param_strategy->double_val_cache.size() == 0);
    }

    // 将参数直接拷贝到节点中
    param->col_index_cache = param_strategy->col_index_cache;
    param->col_index_max = param_strategy->col_index_max;
    param->double_val_cache = param_strategy->double_val_cache;
    param->float_val_cache = param_strategy->float_val_cache;
    param->row_index_cache = param_strategy->row_index_cache;
    param->row_index_max = param_strategy->row_index_max;
    param->val_data_type = param_strategy->val_data_type;
}

void execute_compress_none_param_strategy(compress_none_param_strategy_t* param_strategy, exe_compress_param_t* param, sparse_struct_t* matrix)
{
    assert(param_strategy != NULL && param != NULL && matrix != NULL);
}

void execute_dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy(dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t* param_strategy, exe_dense_row_div_param_t* param, sparse_struct_t* matrix)
{
    assert(param_strategy != NULL && param != NULL && matrix != NULL);

    // 查看子块的编号是不是满足范围
    if (matrix->block_coor_table.item_arr.size() == 0)
    {
        assert(param_strategy->sub_dense_block_id == matrix->block_coor_table.item_arr.size());
    }

    if (matrix->block_coor_table.item_arr.size() > 0)
    {
        assert(param_strategy->sub_dense_block_id < matrix->block_coor_table.item_arr.size());
    }

    // 检查参数是否正确
    assert(param_strategy->expansion_rate > 0 && param_strategy->lowest_nnz_bound_of_row > 0);

    // 将参数覆盖到优化路径的节点
    param->dense_sub_block_id = param_strategy->sub_dense_block_id;

    // 子块的行非零元数量
    vector<unsigned long> row_nnz_vec;

    if (matrix->block_coor_table.item_arr.size() == 0)
    {
        // 整个矩阵的行非零元数量
        // vector<unsigned long> get_nnz_of_each_row_in_spec_range(void *row_index_arr, data_type data_type_of_row_index_arr, unsigned long begin_row_bound, unsigned long end_row_bound, unsigned long global_coo_start, unsigned long global_coo_end)
        row_nnz_vec = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);
        assert(row_nnz_vec.size() == matrix->dense_row_number);
    }
    else
    {
        // 获取一个子块的索引范围和非零元范围
        unsigned long row_begin_index = matrix->block_coor_table.item_arr[param->dense_sub_block_id]->min_dense_row_index;
        unsigned long row_end_index = matrix->block_coor_table.item_arr[param->dense_sub_block_id]->max_dense_row_index;
        unsigned long coo_begin_index = matrix->block_coor_table.item_arr[param->dense_sub_block_id]->begin_coo_index;
        unsigned long coo_end_index = matrix->block_coor_table.item_arr[param->dense_sub_block_id]->end_coo_index;

        assert(coo_end_index <= matrix->nnz - 1);
        assert(row_end_index <= matrix->dense_row_number - 1);

        row_nnz_vec = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, row_begin_index, row_end_index, coo_begin_index, coo_end_index);
        assert(row_nnz_vec.size() == row_end_index - row_begin_index + 1);
    }
    
    // 行非零元数量的区间是不包含上界的但是包含下界
    vector<unsigned long> row_div_position = row_div_position_acc_to_exponential_increase_row_nnz_range(row_nnz_vec, param_strategy->lowest_nnz_bound_of_row, param_strategy->highest_nnz_bound_of_row, param_strategy->expansion_rate);

    param->row_div_position = row_div_position;
}

void execute_param_strategy_node_of_dense_matrix(param_strategy_node_t* node, sparse_struct_t* matrix)
{
    assert(node != NULL);
    assert(node->param != NULL && node->param_strategy != NULL);

    if (node->strategy_type == DENSE_ROW_COARSE_SORT_FIXED_PARAM_STRATEGY)
    {
        assert(node->node_type == DENSE_ROW_COARSE_SORT);
        // 这里要求matrix必须存在
        assert(matrix != NULL);
        execute_dense_row_coarse_sort_fixed_param_strategy((dense_row_coarse_sort_fixed_param_strategy_t *)node->param_strategy, (exe_dense_row_coarse_sort_param_t *)node->param, matrix);
        assert(((exe_dense_row_coarse_sort_param_t *)node->param)->bin_row_nnz_low_bound.size() > 0);
        return;
    }

    // 执行对应的策略
    if (node->strategy_type == DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY)
    {
        assert(node->node_type == BEGIN_MEMORY_CACHE_INPUT_FILE);
        assert(matrix == NULL);
        execute_dense_begin_memory_cache_input_file_direct_param_strategy((dense_begin_memory_cache_input_file_direct_param_strategy *)node->param_strategy, (exe_begin_memory_cache_input_file_param_t *)node->param, matrix);
        assert(((exe_begin_memory_cache_input_file_param_t *)node->param)->col_index_cache.size() > 0);
        assert(((exe_begin_memory_cache_input_file_param_t *)node->param)->row_index_cache.size() > 0);
        return;
    }

    // 执行对应的调参策略
    if (node->strategy_type == COMPRESS_NONE_PARAM_STRATEGY)
    {
        assert(node->node_type == COMPRESS);
        assert(matrix != NULL);
        execute_compress_none_param_strategy((compress_none_param_strategy_t *)node->param_strategy, (exe_compress_param_t*)node->param, matrix);
        return;
    }

    // 执行对应的调参策略
    if (node->strategy_type == DENSE_ROW_DIV_ACC_TO_EXPONENTIAL_INCREASE_ROW_NNZ_PARAM_STRATEGY)
    {
        assert(node->node_type == DENSE_ROW_DIV);
        assert(matrix != NULL);
        execute_dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy((dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t *)node->param_strategy, (exe_dense_row_div_param_t *)node->param, matrix);
        return;
    }

    // 不支持参数设定
    cout << "execute_param_strategy_node_of_dense_matrix: strategy is not supported" << endl;
    assert(false);
}

void execute_param_strategy_node_of_sub_compressed_matrix(param_strategy_node_t* node, sparse_struct_t* matrix, unsigned long sub_matrix_id)
{
    assert(node != NULL && matrix != NULL);
    assert(node->param != NULL && node->param_strategy != NULL);
    assert(sub_matrix_id < matrix->block_coor_table.item_arr.size());
    
    // 执行所有的的参数设定
    if (node->strategy_type == COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY)
    {
        assert(node->node_type == COMPRESSED_ROW_PADDING);
        execute_compressed_row_padding_direct_param_strategy((compressed_row_padding_direct_param_strategy_t *)node->param_strategy, (exe_compress_row_padding_param_t *)node->param, matrix, sub_matrix_id);
        assert(((exe_compress_row_padding_param_t *)node->param)->multiply > 0);
        assert(((exe_compress_row_padding_param_t *)node->param)->padding_row_length > 0);
        return;
    }

    if (node->strategy_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY)
    {
        // 均匀的BLB行分块
        assert(node->node_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
        execute_compressed_tblock_level_row_div_evenly_param_strategy((compressed_tblock_level_row_div_evenly_param_strategy_t *)node->param_strategy, (exe_compress_tblock_level_row_div_param_t *)node->param, matrix, sub_matrix_id);
        assert(((exe_compress_tblock_level_row_div_param_t *)node->param)->row_num_of_each_BLB.size() > 0);
        return;
    }

    if (node->strategy_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY)
    {
        // 按照非零元数量的BLB行分块
        assert(node->node_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV);
        execute_compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy((compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t *)node->param_strategy, (exe_compress_tblock_level_row_div_param_t *)node->param, matrix, sub_matrix_id);
        assert(((exe_compress_tblock_level_row_div_param_t *)node->param)->row_num_of_each_BLB.size() > 0);
        return;
    }

    if (node->strategy_type == COMPRESSED_TBLOCK_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
    {
        // 执行固定长度的BLB分块
        assert(node->node_type == COMPRESSED_TBLOCK_LEVEL_COL_DIV);
        execute_compressed_tblock_level_col_div_fixed_param_strategy((compressed_tblock_level_col_div_fixed_param_strategy_t *)node->param_strategy, (exe_compress_tblock_level_col_div_param_t *)node->param, matrix, sub_matrix_id);
        assert(((exe_compress_tblock_level_col_div_param_t *)node->param)->col_block_nnz_num_of_each_BLB.size() > 0);
        return;
    }

    if (node->strategy_type == COMPRESSED_WARP_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY)
    {
        // 执行固定长度WLB行分块
        assert(node->node_type == COMPRESSED_WARP_LEVEL_ROW_DIV);
        execute_compressed_warp_level_row_div_evenly_param_strategy((compressed_warp_level_row_div_evenly_param_strategy_t *)node->param_strategy, (exe_compress_warp_level_row_div_param_t *)node->param, matrix, sub_matrix_id);
        assert(((exe_compress_warp_level_row_div_param_t *)node->param)->row_num_of_each_WLB_in_BLB.size() > 0);
        return;
    }

    if (node->strategy_type == COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
    {
        // 固定长度的WLB列分块
        assert(node->node_type == COMPRESSED_WARP_LEVEL_COL_DIV);
        execute_compressed_warp_level_col_div_fixed_param_strategy((compressed_warp_level_col_div_fixed_param_strategy_t *)node->param_strategy, (exe_compress_warp_level_col_div_param_t *)node->param, matrix, sub_matrix_id);
        assert(((exe_compress_warp_level_col_div_param_t *)node->param)->col_num_of_WLB_in_each_parent_row_block_or_BLB.size() > 0);
        return;
    }

    if (node->strategy_type == COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY)
    {
        // 一行一个TLB分块
        assert(node->node_type == COMPRESSED_THREAD_LEVEL_ROW_DIV);
        execute_compressed_thread_level_row_div_none_param_strategy((compressed_thread_level_row_div_none_param_strategy_t *)node->param_strategy, (exe_compress_thread_level_row_div_param_t *)node->param, matrix, sub_matrix_id);
        return;
    }

    if (node->strategy_type == COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
    {
        // TLB列分块
        assert(node->node_type == COMPRESSED_THREAD_LEVEL_COL_DIV);
        execute_compressed_thread_level_col_div_fixed_param_strategy((compressed_thread_level_col_div_fixed_param_strategy_t *)node->param_strategy, (exe_compress_thread_level_col_div_param_t *)node->param, matrix, sub_matrix_id);
        assert(((exe_compress_thread_level_col_div_param_t *)node->param)->col_num_of_TLB_in_each_parent_block.size() > 0);
        return;
    }

    if (node->strategy_type == COMPRESSED_THREAD_LEVEL_NNZ_DIV_DIRECT_PARAM_STRATEGY)
    {
        // TLB的非零元数量分块
        assert(node->node_type == COMPRESSED_THREAD_LEVEL_NNZ_DIV);
        execute_compressed_thread_level_nnz_div_direct_param_strategy((compressed_thread_level_nnz_div_direct_param_strategy_t *)node->param_strategy, (exe_compress_thread_level_nnz_div_param_t *)node->param, matrix, sub_matrix_id);
        assert(((exe_compress_thread_level_nnz_div_param_t *)node->param)->TLB_nnz_num > 0);
        return;
    }

    // 不支持参数设定
    cout << "execute_param_strategy_node_of_sub_compressed_matrix: strategy is not supported" << endl;
    assert(false);
}

void del_strategy_of_param_strategy_node(param_strategy_node_t* node)
{
    assert(node->param_strategy != NULL);
    
    // 根据类型执行对应的析构操作
    if (node->strategy_type == COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY)
    {
        delete (compressed_row_padding_direct_param_strategy_t *)node->param_strategy;
        node->param_strategy = NULL;
        return;
    }

    if (node->strategy_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY)
    {
        delete (compressed_tblock_level_row_div_evenly_param_strategy_t *)node->param_strategy;
        node->param_strategy = NULL;
        return;
    }

    if (node->strategy_type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY)
    {
        delete (compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t *)node->param_strategy;
        node->param_strategy = NULL;
        return;
    }

    if (node->strategy_type == COMPRESSED_TBLOCK_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
    {
        delete (compressed_tblock_level_col_div_fixed_param_strategy_t *)node->param_strategy;
        node->param_strategy = NULL;
        return;
    }

    if (node->strategy_type == COMPRESSED_WARP_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY)
    {
        delete (compressed_warp_level_row_div_evenly_param_strategy_t *)node->param_strategy;
        node->param_strategy = NULL;
        return;
    }

    if (node->strategy_type == COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
    {
        delete (compressed_warp_level_col_div_fixed_param_strategy_t *)node->param_strategy;
        node->param_strategy = NULL;
        return;
    }

    if (node->strategy_type == COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY)
    {
        delete (compressed_thread_level_row_div_none_param_strategy_t *)node->param_strategy;
        node->param_strategy = NULL;
        return;
    }

    if (node->strategy_type == COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
    {
        delete (compressed_thread_level_col_div_fixed_param_strategy_t *)node->param_strategy;
        node->param_strategy = NULL;
        return;
    }

    if (node->strategy_type == COMPRESSED_THREAD_LEVEL_NNZ_DIV_DIRECT_PARAM_STRATEGY)
    {
        delete (compressed_thread_level_nnz_div_direct_param_strategy_t *)node->param_strategy;
        node->param_strategy = NULL;
        return;
    }

    if (node->strategy_type == DENSE_ROW_COARSE_SORT_FIXED_PARAM_STRATEGY)
    {
        delete (dense_row_coarse_sort_fixed_param_strategy_t *)node->param_strategy;
        node->param_strategy = NULL;
        return;
    }

    if (node->strategy_type == DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY)
    {
        delete (dense_begin_memory_cache_input_file_direct_param_strategy_t *)node->param_strategy;
        node->param_strategy = NULL;
        return;
    }

    if (node->strategy_type == COMPRESS_NONE_PARAM_STRATEGY)
    {
        delete (compress_none_param_strategy_t *)node->param_strategy;
        node->param_strategy = NULL;
        return;
    }

    if (node->strategy_type == DENSE_ROW_DIV_ACC_TO_EXPONENTIAL_INCREASE_ROW_NNZ_PARAM_STRATEGY)
    {
        delete (dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t *)node->param_strategy;
        node->param_strategy = NULL;
        return;
    }

    cout << "del_strategy_of_param_strategy_node: strategy is not supported" << endl;
    assert(false);
}

param_strategy_node_t init_compressed_row_padding_direct_param_strategy(compressed_row_padding_direct_param_strategy_t param_strategy, exe_compress_row_padding_param_t* param)
{
    assert(param != NULL);

    param_strategy_node_t return_node;
    
    return_node.node_type = COMPRESSED_ROW_PADDING;
    return_node.strategy_type = COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY;
    return_node.param = param;
    
    compressed_row_padding_direct_param_strategy_t* param_strategy_ptr = new compressed_row_padding_direct_param_strategy_t();
    param_strategy_ptr->multiply = param_strategy.multiply;
    param_strategy_ptr->padding_row_length = param_strategy.padding_row_length;

    return_node.param_strategy = param_strategy_ptr;

    return return_node;
}

param_strategy_node_t init_compressed_tblock_level_row_div_evenly_param_strategy(compressed_tblock_level_row_div_evenly_param_strategy_t param_strategy, exe_compress_tblock_level_row_div_param_t* param)
{
    assert(param != NULL);

    param_strategy_node_t return_node;

    return_node.node_type = COMPRESSED_TBLOCK_LEVEL_ROW_DIV;
    return_node.strategy_type = COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY;
    return_node.param = param;

    compressed_tblock_level_row_div_evenly_param_strategy_t* param_strategy_ptr = new compressed_tblock_level_row_div_evenly_param_strategy_t();
    param_strategy_ptr->block_row_num = param_strategy.block_row_num;

    return_node.param_strategy = param_strategy_ptr;

    return return_node;
}

param_strategy_node_t init_compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy(compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t param_strategy, exe_compress_tblock_level_row_div_param_t* param)
{
    assert(param != NULL);

    param_strategy_node_t return_node;

    return_node.node_type = COMPRESSED_TBLOCK_LEVEL_ROW_DIV;
    return_node.strategy_type = COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY;
    return_node.param = param;

    compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t* param_strategy_ptr = new compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t();
    param_strategy_ptr->nnz_low_bound = param_strategy.nnz_low_bound;

    return_node.param_strategy = param_strategy_ptr;

    return return_node;
}

param_strategy_node_t init_compressed_tblock_level_col_div_fixed_param_strategy(compressed_tblock_level_col_div_fixed_param_strategy_t param_strategy, exe_compress_tblock_level_col_div_param_t* param)
{
    assert(param != NULL);

    param_strategy_node_t return_node;

    return_node.node_type = COMPRESSED_TBLOCK_LEVEL_COL_DIV;
    return_node.strategy_type = COMPRESSED_TBLOCK_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY;
    return_node.param = param;

    compressed_tblock_level_col_div_fixed_param_strategy_t* param_strategy_ptr = new compressed_tblock_level_col_div_fixed_param_strategy_t();
    param_strategy_ptr->col_block_nnz_num = param_strategy.col_block_nnz_num;

    return_node.param_strategy = param_strategy_ptr;
    
    return return_node;
}

param_strategy_node_t init_compressed_warp_level_row_div_evenly_param_strategy(compressed_warp_level_row_div_evenly_param_strategy_t param_strategy, exe_compress_warp_level_row_div_param_t* param)
{
    assert(param != NULL);
    
    param_strategy_node_t return_node;
    
    return_node.node_type = COMPRESSED_WARP_LEVEL_ROW_DIV;
    return_node.strategy_type = COMPRESSED_WARP_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY;
    return_node.param = param;
    
    compressed_warp_level_row_div_evenly_param_strategy_t* param_strategy_ptr = new compressed_warp_level_row_div_evenly_param_strategy_t();
    param_strategy_ptr->warp_row_num_of_each_BLB = param_strategy.warp_row_num_of_each_BLB;

    return_node.param_strategy = param_strategy_ptr;

    return return_node;
}

param_strategy_node_t init_compressed_warp_level_col_div_fixed_param_strategy(compressed_warp_level_col_div_fixed_param_strategy_t param_strategy, exe_compress_warp_level_col_div_param_t* param)
{
    assert(param != NULL);

    param_strategy_node_t return_node;

    return_node.node_type = COMPRESSED_WARP_LEVEL_COL_DIV;
    return_node.strategy_type = COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY;
    return_node.param = param;

    compressed_warp_level_col_div_fixed_param_strategy_t* param_strategy_ptr = new compressed_warp_level_col_div_fixed_param_strategy_t();
    param_strategy_ptr->col_block_nnz_num = param_strategy.col_block_nnz_num;

    return_node.param_strategy = param_strategy_ptr;

    return return_node;
}

param_strategy_node_t init_compressed_thread_level_row_div_none_param_strategy(compressed_thread_level_row_div_none_param_strategy_t param_strategy, exe_compress_thread_level_row_div_param_t* param)
{
    assert(param != NULL);

    param_strategy_node_t return_node;

    return_node.node_type = COMPRESSED_THREAD_LEVEL_ROW_DIV;
    return_node.strategy_type = COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY;
    return_node.param = param;

    compressed_thread_level_row_div_none_param_strategy_t* param_strategy_ptr = new compressed_thread_level_row_div_none_param_strategy_t();

    return_node.param_strategy = param_strategy_ptr;
    
    return return_node;
}

param_strategy_node_t init_compressed_thread_level_col_div_fixed_param_strategy(compressed_thread_level_col_div_fixed_param_strategy_t param_strategy, exe_compress_thread_level_col_div_param_t* param)
{
    assert(param != NULL);

    param_strategy_node_t return_node;

    return_node.node_type = COMPRESSED_THREAD_LEVEL_COL_DIV;
    return_node.strategy_type = COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY;
    return_node.param = param;

    compressed_thread_level_col_div_fixed_param_strategy_t* param_strategy_ptr = new compressed_thread_level_col_div_fixed_param_strategy_t();
    param_strategy_ptr->col_block_nnz_num = param_strategy.col_block_nnz_num;

    return_node.param_strategy = param_strategy_ptr;

    return return_node;
}

param_strategy_node_t init_compressed_thread_level_nnz_div_direct_param_strategy(compressed_thread_level_nnz_div_direct_param_strategy_t param_strategy, exe_compress_thread_level_nnz_div_param_t* param)
{
    assert(param != NULL);

    param_strategy_node_t return_node;

    return_node.node_type = COMPRESSED_THREAD_LEVEL_NNZ_DIV;
    return_node.strategy_type = COMPRESSED_THREAD_LEVEL_NNZ_DIV_DIRECT_PARAM_STRATEGY;
    return_node.param = param;

    compressed_thread_level_nnz_div_direct_param_strategy_t* param_strategy_ptr = new compressed_thread_level_nnz_div_direct_param_strategy_t();
    param_strategy_ptr->block_nnz_num = param_strategy.block_nnz_num;

    return_node.param_strategy = param_strategy_ptr;
    
    return return_node;
}

param_strategy_node_t init_dense_row_coarse_sort_fixed_param_strategy(dense_row_coarse_sort_fixed_param_strategy_t param_strategy, exe_dense_row_coarse_sort_param_t* param)
{
    assert(param != NULL);

    param_strategy_node_t return_node;

    return_node.node_type = DENSE_ROW_COARSE_SORT;
    return_node.strategy_type = DENSE_ROW_COARSE_SORT_FIXED_PARAM_STRATEGY;
    return_node.param = param;

    dense_row_coarse_sort_fixed_param_strategy_t* param_strategy_ptr = new dense_row_coarse_sort_fixed_param_strategy_t();
    param_strategy_ptr->row_nnz_low_bound_step_size = param_strategy.row_nnz_low_bound_step_size;

    return_node.param_strategy = param_strategy_ptr;

    return return_node;
}

param_strategy_node_t init_dense_begin_memory_cache_input_file_direct_param_strategy(dense_begin_memory_cache_input_file_direct_param_strategy_t param_strategy, exe_begin_memory_cache_input_file_param_t* param)
{
    assert(param != NULL);

    // 检查一下参数，将策略的参数拷贝到节点的参数中
    assert(param_strategy.row_index_cache.size() > 0 && param_strategy.col_index_max > 0 && param_strategy.row_index_max > 0);
    assert(param_strategy.row_index_cache.size() == param_strategy.col_index_cache.size());

    assert(param_strategy.val_data_type == DOUBLE || param_strategy.val_data_type == FLOAT);

    // 检查值数组的缓存
    if (param_strategy.val_data_type == DOUBLE)
    {
        assert(param_strategy.double_val_cache.size() == param_strategy.row_index_cache.size());
        assert(param_strategy.float_val_cache.size() == 0);
    }

    if (param_strategy.val_data_type == FLOAT)
    {
        assert(param_strategy.float_val_cache.size() == param_strategy.row_index_cache.size());
        assert(param_strategy.double_val_cache.size() == 0);
    }

    param_strategy_node_t return_node;

    return_node.node_type = BEGIN_MEMORY_CACHE_INPUT_FILE;
    return_node.strategy_type = DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY;
    return_node.param = param;

    dense_begin_memory_cache_input_file_direct_param_strategy_t* param_strategy_ptr = new dense_begin_memory_cache_input_file_direct_param_strategy_t();
    param_strategy_ptr->col_index_cache = param_strategy.col_index_cache;
    param_strategy_ptr->col_index_max = param_strategy.col_index_max;
    param_strategy_ptr->double_val_cache = param_strategy.double_val_cache;
    param_strategy_ptr->float_val_cache = param_strategy.float_val_cache;
    param_strategy_ptr->row_index_cache = param_strategy.row_index_cache;
    param_strategy_ptr->row_index_max = param_strategy.row_index_max;
    param_strategy_ptr->val_data_type = param_strategy.val_data_type;

    return_node.param_strategy = param_strategy_ptr;

    return return_node;
}

param_strategy_node_t init_compress_none_param_strategy(compress_none_param_strategy_t param_strategy, exe_compress_param_t* param)
{
    param_strategy_node_t return_node;
    assert(param != NULL);

    // 执行对应的赋值
    return_node.node_type = COMPRESS;
    return_node.strategy_type = COMPRESS_NONE_PARAM_STRATEGY;
    return_node.param = param;

    compress_none_param_strategy_t* param_strategy_ptr = new compress_none_param_strategy_t();
    
    return_node.param_strategy = param_strategy_ptr;
    
    return return_node;
}

param_strategy_node_t init_dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy(dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t param_strategy, exe_dense_row_div_param_t* param)
{
    param_strategy_node_t return_node;
    assert(param != NULL);

    // 执行对应的赋值
    return_node.node_type = DENSE_ROW_DIV;
    return_node.strategy_type = DENSE_ROW_DIV_ACC_TO_EXPONENTIAL_INCREASE_ROW_NNZ_PARAM_STRATEGY;
    return_node.param = param;

    // 检查参数
    assert(param_strategy.expansion_rate > 0);
    assert(param_strategy.lowest_nnz_bound_of_row > 0);

    dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t* param_strategy_ptr = new dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t();
    param_strategy_ptr->expansion_rate = param_strategy.expansion_rate;
    param_strategy_ptr->lowest_nnz_bound_of_row = param_strategy.lowest_nnz_bound_of_row;
    param_strategy_ptr->highest_nnz_bound_of_row = param_strategy.highest_nnz_bound_of_row;
    param_strategy_ptr->sub_dense_block_id = param_strategy.sub_dense_block_id;

    // cout << "init_dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy:" << param_strategy.sub_dense_block_id << endl;

    return_node.param_strategy = param_strategy_ptr;
    
    return return_node;
}

void del_strategy_of_param_strategy_node_in_sub_matrix(param_strategy_of_sub_graph_t* param_strategy_of_sub_matrix)
{
    assert(param_strategy_of_sub_matrix != NULL);

    for (unsigned long i = 0; i < param_strategy_of_sub_matrix->param_strategy_vec.size(); i++)
    {
        // 对应参数都是存在的
        assert(param_strategy_of_sub_matrix->param_strategy_vec[i].param != NULL);
        assert(param_strategy_of_sub_matrix->param_strategy_vec[i].param_strategy != NULL);
        
        del_strategy_of_param_strategy_node(&(param_strategy_of_sub_matrix->param_strategy_vec[i]));

        // 删完之后对应参数归零
        assert(param_strategy_of_sub_matrix->param_strategy_vec[i].param_strategy == NULL);
    }
}

string convert_strategy_param_to_string(void *param_strategy, exe_node_param_set_strategy type)
{
    assert(param_strategy != NULL);

    string return_str = "";

    if (type == COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY)
    {
        compressed_row_padding_direct_param_strategy_t* param_ptr = (compressed_row_padding_direct_param_strategy_t*)param_strategy;

        return_str = return_str + "{\n";

        return_str = return_str + "multiply:" + to_string(param_ptr->multiply) + "\n";
        return_str = return_str + "padding_row_length:" + to_string(param_ptr->padding_row_length) + "\n";

        return_str = return_str + "}\n";
    }
    else if (type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY)
    {
        compressed_tblock_level_row_div_evenly_param_strategy_t* param_ptr = (compressed_tblock_level_row_div_evenly_param_strategy_t*)param_strategy;

        return_str = return_str + "{\n";

        return_str = return_str + "block_row_num:" + to_string(param_ptr->block_row_num) + "\n";

        return_str = return_str + "}\n";
    }
    else if (type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY)
    {
        compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t* param_ptr = (compressed_tblock_level_row_div_acc_to_least_nnz_param_strategy_t*)param_strategy;

        return_str = return_str + "{\n";
        return_str = return_str + "nnz_low_bound:" + to_string(param_ptr->nnz_low_bound) + "\n";
        return_str = return_str + "}\n";
    }
    else if (type == COMPRESSED_TBLOCK_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
    {
        compressed_tblock_level_col_div_fixed_param_strategy_t* param_ptr = (compressed_tblock_level_col_div_fixed_param_strategy_t*)param_strategy;

        return_str = return_str + "{\n";
        return_str = return_str + "col_block_nnz_num:" + to_string(param_ptr->col_block_nnz_num) + "\n";
        return_str = return_str + "}\n";
    }
    else if (type == COMPRESSED_WARP_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY)
    {
        compressed_warp_level_row_div_evenly_param_strategy_t* param_ptr = (compressed_warp_level_row_div_evenly_param_strategy_t*)param_strategy;

        return_str = return_str + "{\n";
        return_str = return_str + "warp_row_num_of_each_BLB:" + to_string(param_ptr->warp_row_num_of_each_BLB) + "\n";
        return_str = return_str + "}\n";
    }
    else if (type == COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
    {
        compressed_warp_level_col_div_fixed_param_strategy_t* param_ptr = (compressed_warp_level_col_div_fixed_param_strategy_t *)param_strategy;
        
        return_str = return_str + "{\n";
        return_str = return_str + "col_block_nnz_num:" + to_string(param_ptr->col_block_nnz_num) + "\n";
        return_str = return_str + "}\n";
    }
    else if (type == COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY)
    {
        compressed_thread_level_row_div_none_param_strategy_t* param_ptr = (compressed_thread_level_row_div_none_param_strategy_t*) param_strategy;

        return_str = return_str + "{\n";
        return_str = return_str + "}\n";
    }
    else if (type == COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
    {
        compressed_thread_level_col_div_fixed_param_strategy_t* param_ptr = (compressed_thread_level_col_div_fixed_param_strategy_t*)param_strategy;

        return_str = return_str + "{\n";
        return_str = return_str + "col_block_nnz_num:" + to_string(param_ptr->col_block_nnz_num) + "\n";
        return_str = return_str + "}\n";
    }
    else if (type == COMPRESSED_THREAD_LEVEL_NNZ_DIV_DIRECT_PARAM_STRATEGY)
    {
        compressed_thread_level_nnz_div_direct_param_strategy_t* param_ptr = (compressed_thread_level_nnz_div_direct_param_strategy_t*)param_strategy;

        return_str = return_str + "{\n";
        return_str = return_str + "block_nnz_num:" + to_string(param_ptr->block_nnz_num) + "\n";
        return_str = return_str + "}\n";
    }
    else if (type == DENSE_ROW_COARSE_SORT_FIXED_PARAM_STRATEGY)
    {
        dense_row_coarse_sort_fixed_param_strategy_t* param_ptr = (dense_row_coarse_sort_fixed_param_strategy_t*)param_strategy;
        
        return_str = return_str + "{\n";
        return_str = return_str + "row_nnz_low_bound_step_size:" + to_string(param_ptr->row_nnz_low_bound_step_size) + "\n";
        return_str = return_str + "}\n";
    }
    else if (type == DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY)
    {
        dense_begin_memory_cache_input_file_direct_param_strategy_t* param_ptr = (dense_begin_memory_cache_input_file_direct_param_strategy_t*)param_strategy;

        assert(param_ptr->col_index_cache.size() == param_ptr->row_index_cache.size());

        return_str = return_str + "{\n";
        return_str = return_str + "nnz:" + to_string(param_ptr->row_index_cache.size()) + "\n";
        return_str = return_str + "data_type:" + convert_data_type_to_string(param_ptr->val_data_type) + "\n";
        return_str = return_str + "col_index_max:" + to_string(param_ptr->col_index_max) + "\n";
        return_str = return_str + "row_index_max:" + to_string(param_ptr->row_index_max) + "\n";
        return_str = return_str + "}\n";
    }
    else if (type == COMPRESS_NONE_PARAM_STRATEGY)
    {
        return_str = return_str + "";
    }
    else if (type == DENSE_ROW_DIV_ACC_TO_EXPONENTIAL_INCREASE_ROW_NNZ_PARAM_STRATEGY)
    {
        dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t* param_ptr = (dense_row_div_acc_to_exponential_increase_row_nnz_param_strategy_t*)param_strategy;
        
        assert(param_ptr->expansion_rate > 0 && param_ptr->lowest_nnz_bound_of_row > 0 && param_ptr->sub_dense_block_id >= 0);

        return_str = return_str + "{\n";
        return_str = return_str + "expansion_rate:" + to_string(param_ptr->expansion_rate) + "\n";
        return_str = return_str + "lowest_nnz_bound_of_row:" + to_string(param_ptr->lowest_nnz_bound_of_row) + "\n";
        return_str = return_str + "highest_nnz_bound_of_row:" + to_string(param_ptr->highest_nnz_bound_of_row) + "\n";
        return_str = return_str + "sub_dense_block_id:" + to_string(param_ptr->sub_dense_block_id) + "\n";
        return_str = return_str + "}\n";
    }
    else
    {
        cout << "convert_strategy_param_to_string: strategy type is not supported" << endl;
        assert(false);
    }

    return return_str;
}

string convert_param_set_strategy_to_string(exe_node_param_set_strategy type)
{
    if (type == COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY)
    {
        return "COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY";
    }
    else if (type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY)
    {
        return "COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY";
    }
    else if (type == COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY)
    {
        return "COMPRESSED_TBLOCK_LEVEL_ROW_DIV_ACC_TO_LEAST_NNZ_PARAM_STRATEGY";
    }
    else if (type == COMPRESSED_TBLOCK_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
    {
        return "COMPRESSED_TBLOCK_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY";
    }
    else if (type == COMPRESSED_WARP_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY)
    {
        return "COMPRESSED_WARP_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY";
    }
    else if (type == COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
    {
        return "COMPRESSED_WARP_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY";
    }
    else if (type == COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY)
    {
        return "COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY";
    }
    else if (type == COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY)
    {
        return "COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY";
    }
    else if (type == COMPRESSED_THREAD_LEVEL_NNZ_DIV_DIRECT_PARAM_STRATEGY)
    {
        return "COMPRESSED_THREAD_LEVEL_NNZ_DIV_DIRECT_PARAM_STRATEGY";
    }
    else if (type == DENSE_ROW_COARSE_SORT_FIXED_PARAM_STRATEGY)
    {
        return "DENSE_ROW_COARSE_SORT_FIXED_PARAM_STRATEGY";
    }
    else if (type == DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY)
    {
        return "DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY";
    }
    else if (type == COMPRESS_NONE_PARAM_STRATEGY)
    {
        return "COMPRESS_NONE_PARAM_STRATEGY";
    }
    else if (type == DENSE_ROW_DIV_ACC_TO_EXPONENTIAL_INCREASE_ROW_NNZ_PARAM_STRATEGY)
    {
        return "DENSE_ROW_DIV_ACC_TO_EXPONENTIAL_INCREASE_ROW_NNZ_PARAM_STRATEGY";
    }
    else
    {
        cout << "convert_strategy_param_to_string: strategy type is not supported" << endl;
        assert(false);
    }
}

string convert_stategy_node_to_string(param_strategy_node_t node)
{
    // 打印一个策略节点内的内容
    assert(node.param != NULL && node.param_strategy != NULL);

    string return_str = "";

    // 打印策略的类型和对应优化节点的类型
    return_str = return_str + "{\n";
    
    return_str = return_str + "strategy_type:" + convert_param_set_strategy_to_string(node.strategy_type) + "\n";

    return_str = return_str + "node_type:" + convert_exe_node_type_to_string(node.node_type) + "\n";

    return_str = return_str + convert_strategy_param_to_string(node.param_strategy, node.strategy_type);
    
    return_str = return_str + "}\n";

    return return_str;
}

string convert_all_stategy_node_of_sub_matrix_to_string(param_strategy_of_sub_graph_t strategy_skeleon_of_sub_matrix)
{
    assert(strategy_skeleon_of_sub_matrix.param_strategy_vec.size() > 0);

    string return_str = "";

    for (unsigned long i = 0; i < strategy_skeleon_of_sub_matrix.param_strategy_vec.size(); i++)
    {
        return_str = return_str + convert_stategy_node_to_string(strategy_skeleon_of_sub_matrix.param_strategy_vec[i]);
    }

    return return_str;
}