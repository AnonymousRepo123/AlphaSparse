#include "matrix_info.hpp"

matrix_info_t get_matrix_info_from_matrix_coo_file(string coo_file_name)
{
    matrix_info_t info;

    sparse_struct_t *matrix = init_sparse_struct_by_coo_file(coo_file_name, FLOAT);

    assert(matrix != NULL);

    info = get_matrix_info_from_sparse_matrix_ptr(matrix);

    // 收集完信息，准备删除matrix
    memory_garbage_manager_t mem_manager;

    delete_sparse_struct_t(&mem_manager, matrix);

    return info;
}


matrix_info_t get_matrix_info_from_sparse_matrix_ptr(sparse_struct_t* matrix)
{
    matrix_info_t info;
    assert(matrix != NULL);
    // 初步的几个指针都在
    assert(matrix->coo_col_index_cache != NULL && matrix->coo_row_index_cache != NULL && matrix->coo_value_cache != NULL);

    info.row_nnz = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);

    assert(info.row_nnz.size() == matrix->dense_row_number);

    info.col_num = matrix->dense_col_number;
    info.row_num = matrix->dense_row_number;
    info.nnz = matrix->nnz;

    // 查看row_nnz是不是升序或者降序排列
    // 查看是不是降序排列
    bool is_descending_sort = true;
    // 查看是不是升序排列
    bool is_ascending_sort = true;

    info.max_row_nnz = info.row_nnz[0];
    info.min_row_nnz = info.row_nnz[0];

    unsigned long row_nnz_sum = 0;

    for (unsigned long i = 0; i < info.row_nnz.size(); i++)
    {
        if (i < info.row_nnz.size() - 1)
        {
            // 如果前面的值大于后面的，那么就不可能是升序的
            if (info.row_nnz[i] > info.row_nnz[i + 1])
            {
                is_ascending_sort = false;
            }

            // 如果是后面的值大于前面的，那就不可能降序
            if (info.row_nnz[i + 1] > info.row_nnz[i])
            {
                is_descending_sort = false;
            }
        }

        if (info.row_nnz[i] < info.min_row_nnz)
        {
            info.min_row_nnz = info.row_nnz[i];
        }

        if (info.row_nnz[i] > info.max_row_nnz)
        {
            info.max_row_nnz = info.row_nnz[i];
        }
        
        row_nnz_sum = row_nnz_sum + info.row_nnz[i];
    }

    info.is_sorted = (is_ascending_sort || is_descending_sort);

    // 行非零元总和和nnz数量相同
    if (row_nnz_sum != info.nnz)
    {
        cout << "row_nnz_sum:" << row_nnz_sum << "info.nnz:" << info.nnz << endl;
        assert(false);
    }

    info.avg_row_nnz = row_nnz_sum / info.row_nnz.size();

    return info;
}

matrix_info_t get_global_matrix_info_from_input_node(exe_begin_memory_cache_input_file_param_t input_matrix_node_param)
{
    exe_graph_t graph;
    // 创造一个新的节点
    add_exe_begin_memory_cache_input_file_node_to_exe_graph(&graph, EXE_DENSE_SUB_GRAPH, input_matrix_node_param, 0, GRAPH_END);

    execute_graph_dense_part(&graph);

    assert(graph.dense_sub_graph.exe_node_vec.size() == 1);
    assert(graph.builder == NULL);
    assert(graph.op_manager != NULL);
    assert(graph.total_compressed_sub_graph.compressed_sub_graph_vec.size() == 0);

    sparse_struct_t* matrix = graph.op_manager->matrix;

    assert(matrix != NULL);

    matrix_info info = get_matrix_info_from_sparse_matrix_ptr(matrix);
    
    // 析构图的节点参数
    del_exe_node_param_of_dense_view_matrix(&(graph.dense_sub_graph));
    
    // 最后析构整个图
    memory_garbage_manager_t mem_manager;
    delete_op_manager(&mem_manager, graph.op_manager);
    
    return info;
}

// 获取一个子块的的基本信息
matrix_info_t get_sub_matrix_info_from_compressed_matrix_block(dense_block_table_item_t* sub_matrix)
{
    assert(sub_matrix != NULL);
    // 这个子块已经被压缩过
    assert(sub_matrix->compressed_block_ptr != NULL);

    matrix_info_t info;

    info.row_nnz = get_nnz_of_each_row_in_compressed_sub_matrix(sub_matrix->compressed_block_ptr);
    info.col_num = sub_matrix->compressed_block_ptr->read_index[0]->max_col_index - sub_matrix->compressed_block_ptr->read_index[0]->min_col_index + 1;
    info.row_num = sub_matrix->compressed_block_ptr->read_index[0]->max_row_index - sub_matrix->compressed_block_ptr->read_index[0]->min_row_index + 1;
    info.nnz = sub_matrix->compressed_block_ptr->size;

    assert(sub_matrix->compressed_block_ptr->size == sub_matrix->compressed_block_ptr->read_index[0]->length);
    assert(sub_matrix->compressed_block_ptr->read_index[0]->min_row_index == sub_matrix->compressed_block_ptr->read_index[1]->min_row_index);
    assert(sub_matrix->compressed_block_ptr->read_index[0]->max_row_index == sub_matrix->compressed_block_ptr->read_index[1]->max_row_index);

    info.max_row_nnz = info.row_nnz[0];
    info.min_row_nnz = info.row_nnz[0];

    // 将每一行的非零元数量加起来
    unsigned long row_nnz_sum = 0;

    // 查看排序的情况，排序有可能在外部已经排序外部，也可能一开始就自带排序
    // 看看是不是降序
    info.is_sorted = true;

    // 查看是不是降序排列
    bool is_descending_sort = true;
    // 查看是不是升序排列
    bool is_ascending_sort = true;

    // 遍历每一行的非零元数量，获取最大值最小值和平均值
    for (unsigned long i = 0; i < info.row_nnz.size(); i++)
    {
        if (info.row_nnz[i] > info.max_row_nnz)
        {
            info.max_row_nnz = info.row_nnz[i];
        }

        if (info.row_nnz[i] < info.min_row_nnz)
        {
            info.min_row_nnz = info.row_nnz[i];
        }

        row_nnz_sum = row_nnz_sum + info.row_nnz[i];

        if (i < info.row_nnz.size() - 1)
        {
            if (info.row_nnz[i] < info.row_nnz[i + 1])
            {
                // 这里说明肯定不是降序
                is_descending_sort = false;
            }

            if (info.row_nnz[i] > info.row_nnz[i + 1])
            {
                // 这里说明肯定不是升序
                is_ascending_sort = false;
            }
        }
    }

    info.is_sorted = (is_ascending_sort || is_descending_sort);

    assert(row_nnz_sum == info.nnz);

    // 行平均非零元数量
    unsigned long avg_row_nnz = row_nnz_sum / info.row_nnz.size();

    info.avg_row_nnz = avg_row_nnz;

    return info;
}