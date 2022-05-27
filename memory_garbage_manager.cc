#include "memory_garbage_manager.hpp"

void delete_sparse_struct_t(memory_garbage_manager_t *mem_manager, sparse_struct_t *matrix)
{
    // cout << "delete_sparse_struct_t" << endl;

    // 所有子块的所有索引的is_sort_arr指针都是NULL
    for (unsigned long i = 0; i < matrix->block_coor_table.item_arr.size(); i++)
    {
        for (unsigned long j = 0; j < matrix->block_coor_table.item_arr[i]->compressed_block_ptr->read_index.size(); j++)
        {
            assert(matrix->block_coor_table.item_arr[i]->compressed_block_ptr->read_index[j]->is_sort_arr == NULL);
        }
    }

    assert(mem_manager != NULL && matrix != NULL);

    // 最外层不能被析构
    assert(is_deleted(mem_manager, matrix) == false);

    // 首先析构dense_block_table
    for (int item_id = 0; item_id < matrix->block_coor_table.item_arr.size(); item_id++)
    {
        // cout << "delete_sparse_struct_t: item_id:" << item_id << endl;
        // 压缩视图和对应表项可能不存在，所有对应表项和压缩视图存在时，才需要进一步析构
        if (matrix->block_coor_table.item_arr[item_id] != NULL && matrix->block_coor_table.item_arr[item_id]->compressed_block_ptr != NULL)
        {
            // 只有压缩视图需要被析构，压缩视图是必然存在的
            compressed_block_t *compressed_view_of_block = matrix->block_coor_table.item_arr[item_id]->compressed_block_ptr;
            assert(compressed_view_of_block != NULL);
            // 肯定没有被析构过
            assert(is_deleted(mem_manager, compressed_view_of_block) == false);

            for (unsigned long j = 0; j < compressed_view_of_block->read_index.size(); j++)
            {
                assert(compressed_view_of_block->read_index[j]->is_sort_arr == NULL);
            }

            // 首先析构所有的read_index
            for (int read_index_id = 0; read_index_id < compressed_view_of_block->read_index.size(); read_index_id++)
            {
                // if (compressed_view_of_block->read_index.size() >= 6)
                // {
                //     cout << "delete_sparse_struct_t: compressed_view_of_block->read_index[5]->is_sort_arr:" << compressed_view_of_block->read_index[5]->is_sort_arr << endl;
                // }

                index_of_compress_block_t *index_ptr = compressed_view_of_block->read_index[read_index_id];

                // 肯定不是空的

                assert(index_ptr != NULL);

                // cout << "delete_sparse_struct_t: read_index_id:" << read_index_id << ", index_ptr->is_sort_arr:" << index_ptr->is_sort_arr << endl;
            
                // 肯定没有被析构过
                // 有可能被析构过。
                // assert(is_deleted(mem_manager, index_ptr) == false);

                // 析构索引中所有的元数据，如果不是空指针，并且没有析构过，就执行析构，并且登记
                // index_arr
                if (index_ptr->index_arr != NULL && is_deleted(mem_manager, index_ptr->index_arr) == false)
                {
                    // cout << "delete_sparse_struct_t: delete index_ptr->index_arr" << endl;
                    delete_arr_with_data_type(index_ptr->index_arr, index_ptr->index_data_type);
                    register_del_ptr(mem_manager, index_ptr->index_arr);
                    index_ptr->index_arr = NULL;
                    // cout << "delete_sparse_struct_t: finish delete index_ptr->index_arr" << endl;
                }

                // if (read_index_id == 3)
                // {
                //     cout << 1 << endl;
                // }
                cout << "WARNNING: delete_sparse_struct_t: sometimes the is_sort_arr is not empty" << endl;
                index_ptr->is_sort_arr = NULL;
                assert(index_ptr->is_sort_arr == NULL);

                // cout << "delete_sparse_struct_t: index_ptr->is_sort_arr:" << index_ptr->is_sort_arr << endl;

                if (index_ptr->is_sort_arr != NULL && is_deleted(mem_manager, index_ptr->is_sort_arr) == false)
                {
                    // cout << "delete_sparse_struct_t: delete index_ptr->is_sort_arr" << endl;
                    delete[] index_ptr->is_sort_arr;
                    register_del_ptr(mem_manager, index_ptr->is_sort_arr);
                    index_ptr->is_sort_arr = NULL;
                }

                // if (read_index_id == 3)
                // {
                //     cout << 2 << endl;
                // }

                if (index_ptr->index_of_the_first_row_arr != NULL && is_deleted(mem_manager, index_ptr->index_of_the_first_row_arr) == false)
                {
                    // cout << "delete_sparse_struct_t: delete index_ptr->index_of_the_first_row_arr" << endl;
                    delete_arr_with_data_type(index_ptr->index_of_the_first_row_arr, index_ptr->data_type_of_index_of_the_first_row_arr);
                    register_del_ptr(mem_manager, index_ptr->index_of_the_first_row_arr);
                    index_ptr->index_of_the_first_row_arr = NULL;
                }

                // if (read_index_id == 3)
                // {
                //     cout << 3 << endl;
                // }

                if (index_ptr->row_number_of_block_arr != NULL && is_deleted(mem_manager, index_ptr->row_number_of_block_arr) == false)
                {
                    // cout << "delete_sparse_struct_t: delete index_ptr->row_number_of_block_arr" << endl;
                    delete_arr_with_data_type(index_ptr->row_number_of_block_arr, index_ptr->data_type_of_row_number_of_block_arr);
                    register_del_ptr(mem_manager, index_ptr->row_number_of_block_arr);
                    index_ptr->row_number_of_block_arr = NULL;
                }

                // if (read_index_id == 3)
                // {
                //     cout << 4 << endl;
                // }

                if (index_ptr->tmp_result_write_index_arr != NULL && is_deleted(mem_manager, index_ptr->tmp_result_write_index_arr) == false)
                {
                    // cout << "delete_sparse_struct_t: delete index_ptr->tmp_result_write_index_arr" << endl;
                    delete_arr_with_data_type(index_ptr->tmp_result_write_index_arr, index_ptr->data_type_of_tmp_result_write_index_arr);
                    register_del_ptr(mem_manager, index_ptr->tmp_result_write_index_arr);
                    index_ptr->tmp_result_write_index_arr = NULL;
                }

                // if (read_index_id == 3)
                // {
                //     cout << 5 << endl;
                // }

                if (index_ptr->coo_begin_index_arr != NULL && is_deleted(mem_manager, index_ptr->coo_begin_index_arr) == false)
                {
                    // cout << "delete_sparse_struct_t: delete index_ptr->tmp_result_write_index_arr" << endl;
                    delete_arr_with_data_type(index_ptr->coo_begin_index_arr, index_ptr->data_type_of_coo_begin_index_arr);
                    register_del_ptr(mem_manager, index_ptr->coo_begin_index_arr);
                    index_ptr->coo_begin_index_arr = NULL;
                }

                // if (read_index_id == 3)
                // {
                //     cout << 6 << endl;
                // }

                if (index_ptr->coo_block_size_arr != NULL && is_deleted(mem_manager, index_ptr->coo_block_size_arr) == false)
                {
                    // cout << "delete_sparse_struct_t: delete index_ptr->coo_block_size_arr" << endl;
                    delete_arr_with_data_type(index_ptr->coo_block_size_arr, index_ptr->data_type_of_coo_block_size_arr);
                    register_del_ptr(mem_manager, index_ptr->coo_block_size_arr);
                    index_ptr->coo_block_size_arr = NULL;
                }

                // if (read_index_id == 3)
                // {
                //     cout << 7 << endl;
                // }

                if (index_ptr->child_tmp_row_csr_index_arr != NULL && is_deleted(mem_manager, index_ptr->child_tmp_row_csr_index_arr) == false)
                {
                    // cout << "delete_sparse_struct_t: delete index_ptr->child_tmp_row_csr_index_arr" << endl;
                    delete_arr_with_data_type(index_ptr->child_tmp_row_csr_index_arr, index_ptr->data_type_of_child_tmp_row_csr_index);
                    register_del_ptr(mem_manager, index_ptr->child_tmp_row_csr_index_arr);
                    index_ptr->child_tmp_row_csr_index_arr = NULL;
                }

                // if (read_index_id == 3)
                // {
                //     cout << 8 << endl;
                // }

                if (index_ptr->begin_index_in_tmp_row_csr_arr_of_block != NULL && is_deleted(mem_manager, index_ptr->begin_index_in_tmp_row_csr_arr_of_block) == false)
                {
                    // cout << "delete_sparse_struct_t: delete index_ptr->begin_index_in_tmp_row_csr_arr_of_block" << endl;
                    delete_arr_with_data_type(index_ptr->begin_index_in_tmp_row_csr_arr_of_block, index_ptr->data_type_of_begin_index_in_tmp_row_csr_arr_of_block);
                    register_del_ptr(mem_manager, index_ptr->begin_index_in_tmp_row_csr_arr_of_block);
                    index_ptr->begin_index_in_tmp_row_csr_arr_of_block = NULL;
                }

                // 查看是不是析构过
                if (index_ptr != NULL && is_deleted(mem_manager, index_ptr) == false)
                {
                    // 析构一个索引本身
                    // cout << "delete_sparse_struct_t: delete index_ptr: " << index_ptr << endl;
                    delete index_ptr;
                    register_del_ptr(mem_manager, index_ptr);
                    compressed_view_of_block->read_index[read_index_id] = NULL;
                    // cout << "delete_sparse_struct_t: finish delete index_ptr" << endl;
                }
            }

            // cout << "delete_sparse_struct_t: compressed_view_of_block->y_write_index.size():" << compressed_view_of_block->y_write_index.size() << endl;

            // 然后析构所有的y_write_index
            for (int write_index_id = 0; write_index_id < compressed_view_of_block->y_write_index.size(); write_index_id++)
            {
                // cout << "delete_sparse_struct_t: write_index_id:" << write_index_id << endl;
                index_of_compress_block_t *index_ptr = compressed_view_of_block->y_write_index[write_index_id];
                // 肯定不是空的
                assert(index_ptr != NULL);
                // 肯定没有被析构过
                // assert(is_deleted(mem_manager, index_ptr) == false);

                // 析构索引中所有的元数据，如果不是空指针，并且没有析构过，就执行析构，并且登记
                // index_arr
                if (index_ptr->index_arr != NULL && is_deleted(mem_manager, index_ptr->index_arr) == false)
                {
                    // cout << "delete_sparse_struct_t: index_ptr->index_data_type" << endl;
                    delete_arr_with_data_type(index_ptr->index_arr, index_ptr->index_data_type);
                    register_del_ptr(mem_manager, index_ptr->index_arr);
                    index_ptr->index_arr = NULL;
                }

                if (index_ptr->is_sort_arr != NULL && is_deleted(mem_manager, index_ptr->is_sort_arr) == false)
                {
                    // cout << "delete_sparse_struct_t: index_ptr->is_sort_arr" << endl;
                    delete[] index_ptr->is_sort_arr;
                    register_del_ptr(mem_manager, index_ptr->is_sort_arr);
                    index_ptr->is_sort_arr = NULL;
                }

                if (index_ptr->index_of_the_first_row_arr != NULL && is_deleted(mem_manager, index_ptr->index_of_the_first_row_arr) == false)
                {
                    // cout << "delete_sparse_struct_t: index_ptr->index_of_the_first_row_arr" << endl;
                    delete_arr_with_data_type(index_ptr->index_of_the_first_row_arr, index_ptr->data_type_of_index_of_the_first_row_arr);
                    register_del_ptr(mem_manager, index_ptr->index_of_the_first_row_arr);
                    index_ptr->index_of_the_first_row_arr = NULL;
                }

                if (index_ptr->row_number_of_block_arr != NULL && is_deleted(mem_manager, index_ptr->row_number_of_block_arr) == false)
                {
                    // cout << "delete_sparse_struct_t: index_ptr->row_number_of_block_arr" << endl;
                    delete_arr_with_data_type(index_ptr->row_number_of_block_arr, index_ptr->data_type_of_row_number_of_block_arr);
                    register_del_ptr(mem_manager, index_ptr->row_number_of_block_arr);
                    index_ptr->row_number_of_block_arr = NULL;
                }

                if (index_ptr->tmp_result_write_index_arr != NULL && is_deleted(mem_manager, index_ptr->tmp_result_write_index_arr) == false)
                {
                    // cout << "delete_sparse_struct_t: index_ptr->tmp_result_write_index_arr" << endl;
                    delete_arr_with_data_type(index_ptr->tmp_result_write_index_arr, index_ptr->data_type_of_tmp_result_write_index_arr);
                    register_del_ptr(mem_manager, index_ptr->tmp_result_write_index_arr);
                    index_ptr->tmp_result_write_index_arr = NULL;
                }

                if (index_ptr->coo_begin_index_arr != NULL && is_deleted(mem_manager, index_ptr->coo_begin_index_arr) == false)
                {
                    // cout << "delete_sparse_struct_t: index_ptr->coo_begin_index_arr" << endl;
                    delete_arr_with_data_type(index_ptr->coo_begin_index_arr, index_ptr->data_type_of_coo_begin_index_arr);
                    register_del_ptr(mem_manager, index_ptr->coo_begin_index_arr);
                    index_ptr->coo_begin_index_arr = NULL;
                }

                if (index_ptr->coo_block_size_arr != NULL && is_deleted(mem_manager, index_ptr->coo_block_size_arr) == false)
                {
                    // cout << "delete_sparse_struct_t: index_ptr->coo_block_size_arr" << endl;
                    delete_arr_with_data_type(index_ptr->coo_block_size_arr, index_ptr->data_type_of_coo_block_size_arr);
                    register_del_ptr(mem_manager, index_ptr->coo_block_size_arr);
                    index_ptr->coo_block_size_arr = NULL;
                }

                if (index_ptr->child_tmp_row_csr_index_arr != NULL && is_deleted(mem_manager, index_ptr->child_tmp_row_csr_index_arr) == false)
                {
                    // cout << "delete_sparse_struct_t: index_ptr->child_tmp_row_csr_index_arr" << endl;
                    delete_arr_with_data_type(index_ptr->child_tmp_row_csr_index_arr, index_ptr->data_type_of_child_tmp_row_csr_index);
                    register_del_ptr(mem_manager, index_ptr->child_tmp_row_csr_index_arr);
                    index_ptr->child_tmp_row_csr_index_arr = NULL;
                }

                if (index_ptr->begin_index_in_tmp_row_csr_arr_of_block != NULL && is_deleted(mem_manager, index_ptr->begin_index_in_tmp_row_csr_arr_of_block) == false)
                {
                    // cout << "delete_sparse_struct_t: index_ptr->begin_index_in_tmp_row_csr_arr_of_block" << endl;
                    delete_arr_with_data_type(index_ptr->begin_index_in_tmp_row_csr_arr_of_block, index_ptr->data_type_of_begin_index_in_tmp_row_csr_arr_of_block);
                    register_del_ptr(mem_manager, index_ptr->begin_index_in_tmp_row_csr_arr_of_block);
                    index_ptr->begin_index_in_tmp_row_csr_arr_of_block = NULL;
                }

                // 本体的析构之前也需要一个检查
                if (index_ptr != NULL && is_deleted(mem_manager, index_ptr) == false)
                {
                    // 析构一个索引本身
                    delete index_ptr;
                    register_del_ptr(mem_manager, index_ptr);
                    compressed_view_of_block->y_write_index[write_index_id] = NULL;
                }
            }

            // cout << "delete_sparse_struct_t: compressed_view_of_block->reduce_help_csr.size():" << compressed_view_of_block->reduce_help_csr.size() << endl;

            // 析构reduce_help_csr，可能没有用，但是也析构了吧
            for (int reduce_help_csr_index_id = 0; reduce_help_csr_index_id < compressed_view_of_block->reduce_help_csr.size(); reduce_help_csr_index_id++)
            {
                index_of_compress_block_t *index_ptr = compressed_view_of_block->reduce_help_csr[reduce_help_csr_index_id];
                // 肯定不是空的
                assert(index_ptr != NULL);
                // 肯定没有被析构过
                // assert(is_deleted(mem_manager, index_ptr) == false);

                // 析构索引中所有的元数据，如果不是空指针，并且没有析构过，就执行析构，并且登记
                // index_arr
                if (index_ptr->index_arr != NULL && is_deleted(mem_manager, index_ptr->index_arr) == false)
                {
                    delete_arr_with_data_type(index_ptr->index_arr, index_ptr->index_data_type);
                    register_del_ptr(mem_manager, index_ptr->index_arr);
                    index_ptr->index_arr = NULL;
                }

                assert(index_ptr->is_sort_arr == NULL);

                if (index_ptr->is_sort_arr != NULL && is_deleted(mem_manager, index_ptr->is_sort_arr) == false)
                {
                    delete[] index_ptr->is_sort_arr;
                    register_del_ptr(mem_manager, index_ptr->is_sort_arr);
                    index_ptr->is_sort_arr = NULL;
                }

                if (index_ptr->index_of_the_first_row_arr != NULL && is_deleted(mem_manager, index_ptr->index_of_the_first_row_arr) == false)
                {
                    delete_arr_with_data_type(index_ptr->index_of_the_first_row_arr, index_ptr->data_type_of_index_of_the_first_row_arr);
                    register_del_ptr(mem_manager, index_ptr->index_of_the_first_row_arr);
                    index_ptr->index_of_the_first_row_arr = NULL;
                }

                if (index_ptr->row_number_of_block_arr != NULL && is_deleted(mem_manager, index_ptr->row_number_of_block_arr) == false)
                {
                    delete_arr_with_data_type(index_ptr->row_number_of_block_arr, index_ptr->data_type_of_row_number_of_block_arr);
                    register_del_ptr(mem_manager, index_ptr->row_number_of_block_arr);
                    index_ptr->row_number_of_block_arr = NULL;
                }

                if (index_ptr->tmp_result_write_index_arr != NULL && is_deleted(mem_manager, index_ptr->tmp_result_write_index_arr) == false)
                {
                    delete_arr_with_data_type(index_ptr->tmp_result_write_index_arr, index_ptr->data_type_of_tmp_result_write_index_arr);
                    register_del_ptr(mem_manager, index_ptr->tmp_result_write_index_arr);
                    index_ptr->tmp_result_write_index_arr = NULL;
                }

                if (index_ptr->coo_begin_index_arr != NULL && is_deleted(mem_manager, index_ptr->coo_begin_index_arr) == false)
                {
                    delete_arr_with_data_type(index_ptr->coo_begin_index_arr, index_ptr->data_type_of_coo_begin_index_arr);
                    register_del_ptr(mem_manager, index_ptr->coo_begin_index_arr);
                    index_ptr->coo_begin_index_arr = NULL;
                }

                if (index_ptr->coo_block_size_arr != NULL && is_deleted(mem_manager, index_ptr->coo_block_size_arr) == false)
                {
                    delete_arr_with_data_type(index_ptr->coo_block_size_arr, index_ptr->data_type_of_coo_block_size_arr);
                    register_del_ptr(mem_manager, index_ptr->coo_block_size_arr);
                    index_ptr->coo_block_size_arr = NULL;
                }

                if (index_ptr->child_tmp_row_csr_index_arr != NULL && is_deleted(mem_manager, index_ptr->child_tmp_row_csr_index_arr) == false)
                {
                    delete_arr_with_data_type(index_ptr->child_tmp_row_csr_index_arr, index_ptr->data_type_of_child_tmp_row_csr_index);
                    register_del_ptr(mem_manager, index_ptr->child_tmp_row_csr_index_arr);
                    index_ptr->child_tmp_row_csr_index_arr = NULL;
                }

                if (index_ptr->begin_index_in_tmp_row_csr_arr_of_block != NULL && is_deleted(mem_manager, index_ptr->begin_index_in_tmp_row_csr_arr_of_block) == false)
                {
                    delete_arr_with_data_type(index_ptr->begin_index_in_tmp_row_csr_arr_of_block, index_ptr->data_type_of_begin_index_in_tmp_row_csr_arr_of_block);
                    register_del_ptr(mem_manager, index_ptr->begin_index_in_tmp_row_csr_arr_of_block);
                    index_ptr->begin_index_in_tmp_row_csr_arr_of_block = NULL;
                }

                if (index_ptr != NULL && is_deleted(mem_manager, index_ptr) == false)
                {
                    // 析构一个索引本身
                    delete index_ptr;
                    register_del_ptr(mem_manager, index_ptr);
                    compressed_view_of_block->reduce_help_csr[reduce_help_csr_index_id] = NULL;
                }
            }

            // cout << "delete_sparse_struct_t: begin del other info" << endl;

            // 析构几个值数组
            if (compressed_view_of_block->val_arr != NULL && is_deleted(mem_manager, compressed_view_of_block->val_arr) == false)
            {
                // cout << "delete_sparse_struct_t: compressed_view_of_block->val_arr" << endl;
                delete_arr_with_data_type(compressed_view_of_block->val_arr, compressed_view_of_block->val_data_type);
                register_del_ptr(mem_manager, compressed_view_of_block->val_arr);
                compressed_view_of_block->val_arr = NULL;
            }

            if (compressed_view_of_block->padding_val_arr != NULL && is_deleted(mem_manager, compressed_view_of_block->padding_val_arr) == false)
            {
                // cout << "delete_sparse_struct_t: compressed_view_of_block->padding_val_arr" << endl;
                delete_arr_with_data_type(compressed_view_of_block->padding_val_arr, compressed_view_of_block->val_data_type);
                register_del_ptr(mem_manager, compressed_view_of_block->padding_val_arr);
                compressed_view_of_block->padding_val_arr = NULL;
            }

            if (compressed_view_of_block->staggered_padding_val_arr != NULL && is_deleted(mem_manager, compressed_view_of_block->staggered_padding_val_arr) == false)
            {
                // cout << "delete_sparse_struct_t: compressed_view_of_block->staggered_padding_val_arr" << endl;
                delete_arr_with_data_type(compressed_view_of_block->staggered_padding_val_arr, compressed_view_of_block->val_data_type);
                register_del_ptr(mem_manager, compressed_view_of_block->staggered_padding_val_arr);
                compressed_view_of_block->staggered_padding_val_arr = NULL;
            }

            // 析构表项和自己
            if (compressed_view_of_block != NULL && is_deleted(mem_manager, compressed_view_of_block) == false)
            {
                // cout << "delete_sparse_struct_t: compressed_view_of_block:" << compressed_view_of_block << endl;
                // 析构block自己
                delete compressed_view_of_block;
                // 登记已经被析构的指针
                register_del_ptr(mem_manager, compressed_view_of_block);
                matrix->block_coor_table.item_arr[item_id]->compressed_block_ptr = NULL;
            }
        }

        if (matrix->block_coor_table.item_arr[item_id] != NULL && is_deleted(mem_manager, matrix->block_coor_table.item_arr[item_id]) == false)
        {

            // 析构当前的子矩阵表的索引项
            delete (matrix->block_coor_table.item_arr[item_id]);
            register_del_ptr(mem_manager, matrix->block_coor_table.item_arr[item_id]);
            matrix->block_coor_table.item_arr[item_id] = NULL;
        }
    }

    // 记录排序的数组的析构
    if (matrix->sorted_row_index != NULL && is_deleted(mem_manager, matrix->sorted_row_index) == false)
    {
        delete_arr_with_data_type(matrix->sorted_row_index, matrix->data_type_of_sorted_row_index);
        register_del_ptr(mem_manager, matrix->sorted_row_index);
        matrix->sorted_row_index = NULL;
    }

    // 有一个抛弃的指针
    assert(matrix->compressed_block_arr == NULL);

    // 剩下几个索引
    assert(matrix->coo_row_index_cache != NULL && matrix->coo_col_index_cache != NULL && matrix->coo_value_cache != NULL);

    if (is_deleted(mem_manager, matrix->coo_row_index_cache) == false)
    {
        delete[] matrix->coo_row_index_cache;
        register_del_ptr(mem_manager, matrix->coo_row_index_cache);
        matrix->coo_row_index_cache = NULL;
    }

    if (is_deleted(mem_manager, matrix->coo_col_index_cache) == false)
    {
        delete[] matrix->coo_col_index_cache;
        register_del_ptr(mem_manager, matrix->coo_col_index_cache);
        matrix->coo_col_index_cache = NULL;
    }

    if (is_deleted(mem_manager, matrix->coo_value_cache) == false)
    {
        delete_arr_with_data_type(matrix->coo_value_cache, matrix->val_data_type);
        register_del_ptr(mem_manager, matrix->coo_value_cache);
        matrix->coo_value_cache = NULL;
    }

    // cout << 1 << endl;

    if (matrix->coo_x_cache.x_arr != NULL && is_deleted(mem_manager, matrix->coo_x_cache.x_arr) == false)
    {
        delete_arr_with_data_type(matrix->coo_x_cache.x_arr, matrix->coo_x_cache.x_data_type);
        register_del_ptr(mem_manager, matrix->coo_x_cache.x_arr);
        matrix->coo_x_cache.x_arr = NULL;
    }

    // cout << 2 << endl;

    // 析构自己
    if (matrix != NULL && is_deleted(mem_manager, matrix) == false)
    {
        delete matrix;
        // 不需要自己登记自己，
        // register_del_ptr(mem_manager, matrix);
        // 归零没啥用
    }

    // cout << "delete_sparse_struct_t: finish" << endl;

    // cout << 3 << endl;
}

void register_del_ptr(memory_garbage_manager_t *mem_manager, void *del_ptr)
{
    // 这个指针不可能已经被登记，并且不可能是空指针
    assert(mem_manager != NULL && mem_manager->ptr_set.count(del_ptr) == 0 && del_ptr != NULL);

    // 登记一个指针
    mem_manager->ptr_set.insert(del_ptr);
}

bool is_deleted(memory_garbage_manager_t *mem_manager, void *del_ptr)
{
    if (mem_manager->ptr_set.count(del_ptr) == 0)
    {
        // 还没有被析构
        return false;
    }
    else
    {
        cout << "is_deleted: is deleted before:" << del_ptr << endl;
        return true;
    }
}

void print_all_register_ptr(memory_garbage_manager_t *mem_manager)
{
    cout << "ptr set size:" << mem_manager->ptr_set.size() << endl;

    for (set<void *>::iterator it = mem_manager->ptr_set.begin(); it != mem_manager->ptr_set.end(); it++)
    {
        cout << *it << ",";
    }
    cout << endl;
}

void delete_direct_atom_template(memory_garbage_manager_t *mem_manager, direct_atom_template_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    // 首先析构matrix
    assert(del_template->matrix != NULL);
    if (is_deleted(mem_manager, del_template->matrix) == false)
    {
        delete_sparse_struct_t(mem_manager, del_template->matrix);
        register_del_ptr(mem_manager, del_template->matrix);
        del_template->matrix = NULL;
    }

    // cout << 1 << endl;

    // 析构一系列元数据
    if (del_template->global_row_index_of_thread_level_block != NULL && is_deleted(mem_manager, del_template->global_row_index_of_thread_level_block) == false)
    {
        delete_arr_with_data_type(del_template->global_row_index_of_thread_level_block, del_template->data_type_of_global_row_index_of_thread_level_block);
        register_del_ptr(mem_manager, del_template->global_row_index_of_thread_level_block);
        del_template->global_row_index_of_thread_level_block = NULL;
    }

    // cout << 1 << endl;

    if (del_template->block_begin_warp_index_offset != NULL && is_deleted(mem_manager, del_template->block_begin_warp_index_offset) == false)
    {
        delete_arr_with_data_type(del_template->block_begin_warp_index_offset, del_template->data_type_of_block_begin_warp_index_offset);
        register_del_ptr(mem_manager, del_template->block_begin_warp_index_offset);
        del_template->block_begin_warp_index_offset = NULL;
    }

    // cout << 1 << endl;

    if (del_template->warp_begin_thread_index_offset != NULL && is_deleted(mem_manager, del_template->warp_begin_thread_index_offset) == false)
    {
        delete_arr_with_data_type(del_template->warp_begin_thread_index_offset, del_template->data_type_of_warp_begin_thread_index_offset);
        register_del_ptr(mem_manager, del_template->warp_begin_thread_index_offset);
        del_template->warp_begin_thread_index_offset = NULL;
    }

    if (del_template->thread_block_size_in_warp != NULL && is_deleted(mem_manager, del_template->thread_block_size_in_warp) == false)
    {
        delete_arr_with_data_type(del_template->thread_block_size_in_warp, del_template->data_type_of_thread_block_size_in_warp);
        register_del_ptr(mem_manager, del_template->thread_block_size_in_warp);
        del_template->thread_block_size_in_warp = NULL;
    }

    if (del_template->row_index_before_sort != NULL && is_deleted(mem_manager, del_template->row_index_before_sort) == false)
    {
        delete_arr_with_data_type(del_template->row_index_before_sort, del_template->data_type_of_row_index_before_sort);
        register_del_ptr(mem_manager, del_template->row_index_before_sort);
        del_template->row_index_before_sort = NULL;
    }

    // cout << 1 << endl;

    if (del_template->block_nz_begin_offset != NULL && is_deleted(mem_manager, del_template->block_nz_begin_offset) == false)
    {
        delete_arr_with_data_type(del_template->block_nz_begin_offset, del_template->data_type_of_block_nz_begin_offset);
        register_del_ptr(mem_manager, del_template->block_nz_begin_offset);
        del_template->block_nz_begin_offset = NULL;
    }

    if (del_template->warp_nz_begin_offset != NULL && is_deleted(mem_manager, del_template->warp_nz_begin_offset) == false)
    {
        delete_arr_with_data_type(del_template->warp_nz_begin_offset, del_template->data_type_of_warp_nz_begin_offset);
        register_del_ptr(mem_manager, del_template->warp_nz_begin_offset);
        del_template->warp_nz_begin_offset = NULL;
    }

    if (del_template->val_arr != NULL && is_deleted(mem_manager, del_template->val_arr) == false)
    {
        delete_arr_with_data_type(del_template->val_arr, del_template->data_type_of_val_arr);
        register_del_ptr(mem_manager, del_template->val_arr);
        del_template->val_arr = NULL;
    }

    if (del_template->col_index_arr != NULL && is_deleted(mem_manager, del_template->col_index_arr) == false)
    {
        delete_arr_with_data_type(del_template->col_index_arr, del_template->data_type_of_col_index_arr);
        register_del_ptr(mem_manager, del_template->col_index_arr);
        del_template->col_index_arr = NULL;
    }

    // 所有的压缩元数据
    if (del_template->global_row_index_compress_meta != NULL && is_deleted(mem_manager, del_template->global_row_index_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->global_row_index_compress_meta, del_template->global_row_index_compress);
        register_del_ptr(mem_manager, del_template->global_row_index_compress_meta);
        del_template->global_row_index_compress_meta = NULL;
    }

    if (del_template->block_begin_warp_index_compress_meta != NULL && is_deleted(mem_manager, del_template->block_begin_warp_index_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->block_begin_warp_index_compress_meta, del_template->block_begin_warp_index_compress);
        register_del_ptr(mem_manager, del_template->block_begin_warp_index_compress_meta);
        del_template->block_begin_warp_index_compress_meta = NULL;
    }

    if (del_template->warp_begin_thread_index_compress_meta != NULL && is_deleted(mem_manager, del_template->warp_begin_thread_index_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->warp_begin_thread_index_compress_meta, del_template->warp_begin_thread_index_compress);
        register_del_ptr(mem_manager, del_template->warp_begin_thread_index_compress_meta);
        del_template->warp_begin_thread_index_compress_meta = NULL;
    }

    if (del_template->thread_block_size_compress_meta != NULL && is_deleted(mem_manager, del_template->thread_block_size_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->thread_block_size_compress_meta, del_template->thread_block_size_compress);
        register_del_ptr(mem_manager, del_template->thread_block_size_compress_meta);
        del_template->thread_block_size_compress_meta = NULL;
    }

    if (del_template->row_index_before_sort_compress_meta != NULL && is_deleted(mem_manager, del_template->row_index_before_sort_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->row_index_before_sort_compress_meta, del_template->row_index_before_sort_compress);
        register_del_ptr(mem_manager, del_template->row_index_before_sort_compress_meta);
        del_template->row_index_before_sort_compress_meta = NULL;
    }

    if (del_template->block_nz_begin_offset_compress_meta != NULL && is_deleted(mem_manager, del_template->block_nz_begin_offset_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->block_nz_begin_offset_compress_meta, del_template->block_nz_begin_offset_compress);
        register_del_ptr(mem_manager, del_template->block_nz_begin_offset_compress_meta);
        del_template->block_nz_begin_offset_compress_meta = NULL;
    }

    if (del_template->warp_nz_begin_offset_compress_meta != NULL && is_deleted(mem_manager, del_template->warp_nz_begin_offset_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->warp_nz_begin_offset_compress_meta, del_template->warp_nz_begin_offset_compress);
        register_del_ptr(mem_manager, del_template->warp_nz_begin_offset_compress_meta);
        del_template->warp_nz_begin_offset_compress_meta = NULL;
    }

    // 如果析构过才析构自己，就不用再析构自己了
    if (del_template != NULL && is_deleted(mem_manager, del_template) == false)
    {
        // 析构模板自己
        delete del_template;
        // 析构自己不用登记
        // register_del_ptr(mem_manager, del_template);
        del_template = NULL;
    }
}

void delete_direct_atom_template_warp_compress(memory_garbage_manager_t *mem_manager, direct_atom_template_warp_compress_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    // 首先析构matrix
    assert(del_template->matrix != NULL);
    if (is_deleted(mem_manager, del_template->matrix) == false)
    {
        delete_sparse_struct_t(mem_manager, del_template->matrix);
        register_del_ptr(mem_manager, del_template->matrix);
        del_template->matrix = NULL;
    }

    if (del_template->global_row_index_of_thread_level_block != NULL && is_deleted(mem_manager, del_template->global_row_index_of_thread_level_block) == false)
    {
        delete_arr_with_data_type(del_template->global_row_index_of_thread_level_block, del_template->data_type_of_global_row_index_of_thread_level_block);
        register_del_ptr(mem_manager, del_template->global_row_index_of_thread_level_block);
        del_template->global_row_index_of_thread_level_block = NULL;
    }

    if (del_template->block_begin_thread_index_offset != NULL && is_deleted(mem_manager, del_template->block_begin_thread_index_offset) == false)
    {
        delete_arr_with_data_type(del_template->block_begin_thread_index_offset, del_template->data_type_of_block_begin_thread_index_offset);
        register_del_ptr(mem_manager, del_template->block_begin_thread_index_offset);
        del_template->block_begin_thread_index_offset = NULL;
    }

    if (del_template->block_nz_begin_offset != NULL && is_deleted(mem_manager, del_template->block_nz_begin_offset) == false)
    {
        delete_arr_with_data_type(del_template->block_nz_begin_offset, del_template->data_type_of_block_nz_begin_offset);
        register_del_ptr(mem_manager, del_template->block_nz_begin_offset);
        del_template->block_nz_begin_offset = NULL;
    }

    if (del_template->thread_block_size_in_block != NULL && is_deleted(mem_manager, del_template->thread_block_size_in_block) == false)
    {
        delete_arr_with_data_type(del_template->thread_block_size_in_block, del_template->data_type_of_thread_block_size_in_block);
        register_del_ptr(mem_manager, del_template->thread_block_size_in_block);
        del_template->thread_block_size_in_block = NULL;
    }

    if (del_template->row_index_before_sort != NULL && is_deleted(mem_manager, del_template->row_index_before_sort) == false)
    {
        delete_arr_with_data_type(del_template->row_index_before_sort, del_template->data_type_of_row_index_before_sort);
        register_del_ptr(mem_manager, del_template->row_index_before_sort);
        del_template->row_index_before_sort = NULL;
    }

    if (del_template->val_arr != NULL && is_deleted(mem_manager, del_template->val_arr) == false)
    {
        delete_arr_with_data_type(del_template->val_arr, del_template->data_type_of_val_arr);
        register_del_ptr(mem_manager, del_template->val_arr);
        del_template->val_arr = NULL;
    }

    if (del_template->col_index_arr != NULL && is_deleted(mem_manager, del_template->col_index_arr) == false)
    {
        delete_arr_with_data_type(del_template->col_index_arr, del_template->data_type_of_col_index_arr);
        register_del_ptr(mem_manager, del_template->col_index_arr);
        del_template->col_index_arr = NULL;
    }

    if (del_template->global_row_index_compress_meta != NULL && is_deleted(mem_manager, del_template->global_row_index_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->global_row_index_compress_meta, del_template->global_row_index_compress);
        register_del_ptr(mem_manager, del_template->global_row_index_compress_meta);
        del_template->global_row_index_compress_meta = NULL;
    }

    if (del_template->block_begin_thread_index_offset_compress_meta != NULL && is_deleted(mem_manager, del_template->block_begin_thread_index_offset_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->block_begin_thread_index_offset_compress_meta, del_template->block_begin_thread_index_offset_compress);
        register_del_ptr(mem_manager, del_template->block_begin_thread_index_offset_compress_meta);
        del_template->block_begin_thread_index_offset_compress_meta = NULL;
    }

    if (del_template->block_nz_begin_offset_compress_meta != NULL && is_deleted(mem_manager, del_template->block_nz_begin_offset_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->block_nz_begin_offset_compress_meta, del_template->block_nz_begin_offset_compress);
        register_del_ptr(mem_manager, del_template->block_nz_begin_offset_compress_meta);
        del_template->block_nz_begin_offset_compress_meta = NULL;
    }

    if (del_template->thread_block_size_in_block_compress_meta != NULL && is_deleted(mem_manager, del_template->thread_block_size_in_block_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->thread_block_size_in_block_compress_meta, del_template->thread_block_size_in_block_compress);
        register_del_ptr(mem_manager, del_template->thread_block_size_in_block_compress_meta);
        del_template->thread_block_size_in_block_compress_meta = NULL;
    }

    if (del_template->row_index_before_sort_compress_meta != NULL && is_deleted(mem_manager, del_template->row_index_before_sort_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->row_index_before_sort_compress_meta, del_template->row_index_before_sort_compress);
        register_del_ptr(mem_manager, del_template->row_index_before_sort_compress_meta);
        del_template->row_index_before_sort_compress_meta = NULL;
    }

    if (del_template != NULL && is_deleted(mem_manager, del_template) == false)
    {
        // 删除模板本身
        delete del_template;
        // 不用登记，不用清零
    }
}

void delete_direct_atom_template_warp_block_compress(memory_garbage_manager_t *mem_manager, direct_atom_template_warp_block_compress_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    // 首先析构matrix
    assert(del_template->matrix != NULL);
    if (is_deleted(mem_manager, del_template->matrix) == false)
    {
        delete_sparse_struct_t(mem_manager, del_template->matrix);
        register_del_ptr(mem_manager, del_template->matrix);
        del_template->matrix = NULL;
    }

    if (del_template->global_row_index_of_thread_level_block != NULL && is_deleted(mem_manager, del_template->global_row_index_of_thread_level_block) == false)
    {
        delete_arr_with_data_type(del_template->global_row_index_of_thread_level_block, del_template->data_type_of_global_row_index_of_thread_level_block);
        register_del_ptr(mem_manager, del_template->global_row_index_of_thread_level_block);
        del_template->global_row_index_of_thread_level_block = NULL;
    }

    if (del_template->row_index_before_sort != NULL && is_deleted(mem_manager, del_template->row_index_before_sort) == false)
    {
        delete_arr_with_data_type(del_template->row_index_before_sort, del_template->data_type_of_row_index_before_sort);
        register_del_ptr(mem_manager, del_template->row_index_before_sort);
        del_template->row_index_before_sort = NULL;
    }

    if (del_template->val_arr != NULL && is_deleted(mem_manager, del_template->val_arr) == false)
    {
        delete_arr_with_data_type(del_template->val_arr, del_template->data_type_of_val_arr);
        register_del_ptr(mem_manager, del_template->val_arr);
        del_template->val_arr = NULL;
    }

    if (del_template->col_index_arr != NULL && is_deleted(mem_manager, del_template->col_index_arr) == false)
    {
        delete_arr_with_data_type(del_template->col_index_arr, del_template->data_type_of_col_index_arr);
        register_del_ptr(mem_manager, del_template->col_index_arr);
        del_template->col_index_arr = NULL;
    }

    if (del_template->global_row_index_compress_meta != NULL && is_deleted(mem_manager, del_template->global_row_index_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->global_row_index_compress_meta, del_template->global_row_index_compress);
        register_del_ptr(mem_manager, del_template->global_row_index_compress_meta);
        del_template->global_row_index_compress_meta = NULL;
    }

    if (del_template->row_index_before_sort_compress_meta != NULL && is_deleted(mem_manager, del_template->row_index_before_sort_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->row_index_before_sort_compress_meta, del_template->row_index_before_sort_compress);
        register_del_ptr(mem_manager, del_template->row_index_before_sort_compress_meta);
        del_template->row_index_before_sort_compress_meta = NULL;
    }

    if (del_template != NULL && is_deleted(mem_manager, del_template) == false)
    {
        // 删除模板本身
        delete del_template;
        // 不用登记，不用清零
    }
}

void delete_direct_atom_total_warp_reduce_template(memory_garbage_manager_t *mem_manager, direct_atom_total_warp_reduce_template_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    // 首先析构matrix
    assert(del_template->matrix != NULL);
    if (is_deleted(mem_manager, del_template->matrix) == false)
    {
        delete_sparse_struct_t(mem_manager, del_template->matrix);
        register_del_ptr(mem_manager, del_template->matrix);
        del_template->matrix = NULL;
    }

    if (del_template->global_row_index_of_warp_level_block != NULL && is_deleted(mem_manager, del_template->global_row_index_of_warp_level_block) == false)
    {
        delete_arr_with_data_type(del_template->global_row_index_of_warp_level_block, del_template->data_type_of_global_row_index_of_warp_level_block);
        register_del_ptr(mem_manager, del_template->global_row_index_of_warp_level_block);
        del_template->global_row_index_of_warp_level_block = NULL;
    }

    if (del_template->global_warp_nz_begin_offset != NULL && is_deleted(mem_manager, del_template->global_warp_nz_begin_offset) == false)
    {
        delete_arr_with_data_type(del_template->global_warp_nz_begin_offset, del_template->data_type_of_global_warp_nz_begin_offset);
        register_del_ptr(mem_manager, del_template->global_warp_nz_begin_offset);
        del_template->global_warp_nz_begin_offset = NULL;
    }

    if (del_template->row_index_before_sort != NULL && is_deleted(mem_manager, del_template->row_index_before_sort) == false)
    {
        delete_arr_with_data_type(del_template->row_index_before_sort, del_template->data_type_of_row_index_before_sort);
        register_del_ptr(mem_manager, del_template->row_index_before_sort);
        del_template->row_index_before_sort = NULL;
    }

    if (del_template->val_arr != NULL && is_deleted(mem_manager, del_template->val_arr) == false)
    {
        delete_arr_with_data_type(del_template->val_arr, del_template->data_type_of_val_arr);
        register_del_ptr(mem_manager, del_template->val_arr);
        del_template->val_arr = NULL;
    }

    if (del_template->col_index_arr != NULL && is_deleted(mem_manager, del_template->col_index_arr) == false)
    {
        delete_arr_with_data_type(del_template->col_index_arr, del_template->data_type_of_col_index_arr);
        register_del_ptr(mem_manager, del_template->col_index_arr);
        del_template->col_index_arr = NULL;
    }

    if (del_template->global_row_index_of_warp_level_block_compress_meta != NULL && is_deleted(mem_manager, del_template->global_row_index_of_warp_level_block_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->global_row_index_of_warp_level_block_compress_meta, del_template->global_row_index_of_warp_level_block_compress);
        register_del_ptr(mem_manager, del_template->global_row_index_of_warp_level_block_compress_meta);
        del_template->global_row_index_of_warp_level_block_compress_meta = NULL;
    }

    if (del_template->row_index_before_sort_compress_meta != NULL && is_deleted(mem_manager, del_template->row_index_before_sort_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->row_index_before_sort_compress_meta, del_template->row_index_before_sort_compress);
        register_del_ptr(mem_manager, del_template->row_index_before_sort_compress_meta);
        del_template->row_index_before_sort_compress_meta = NULL;
    }

    if (del_template != NULL && is_deleted(mem_manager, del_template) == false)
    {
        // 删除模板本身
        delete del_template;
        // 不用登记，不用清零
    }
}

void delete_shared_memory_long_row_template(memory_garbage_manager_t *mem_manager, shared_memory_long_row_template_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    // 首先析构matrix
    assert(del_template->matrix != NULL);
    if (is_deleted(mem_manager, del_template->matrix) == false)
    {
        delete_sparse_struct_t(mem_manager, del_template->matrix);
        register_del_ptr(mem_manager, del_template->matrix);
        del_template->matrix = NULL;
    }

    if (del_template->row_index_of_block_level_block != NULL && is_deleted(mem_manager, del_template->row_index_of_block_level_block) == false)
    {
        delete_arr_with_data_type(del_template->row_index_of_block_level_block, del_template->data_type_of_row_index_of_block_level_block);
        register_del_ptr(mem_manager, del_template->row_index_of_block_level_block);
        del_template->row_index_of_block_level_block = NULL;
    }

    if (del_template->block_nz_begin_offset != NULL && is_deleted(mem_manager, del_template->block_nz_begin_offset) == false)
    {
        delete_arr_with_data_type(del_template->block_nz_begin_offset, del_template->data_type_of_block_nz_begin_offset);
        register_del_ptr(mem_manager, del_template->block_nz_begin_offset);
        del_template->block_nz_begin_offset = NULL;
    }

    if (del_template->row_index_before_sort != NULL && is_deleted(mem_manager, del_template->row_index_before_sort) == false)
    {
        delete_arr_with_data_type(del_template->row_index_before_sort, del_template->data_type_of_row_index_before_sort);
        register_del_ptr(mem_manager, del_template->row_index_before_sort);
        del_template->row_index_before_sort = NULL;
    }

    if (del_template->val_arr != NULL && is_deleted(mem_manager, del_template->val_arr) == false)
    {
        delete_arr_with_data_type(del_template->val_arr, del_template->data_type_of_val_arr);
        register_del_ptr(mem_manager, del_template->val_arr);
        del_template->val_arr = NULL;
    }

    if (del_template->col_index_arr != NULL && is_deleted(mem_manager, del_template->col_index_arr) == false)
    {
        delete_arr_with_data_type(del_template->col_index_arr, del_template->data_type_of_col_index_arr);
        register_del_ptr(mem_manager, del_template->col_index_arr);
        del_template->col_index_arr = NULL;
    }

    if (del_template->row_index_of_block_level_block_compress_meta != NULL && is_deleted(mem_manager, del_template->row_index_of_block_level_block_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->row_index_of_block_level_block_compress_meta, del_template->row_index_of_block_level_block_compress);
        register_del_ptr(mem_manager, del_template->row_index_of_block_level_block_compress_meta);
        del_template->row_index_of_block_level_block_compress_meta = NULL;
    }

    if (del_template->block_nz_begin_offset_compress_meta != NULL && is_deleted(mem_manager, del_template->block_nz_begin_offset_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->block_nz_begin_offset_compress_meta, del_template->block_nz_begin_offset_compress);
        register_del_ptr(mem_manager, del_template->block_nz_begin_offset_compress_meta);
        del_template->block_nz_begin_offset_compress_meta = NULL;
    }

    if (del_template->row_index_before_sort_compress_meta != NULL && is_deleted(mem_manager, del_template->row_index_before_sort_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->row_index_before_sort_compress_meta, del_template->row_index_before_sort_compress);
        register_del_ptr(mem_manager, del_template->row_index_before_sort_compress_meta);
        del_template->row_index_before_sort_compress_meta = NULL;
    }

    if (del_template != NULL && is_deleted(mem_manager, del_template) == false)
    {
        // 删除模板本身
        delete del_template;
        // 不用登记，不用清零
    }
}

void delete_shared_memory_template_warp_compress(memory_garbage_manager_t *mem_manager, shared_memory_template_warp_compress_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    // 首先析构matrix
    assert(del_template->matrix != NULL);
    if (is_deleted(mem_manager, del_template->matrix) == false)
    {
        delete_sparse_struct_t(mem_manager, del_template->matrix);
        register_del_ptr(mem_manager, del_template->matrix);
        del_template->matrix = NULL;
    }

    if (del_template->row_offset_in_thread_tmp_result != NULL && is_deleted(mem_manager, del_template->row_offset_in_thread_tmp_result) == false)
    {
        delete_arr_with_data_type(del_template->row_offset_in_thread_tmp_result, del_template->data_type_of_row_offset_in_thread_tmp_result);
        register_del_ptr(mem_manager, del_template->row_offset_in_thread_tmp_result);
        del_template->row_offset_in_thread_tmp_result = NULL;
    }

    if (del_template->block_first_row_index != NULL && is_deleted(mem_manager, del_template->block_first_row_index) == false)
    {
        delete_arr_with_data_type(del_template->block_first_row_index, del_template->data_type_of_block_first_row_index);
        register_del_ptr(mem_manager, del_template->block_first_row_index);
        del_template->block_first_row_index = NULL;
    }

    if (del_template->block_begin_thread_index_offset != NULL && is_deleted(mem_manager, del_template->block_begin_thread_index_offset) == false)
    {
        delete_arr_with_data_type(del_template->block_begin_thread_index_offset, del_template->data_type_of_block_begin_thread_index_offset);
        register_del_ptr(mem_manager, del_template->block_begin_thread_index_offset);
        del_template->block_begin_thread_index_offset = NULL;
    }

    if (del_template->thread_block_size_in_block != NULL && is_deleted(mem_manager, del_template->thread_block_size_in_block) == false)
    {
        delete_arr_with_data_type(del_template->thread_block_size_in_block, del_template->data_type_of_thread_block_size_in_block);
        register_del_ptr(mem_manager, del_template->thread_block_size_in_block);
        del_template->thread_block_size_in_block = NULL;
    }

    if (del_template->row_index_before_sort != NULL && is_deleted(mem_manager, del_template->row_index_before_sort) == false)
    {
        delete_arr_with_data_type(del_template->row_index_before_sort, del_template->data_type_of_row_index_before_sort);
        register_del_ptr(mem_manager, del_template->row_index_before_sort);
        del_template->row_index_before_sort = NULL;
    }

    if (del_template->block_nz_begin_offset != NULL && is_deleted(mem_manager, del_template->block_nz_begin_offset) == false)
    {
        delete_arr_with_data_type(del_template->block_nz_begin_offset, del_template->data_type_of_block_nz_begin_offset);
        register_del_ptr(mem_manager, del_template->block_nz_begin_offset);
        del_template->block_nz_begin_offset = NULL;
    }

    if (del_template->val_arr != NULL && is_deleted(mem_manager, del_template->val_arr) == false)
    {
        delete_arr_with_data_type(del_template->val_arr, del_template->data_type_of_val_arr);
        register_del_ptr(mem_manager, del_template->val_arr);
        del_template->val_arr = NULL;
    }

    if (del_template->col_index_arr != NULL && is_deleted(mem_manager, del_template->col_index_arr) == false)
    {
        delete_arr_with_data_type(del_template->col_index_arr, del_template->data_type_of_col_index_arr);
        register_del_ptr(mem_manager, del_template->col_index_arr);
        del_template->col_index_arr = NULL;
    }

    if (del_template->row_offset_in_thread_tmp_result_compress_meta != NULL && is_deleted(mem_manager, del_template->row_offset_in_thread_tmp_result_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->row_offset_in_thread_tmp_result_compress_meta, del_template->row_offset_in_thread_tmp_result_compress);
        register_del_ptr(mem_manager, del_template->row_offset_in_thread_tmp_result_compress_meta);
        del_template->row_offset_in_thread_tmp_result_compress_meta = NULL;
    }

    if (del_template->block_first_row_index_compress_meta != NULL && is_deleted(mem_manager, del_template->block_first_row_index_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->block_first_row_index_compress_meta, del_template->block_first_row_index_compress);
        register_del_ptr(mem_manager, del_template->block_first_row_index_compress_meta);
        del_template->block_first_row_index_compress_meta = NULL;
    }

    if (del_template->block_begin_thread_index_offset_compress_meta != NULL && is_deleted(mem_manager, del_template->block_begin_thread_index_offset_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->block_begin_thread_index_offset_compress_meta, del_template->block_begin_thread_index_offset_compress);
        register_del_ptr(mem_manager, del_template->block_begin_thread_index_offset_compress_meta);
        del_template->block_begin_thread_index_offset_compress_meta = NULL;
    }

    if (del_template->thread_block_size_in_block_compress_meta != NULL && is_deleted(mem_manager, del_template->thread_block_size_in_block_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->thread_block_size_in_block_compress_meta, del_template->thread_block_size_in_block_compress);
        register_del_ptr(mem_manager, del_template->thread_block_size_in_block_compress_meta);
        del_template->thread_block_size_in_block_compress_meta = NULL;
    }

    if (del_template->row_index_before_sort_compress_meta != NULL && is_deleted(mem_manager, del_template->row_index_before_sort_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->row_index_before_sort_compress_meta, del_template->row_index_before_sort_compress);
        register_del_ptr(mem_manager, del_template->row_index_before_sort_compress_meta);
        del_template->row_index_before_sort_compress_meta = NULL;
    }

    if (del_template->block_nz_begin_offset_compress_meta != NULL && is_deleted(mem_manager, del_template->block_nz_begin_offset_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->block_nz_begin_offset_compress_meta, del_template->block_nz_begin_offset_compress);
        register_del_ptr(mem_manager, del_template->block_nz_begin_offset_compress_meta);
        del_template->block_nz_begin_offset_compress_meta = NULL;
    }

    if (del_template != NULL && is_deleted(mem_manager, del_template) == false)
    {
        // 删除模板本身
        delete del_template;
        // 不用登记，不用清零
    }
}

void delete_shared_memory_template(memory_garbage_manager_t *mem_manager, shared_memory_template_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    // 首先析构matrix
    assert(del_template->matrix != NULL);
    if (is_deleted(mem_manager, del_template->matrix) == false)
    {
        delete_sparse_struct_t(mem_manager, del_template->matrix);
        register_del_ptr(mem_manager, del_template->matrix);
        del_template->matrix = NULL;
    }

    if (del_template->row_offset_in_thread_tmp_result != NULL && is_deleted(mem_manager, del_template->row_offset_in_thread_tmp_result) == false)
    {
        delete_arr_with_data_type(del_template->row_offset_in_thread_tmp_result, del_template->data_type_of_row_offset_in_thread_tmp_result);
        register_del_ptr(mem_manager, del_template->row_offset_in_thread_tmp_result);
        del_template->row_offset_in_thread_tmp_result = NULL;
    }

    if (del_template->block_first_row_index != NULL && is_deleted(mem_manager, del_template->block_first_row_index) == false)
    {
        delete_arr_with_data_type(del_template->block_first_row_index, del_template->data_type_of_block_first_row_index);
        register_del_ptr(mem_manager, del_template->block_first_row_index);
        del_template->block_first_row_index = NULL;
    }

    if (del_template->block_begin_warp_index_offset != NULL && is_deleted(mem_manager, del_template->block_begin_warp_index_offset) == false)
    {
        delete_arr_with_data_type(del_template->block_begin_warp_index_offset, del_template->data_type_of_block_begin_warp_index_offset);
        register_del_ptr(mem_manager, del_template->block_begin_warp_index_offset);
        del_template->block_begin_warp_index_offset = NULL;
    }

    if (del_template->warp_begin_thread_index_offset != NULL && is_deleted(mem_manager, del_template->warp_begin_thread_index_offset) == false)
    {
        delete_arr_with_data_type(del_template->warp_begin_thread_index_offset, del_template->data_type_of_warp_begin_thread_index_offset);
        register_del_ptr(mem_manager, del_template->warp_begin_thread_index_offset);
        del_template->warp_begin_thread_index_offset = NULL;
    }

    if (del_template->thread_block_size_in_warp != NULL && is_deleted(mem_manager, del_template->thread_block_size_in_warp) == false)
    {
        delete_arr_with_data_type(del_template->thread_block_size_in_warp, del_template->data_type_of_thread_block_size_in_warp);
        register_del_ptr(mem_manager, del_template->thread_block_size_in_warp);
        del_template->thread_block_size_in_warp = NULL;
    }

    if (del_template->row_index_before_sort != NULL && is_deleted(mem_manager, del_template->row_index_before_sort) == false)
    {
        delete_arr_with_data_type(del_template->row_index_before_sort, del_template->data_type_of_row_index_before_sort);
        register_del_ptr(mem_manager, del_template->row_index_before_sort);
        del_template->row_index_before_sort = NULL;
    }

    if (del_template->block_nz_begin_offset != NULL && is_deleted(mem_manager, del_template->block_nz_begin_offset) == false)
    {
        delete_arr_with_data_type(del_template->block_nz_begin_offset, del_template->data_type_of_block_nz_begin_offset);
        register_del_ptr(mem_manager, del_template->block_nz_begin_offset);
        del_template->block_nz_begin_offset = NULL;
    }

    if (del_template->warp_nz_begin_offset != NULL && is_deleted(mem_manager, del_template->warp_nz_begin_offset) == false)
    {
        delete_arr_with_data_type(del_template->warp_nz_begin_offset, del_template->data_type_of_warp_nz_begin_offset);
        register_del_ptr(mem_manager, del_template->warp_nz_begin_offset);
        del_template->warp_nz_begin_offset = NULL;
    }

    if (del_template->val_arr != NULL && is_deleted(mem_manager, del_template->val_arr) == false)
    {
        delete_arr_with_data_type(del_template->val_arr, del_template->data_type_of_val_arr);
        register_del_ptr(mem_manager, del_template->val_arr);
        del_template->val_arr = NULL;
    }

    if (del_template->col_index_arr != NULL && is_deleted(mem_manager, del_template->col_index_arr) == false)
    {
        delete_arr_with_data_type(del_template->col_index_arr, del_template->data_type_of_col_index_arr);
        register_del_ptr(mem_manager, del_template->col_index_arr);
        del_template->col_index_arr = NULL;
    }

    if (del_template->row_offset_in_thread_tmp_result_compress_meta != NULL && is_deleted(mem_manager, del_template->row_offset_in_thread_tmp_result_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->row_offset_in_thread_tmp_result_compress_meta, del_template->row_offset_in_thread_tmp_result_compress);
        register_del_ptr(mem_manager, del_template->row_offset_in_thread_tmp_result_compress_meta);
        del_template->row_offset_in_thread_tmp_result_compress_meta = NULL;
    }

    if (del_template->block_first_row_index_compress_meta != NULL && is_deleted(mem_manager, del_template->block_first_row_index_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->block_first_row_index_compress_meta, del_template->block_first_row_index_compress);
        register_del_ptr(mem_manager, del_template->block_first_row_index_compress_meta);
        del_template->block_first_row_index_compress_meta = NULL;
    }

    if (del_template->block_begin_warp_index_offset_compress_meta != NULL && is_deleted(mem_manager, del_template->block_begin_warp_index_offset_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->block_begin_warp_index_offset_compress_meta, del_template->block_begin_warp_index_offset_compress);
        register_del_ptr(mem_manager, del_template->block_begin_warp_index_offset_compress_meta);
        del_template->block_begin_warp_index_offset_compress_meta = NULL;
    }

    if (del_template->warp_begin_thread_index_offset_compress_meta != NULL && is_deleted(mem_manager, del_template->warp_begin_thread_index_offset_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->warp_begin_thread_index_offset_compress_meta, del_template->warp_begin_thread_index_offset_compress);
        register_del_ptr(mem_manager, del_template->warp_begin_thread_index_offset_compress_meta);
        del_template->warp_begin_thread_index_offset_compress_meta = NULL;
    }

    if (del_template->thread_block_size_in_warp_compress_meta != NULL && is_deleted(mem_manager, del_template->thread_block_size_in_warp_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->thread_block_size_in_warp_compress_meta, del_template->thread_block_size_in_warp_compress);
        register_del_ptr(mem_manager, del_template->thread_block_size_in_warp_compress_meta);
        del_template->thread_block_size_in_warp_compress_meta = NULL;
    }

    if (del_template->row_index_before_sort_compress_meta != NULL && is_deleted(mem_manager, del_template->row_index_before_sort_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->row_index_before_sort_compress_meta, del_template->row_index_before_sort_compress);
        register_del_ptr(mem_manager, del_template->row_index_before_sort_compress_meta);
        del_template->row_index_before_sort_compress_meta = NULL;
    }

    if (del_template->block_nz_begin_offset_compress_meta != NULL && is_deleted(mem_manager, del_template->block_nz_begin_offset_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->block_nz_begin_offset_compress_meta, del_template->block_nz_begin_offset_compress);
        register_del_ptr(mem_manager, del_template->block_nz_begin_offset_compress_meta);
        del_template->block_nz_begin_offset_compress_meta = NULL;
    }

    if (del_template->warp_nz_begin_offset_compress_meta != NULL && is_deleted(mem_manager, del_template->warp_nz_begin_offset_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->warp_nz_begin_offset_compress_meta, del_template->warp_nz_begin_offset_compress);
        register_del_ptr(mem_manager, del_template->warp_nz_begin_offset_compress_meta);
        del_template->warp_nz_begin_offset_compress_meta = NULL;
    }

    if (del_template != NULL && is_deleted(mem_manager, del_template) == false)
    {
        // 删除模板本身
        delete del_template;
        // 不用登记，不用清零
    }
}

void delete_shared_memory_total_warp_reduce_template(memory_garbage_manager_t *mem_manager, shared_memory_total_warp_reduce_template_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    // 首先析构matrix
    assert(del_template->matrix != NULL);
    if (is_deleted(mem_manager, del_template->matrix) == false)
    {
        delete_sparse_struct_t(mem_manager, del_template->matrix);
        register_del_ptr(mem_manager, del_template->matrix);
        del_template->matrix = NULL;
    }

    if (del_template->row_offset_in_warp_tmp_result != NULL && is_deleted(mem_manager, del_template->row_offset_in_warp_tmp_result) == false)
    {
        delete_arr_with_data_type(del_template->row_offset_in_warp_tmp_result, del_template->data_type_of_row_offset_in_warp_tmp_result);
        register_del_ptr(mem_manager, del_template->row_offset_in_warp_tmp_result);
        del_template->row_offset_in_warp_tmp_result = NULL;
    }

    if (del_template->block_first_row_index != NULL && is_deleted(mem_manager, del_template->block_first_row_index) == false)
    {
        delete_arr_with_data_type(del_template->block_first_row_index, del_template->data_type_of_block_first_row_index);
        register_del_ptr(mem_manager, del_template->block_first_row_index);
        del_template->block_first_row_index = NULL;
    }

    if (del_template->block_begin_warp_index_offset != NULL && is_deleted(mem_manager, del_template->block_begin_warp_index_offset) == false)
    {
        delete_arr_with_data_type(del_template->block_begin_warp_index_offset, del_template->data_type_of_block_begin_warp_index_offset);
        register_del_ptr(mem_manager, del_template->block_begin_warp_index_offset);
        del_template->block_begin_warp_index_offset = NULL;
    }

    if (del_template->row_index_before_sort != NULL && is_deleted(mem_manager, del_template->row_index_before_sort) == false)
    {
        delete_arr_with_data_type(del_template->row_index_before_sort, del_template->data_type_of_row_index_before_sort);
        register_del_ptr(mem_manager, del_template->row_index_before_sort);
        del_template->row_index_before_sort = NULL;
    }

    if (del_template->global_warp_block_first_nz != NULL && is_deleted(mem_manager, del_template->global_warp_block_first_nz) == false)
    {
        delete_arr_with_data_type(del_template->global_warp_block_first_nz, del_template->data_type_of_global_warp_block_first_nz);
        register_del_ptr(mem_manager, del_template->global_warp_block_first_nz);
        del_template->global_warp_block_first_nz = NULL;
    }

    if (del_template->val_arr != NULL && is_deleted(mem_manager, del_template->val_arr) == false)
    {
        delete_arr_with_data_type(del_template->val_arr, del_template->data_type_of_val_arr);
        register_del_ptr(mem_manager, del_template->val_arr);
        del_template->val_arr = NULL;
    }

    if (del_template->col_index_arr != NULL && is_deleted(mem_manager, del_template->col_index_arr) == false)
    {
        delete_arr_with_data_type(del_template->col_index_arr, del_template->data_type_of_col_index_arr);
        register_del_ptr(mem_manager, del_template->col_index_arr);
        del_template->col_index_arr = NULL;
    }

    if (del_template->row_offset_in_warp_tmp_result_compress_meta != NULL && is_deleted(mem_manager, del_template->row_offset_in_warp_tmp_result_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->row_offset_in_warp_tmp_result_compress_meta, del_template->row_offset_in_warp_tmp_result_compress);
        register_del_ptr(mem_manager, del_template->row_offset_in_warp_tmp_result_compress_meta);
        del_template->row_offset_in_warp_tmp_result_compress_meta = NULL;
    }

    if (del_template->block_first_row_index_compress_meta != NULL && is_deleted(mem_manager, del_template->block_first_row_index_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->block_first_row_index_compress_meta, del_template->block_first_row_index_compress);
        register_del_ptr(mem_manager, del_template->block_first_row_index_compress_meta);
        del_template->block_first_row_index_compress_meta = NULL;
    }

    if (del_template->block_begin_warp_index_offset_compress_meta != NULL && is_deleted(mem_manager, del_template->block_begin_warp_index_offset_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->block_begin_warp_index_offset_compress_meta, del_template->block_begin_warp_index_offset_compress);
        register_del_ptr(mem_manager, del_template->block_begin_warp_index_offset_compress_meta);
        del_template->block_begin_warp_index_offset_compress_meta = NULL;
    }

    if (del_template->global_warp_block_first_nz_compress_meta != NULL && is_deleted(mem_manager, del_template->global_warp_block_first_nz_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->global_warp_block_first_nz_compress_meta, del_template->global_warp_block_first_nz_compress);
        register_del_ptr(mem_manager, del_template->global_warp_block_first_nz_compress_meta);
        del_template->global_warp_block_first_nz_compress_meta = NULL;
    }

    if (del_template->row_index_before_sort_compress_meta != NULL && is_deleted(mem_manager, del_template->row_index_before_sort_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->row_index_before_sort_compress_meta, del_template->row_index_before_sort_compress);
        register_del_ptr(mem_manager, del_template->row_index_before_sort_compress_meta);
        del_template->row_index_before_sort_compress_meta = NULL;
    }

    if (del_template != NULL && is_deleted(mem_manager, del_template) == false)
    {
        // 删除模板本身
        delete del_template;
        // 不用登记，不用清零
    }
}

void delete_unaligned_warp_reduce_same_TLB_size_template(memory_garbage_manager_t *mem_manager, unaligned_warp_reduce_same_TLB_size_template_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 当前模板还没被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    if (del_template->matrix != NULL && is_deleted(mem_manager, del_template->matrix) == false)
    {
        delete_sparse_struct_t(mem_manager, del_template->matrix);
        register_del_ptr(mem_manager, del_template->matrix);
        del_template->matrix = NULL;
    }

    if (del_template->global_first_row_index_of_warp_level_block != NULL && is_deleted(mem_manager, del_template->global_first_row_index_of_warp_level_block) == false)
    {
        delete_arr_with_data_type(del_template->global_first_row_index_of_warp_level_block, del_template->data_type_of_global_first_row_index_of_warp_level_block);
        register_del_ptr(mem_manager, del_template->global_first_row_index_of_warp_level_block);
        del_template->global_first_row_index_of_warp_level_block = NULL;
    }

    if (del_template->first_relative_reduce_row_of_thread_level_block != NULL && is_deleted(mem_manager, del_template->first_relative_reduce_row_of_thread_level_block) == false)
    {
        delete_arr_with_data_type(del_template->first_relative_reduce_row_of_thread_level_block, del_template->data_type_of_first_relative_reduce_row_of_thread_level_block);
        register_del_ptr(mem_manager, del_template->first_relative_reduce_row_of_thread_level_block);
        del_template->first_relative_reduce_row_of_thread_level_block = NULL;
    }

    if (del_template->tmp_result_reduce_offset_of_thread_level_block != NULL && is_deleted(mem_manager, del_template->tmp_result_reduce_offset_of_thread_level_block) == false)
    {
        delete_arr_with_data_type(del_template->tmp_result_reduce_offset_of_thread_level_block, del_template->data_type_of_tmp_result_reduce_offset_of_thread_level_block);
        register_del_ptr(mem_manager, del_template->tmp_result_reduce_offset_of_thread_level_block);
        del_template->tmp_result_reduce_offset_of_thread_level_block = NULL;
    }

    if (del_template->combine_meta_of_thread_level_block != NULL && is_deleted(mem_manager, del_template->combine_meta_of_thread_level_block) == false)
    {
        delete_arr_with_data_type(del_template->combine_meta_of_thread_level_block, del_template->data_type_of_combine_meta_of_thread_level_block);
        register_del_ptr(mem_manager, del_template->combine_meta_of_thread_level_block);
        del_template->combine_meta_of_thread_level_block = NULL;
    }

    if (del_template->row_index_before_sort != NULL && is_deleted(mem_manager, del_template->row_index_before_sort) == false)
    {
        delete_arr_with_data_type(del_template->row_index_before_sort, del_template->data_type_of_row_index_before_sort);
        register_del_ptr(mem_manager, del_template->row_index_before_sort);
        del_template->row_index_before_sort = NULL;
    }

    if (del_template->val_arr != NULL && is_deleted(mem_manager, del_template->val_arr) == false)
    {
        delete_arr_with_data_type(del_template->val_arr, del_template->data_type_of_val_arr);
        register_del_ptr(mem_manager, del_template->val_arr);
        del_template->val_arr = NULL;
    }

    if (del_template->col_index_arr != NULL && is_deleted(mem_manager, del_template->col_index_arr) == false)
    {
        delete_arr_with_data_type(del_template->col_index_arr, del_template->data_type_of_col_index_arr);
        register_del_ptr(mem_manager, del_template->col_index_arr);
        del_template->col_index_arr = NULL;
    }

    // 压缩所有的压缩器
    if (del_template->global_first_row_index_of_warp_level_block_compress_meta != NULL && is_deleted(mem_manager, del_template->global_first_row_index_of_warp_level_block_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->global_first_row_index_of_warp_level_block_compress_meta, del_template->global_first_row_index_of_warp_level_block_compress);
        register_del_ptr(mem_manager, del_template->global_first_row_index_of_warp_level_block_compress_meta);
        del_template->global_first_row_index_of_warp_level_block_compress_meta = NULL;
    }

    if (del_template->row_index_before_sort_compress_meta != NULL && is_deleted(mem_manager, del_template->row_index_before_sort_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->row_index_before_sort_compress_meta, del_template->row_index_before_sort_compress);
        register_del_ptr(mem_manager, del_template->row_index_before_sort_compress_meta);
        del_template->row_index_before_sort_compress_meta = NULL;
    }

    if (del_template != NULL && is_deleted(mem_manager, del_template) == false)
    {
        // 删除模板本身
        delete del_template;
        // 不用登记，不用清零
    }
}

void delete_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce(memory_garbage_manager_t *mem_manager, unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 当前模板还没被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    if (del_template->matrix != NULL && is_deleted(mem_manager, del_template->matrix) == false)
    {
        delete_sparse_struct_t(mem_manager, del_template->matrix);
        register_del_ptr(mem_manager, del_template->matrix);
        del_template->matrix = NULL;
    }

    if (del_template->global_first_row_index_of_warp_level_block != NULL && is_deleted(mem_manager, del_template->global_first_row_index_of_warp_level_block) == false)
    {
        delete_arr_with_data_type(del_template->global_first_row_index_of_warp_level_block, del_template->data_type_of_global_first_row_index_of_warp_level_block);
        register_del_ptr(mem_manager, del_template->global_first_row_index_of_warp_level_block);
        del_template->global_first_row_index_of_warp_level_block = NULL;
    }

    if (del_template->first_relative_reduce_row_of_thread_level_block != NULL && is_deleted(mem_manager, del_template->first_relative_reduce_row_of_thread_level_block) == false)
    {
        delete_arr_with_data_type(del_template->first_relative_reduce_row_of_thread_level_block, del_template->data_type_of_first_relative_reduce_row_of_thread_level_block);
        register_del_ptr(mem_manager, del_template->first_relative_reduce_row_of_thread_level_block);
        del_template->first_relative_reduce_row_of_thread_level_block = NULL;
    }

    if (del_template->tmp_result_reduce_offset_of_thread_level_block != NULL && is_deleted(mem_manager, del_template->tmp_result_reduce_offset_of_thread_level_block) == false)
    {
        delete_arr_with_data_type(del_template->tmp_result_reduce_offset_of_thread_level_block, del_template->data_type_of_tmp_result_reduce_offset_of_thread_level_block);
        register_del_ptr(mem_manager, del_template->tmp_result_reduce_offset_of_thread_level_block);
        del_template->tmp_result_reduce_offset_of_thread_level_block = NULL;
    }

    if (del_template->combine_meta_of_thread_level_block != NULL && is_deleted(mem_manager, del_template->combine_meta_of_thread_level_block) == false)
    {
        delete_arr_with_data_type(del_template->combine_meta_of_thread_level_block, del_template->data_type_of_combine_meta_of_thread_level_block);
        register_del_ptr(mem_manager, del_template->combine_meta_of_thread_level_block);
        del_template->combine_meta_of_thread_level_block = NULL;
    }

    if (del_template->row_index_before_sort != NULL && is_deleted(mem_manager, del_template->row_index_before_sort) == false)
    {
        delete_arr_with_data_type(del_template->row_index_before_sort, del_template->data_type_of_row_index_before_sort);
        register_del_ptr(mem_manager, del_template->row_index_before_sort);
        del_template->row_index_before_sort = NULL;
    }

    if (del_template->val_arr != NULL && is_deleted(mem_manager, del_template->val_arr) == false)
    {
        delete_arr_with_data_type(del_template->val_arr, del_template->data_type_of_val_arr);
        register_del_ptr(mem_manager, del_template->val_arr);
        del_template->val_arr = NULL;
    }

    if (del_template->col_index_arr != NULL && is_deleted(mem_manager, del_template->col_index_arr) == false)
    {
        delete_arr_with_data_type(del_template->col_index_arr, del_template->data_type_of_col_index_arr);
        register_del_ptr(mem_manager, del_template->col_index_arr);
        del_template->col_index_arr = NULL;
    }

    // 压缩所有的压缩器
    if (del_template->global_first_row_index_of_warp_level_block_compress_meta != NULL && is_deleted(mem_manager, del_template->global_first_row_index_of_warp_level_block_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->global_first_row_index_of_warp_level_block_compress_meta, del_template->global_first_row_index_of_warp_level_block_compress);
        register_del_ptr(mem_manager, del_template->global_first_row_index_of_warp_level_block_compress_meta);
        del_template->global_first_row_index_of_warp_level_block_compress_meta = NULL;
    }

    if (del_template->row_index_before_sort_compress_meta != NULL && is_deleted(mem_manager, del_template->row_index_before_sort_compress_meta) == false)
    {
        delete_compressor_with_type(del_template->row_index_before_sort_compress_meta, del_template->row_index_before_sort_compress);
        register_del_ptr(mem_manager, del_template->row_index_before_sort_compress_meta);
        del_template->row_index_before_sort_compress_meta = NULL;
    }

    if (del_template != NULL && is_deleted(mem_manager, del_template) == false)
    {
        // 删除模板本身
        delete del_template;
        // 不用登记，不用清零
    }
}

// 在这个函数外面执行指针登记
void delete_template_with_type(memory_garbage_manager_t *mem_manager, void *template_ptr, template_type type)
{
    assert(mem_manager != NULL && template_ptr != NULL);

    // 所有的模板析构完之后都要登记一下
    if (type == DIRECT_ATOM_TEMPLATE)
    {
        delete_direct_atom_template(mem_manager, (direct_atom_template_t *)template_ptr);
        return;
    }

    if (type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
    {
        delete_direct_atom_template_warp_compress(mem_manager, (direct_atom_template_warp_compress_t *)template_ptr);
        return;
    }

    if (type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
    {
        delete_direct_atom_template_warp_block_compress(mem_manager, (direct_atom_template_warp_block_compress_t *)template_ptr);
        return;
    }

    if (type == SHARED_MEMORY_TEMPLATE)
    {
        delete_shared_memory_template(mem_manager, (shared_memory_template_t *)template_ptr);
        return;
    }

    if (type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
    {
        delete_shared_memory_template_warp_compress(mem_manager, (shared_memory_template_warp_compress_t *)template_ptr);
        return;
    }

    if (type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
    {
        delete_shared_memory_long_row_template(mem_manager, (shared_memory_long_row_template_t *)template_ptr);
        return;
    }

    if (type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        delete_shared_memory_total_warp_reduce_template(mem_manager, (shared_memory_total_warp_reduce_template_t *)template_ptr);
        return;
    }

    if (type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        delete_direct_atom_total_warp_reduce_template(mem_manager, (direct_atom_total_warp_reduce_template_t *)template_ptr);
        return;
    }

    if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
    {
        delete_unaligned_warp_reduce_same_TLB_size_template(mem_manager, (unaligned_warp_reduce_same_TLB_size_template_t *)template_ptr);
        return;
    }

    if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
    {
        delete_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce(mem_manager, (unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *)template_ptr);
        return;
    }

    cout << "delete_template_with_type:template type is not supported" << endl;
    assert(false);
}

// 析构操作管理器
void delete_op_manager(memory_garbage_manager_t *mem_manager, operator_manager_t *op_manager)
{
    assert(mem_manager != NULL && op_manager != NULL);

    // 析构矩阵
    assert(op_manager->matrix != NULL);

    if (is_deleted(mem_manager, op_manager->matrix) == false)
    {
        delete_sparse_struct_t(mem_manager, op_manager->matrix);
        register_del_ptr(mem_manager, op_manager->matrix);
        op_manager->matrix = NULL;
    }

    // 析构自己，但是不登记，等级永远是在外面
    delete op_manager;
}

// 析构代码生成器
void delete_code_builder(memory_garbage_manager_t *mem_manager, code_builder_t *builder)
{
    assert(mem_manager != NULL && builder != NULL);

    // 首先析构操作管理器，操作管理器的指针是必然存在的
    assert(builder->op_manager != NULL);
    if (is_deleted(mem_manager, builder->op_manager) == false)
    {
        delete_op_manager(mem_manager, builder->op_manager);
        register_del_ptr(mem_manager, builder->op_manager);
        builder->op_manager = NULL;
    }

    // 析构模板
    assert(builder->template_type_vec.size() == builder->template_vec.size());
    assert(builder->template_type_vec.size() > 0);

    for (int template_id = 0; template_id < builder->template_vec.size(); template_id++)
    {
        // 析构模板
        void *template_ptr = builder->template_vec[template_id];
        template_type type = builder->template_type_vec[template_id];
        // 如果模板是不存在的，就不析构了
        if (template_ptr == NULL)
        {
            assert(type == NONE_TEMPLATE);
            // 直接跳过
            continue;
        }
        assert(template_ptr != NULL);
        delete_template_with_type(mem_manager, template_ptr, type);
        register_del_ptr(mem_manager, template_ptr);
        builder->template_vec[template_id] = NULL;
    }

    // 析构自己，但是不登记
    delete builder;
}

// 析构一个代码生成器除了矩阵的部分
void delete_code_builder_without_operator_manager(memory_garbage_manager *mem_manager, code_builder_t *builder)
{
    assert(mem_manager != NULL && builder != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;

    assert(matrix != NULL);

    // 将所有的矩阵子块全都登记一下
    vector<int> index_of_sub_matrix;

    for (int i = 0; i < matrix->block_coor_table.item_arr.size(); i++)
    {
        index_of_sub_matrix.push_back(i);
    }

    // 收集所有matrix矩阵的指针
    set<void *> ptr_set_of_matrix = get_all_mem_ptr_from_matrix_dense_view_and_some_compressed_sub_block(matrix, index_of_sub_matrix);

    // 操作管理器也放进来
    ptr_set_of_matrix.insert(builder->op_manager);

    // 放到内存管理器中
    mem_manager->ptr_set.insert(ptr_set_of_matrix.begin(), ptr_set_of_matrix.end());

    // 执行析构程序
    delete_code_builder(mem_manager, builder);
}

set<void *> get_all_mem_ptr_from_matrix_dense_view_and_some_compressed_sub_block(sparse_struct_t *matrix, vector<int> not_need_to_del_compressed_block_id)
{
    assert(matrix != NULL);

    // 从自己开始注册
    set<void *> ptr_need_to_be_protect_set;

    // 把自己注册了
    ptr_need_to_be_protect_set.insert(matrix);

    // 遍历需要被保护的子块
    for (int i = 0; i < not_need_to_del_compressed_block_id.size(); i++)
    {
        int item_id = not_need_to_del_compressed_block_id[i];

        assert(item_id < matrix->block_coor_table.item_arr.size());

        // 如果当前子块是空的，那就不注册了
        dense_block_table_item_t *item_ptr = matrix->block_coor_table.item_arr[item_id];
        if (item_ptr == NULL)
        {
            // 没必要进一步析构了
            continue;
        }

        // 当前子块不空，需要进一步需要登记保护起来
        ptr_need_to_be_protect_set.insert(item_ptr);

        // 压缩之后的指针
        compressed_block_t *compressed_block_ptr = matrix->block_coor_table.item_arr[item_id]->compressed_block_ptr;

        if (compressed_block_ptr == NULL)
        {
            continue;
        }

        ptr_need_to_be_protect_set.insert(compressed_block_ptr);

        // 当前压缩子块不空可以进一步获取其中的内容
        for (int read_index_id = 0; read_index_id < compressed_block_ptr->read_index.size(); read_index_id++)
        {
            index_of_compress_block_t *index_ptr = compressed_block_ptr->read_index[read_index_id];
            // 肯定不是空的
            assert(index_ptr != NULL);

            ptr_need_to_be_protect_set.insert(index_ptr);

            // 析构压缩索引中的所有内容
            if (index_ptr->index_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->index_arr);
            }

            if (index_ptr->is_sort_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->is_sort_arr);
            }

            if (index_ptr->index_of_the_first_row_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->index_of_the_first_row_arr);
            }

            if (index_ptr->row_number_of_block_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->row_number_of_block_arr);
            }

            if (index_ptr->tmp_result_write_index_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->tmp_result_write_index_arr);
            }

            if (index_ptr->coo_begin_index_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->coo_begin_index_arr);
            }

            if (index_ptr->coo_block_size_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->coo_block_size_arr);
            }

            if (index_ptr->child_tmp_row_csr_index_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->child_tmp_row_csr_index_arr);
            }

            if (index_ptr->begin_index_in_tmp_row_csr_arr_of_block != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->begin_index_in_tmp_row_csr_arr_of_block);
            }
        }

        // 搜集当前子块所有的y_write_index
        for (int write_index_id = 0; write_index_id < compressed_block_ptr->y_write_index.size(); write_index_id++)
        {
            index_of_compress_block_t *index_ptr = compressed_block_ptr->y_write_index[write_index_id];

            assert(index_ptr != NULL);

            // 注册
            ptr_need_to_be_protect_set.insert(index_ptr);

            // 析构压缩索引中的所有内容
            if (index_ptr->index_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->index_arr);
            }

            if (index_ptr->is_sort_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->is_sort_arr);
            }

            if (index_ptr->index_of_the_first_row_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->index_of_the_first_row_arr);
            }

            if (index_ptr->row_number_of_block_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->row_number_of_block_arr);
            }

            if (index_ptr->tmp_result_write_index_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->tmp_result_write_index_arr);
            }

            if (index_ptr->coo_begin_index_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->coo_begin_index_arr);
            }

            if (index_ptr->coo_block_size_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->coo_block_size_arr);
            }

            if (index_ptr->child_tmp_row_csr_index_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->child_tmp_row_csr_index_arr);
            }

            if (index_ptr->begin_index_in_tmp_row_csr_arr_of_block != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->begin_index_in_tmp_row_csr_arr_of_block);
            }
        }

        // 注册reduce_help_csr
        for (int reduce_help_csr_index_id = 0; reduce_help_csr_index_id < compressed_block_ptr->reduce_help_csr.size(); reduce_help_csr_index_id++)
        {
            index_of_compress_block_t *index_ptr = compressed_block_ptr->reduce_help_csr[reduce_help_csr_index_id];

            assert(index_ptr != NULL);

            // 注册
            ptr_need_to_be_protect_set.insert(index_ptr);

            // 析构压缩索引中的所有内容
            if (index_ptr->index_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->index_arr);
            }

            if (index_ptr->is_sort_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->is_sort_arr);
            }

            if (index_ptr->index_of_the_first_row_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->index_of_the_first_row_arr);
            }

            if (index_ptr->row_number_of_block_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->row_number_of_block_arr);
            }

            if (index_ptr->tmp_result_write_index_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->tmp_result_write_index_arr);
            }

            if (index_ptr->coo_begin_index_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->coo_begin_index_arr);
            }

            if (index_ptr->coo_block_size_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->coo_block_size_arr);
            }

            if (index_ptr->child_tmp_row_csr_index_arr != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->child_tmp_row_csr_index_arr);
            }

            if (index_ptr->begin_index_in_tmp_row_csr_arr_of_block != NULL)
            {
                ptr_need_to_be_protect_set.insert(index_ptr->begin_index_in_tmp_row_csr_arr_of_block);
            }
        }

        // 注册compressed block的几个数组
        if (compressed_block_ptr->val_arr != NULL)
        {
            ptr_need_to_be_protect_set.insert(compressed_block_ptr->val_arr);
        }

        if (compressed_block_ptr->padding_val_arr != NULL)
        {
            ptr_need_to_be_protect_set.insert(compressed_block_ptr->padding_val_arr);
        }

        if (compressed_block_ptr->staggered_padding_val_arr != NULL)
        {
            ptr_need_to_be_protect_set.insert(compressed_block_ptr->staggered_padding_val_arr);
        }
    }

    // 在矩阵中的其他数组
    if (matrix->sorted_row_index != NULL)
    {
        ptr_need_to_be_protect_set.insert(matrix->sorted_row_index);
    }

    if (matrix->compressed_block_arr != NULL)
    {
        ptr_need_to_be_protect_set.insert(matrix->compressed_block_arr);
    }

    if (matrix->coo_row_index_cache != NULL)
    {
        ptr_need_to_be_protect_set.insert(matrix->coo_row_index_cache);
    }

    if (matrix->coo_col_index_cache != NULL)
    {
        ptr_need_to_be_protect_set.insert(matrix->coo_col_index_cache);
    }

    if (matrix->coo_value_cache != NULL)
    {
        ptr_need_to_be_protect_set.insert(matrix->coo_value_cache);
    }

    if (matrix->coo_x_cache.x_arr != NULL)
    {
        ptr_need_to_be_protect_set.insert(matrix->coo_x_cache.x_arr);
    }

    return ptr_need_to_be_protect_set;
}

void delete_template_without_matrix_with_type(memory_garbage_manager_t *mem_manager, code_builder_t *builder, int template_id_in_code_builder)
{
    assert(mem_manager != NULL && builder != NULL && template_id_in_code_builder < builder->template_vec.size());

    template_type type = builder->template_type_vec[template_id_in_code_builder];
    void *template_ptr = builder->template_vec[template_id_in_code_builder];

    delete_template_without_matrix_with_type(mem_manager, template_ptr, type);

    builder->template_vec[template_id_in_code_builder] = NULL;
    builder->template_type_vec[template_id_in_code_builder] = NONE_TEMPLATE;
}

void delete_template_without_matrix_with_type(memory_garbage_manager_t *mem_manager, void *template_ptr, template_type type)
{
    assert(mem_manager != NULL && template_ptr != NULL);

    // 所有的模板析构完之后都要登记一下
    if (type == DIRECT_ATOM_TEMPLATE)
    {
        delete_direct_atom_template_without_matrix(mem_manager, (direct_atom_template_t *)template_ptr);
        return;
    }

    if (type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
    {
        delete_direct_atom_template_warp_compress_without_matrix(mem_manager, (direct_atom_template_warp_compress_t *)template_ptr);
        return;
    }

    if (type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
    {
        delete_direct_atom_template_warp_block_compress_without_matrix(mem_manager, (direct_atom_template_warp_block_compress_t *)template_ptr);
        return;
    }

    if (type == SHARED_MEMORY_TEMPLATE)
    {
        delete_shared_memory_template_without_matrix(mem_manager, (shared_memory_template_t *)template_ptr);
        return;
    }

    if (type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
    {
        delete_shared_memory_template_warp_compress_without_matrix(mem_manager, (shared_memory_template_warp_compress_t *)template_ptr);
        return;
    }

    if (type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
    {
        delete_shared_memory_long_row_template_without_matrix(mem_manager, (shared_memory_long_row_template_t *)template_ptr);
        return;
    }

    if (type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        delete_shared_memory_total_warp_reduce_template_without_matrix(mem_manager, (shared_memory_total_warp_reduce_template_t *)template_ptr);
        return;
    }

    if (type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        delete_direct_atom_total_warp_reduce_template_without_matrix(mem_manager, (direct_atom_total_warp_reduce_template_t *)template_ptr);
        return;
    }

    if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
    {
        delete_unaligned_warp_reduce_same_TLB_size_template_without_matrix(mem_manager, (unaligned_warp_reduce_same_TLB_size_template_t *)template_ptr);
        return;
    }

    if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
    {
        delete_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_without_matrix(mem_manager, (unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *)template_ptr);
        return;
    }

    cout << "delete_template_with_type_without_matrix:template type is not supported" << endl;
    assert(false);
}

void delete_direct_atom_template_without_matrix(memory_garbage_manager_t *mem_manager, direct_atom_template_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    assert(del_template->matrix != NULL);

    vector<int> not_need_to_del_compressed_block_id;

    for (int i = 0; i < del_template->matrix->block_coor_table.item_arr.size(); i++)
    {
        not_need_to_del_compressed_block_id.push_back(i);
    }

    // 首先收集和matrix相关的指针，避免被析构
    set<void *> protect_matrix_ptr_set = get_all_mem_ptr_from_matrix_dense_view_and_some_compressed_sub_block(del_template->matrix, not_need_to_del_compressed_block_id);

    // 将避免被析构的指针放到mem_manager中
    mem_manager->ptr_set.insert(protect_matrix_ptr_set.begin(), protect_matrix_ptr_set.end());

    assert(mem_manager->ptr_set.count(del_template->matrix) != 0);
    // register_del_ptr(mem_manager, del_template->matrix);

    // 删除模板
    delete_direct_atom_template(mem_manager, del_template);
}

void delete_direct_atom_template_warp_compress_without_matrix(memory_garbage_manager_t *mem_manager, direct_atom_template_warp_compress_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    assert(del_template->matrix != NULL);

    vector<int> not_need_to_del_compressed_block_id;

    for (int i = 0; i < del_template->matrix->block_coor_table.item_arr.size(); i++)
    {
        not_need_to_del_compressed_block_id.push_back(i);
    }

    // 首先收集和matrix相关的指针，避免被析构
    set<void *> protect_matrix_ptr_set = get_all_mem_ptr_from_matrix_dense_view_and_some_compressed_sub_block(del_template->matrix, not_need_to_del_compressed_block_id);

    // 将避免被析构的指针放到mem_manager中
    mem_manager->ptr_set.insert(protect_matrix_ptr_set.begin(), protect_matrix_ptr_set.end());

    assert(mem_manager->ptr_set.count(del_template->matrix) != 0);
    // register_del_ptr(mem_manager, del_template->matrix);

    delete_direct_atom_template_warp_compress(mem_manager, del_template);
}

void delete_direct_atom_template_warp_block_compress_without_matrix(memory_garbage_manager_t *mem_manager, direct_atom_template_warp_block_compress_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    assert(del_template->matrix != NULL);

    vector<int> not_need_to_del_compressed_block_id;

    for (int i = 0; i < del_template->matrix->block_coor_table.item_arr.size(); i++)
    {
        not_need_to_del_compressed_block_id.push_back(i);
    }

    // 首先收集和matrix相关的指针，避免被析构
    set<void *> protect_matrix_ptr_set = get_all_mem_ptr_from_matrix_dense_view_and_some_compressed_sub_block(del_template->matrix, not_need_to_del_compressed_block_id);

    // 将避免被析构的指针放到mem_manager中
    mem_manager->ptr_set.insert(protect_matrix_ptr_set.begin(), protect_matrix_ptr_set.end());

    assert(mem_manager->ptr_set.count(del_template->matrix) != 0);
    // register_del_ptr(mem_manager, del_template->matrix);

    delete_direct_atom_template_warp_block_compress(mem_manager, del_template);
}

void delete_direct_atom_total_warp_reduce_template_without_matrix(memory_garbage_manager_t *mem_manager, direct_atom_total_warp_reduce_template_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    assert(del_template->matrix != NULL);

    vector<int> not_need_to_del_compressed_block_id;

    for (int i = 0; i < del_template->matrix->block_coor_table.item_arr.size(); i++)
    {
        not_need_to_del_compressed_block_id.push_back(i);
    }

    // 首先收集和matrix相关的指针，避免被析构
    set<void *> protect_matrix_ptr_set = get_all_mem_ptr_from_matrix_dense_view_and_some_compressed_sub_block(del_template->matrix, not_need_to_del_compressed_block_id);

    // 将避免被析构的指针放到mem_manager中
    mem_manager->ptr_set.insert(protect_matrix_ptr_set.begin(), protect_matrix_ptr_set.end());

    assert(mem_manager->ptr_set.count(del_template->matrix) != 0);
    // register_del_ptr(mem_manager, del_template->matrix);

    delete_direct_atom_total_warp_reduce_template(mem_manager, del_template);
}

// 析构长行
void delete_shared_memory_long_row_template_without_matrix(memory_garbage_manager_t *mem_manager, shared_memory_long_row_template_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    assert(del_template->matrix != NULL);

    vector<int> not_need_to_del_compressed_block_id;

    for (int i = 0; i < del_template->matrix->block_coor_table.item_arr.size(); i++)
    {
        not_need_to_del_compressed_block_id.push_back(i);
    }

    // 首先收集和matrix相关的指针，避免被析构
    set<void *> protect_matrix_ptr_set = get_all_mem_ptr_from_matrix_dense_view_and_some_compressed_sub_block(del_template->matrix, not_need_to_del_compressed_block_id);

    // 将避免被析构的指针放到mem_manager中
    mem_manager->ptr_set.insert(protect_matrix_ptr_set.begin(), protect_matrix_ptr_set.end());

    assert(mem_manager->ptr_set.count(del_template->matrix) != 0);
    // register_del_ptr(mem_manager, del_template->matrix);

    delete_shared_memory_long_row_template(mem_manager, del_template);
}

void delete_shared_memory_template_warp_compress_without_matrix(memory_garbage_manager_t *mem_manager, shared_memory_template_warp_compress_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    assert(del_template->matrix != NULL);

    vector<int> not_need_to_del_compressed_block_id;

    for (int i = 0; i < del_template->matrix->block_coor_table.item_arr.size(); i++)
    {
        not_need_to_del_compressed_block_id.push_back(i);
    }

    // 首先收集和matrix相关的指针，避免被析构
    set<void *> protect_matrix_ptr_set = get_all_mem_ptr_from_matrix_dense_view_and_some_compressed_sub_block(del_template->matrix, not_need_to_del_compressed_block_id);

    // 将避免被析构的指针放到mem_manager中
    mem_manager->ptr_set.insert(protect_matrix_ptr_set.begin(), protect_matrix_ptr_set.end());

    assert(mem_manager->ptr_set.count(del_template->matrix) != 0);
    // register_del_ptr(mem_manager, del_template->matrix);

    delete_shared_memory_template_warp_compress(mem_manager, del_template);
}

void delete_shared_memory_template_without_matrix(memory_garbage_manager_t *mem_manager, shared_memory_template_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    assert(del_template->matrix != NULL);

    vector<int> not_need_to_del_compressed_block_id;

    for (int i = 0; i < del_template->matrix->block_coor_table.item_arr.size(); i++)
    {
        not_need_to_del_compressed_block_id.push_back(i);
    }

    // 首先收集和matrix相关的指针，避免被析构
    set<void *> protect_matrix_ptr_set = get_all_mem_ptr_from_matrix_dense_view_and_some_compressed_sub_block(del_template->matrix, not_need_to_del_compressed_block_id);

    // 将避免被析构的指针放到mem_manager中
    mem_manager->ptr_set.insert(protect_matrix_ptr_set.begin(), protect_matrix_ptr_set.end());

    assert(mem_manager->ptr_set.count(del_template->matrix) != 0);
    // register_del_ptr(mem_manager, del_template->matrix);

    delete_shared_memory_template(mem_manager, del_template);
}

void delete_shared_memory_total_warp_reduce_template_without_matrix(memory_garbage_manager_t *mem_manager, shared_memory_total_warp_reduce_template_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    assert(del_template->matrix != NULL);

    vector<int> not_need_to_del_compressed_block_id;

    for (int i = 0; i < del_template->matrix->block_coor_table.item_arr.size(); i++)
    {
        not_need_to_del_compressed_block_id.push_back(i);
    }

    // 首先收集和matrix相关的指针，避免被析构
    set<void *> protect_matrix_ptr_set = get_all_mem_ptr_from_matrix_dense_view_and_some_compressed_sub_block(del_template->matrix, not_need_to_del_compressed_block_id);

    // 将避免被析构的指针放到mem_manager中
    mem_manager->ptr_set.insert(protect_matrix_ptr_set.begin(), protect_matrix_ptr_set.end());

    assert(mem_manager->ptr_set.count(del_template->matrix) != 0);
    // register_del_ptr(mem_manager, del_template->matrix);

    delete_shared_memory_total_warp_reduce_template(mem_manager, del_template);
}

void delete_unaligned_warp_reduce_same_TLB_size_template_without_matrix(memory_garbage_manager_t *mem_manager, unaligned_warp_reduce_same_TLB_size_template_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    assert(del_template->matrix != NULL);

    vector<int> not_need_to_del_compressed_block_id;

    for (int i = 0; i < del_template->matrix->block_coor_table.item_arr.size(); i++)
    {
        not_need_to_del_compressed_block_id.push_back(i);
    }

    // 首先收集和matrix相关的指针，避免被析构
    set<void *> protect_matrix_ptr_set = get_all_mem_ptr_from_matrix_dense_view_and_some_compressed_sub_block(del_template->matrix, not_need_to_del_compressed_block_id);

    // 将避免被析构的指针放到mem_manager中
    mem_manager->ptr_set.insert(protect_matrix_ptr_set.begin(), protect_matrix_ptr_set.end());

    assert(mem_manager->ptr_set.count(del_template->matrix) != 0);
    // register_del_ptr(mem_manager, del_template->matrix);

    delete_unaligned_warp_reduce_same_TLB_size_template(mem_manager, del_template);
}

void delete_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_without_matrix(memory_garbage_manager_t *mem_manager, unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *del_template)
{
    assert(mem_manager != NULL && del_template != NULL);

    // 最外层数据结构没有被析构过
    assert(is_deleted(mem_manager, del_template) == false);

    assert(del_template->matrix != NULL);

    vector<int> not_need_to_del_compressed_block_id;

    for (int i = 0; i < del_template->matrix->block_coor_table.item_arr.size(); i++)
    {
        not_need_to_del_compressed_block_id.push_back(i);
    }

    // 首先收集和matrix相关的指针，避免被析构
    set<void *> protect_matrix_ptr_set = get_all_mem_ptr_from_matrix_dense_view_and_some_compressed_sub_block(del_template->matrix, not_need_to_del_compressed_block_id);

    // 将避免被析构的指针放到mem_manager中
    mem_manager->ptr_set.insert(protect_matrix_ptr_set.begin(), protect_matrix_ptr_set.end());

    assert(mem_manager->ptr_set.count(del_template->matrix) != 0);

    // register_del_ptr(mem_manager, del_template->matrix);

    delete_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce(mem_manager, del_template);
}