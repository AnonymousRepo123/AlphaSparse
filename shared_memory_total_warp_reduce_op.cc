#include "shared_memory_total_warp_reduce_op.hpp"

shared_memory_total_warp_reduce_template_t *init_shared_memory_total_warp_reduce_template(code_builder_t *builder, unsigned long dense_block_id)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;
    assert(matrix->block_coor_table.item_arr.size() > dense_block_id);

    // 创建对应的模板
    shared_memory_total_warp_reduce_template_t *new_template = new shared_memory_total_warp_reduce_template_t();

    new_template->dense_block_index = dense_block_id;
    new_template->matrix = matrix;

    new_template->kernal_first_row_index = matrix->block_coor_table.item_arr[dense_block_id]->min_dense_row_index;
    new_template->kernal_first_col_index = matrix->block_coor_table.item_arr[dense_block_id]->min_dense_col_index;

    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr;

    if (matrix->block_coor_table.item_arr[dense_block_id]->min_dense_col_index == 0 && matrix->block_coor_table.item_arr[dense_block_id]->max_dense_col_index == matrix->dense_col_number - 1)
    {
        // 稠密子块之间没有共享的行
    }
    else
    {
        new_template->is_atom_add = true;
    }

    // 首先处理每一线程的全局行索引，将全局行索引搞出来
    // 分别遍历三个层次的索引
    index_of_compress_block_t *block_level_index = compressed_block_view->read_index[2];
    index_of_compress_block_t *warp_level_index = compressed_block_view->read_index[3];
    index_of_compress_block_t *thread_level_index = compressed_block_view->read_index[4];
    assert(block_level_index->level_of_this_index == TBLOCK_LEVEL);
    assert(warp_level_index->level_of_this_index == WRAP_LEVEL);
    assert(thread_level_index->level_of_this_index == THREAD_LEVEL);

    // 行最大值相同
    assert(matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index <= block_level_index->max_row_index);
    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[0]->max_row_index == block_level_index->max_row_index);

    // 有效的部分应该只有
    new_template->effective_row_num = matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index - matrix->block_coor_table.item_arr[dense_block_id]->min_dense_row_index + 1;

    if (thread_level_index->row_number_of_block_arr != NULL)
    {
        cout << "row num in thread level block must be 1, thread level index shouldn't have this metadata" << endl;
        assert(false);
    }

    // 检查warp粒度的块包含的行数量，每个warp粒度的块包含一行
    // for (unsigned long warp_level_block_id = 0; warp_level_block_id < warp_level_index->block_num; warp_level_block_id++)
    // {
    //     unsigned long block_row_num = read_from_array_with_data_type(warp_level_index->row_number_of_block_arr, warp_level_index->data_type_of_row_number_of_block_arr, warp_level_block_id);
    //     if (block_row_num != 1)
    //     {
    //         cout << "row num in warp level block must be 1" << endl;
    //         assert(false);
    //     }
    // }

    // 查看线程块粒度的块包含的行，查看线程块粒度的块之间有没有共享行，判断是否需要全局内存中的原子加归约
    unsigned long global_min_row_index = read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, 0);
    unsigned long global_max_row_index = global_min_row_index + read_from_array_with_data_type(block_level_index->row_number_of_block_arr, block_level_index->data_type_of_row_number_of_block_arr, 0) - 1;
    for (unsigned long index_of_block_level_index = 1; index_of_block_level_index < block_level_index->block_num; index_of_block_level_index++)
    {
        // 当前块的起始行号和包含的行的数量，
        unsigned long cur_min_row_index = read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, index_of_block_level_index);

        if (cur_min_row_index <= global_max_row_index)
        {
            // 代表有重合的部分
            new_template->is_atom_add = true;

            // 这里的重合代表一行可能有多个线程块，就不支持这个模板了
            cout << "several tblock level block in one row, not supported in this template" << endl;
            assert(false);

            break;
        }

        unsigned long cur_row_num = read_from_array_with_data_type(block_level_index->row_number_of_block_arr, block_level_index->data_type_of_row_number_of_block_arr, index_of_block_level_index);

        if (cur_row_num + cur_min_row_index - 1 > global_max_row_index)
        {
            // 没有重合就修改元数据
            global_max_row_index = cur_min_row_index + cur_row_num - 1;
        }
    }

    // 当前稠密子块的行数量，这个值也是块最大行号+1
    unsigned long total_row_num = compressed_block_view->read_index[0]->max_row_index - compressed_block_view->read_index[0]->min_row_index + 1;

    vector<unsigned long> new_row_offset_in_warp_tmp_result_vec;
    new_row_offset_in_warp_tmp_result_vec.push_back(0);

    // 用一个数组记录每一行中间结果的数量，
    vector<unsigned long> warp_level_result_num_of_each_row(total_row_num);
    // 每个warp粒度的块的首个非零元的索引，使用CSR的索引，一共是warp块的数量+1，这一点和其他的模板不一样
    vector<unsigned long> global_warp_block_first_nz_vec(warp_level_index->block_num + 1);

    // 每一行的warp粒度的块的数量一开始全都初始化为0
    for (unsigned long i = 0; i < total_row_num; i++)
    {
        warp_level_result_num_of_each_row[i] = 0;
    }

    // 分别遍历三个层次
    // 遍历三个层次的索引，计算每一行warp结果的数量，并且计算每个warp粒度的块的非零元起始索引
    for (unsigned long index_of_block_level_index = 0; index_of_block_level_index < block_level_index->block_num; index_of_block_level_index++)
    {
        // 当前block的首行行号
        unsigned long block_first_row_index = read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, index_of_block_level_index);
        // 当前block第一个非零元索引
        unsigned long block_first_nz_index = read_from_array_with_data_type(block_level_index->coo_begin_index_arr, block_level_index->data_type_of_coo_begin_index_arr, index_of_block_level_index);

        // block中第一个warp号和下一个block的首warp
        unsigned long this_block_first_warp_index = read_from_array_with_data_type(block_level_index->index_arr, block_level_index->index_data_type, index_of_block_level_index);
        unsigned long next_block_first_warp_index = read_from_array_with_data_type(block_level_index->index_arr, block_level_index->index_data_type, index_of_block_level_index + 1);

        // 当前线程块粒度的块的第一个warp粒度的块的线程粒度的块的大小
        assert(this_block_first_warp_index < warp_level_index->block_num && thread_level_index->coo_block_size_arr != NULL);

        // 遍历所有warp粒度的索引
        for (unsigned long index_of_warp_level_index = this_block_first_warp_index; index_of_warp_level_index < next_block_first_warp_index; index_of_warp_level_index++)
        {
            assert(index_of_warp_level_index < warp_level_index->block_num);
            // 首先查看一个warp的相对行索引
            unsigned long warp_first_row_index = read_from_array_with_data_type(warp_level_index->index_of_the_first_row_arr, warp_level_index->data_type_of_index_of_the_first_row_arr, index_of_warp_level_index);
            unsigned long warp_first_nz_index = read_from_array_with_data_type(warp_level_index->coo_begin_index_arr, warp_level_index->data_type_of_coo_begin_index_arr, index_of_warp_level_index);
            unsigned long block_row_num = read_from_array_with_data_type(warp_level_index->row_number_of_block_arr, warp_level_index->data_type_of_row_number_of_block_arr, index_of_warp_level_index);

            if (block_row_num != 1)
            {
                cout << "row num in warp level block must be 1" << endl;
                assert(false);
            }

            // 没必要进行thread层次的遍历，计算当前全局行号
            unsigned long global_warp_row_index = block_first_row_index + warp_first_row_index;
            unsigned long global_warp_first_nz_index = warp_first_nz_index + block_first_nz_index;

            warp_level_result_num_of_each_row[global_warp_row_index]++;
            global_warp_block_first_nz_vec[index_of_warp_level_index] = global_warp_first_nz_index;

            // 检查非零元的数量是32的倍数
            if (index_of_warp_level_index > 0)
            {
                assert((global_warp_block_first_nz_vec[index_of_warp_level_index] - global_warp_block_first_nz_vec[index_of_warp_level_index - 1]) % 32 == 0);
            }
        }
    }

    // global_warp_block_first_nz_vec最后一位填的是nnz
    global_warp_block_first_nz_vec[global_warp_block_first_nz_vec.size() - 1] = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->padding_arr_size;
    assert((global_warp_block_first_nz_vec[global_warp_block_first_nz_vec.size() - 1] - global_warp_block_first_nz_vec[global_warp_block_first_nz_vec.size() - 2]) % 32 == 0);

    // 如果每一行的中间结果只有一个，就不需要shared memory层次的归约
    // 遍历每一行结果的数量，从而得出每一行在中间结果中的偏移量
    for (unsigned long row_index = 0; row_index < warp_level_result_num_of_each_row.size(); row_index++)
    {
        new_row_offset_in_warp_tmp_result_vec.push_back(new_row_offset_in_warp_tmp_result_vec[new_row_offset_in_warp_tmp_result_vec.size() - 1] + warp_level_result_num_of_each_row[row_index]);
    }

    assert(new_row_offset_in_warp_tmp_result_vec.size() == total_row_num + 1);

    // 交错存储是没必要的，因为warp内的内容同属于一行
    new_template->data_type_of_row_offset_in_warp_tmp_result = find_most_suitable_data_type(new_row_offset_in_warp_tmp_result_vec[new_row_offset_in_warp_tmp_result_vec.size() - 1]);
    new_template->size_of_row_offset_in_warp_tmp_result = new_row_offset_in_warp_tmp_result_vec.size();
    // 申请数组存储归约信息
    new_template->row_offset_in_warp_tmp_result = malloc_arr(new_template->size_of_row_offset_in_warp_tmp_result, new_template->data_type_of_row_offset_in_warp_tmp_result);
    copy_unsigned_long_arr_to_others(&(new_row_offset_in_warp_tmp_result_vec[0]), new_template->row_offset_in_warp_tmp_result, new_template->data_type_of_row_offset_in_warp_tmp_result, new_template->size_of_row_offset_in_warp_tmp_result);

    // 考虑到空行，最多也是等于最后一个数据
    assert(total_row_num >= read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, block_level_index->block_num - 1));

    // 记录每个block的起始行号，在归约的时候使用，直接就有现成的，拷贝进来即可，但是最后一个需要是所有的行数量，使用的是CSR类型的索引
    new_template->data_type_of_block_first_row_index = find_most_suitable_data_type(total_row_num);
    new_template->size_of_block_first_row_index = block_level_index->length;
    new_template->block_first_row_index = malloc_arr(new_template->size_of_block_first_row_index, new_template->data_type_of_block_first_row_index);

    // 不同类型数组之间的拷贝，从block_level_index中拷贝到new_template中
    for (unsigned long block_id = 0; block_id < block_level_index->block_num; block_id++)
    {
        unsigned long source_arr_content = read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, block_id);
        // 写数据
        write_to_array_with_data_type(new_template->block_first_row_index, new_template->data_type_of_block_first_row_index, block_id, source_arr_content);
    }

    // 最后写一个数据，写的的是行的数量
    write_to_array_with_data_type(new_template->block_first_row_index, new_template->data_type_of_block_first_row_index, block_level_index->block_num, total_row_num);

    // 每个block的warp粒度的块的数量
    new_template->data_type_of_block_begin_warp_index_offset = block_level_index->index_data_type;
    new_template->size_of_block_begin_warp_index_offset = block_level_index->length;
    new_template->block_begin_warp_index_offset = block_level_index->index_arr;

    // 排序相关
    // 一些排序的数据
    // 最后给出排序索引类型和具体的数组
    if (compressed_block_view->y_write_index.size() > 0)
    {
        // 在子块内排序了
        assert(compressed_block_view->is_sorted == true && builder->sub_block_sort_type_vec[dense_block_id] == SUB_BLOCK_SORT && matrix->is_sorted == false);
        new_template->global_sort_index = false;
        new_template->local_sort_index = true;

        // 拷贝
        new_template->data_type_of_row_index_before_sort = compressed_block_view->y_write_index[0]->index_data_type;
        new_template->row_index_before_sort = compressed_block_view->y_write_index[0]->index_arr;
        new_template->size_of_row_index_before_sort = compressed_block_view->y_write_index[0]->length;
    }
    else if (matrix->sorted_row_index != NULL)
    {
        cout << "have global sort" << endl;
        // 在全局范围内有排序
        assert(compressed_block_view->is_sorted == false && matrix->is_sorted == true && builder->sub_block_sort_type_vec[dense_block_id] == GLOBAL_SORT);
        new_template->global_sort_index = true;
        new_template->local_sort_index = false;

        // 拷贝
        new_template->data_type_of_row_index_before_sort = matrix->data_type_of_sorted_row_index;
        new_template->row_index_before_sort = matrix->sorted_row_index;
        new_template->size_of_row_index_before_sort = matrix->dense_row_number;
    }

    // 每个warp粒度的块的第一个非零元的索引。不需要block索引，采用的CSR的索引方法
    new_template->size_of_global_warp_block_first_nz = global_warp_block_first_nz_vec.size();
    new_template->data_type_of_global_warp_block_first_nz = find_most_suitable_data_type(global_warp_block_first_nz_vec[global_warp_block_first_nz_vec.size() - 1]);
    new_template->global_warp_block_first_nz = malloc_arr(new_template->size_of_global_warp_block_first_nz, new_template->data_type_of_global_warp_block_first_nz);
    // 将数据拷贝进来，从global_warp_block_first_nz_vec中拷贝进来，数量比warp粒度的块的数量多一个。
    for (unsigned long i = 0; i < global_warp_block_first_nz_vec.size(); i++)
    {
        unsigned long nz_index = global_warp_block_first_nz_vec[i];
        write_to_array_with_data_type(new_template->global_warp_block_first_nz, new_template->data_type_of_global_warp_block_first_nz, i, nz_index);
    }

    // 列索引的值直接做一个拷贝，可以拷贝交错存储的版本
    // 值
    new_template->data_type_of_val_arr = compressed_block_view->val_data_type;
    new_template->val_arr = compressed_block_view->staggered_padding_val_arr;
    new_template->size_of_val_arr = compressed_block_view->staggered_padding_val_arr_size;

    // 这两个数组的大小和warpnz的最后一个非零元大小相同
    assert(new_template->size_of_val_arr == read_from_array_with_data_type(new_template->global_warp_block_first_nz, new_template->data_type_of_global_warp_block_first_nz, new_template->size_of_global_warp_block_first_nz - 1));

    // 列
    new_template->data_type_of_col_index_arr = compressed_block_view->read_index[6]->index_data_type;
    new_template->col_index_arr = compressed_block_view->read_index[6]->index_arr;
    new_template->size_of_col_index_arr = compressed_block_view->read_index[6]->length;

    assert(new_template->size_of_val_arr == new_template->size_of_col_index_arr);

    return new_template;
}

bool is_supported_by_shared_memory_total_warp_reduce_template(sparse_struct_t *matrix, unsigned long dense_block_id)
{
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

    if (thread_level_index->row_number_of_block_arr != NULL)
    {
        return false;
    }

    // 查看线程块粒度的块包含的行，查看线程块粒度的块之间有没有共享行，判断是否需要全局内存中的原子加归约
    unsigned long global_min_row_index = read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, 0);
    unsigned long global_max_row_index = global_min_row_index + read_from_array_with_data_type(block_level_index->row_number_of_block_arr, block_level_index->data_type_of_row_number_of_block_arr, 0) - 1;
    for (unsigned long index_of_block_level_index = 1; index_of_block_level_index < block_level_index->block_num; index_of_block_level_index++)
    {
        // 当前块的起始行号和包含的行的数量，
        unsigned long cur_min_row_index = read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, index_of_block_level_index);

        if (cur_min_row_index <= global_max_row_index)
        {
            // 这里的重合代表一行可能有多个线程块，就不支持这个模板了
            return false;
        }

        unsigned long cur_row_num = read_from_array_with_data_type(block_level_index->row_number_of_block_arr, block_level_index->data_type_of_row_number_of_block_arr, index_of_block_level_index);

        if (cur_row_num + cur_min_row_index - 1 > global_max_row_index)
        {
            // 没有重合就修改元数据
            global_max_row_index = cur_min_row_index + cur_row_num - 1;
        }
    }

    // 用一个bool判断是不是每个warp对应一行，
    bool many_warp_one_row = false;
    
    // 判断当前WLB是不是第一个WLB
    bool is_first_WLB = true;
    unsigned long last_WLB_first_global_row_index = 0;

    // 遍历三个层次
    // 遍历三个层次的索引，计算每一行warp结果的数量，并且计算每个warp粒度的块的非零元起始索引
    for (unsigned long index_of_block_level_index = 0; index_of_block_level_index < block_level_index->block_num; index_of_block_level_index++)
    {
        // 当前block的首行行号
        unsigned long block_first_row_index = read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, index_of_block_level_index);
        // 当前block第一个非零元索引
        unsigned long block_first_nz_index = read_from_array_with_data_type(block_level_index->coo_begin_index_arr, block_level_index->data_type_of_coo_begin_index_arr, index_of_block_level_index);

        // block中第一个warp号和下一个block的首warp
        unsigned long this_block_first_warp_index = read_from_array_with_data_type(block_level_index->index_arr, block_level_index->index_data_type, index_of_block_level_index);
        unsigned long next_block_first_warp_index = read_from_array_with_data_type(block_level_index->index_arr, block_level_index->index_data_type, index_of_block_level_index + 1);

        // 当前线程块粒度的块的第一个warp粒度的块的线程粒度的块的大小
        assert(this_block_first_warp_index < warp_level_index->block_num && thread_level_index->coo_block_size_arr != NULL);

        // 遍历所有warp粒度的索引
        for (unsigned long index_of_warp_level_index = this_block_first_warp_index; index_of_warp_level_index < next_block_first_warp_index; index_of_warp_level_index++)
        {
            assert(index_of_warp_level_index < warp_level_index->block_num);
            // 首先查看一个warp的相对行索引
            unsigned long warp_first_row_index = read_from_array_with_data_type(warp_level_index->index_of_the_first_row_arr, warp_level_index->data_type_of_index_of_the_first_row_arr, index_of_warp_level_index);
            unsigned long warp_first_nz_index = read_from_array_with_data_type(warp_level_index->coo_begin_index_arr, warp_level_index->data_type_of_coo_begin_index_arr, index_of_warp_level_index);
            unsigned long block_row_num = read_from_array_with_data_type(warp_level_index->row_number_of_block_arr, warp_level_index->data_type_of_row_number_of_block_arr, index_of_warp_level_index);

            // cout << "block_row_num:" << block_row_num << endl;

            if (block_row_num != 1)
            {
                // 一个warp只能处理一行
                return false;
            }

            // 没必要进行thread层次的遍历，计算当前全局行号
            unsigned long global_warp_row_index = block_first_row_index + warp_first_row_index;
            unsigned long global_warp_first_nz_index = warp_first_nz_index + block_first_nz_index;

            //判断行号是否重合
            if (is_first_WLB == true)
            {
                last_WLB_first_global_row_index = global_warp_row_index;
                is_first_WLB = false;
            }
            else
            {
                if (last_WLB_first_global_row_index == global_warp_row_index)
                {
                    many_warp_one_row = true;
                }

                last_WLB_first_global_row_index == global_warp_row_index;
            }
        }
    }

    // 如果不是多个warp处理一行，那么检查不通过
    if (many_warp_one_row == false)
    {
        return false;
    }

    // 这里做一个比较粗的过滤，只要每一个BLB内的WLB数量超过阈值，代表shared memory爆炸了
    for (unsigned long BLB_id = 0; BLB_id < block_level_index->block_num; BLB_id++)
    {
        // 减一下，得出当前BLB中WLB的数量
        unsigned long first_WLB_index_of_cur_BLB = read_from_array_with_data_type(block_level_index->index_arr, block_level_index->index_data_type, BLB_id);
        unsigned long first_WLB_index_of_next_BLB = read_from_array_with_data_type(block_level_index->index_arr, block_level_index->index_data_type, BLB_id);
        
        assert(first_WLB_index_of_next_BLB > first_WLB_index_of_cur_BLB);
        // BLB中WLB的数量
        unsigned long WLB_num_of_cur_BLB = first_WLB_index_of_next_BLB - first_WLB_index_of_cur_BLB;

        // WLB_num_of_cur_BLB是共享内存的大小，
        if (WLB_num_of_cur_BLB > get_config()["SHARED_MEM_TOTAL_SIZE"].as_integer() - 5)
        {
            return false;
        }
    }

    return true;
}

// 每个warp最多只能负责一行、不能一个warp只负责一行、一行不能被两个线程块处理。
bool is_supported_by_shared_memory_total_warp_reduce_template(code_builder_t *builder, unsigned long dense_block_id)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;

    return is_supported_by_shared_memory_total_warp_reduce_template(matrix, dense_block_id);
}

void store_template_data(shared_memory_total_warp_reduce_template_t *output_template, string output_dir, bool force_not_share_global_sort_index)
{
    assert(output_template != NULL);

    srand(time(0));
    unsigned long matrix_id = rand() + time(0) % 1000;

    output_dir = output_dir + "/" + to_string(matrix_id) + "_" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer());

    // 创建这个文件夹
    system(("mkdir " + output_dir).c_str());

    // 只要不压缩，就持久化
    if (output_template->row_offset_in_warp_tmp_result_compress == NONE_COMPRESS)
    {
        assert(output_template->row_offset_in_warp_tmp_result != NULL);
        print_arr_to_file_with_data_type(output_template->row_offset_in_warp_tmp_result, output_template->data_type_of_row_offset_in_warp_tmp_result, output_template->size_of_row_offset_in_warp_tmp_result, output_dir + "/row_offset_in_warp_tmp_result");
    }

    if (output_template->block_first_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->block_first_row_index != NULL);
        print_arr_to_file_with_data_type(output_template->block_first_row_index, output_template->data_type_of_block_first_row_index, output_template->size_of_block_first_row_index, output_dir + "/block_first_row_index");
    }

    if (output_template->block_begin_warp_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_warp_index_offset != NULL);
        print_arr_to_file_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, output_template->size_of_block_begin_warp_index_offset, output_dir + "/block_begin_warp_index_offset");
    }

    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->row_index_before_sort != NULL)
    {
        assert(output_template->row_index_before_sort != NULL);
        // 如果是全局排序，只有第一个才需要存排序之后的行索引
        if (output_template->local_sort_index == true)
        {
            assert(output_template->global_sort_index == false);
            print_arr_to_file_with_data_type(output_template->row_index_before_sort, output_template->data_type_of_row_index_before_sort, output_template->size_of_row_index_before_sort, output_dir + "/row_index_before_sort");
        }
        else if (output_template->global_sort_index == true && (output_template->dense_block_index == 0 || force_not_share_global_sort_index == true))
        {
            assert(output_template->local_sort_index == false);
            print_arr_to_file_with_data_type(output_template->row_index_before_sort, output_template->data_type_of_row_index_before_sort, output_template->size_of_row_index_before_sort, output_dir + "/row_index_before_sort");
        }
    }

    if (output_template->global_warp_block_first_nz_compress == NONE_COMPRESS)
    {
        assert(output_template->global_warp_block_first_nz != NULL);
        print_arr_to_file_with_data_type(output_template->global_warp_block_first_nz, output_template->data_type_of_global_warp_block_first_nz, output_template->size_of_global_warp_block_first_nz, output_dir + "/global_warp_block_first_nz");
    }

    // 值
    assert(output_template->val_arr != NULL);
    print_arr_to_file_with_data_type(output_template->val_arr, output_template->data_type_of_val_arr, output_template->size_of_val_arr, output_dir + "/val_arr");

    // 列
    assert(output_template->col_index_arr != NULL);
    print_arr_to_file_with_data_type(output_template->col_index_arr, output_template->data_type_of_col_index_arr, output_template->size_of_col_index_arr, output_dir + "/col_index_arr");

    output_template->hash_of_this_template = matrix_id;
}

string code_of_template_data_struct(shared_memory_total_warp_reduce_template_t *output_template, unsigned long dense_block_id)
{
    string return_str = "typedef struct compressed_dense_block_" + to_string(dense_block_id) + "\n{\n";

    // 对应的位置分别存储行号和块号
    if (output_template->row_offset_in_warp_tmp_result_compress == NONE_COMPRESS)
    {
        assert(output_template->row_offset_in_warp_tmp_result != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_row_offset_in_warp_tmp_result, code_of_arr_var_name(dense_block_id, -1, "row_offset_in_warp_tmp_result"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "row_offset_in_warp_tmp_result") + " = " + to_string(output_template->size_of_row_offset_in_warp_tmp_result) + ";\n";
    }

    return_str = return_str + "\n";

    if (output_template->block_first_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->block_first_row_index != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_block_first_row_index, code_of_arr_var_name(dense_block_id, -1, "block_first_row_index"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "block_first_row_index") + " = " + to_string(output_template->size_of_block_first_row_index) + ";\n";
    }

    return_str = return_str + "\n";

    if (output_template->block_begin_warp_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_warp_index_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_block_begin_warp_index_offset, code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset") + " = " + to_string(output_template->size_of_block_begin_warp_index_offset) + ";\n";
    }

    return_str = return_str + "\n";

    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->row_index_before_sort != NULL)
    {
        assert(output_template->row_index_before_sort != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_row_index_before_sort, code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort") + " = " + to_string(output_template->size_of_row_index_before_sort) + ";\n";
    }

    return_str = return_str + "\n";

    if (output_template->global_warp_block_first_nz_compress == NONE_COMPRESS)
    {
        assert(output_template->global_warp_block_first_nz != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_global_warp_block_first_nz, code_of_arr_var_name(dense_block_id, -1, "global_warp_block_first_nz"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "global_warp_block_first_nz") + " = " + to_string(output_template->size_of_global_warp_block_first_nz) + ";\n";
    }

    return_str = return_str + "\n";

    return_str = return_str + "\n";
    assert(output_template->val_arr != NULL);
    return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_val_arr, code_of_arr_var_name(dense_block_id, -1, "val_arr"));
    return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "val_arr") + " = " + to_string(output_template->size_of_val_arr) + ";\n";

    return_str = return_str + "\n";
    assert(output_template->col_index_arr != NULL);
    return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_col_index_arr, code_of_arr_var_name(dense_block_id, -1, "col_index_arr"));
    return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "col_index_arr") + " = " + to_string(output_template->size_of_col_index_arr) + ";\n";

    return_str = return_str + "}";
    return_str = return_str + "compressed_dense_block_" + to_string(dense_block_id) + "_t;\n";

    return return_str;
}

string code_of_read_template_data_from_file_func_define(shared_memory_total_warp_reduce_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index)
{
    string return_str = "compressed_dense_block_" + to_string(dense_block_id) + "_t* read_dense_block_" + to_string(dense_block_id) + "_from_file(string file_name_prefix)\n{\n";

    return_str = return_str + "compressed_dense_block_" + to_string(dense_block_id) + "_t *template_data = new " + "compressed_dense_block_" + to_string(dense_block_id) + "_t();\n";

    // 对应的位置分别存储行号和块号
    if (output_template->row_offset_in_warp_tmp_result_compress == NONE_COMPRESS)
    {
        assert(output_template->row_offset_in_warp_tmp_result != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "row_offset_in_warp_tmp_result") + " = (" + code_of_data_type(output_template->data_type_of_row_offset_in_warp_tmp_result) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "row_offset_in_warp_tmp_result") + ", " + convert_data_type_to_string(output_template->data_type_of_row_offset_in_warp_tmp_result) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/row_offset_in_warp_tmp_result\");\n";
    }

    return_str = return_str + "\n";

    if (output_template->block_first_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->block_first_row_index != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "block_first_row_index") + " = (" + code_of_data_type(output_template->data_type_of_block_first_row_index) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "block_first_row_index") + ", " + convert_data_type_to_string(output_template->data_type_of_block_first_row_index) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/block_first_row_index\");\n";
    }

    return_str = return_str + "\n";

    if (output_template->block_begin_warp_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_warp_index_offset != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset") + " = (" + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset") + ", " + convert_data_type_to_string(output_template->data_type_of_block_begin_warp_index_offset) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/block_begin_warp_index_offset\");\n";
    }

    return_str = return_str + "\n";

    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->row_index_before_sort != NULL)
    {
        // 如果有全局的排序索引，只有0号块需要存储
        if (output_template->global_sort_index == true)
        {
            if (dense_block_id == 0 || force_not_share_global_sort_index == true)
            {
                // 存一个全局的排序
                assert(output_template->row_index_before_sort != NULL);
                return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort") + " = (" + code_of_data_type(output_template->data_type_of_row_index_before_sort) + " *)";
                return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort") + ", " + convert_data_type_to_string(output_template->data_type_of_row_index_before_sort) + ", ";
                // 要读的文件名
                return_str = return_str + "file_name_prefix + \"/row_index_before_sort\");\n";
            }
            else
            {
                // 如果已经有了就直接拷贝全局的排序
                return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort") + " = NULL;\n";
            }
        }
        else if (output_template->local_sort_index == true)
        {
            assert(output_template->row_index_before_sort != NULL);
            return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort") + " = (" + code_of_data_type(output_template->data_type_of_row_index_before_sort) + " *)";
            return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort") + ", " + convert_data_type_to_string(output_template->data_type_of_row_index_before_sort) + ", ";
            // 要读的文件名
            return_str = return_str + "file_name_prefix + \"/row_index_before_sort\");\n";
        }
        else
        {
            cout << "error" << endl;
            assert(false);
        }
    }

    return_str = return_str + "\n";

    // block和warp的收个非零元索引
    if (output_template->global_warp_block_first_nz_compress == NONE_COMPRESS)
    {
        assert(output_template->global_warp_block_first_nz != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "global_warp_block_first_nz") + " = (" + code_of_data_type(output_template->data_type_of_global_warp_block_first_nz) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "global_warp_block_first_nz") + ", " + convert_data_type_to_string(output_template->data_type_of_global_warp_block_first_nz) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/global_warp_block_first_nz\");\n";
    }

    return_str = return_str + "\n";

    return_str = return_str + "\n";
    assert(output_template->val_arr != NULL);
    return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "val_arr") + " = (" + code_of_data_type(output_template->data_type_of_val_arr) + " *)";
    return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "val_arr") + ", " + convert_data_type_to_string(output_template->data_type_of_val_arr) + ", ";
    // 要读的文件名
    return_str = return_str + "file_name_prefix + \"/val_arr\");\n";

    return_str = return_str + "\n";
    assert(output_template->col_index_arr != NULL);
    return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "col_index_arr") + " = (" + code_of_data_type(output_template->data_type_of_col_index_arr) + " *)";
    return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "col_index_arr") + ", " + convert_data_type_to_string(output_template->data_type_of_col_index_arr) + ", ";
    // 要读的文件名
    return_str = return_str + "file_name_prefix + \"/col_index_arr\");\n";

    return_str = return_str + "return template_data;\n";

    return_str = return_str + "}\n";

    return return_str;
}

// kernal，注意归约
string code_of_template_kernal(shared_memory_total_warp_reduce_template_t *output_template, unsigned long dense_block_id)
{
    if (output_template->thread_num_of_row_reduce != get_config()["HALF_MAX_ROW_REDUCE_THREAD"].as_integer() && output_template->thread_num_of_row_reduce != get_config()["MAX_ROW_REDUCE_THREAD"].as_integer())
    {
        assert(output_template->thread_num_in_block % output_template->thread_num_of_row_reduce == 0);
    }

    assert(output_template->thread_num_in_block % 32 == 0);

    // 内核函数的声明
    string return_str = "__global__ void spmv_" + to_string(dense_block_id) + "(";

    // 用一个变量表明当前形参是不是第一个，如果是第一个就不用点逗号
    bool is_first_param = true;

    // 这里加入形参的声明
    if (output_template->row_offset_in_warp_tmp_result_compress == NONE_COMPRESS)
    {
        assert(output_template->row_offset_in_warp_tmp_result != NULL);
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_row_offset_in_warp_tmp_result, "* row_offset_in_warp_tmp_result");
        is_first_param = false;
    }

    if (output_template->block_first_row_index_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }

        assert(output_template->block_first_row_index != NULL);
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_block_first_row_index, "* block_first_row_index");
    }

    if (output_template->block_begin_warp_index_offset_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }

        assert(output_template->block_begin_warp_index_offset != NULL);
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_block_begin_warp_index_offset, "* block_begin_warp_index_offset");
    }

    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->row_index_before_sort != NULL)
    {
        // 这里代表有排序过
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_row_index_before_sort, "* row_index_before_sort");
    }

    if (output_template->global_warp_block_first_nz_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }

        assert(output_template->global_warp_block_first_nz != NULL);
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_global_warp_block_first_nz, "* global_warp_block_first_nz");
    }

    if (is_first_param == false)
    {
        return_str = return_str + ", ";
    }
    else
    {
        is_first_param = false;
    }
    assert(output_template->val_arr != NULL);
    return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_val_arr, "* val_arr");

    return_str = return_str + ", ";
    assert(output_template->col_index_arr != NULL);
    return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_col_index_arr, "* col_index_arr");

    // x的值
    return_str = return_str + ", ";
    return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_val_arr, "* device_x_arr");

    // y的值
    return_str = return_str + ", ";
    return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_val_arr, "* device_y_arr");

    return_str = return_str + ")\n{\n";

    // 这里是kernal的本体，首先判断哪些变量是需要的，哪些是不需要的

    // 下一个BLB的第一个WLB的索引，减少一次显存读，当每个BLB中WLB的数量一样，并且等于线程块中warp数量的时候，就不需要下一个BLB的第一个WLB的索引
    bool need_next_block_first_warp_index = true;

    if (output_template->block_begin_warp_index_offset_compress == LINEAR_COMPRESS)
    {
        assert(output_template->block_begin_warp_index_offset_compress_meta != NULL);

        linear_compress_t *compressor = (linear_compress_t *)output_template->block_begin_warp_index_offset_compress_meta;

        if (compressor->coefficient == output_template->thread_num_in_block / 32)
        {
            // 节省一次读显存
            need_next_block_first_warp_index = false;
        }
    }

    // 如果一个WLB中只有32个非零元，就不需要next_warp_block_first_nz用来作为WLB遍历自己内容的下界
    bool need_next_warp_block_first_nz = true;

    if (output_template->global_warp_block_first_nz_compress == LINEAR_COMPRESS)
    {
        assert(output_template->global_warp_block_first_nz_compress_meta != NULL);

        linear_compress_t *compressor = (linear_compress_t *)output_template->global_warp_block_first_nz_compress_meta;

        if (compressor->coefficient == 32)
        {
            need_next_warp_block_first_nz = false;
        }
    }

    // 线程索引
    return_str = return_str + "int bid = blockIdx.x;\n";
    return_str = return_str + "int tid_in_block = threadIdx.x;\n";
    return_str = return_str + "int wid_in_block = threadIdx.x / 32;\n";
    return_str = return_str + "int tid_in_warp = threadIdx.x % 32;\n";

    if (need_next_block_first_warp_index == true)
    {
        return_str = return_str + "int warp_num_in_block = blockDim.x / 32;\n";
    }

    // 查看是否需要首行和首列索引
    if (output_template->kernal_first_row_index != 0)
    {
        return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->kernal_first_row_index + 1)) + " kernal_first_row_index = " + to_string(output_template->kernal_first_row_index) + ";\n";
    }

    if (output_template->kernal_first_col_index != 0)
    {
        return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->kernal_first_col_index + 1)) + " kernal_first_col_index = " + to_string(output_template->kernal_first_col_index) + ";\n";
    }

    // 当前shared memory的占用
    unsigned long shared_memory_used_size = 0;

    // 最大的WLB数量
    unsigned long max_WLB_num_of_BLB = 0;

    // 遍历所有的BLB，找出WLB数量最多的BLB中WLB的数量
    for (unsigned long block_level_block_id = 0; block_level_block_id < output_template->size_of_block_begin_warp_index_offset - 1; block_level_block_id++)
    {
        // 当前BLB的WLB数量
        unsigned long WLB_index_of_this_BLB = read_from_array_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, block_level_block_id);
        unsigned long WLB_index_of_next_BLB = read_from_array_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, block_level_block_id + 1);
        unsigned long WLB_num_of_cur_BLB = WLB_index_of_next_BLB - WLB_index_of_this_BLB;

        if (WLB_num_of_cur_BLB > max_WLB_num_of_BLB)
        {
            max_WLB_num_of_BLB = WLB_num_of_cur_BLB;
        }
    }

    return_str = return_str + "\n";

    // 申请一个共享内存存储所有warp的中间结果
    return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_val_arr) + " warp_tmp_result_inner_block[" + to_string(max_WLB_num_of_BLB) + "];\n";

    shared_memory_used_size = shared_memory_used_size + max_WLB_num_of_BLB;

    return_str = return_str + "\n";

    // 只要WLB的首行行号没有被压缩，就需要使用共享内存执行广播
    if (output_template->block_first_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->block_first_row_index_compress_meta == NULL);
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_first_row_index) + " this_block_first_row_index_shared[1];\n";
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_first_row_index) + " next_block_first_row_index_shared[1];\n";

        shared_memory_used_size = shared_memory_used_size + 2;
    }

    return_str = return_str + "\n";

    // BLB的首个WLB索引没有被压缩，那就要用共享内存来广播
    if (output_template->block_begin_warp_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_warp_index_offset_compress_meta == NULL);
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " this_block_first_warp_index_shared[1];\n";

        shared_memory_used_size++;

        if (need_next_block_first_warp_index == true)
        {
            return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " next_block_first_warp_index_shared[1];\n";
            shared_memory_used_size++;
        }
    }

    return_str = return_str + "\n";

    // 只要warp中间结果的行偏移索引没有压缩，就需要用shared mem来广播
    if (output_template->row_offset_in_warp_tmp_result_compress == NONE_COMPRESS)
    {
        assert(output_template->row_offset_in_warp_tmp_result_compress_meta == NULL);
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_row_offset_in_warp_tmp_result) + " this_block_first_tmp_result_shared[1];\n";
        shared_memory_used_size++;
    }

    // 实际共享内存的占用不能超标
    // 尝试不去处理共享内存的超标问题，
    if (shared_memory_used_size >= get_config()["SHARED_MEM_TOTAL_SIZE"].as_integer())
    {
        cout << "shared memory overflow" << endl;
        cout << "shared_memory_used_size:" << shared_memory_used_size << endl;
        // assert(false);
    }

    return_str = return_str + "\n";

    // 根据block的数量和BLB的数量是否一致，决定要不要，加入for循环
    if (output_template->tblock_num == output_template->size_of_block_begin_warp_index_offset - 1)
    {
        return_str = return_str + "{\n";
        // BLB的索引
        return_str = return_str + "unsigned int block_level_block_id = bid;\n";
    }
    else
    {
        return_str = return_str + "for (unsigned int block_level_block_id = bid; block_level_block_id < " + to_string(output_template->size_of_block_begin_warp_index_offset - 1) + "; block_level_block_id = block_level_block_id + gridDim.x)\n{\n";
    }

    // 所有block级别元数据的声明
    return_str = return_str + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " this_block_first_warp_index;\n";

    if (need_next_block_first_warp_index == true)
    {
        return_str = return_str + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " next_block_first_warp_index;\n";
    }

    return_str = return_str + code_of_data_type(output_template->data_type_of_block_first_row_index) + " this_block_first_row_index;\n";
    return_str = return_str + code_of_data_type(output_template->data_type_of_block_first_row_index) + " next_block_first_row_index;\n";

    return_str = return_str + "\n";

    return_str = return_str + code_of_data_type(output_template->data_type_of_row_offset_in_warp_tmp_result) + " this_block_first_tmp_result;\n";

    return_str = return_str + "\n";

    return_str = return_str + "__syncthreads();\n\n";

    // 块级别的几个元数据的获取
    // 只有有东西没被压缩的时候，才需要获取元数据
    if (output_template->block_first_row_index_compress == NONE_COMPRESS || output_template->block_begin_warp_index_offset_compress == NONE_COMPRESS)
    {
        // 当前线程块的第一个线程取数
        return_str = return_str + "if (tid_in_block == 0)\n{\n";

        // warp的索引如果没有压缩，就需要从全局内存中取
        if (output_template->block_begin_warp_index_offset_compress == NONE_COMPRESS)
        {
            return_str = return_str + "this_block_first_warp_index_shared[0] = block_begin_warp_index_offset[block_level_block_id];\n";

            if (need_next_block_first_warp_index == true)
            {
                return_str = return_str + "next_block_first_warp_index_shared[0] = block_begin_warp_index_offset[block_level_block_id + 1];\n";
            }
        }

        if (output_template->block_first_row_index_compress == NONE_COMPRESS)
        {
            // 块的首行索引
            return_str = return_str + "this_block_first_row_index_shared[0] = block_first_row_index[block_level_block_id];\n";
            return_str = return_str + "next_block_first_row_index_shared[0] = block_first_row_index[block_level_block_id + 1];\n";
        }

        return_str = return_str + "}\n";

        return_str = return_str + "__syncthreads();\n\n";
    }

    // 同步获得行索引和warp索引

    // warp索引，压缩和不压缩
    if (output_template->block_begin_warp_index_offset_compress == NONE_COMPRESS)
    {
        return_str = return_str + "this_block_first_warp_index = this_block_first_warp_index_shared[0];\n";

        if (need_next_block_first_warp_index == true)
        {
            return_str = return_str + "next_block_first_warp_index = next_block_first_warp_index_shared[0];\n";
        }
    }
    else if (output_template->block_begin_warp_index_offset_compress == LINEAR_COMPRESS)
    {
        linear_compress_t *compressor = (linear_compress_t *)output_template->block_begin_warp_index_offset_compress_meta;

        assert(compressor != NULL);

        return_str = return_str + code_of_arr_read(compressor, "this_block_first_warp_index", "block_level_block_id") + ";\n";

        if (need_next_block_first_warp_index == true)
        {
            // 直接累加斜率，性能更高
            return_str = return_str + "next_block_first_warp_index = this_block_first_warp_index + " + to_string(compressor->coefficient) + ";\n";
            // return_str = return_str + code_of_arr_read(compressor, "next_block_first_warp_index", "(block_level_block_id + 1)") + ";\n";
        }
    }
    else
    {
        cout << "compress type is not supported" << endl;
        assert(false);
    }

    // 根据块首行索引的压缩情况执行行索引的计算
    if (output_template->block_first_row_index_compress == NONE_COMPRESS)
    {
        return_str = return_str + "this_block_first_row_index = this_block_first_row_index_shared[0];\n";
        return_str = return_str + "next_block_first_row_index = next_block_first_row_index_shared[0];\n";
    }
    else if (output_template->block_first_row_index_compress == LINEAR_COMPRESS)
    {
        linear_compress_t *compressor = (linear_compress_t *)output_template->block_first_row_index_compress_meta;
        assert(compressor != NULL);

        return_str = return_str + code_of_arr_read(compressor, "this_block_first_row_index", "block_level_block_id") + ";\n";

        // 线性压缩的第二个直接用累加的方式
        return_str = return_str + "next_block_first_row_index = this_block_first_row_index + " + to_string(compressor->coefficient) + ";\n";
        // return_str = return_str + code_of_arr_read(compressor, "this_block_first_row_index", "(block_level_block_id + 1)") + ";\n";
    }
    else
    {
        cout << "compress type is not supported" << endl;
        assert(false);
    }

    return_str = return_str + "\n";

    // 如果中间结果的行偏移是不能压缩的，就需要从全局内存中取，然后同步
    if (output_template->row_offset_in_warp_tmp_result_compress == NONE_COMPRESS)
    {
        // 块的第一个线程，获取块首个结果的行偏移量
        return_str = return_str + "if (tid_in_block == 0)\n{\n";

        return_str = return_str + "this_block_first_tmp_result_shared[0] = row_offset_in_warp_tmp_result[this_block_first_row_index];\n";

        return_str = return_str + "}\n";

        return_str = return_str + "__syncthreads();\n";
    }

    // 计算块的首行非零元的索引
    if (output_template->row_offset_in_warp_tmp_result_compress == NONE_COMPRESS)
    {
        return_str = return_str + "this_block_first_tmp_result = this_block_first_tmp_result_shared[0];\n";
    }
    else if (output_template->row_offset_in_warp_tmp_result_compress == LINEAR_COMPRESS)
    {
        linear_compress_t *compressor = (linear_compress_t *)output_template->row_offset_in_warp_tmp_result_compress_meta;
        assert(compressor != NULL);

        return_str = return_str + code_of_arr_read(compressor, "this_block_first_tmp_result", "this_block_first_row_index") + ";\n";
    }
    else
    {
        cout << "compress type is not supported" << endl;
        assert(false);
    }

    // warp级别的遍历
    if (need_next_block_first_warp_index == false)
    {
        return_str = return_str + "{\n";

        return_str = return_str + "unsigned int warp_level_block_id = this_block_first_warp_index + wid_in_block;\n";
    }
    else
    {
        return_str = return_str + "for (unsigned int warp_level_block_id = this_block_first_warp_index + wid_in_block; warp_level_block_id < next_block_first_warp_index; warp_level_block_id = warp_level_block_id + warp_num_in_block)\n{\n";
    }

    // warp即便元数据的声明
    return_str = return_str + code_of_data_type(output_template->data_type_of_global_warp_block_first_nz) + " this_warp_block_first_nz;\n";

    if (need_next_warp_block_first_nz == true)
    {
        return_str = return_str + code_of_data_type(output_template->data_type_of_global_warp_block_first_nz) + " next_warp_block_first_nz;\n";
    }

    // 如果warp的第一个非零元索引没有压缩，warp的第一个线程从全局内存中取数据
    if (output_template->global_warp_block_first_nz_compress == NONE_COMPRESS)
    {
        return_str = return_str + "if (tid_in_warp == 0)\n{\n";

        return_str = return_str + "this_warp_block_first_nz = global_warp_block_first_nz[warp_level_block_id];\n";

        if (need_next_warp_block_first_nz == true)
        {
            return_str = return_str + "next_warp_block_first_nz = global_warp_block_first_nz[warp_level_block_id + 1];\n";
        }

        return_str = return_str + "}\n";
    }

    // 如果没有压缩，warp的非零元范围就要从全局内存获取
    if (output_template->global_warp_block_first_nz_compress == NONE_COMPRESS)
    {
        // 用warp reduce广播结果
        return_str = return_str + "this_warp_block_first_nz = __shfl_sync(0xFFFFFFFF, this_warp_block_first_nz, 0, 32);\n";

        if (need_next_warp_block_first_nz == true)
        {
            return_str = return_str + "next_warp_block_first_nz = __shfl_sync(0xFFFFFFFF, next_warp_block_first_nz, 0, 32);\n";
        }
    }
    else if (output_template->global_warp_block_first_nz_compress == LINEAR_COMPRESS)
    {
        // 用线性压缩直接计算当前warp的首个非零元索引
        linear_compress_t *compressor = (linear_compress_t *)output_template->global_warp_block_first_nz_compress_meta;
        assert(compressor != NULL);

        return_str = return_str + code_of_arr_read(compressor, "this_warp_block_first_nz", "warp_level_block_id") + ";\n";

        if (need_next_warp_block_first_nz == true)
        {
            return_str = return_str + "next_warp_block_first_nz = this_warp_block_first_nz + " + to_string(compressor->coefficient) + ";\n";
        }
    }
    else
    {
        cout << "compress type is not supported" << endl;
        assert(false);
    }

    return_str = return_str + "\n";

    return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " result_tmp_result = 0;\n\n";

    // 根据是否有列号，执行不同的非零元计算结果的累加
    string kernal_first_col_var_name_and_code = "";

    if (output_template->kernal_first_col_index != 0)
    {
        kernal_first_col_var_name_and_code = "kernal_first_col_index + ";
    }

    // 根据一个线程负责的非零元是不是多于一个决定是否要加入累加
    string result_tmp_result_var_name_and_code = "";

    // 非零元计算，并得到中间结果
    if (need_next_warp_block_first_nz == false)
    {
        return_str = return_str + "{\n";
        return_str = return_str + "unsigned int global_nz_index = this_warp_block_first_nz + tid_in_warp;\n";
    }
    else
    {
        return_str = return_str + "for (unsigned int global_nz_index = this_warp_block_first_nz + tid_in_warp; global_nz_index < next_warp_block_first_nz; global_nz_index = global_nz_index + 32)\n{\n";

        // 这里需要执行累加
        result_tmp_result_var_name_and_code = "result_tmp_result + ";
    }

    // 执行累加
    return_str = return_str + "result_tmp_result = " + result_tmp_result_var_name_and_code + " val_arr[global_nz_index] * __ldg(&(device_x_arr[" + kernal_first_col_var_name_and_code + " col_index_arr[global_nz_index]]));\n";

    // warp内非零元级别
    return_str = return_str + "\n}\n";

    // 执行warp内的归约
    return_str = return_str + "for (int offset = 16; offset > 0; offset = offset / 2)\n{\n";

    return_str = return_str + "result_tmp_result = result_tmp_result + __shfl_down_sync(0xFFFFFFFF, result_tmp_result, offset);\n";

    // warp reduce
    return_str = return_str + "\n}\n";

    // 将warp归约的结果放回共享内存
    // 只有一个warp的第一个线程
    return_str = return_str + "if (tid_in_warp == 0)\n{\n";
    if (need_next_block_first_warp_index == false)
    {
        return_str = return_str + "warp_tmp_result_inner_block[wid_in_block] = result_tmp_result;\n";
    }
    else
    {
        // 如果一个warp要处理多个WLB，就需要减一下来获取中间结果的写的位置
        return_str = return_str + "warp_tmp_result_inner_block[warp_level_block_id - this_block_first_warp_index] = result_tmp_result;\n";
    }

    return_str = return_str + "}\n";

    // warp级别
    return_str = return_str + "\n}\n";

    // block粒度归约前必然的同步
    return_str = return_str + "__syncthreads();\n\n";

    // 在block级别做归约，分为32个线程之内的多线程归约和单线程归约
    if (output_template->thread_num_of_row_reduce == 1)
    {
        // 每个线程负责一行
        return_str = return_str + "for (unsigned int row_id = this_block_first_row_index + tid_in_block; row_id < next_block_first_row_index && row_id < " + to_string(output_template->effective_row_num) + "; row_id = row_id + blockDim.x)\n{\n";

        // 声明遍历的上界和下界的变量
        return_str = return_str + code_of_data_type(output_template->data_type_of_row_offset_in_warp_tmp_result) + " this_row_local_first_tmp_result;\n";
        return_str = return_str + code_of_data_type(output_template->data_type_of_row_offset_in_warp_tmp_result) + " next_row_local_first_tmp_result;\n";

        // 获取当前行结果的全局偏移量
        if (output_template->row_offset_in_warp_tmp_result_compress == NONE_COMPRESS)
        {
            return_str = return_str + "this_row_local_first_tmp_result = row_offset_in_warp_tmp_result[row_id];\n";
            return_str = return_str + "next_row_local_first_tmp_result = row_offset_in_warp_tmp_result[row_id + 1];\n";
        }
        else if (output_template->row_offset_in_warp_tmp_result_compress == LINEAR_COMPRESS)
        {
            linear_compress_t *compressor = (linear_compress_t *)output_template->row_offset_in_warp_tmp_result_compress_meta;
            assert(compressor != NULL);

            // 每一行中间结果的上界和下界
            return_str = return_str + code_of_arr_read(compressor, "this_row_local_first_tmp_result", "row_id") + ";\n";
            return_str = return_str + "next_row_local_first_tmp_result = this_row_local_first_tmp_result + " + to_string(compressor->coefficient) + ";\n";
        }
        else
        {
            cout << "compress type is not supported" << endl;
            assert(false);
        }

        // 将全局偏移量变为局部偏移量
        return_str = return_str + "this_row_local_first_tmp_result = this_row_local_first_tmp_result - this_block_first_tmp_result;\n";
        return_str = return_str + "next_row_local_first_tmp_result = next_row_local_first_tmp_result - this_block_first_tmp_result;\n";

        return_str = return_str + "\n";

        // 用一个变量存当前行的中间结果
        return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " row_tmp_result = 0;\n";

        // 遍历一行的所有非零元
        return_str = return_str + "for (unsigned int temp_result_id = this_row_local_first_tmp_result; temp_result_id < next_row_local_first_tmp_result; temp_result_id = temp_result_id + 1)\n{\n";
        return_str = return_str + "row_tmp_result = row_tmp_result + warp_tmp_result_inner_block[temp_result_id];\n";
        return_str = return_str + "\n}\n";

        // 这个时候在row_tmp_result中存储了这一行的最终结果，逻辑行号是row_id，找出实际行号，并最后获得写的结果的位置
        string var_name_of_global_row_index = "row_id";
        
        // 如果排序就需要新的变量
        if (output_template->local_sort_index == false && output_template->global_sort_index == false)
        {
            // 不排序，直接加上偏移量获得真实的行号
            if (output_template->kernal_first_row_index != 0)
            {
                return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->matrix->dense_row_number)) + " global_row_index = " + var_name_of_global_row_index + " + kernal_first_row_index" + ";\n";
                var_name_of_global_row_index = "global_row_index";
            }
        }
        else
        {
            return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->matrix->dense_row_number)) + " global_row_index = " + var_name_of_global_row_index + ";\n";

            if (output_template->local_sort_index == true)
            {
                assert(output_template->global_sort_index == false);
                // 获取真实的行索引
                if (output_template->kernal_first_row_index != 0)
                {
                    return_str = return_str + "global_row_index = row_index_before_sort[global_row_index] + kernal_first_row_index;\n";
                }
                else
                {
                    return_str = return_str + "global_row_index = row_index_before_sort[global_row_index];\n";
                }
            }

            if (output_template->global_sort_index == true)
            {
                assert(output_template->local_sort_index == false);

                if (output_template->kernal_first_row_index != 0)
                {
                    return_str = return_str + "global_row_index = row_index_before_sort[global_row_index + kernal_first_row_index];\n";
                }
                else
                {
                    return_str = return_str + "global_row_index = row_index_before_sort[global_row_index];\n";
                }
            }

            var_name_of_global_row_index = "global_row_index";
        }

        // 一行的结果的变量，存在
        string reduce_result_var_name = "row_tmp_result";

        // 将结果归约到全局内存
        // 原子加
        if (output_template->is_atom_add == true)
        {
            // 原子加
            return_str = return_str + "atomicAdd(&(device_y_arr[" + var_name_of_global_row_index + "]), " + reduce_result_var_name + ");\n";
        }
        else
        {
            // 赋值
            return_str = return_str + "device_y_arr[" + var_name_of_global_row_index + "] = " + reduce_result_var_name + ";\n";
        }

        // block reduce级别，
        return_str = return_str + "\n}\n";
    }
    else
    {
        assert(output_template->thread_num_of_row_reduce == 2 || output_template->thread_num_of_row_reduce == 4 || output_template->thread_num_of_row_reduce == 8 ||
               output_template->thread_num_of_row_reduce == 16 || output_template->thread_num_of_row_reduce == 32);

        return_str = return_str + "unsigned char active_row_reduce_thread_num = " + to_string(output_template->thread_num_of_row_reduce) + ";\n\n";

        // 这里不做压缩，就假设用来归约的线程数量和一个BLB中的线程数量是不匹配的，使用一个for循环来遍历这个线程负责的行，并执行每一行的归约
        // 需要for循环，首先计算遍历的步长
        return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->thread_num_in_block)) + " step_size = blockDim.x / " + to_string(output_template->thread_num_of_row_reduce) + ";\n";

        // 计算这个线程的行内索引
        return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->thread_num_of_row_reduce)) + " tid_in_row = tid_in_block % active_row_reduce_thread_num;\n\n";

        // 执行for循环
        return_str = return_str + "for(";
        return_str = return_str + "unsigned int" + " row_id = this_block_first_row_index + tid_in_block / active_row_reduce_thread_num; row_id < next_block_first_row_index && row_id < " + to_string(output_template->effective_row_num) + "; row_id = row_id + step_size)\n{\n";

        // 结果在共享内存中的首地址
        return_str = return_str + code_of_data_type(output_template->data_type_of_row_offset_in_warp_tmp_result) + " this_row_local_first_tmp_result;\n";
        return_str = return_str + code_of_data_type(output_template->data_type_of_row_offset_in_warp_tmp_result) + " next_row_local_first_tmp_result;\n\n";

        // 获取当前行结果的全局偏移量
        if (output_template->row_offset_in_warp_tmp_result_compress == NONE_COMPRESS)
        {
            return_str = return_str + "this_row_local_first_tmp_result = row_offset_in_warp_tmp_result[row_id];\n";
            return_str = return_str + "next_row_local_first_tmp_result = row_offset_in_warp_tmp_result[row_id + 1];\n";
        }
        else if (output_template->row_offset_in_warp_tmp_result_compress == LINEAR_COMPRESS)
        {
            linear_compress_t *compressor = (linear_compress_t *)output_template->row_offset_in_warp_tmp_result_compress_meta;
            assert(compressor != NULL);

            // 每一行中间结果的上界和下界
            return_str = return_str + code_of_arr_read(compressor, "this_row_local_first_tmp_result", "row_id") + ";\n";
            return_str = return_str + "next_row_local_first_tmp_result = this_row_local_first_tmp_result + " + to_string(compressor->coefficient) + ";\n";
        }
        else
        {
            cout << "compress type is not supported" << endl;
            assert(false);
        }

        // 将全局偏移量变为局部偏移量
        return_str = return_str + "this_row_local_first_tmp_result = this_row_local_first_tmp_result - this_block_first_tmp_result;\n";
        return_str = return_str + "next_row_local_first_tmp_result = next_row_local_first_tmp_result - this_block_first_tmp_result;\n";

        return_str = return_str + "\n";

        // 用一个变量存当前行的中间结果
        // return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " row_tmp_result = 0;\n";

        // 树状归约的最外层遍历，遍历树状归约的某一层
        return_str = return_str + "for(";
        return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->thread_num_of_row_reduce)) + " cur_active_row_reduce_thread_num = active_row_reduce_thread_num; cur_active_row_reduce_thread_num >= 1; cur_active_row_reduce_thread_num = cur_active_row_reduce_thread_num / 2)\n{\n";

        // 只有活跃的线程要执行归约
        return_str = return_str + "if (tid_in_row < cur_active_row_reduce_thread_num)\n{\n";

        // 执行一层内部的归约
        return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " row_tmp_result = 0;\n";

        return_str = return_str + "for (";

        return_str = return_str + "unsigned int" + " temp_result_id = this_row_local_first_tmp_result + tid_in_row; temp_result_id < next_row_local_first_tmp_result; temp_result_id = temp_result_id + cur_active_row_reduce_thread_num)\n{\n";

        return_str = return_str + "row_tmp_result = row_tmp_result + warp_tmp_result_inner_block[temp_result_id];\n";

        return_str = return_str + "}\n";

        // 将一个线程的结果写回共享内存，并且确定下一层的归约范围
        return_str = return_str + "warp_tmp_result_inner_block[this_row_local_first_tmp_result + tid_in_row] = row_tmp_result;\n";
        return_str = return_str + "next_row_local_first_tmp_result = this_row_local_first_tmp_result + cur_active_row_reduce_thread_num;\n";

        return_str = return_str + "}\n";
        return_str = return_str + "}\n";

        string reduce_result_var_name = "warp_tmp_result_inner_block[this_row_local_first_tmp_result]";

        // 行的第一个线程归约数据
        return_str = return_str + "if (tid_in_row == 0)\n{\n";

        // 查找排序之前的行索引之后写回
        string var_name_of_global_row_index = "row_id";

        // 如果排序就需要新的变量
        if (output_template->local_sort_index == false && output_template->global_sort_index == false)
        {
            // 不排序，直接加上偏移量获得真实的行号
            if (output_template->kernal_first_row_index != 0)
            {
                return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->matrix->dense_row_number)) + " global_row_index = " + var_name_of_global_row_index + " + kernal_first_row_index" + ";\n";
                var_name_of_global_row_index = "global_row_index";
            }
        }
        else
        {
            return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->matrix->dense_row_number)) + " global_row_index = " + var_name_of_global_row_index + ";\n";

            if (output_template->local_sort_index == true)
            {
                assert(output_template->global_sort_index == false);
                // 获取真实的行索引
                if (output_template->kernal_first_row_index != 0)
                {
                    return_str = return_str + "global_row_index = row_index_before_sort[global_row_index] + kernal_first_row_index;\n";
                }
                else
                {
                    return_str = return_str + "global_row_index = row_index_before_sort[global_row_index];\n";
                }
            }

            if (output_template->global_sort_index == true)
            {
                assert(output_template->local_sort_index == false);

                if (output_template->kernal_first_row_index != 0)
                {
                    return_str = return_str + "global_row_index = row_index_before_sort[global_row_index + kernal_first_row_index];\n";
                }
                else
                {
                    return_str = return_str + "global_row_index = row_index_before_sort[global_row_index];\n";
                }
            }

            var_name_of_global_row_index = "global_row_index";
        }

        // 将结果归约到全局内存
        // 原子加
        if (output_template->is_atom_add == true)
        {
            // 原子加
            return_str = return_str + "atomicAdd(&(device_y_arr[" + var_name_of_global_row_index + "]), " + reduce_result_var_name + ");\n";
        }
        else
        {
            // 赋值
            return_str = return_str + "device_y_arr[" + var_name_of_global_row_index + "] = " + reduce_result_var_name + ";\n";
        }

        return_str = return_str + "}\n";

        // 一行结果的归约
        return_str = return_str + "}\n";
    }

    // block级别
    return_str = return_str + "\n}\n";
    // 全局级别
    return_str = return_str + "\n}\n";

    return return_str;
}

string code_of_kernal_function_call(shared_memory_total_warp_reduce_template_t *output_template, unsigned long dense_block_id)
{
    assert(output_template != NULL);
    // 线程块的数量和线程的数量不能超标
    assert(output_template->tblock_num <= get_config()["MAX_TBLOCK_NUM"].as_integer() && output_template->thread_num_in_block <= get_config()["MAX_THREAD_NUM_IN_BLOCK"].as_integer());

    string return_str = "spmv_" + to_string(dense_block_id) + "<<<" + to_string(output_template->tblock_num) + ", " + to_string(output_template->thread_num_in_block) + ", 0, stream_arr[" + to_string(dense_block_id) + "]>>>(";

    bool is_first_param = true;

    // 这里加入形参的声明
    if (output_template->row_offset_in_warp_tmp_result_compress == NONE_COMPRESS)
    {
        assert(output_template->row_offset_in_warp_tmp_result != NULL);
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "row_offset_in_warp_tmp_result");
        is_first_param = false;
    }

    if (output_template->block_first_row_index_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }
        assert(output_template->block_first_row_index != NULL);
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "block_first_row_index");
    }

    if (output_template->block_begin_warp_index_offset_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }
        assert(output_template->block_begin_warp_index_offset != NULL);
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset");
    }

    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->row_index_before_sort != NULL)
    {
        // 这里代表有排序过
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort");
    }

    if (output_template->global_warp_block_first_nz_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }
        assert(output_template->global_warp_block_first_nz != NULL);
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "global_warp_block_first_nz");
    }

    if (is_first_param == false)
    {
        return_str = return_str + ", ";
    }
    else
    {
        is_first_param = false;
    }
    assert(output_template->val_arr != NULL);
    return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "val_arr");

    return_str = return_str + ", ";
    assert(output_template->col_index_arr != NULL);
    return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "col_index_arr");

    // x的值
    return_str = return_str + ", ";
    return_str = return_str + "device_x_arr";

    // y的值
    return_str = return_str + ", ";
    return_str = return_str + "device_y_arr";

    return_str = return_str + ");\n";

    return return_str;
}

string  code_of_write_template_data_to_gpu(shared_memory_total_warp_reduce_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index)
{
    string template_data_name = "dense_block_" + to_string(dense_block_id) + "_template_data";

    string return_str = "compressed_dense_block_" + to_string(dense_block_id) + "_t *" + template_data_name + " = read_dense_block_" + to_string(dense_block_id) + "_from_file(" + "\"" + string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source/" + to_string(output_template->hash_of_this_template) + "_" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer()) + "\");\n\n";

    // 全局排序的数组取一个特殊的名字，并且只处理一次，剩下的从这里拷贝即可
    if (output_template->global_sort_index == true)
    {
        if (output_template->dense_block_index == 0 && force_not_share_global_sort_index == false)
        {
            return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_row_index_before_sort, "device_global_sort_index");
            // 申请、拷贝、一气呵成
            return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_row_index_before_sort, to_string(output_template->size_of_row_index_before_sort), "device_global_sort_index");
            return_str = return_str + code_line_of_cuda_memcpy("device_global_sort_index", template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"), output_template->data_type_of_row_index_before_sort, to_string(output_template->size_of_row_index_before_sort), "cudaMemcpyHostToDevice") + "\n";
        }
    }

    if (output_template->row_offset_in_warp_tmp_result_compress == NONE_COMPRESS)
    {
        assert(output_template->row_offset_in_warp_tmp_result != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_row_offset_in_warp_tmp_result, "device_" + code_of_arr_var_name(dense_block_id, -1, "row_offset_in_warp_tmp_result"));
    }

    if (output_template->block_first_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->block_first_row_index != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_block_first_row_index, "device_" + code_of_arr_var_name(dense_block_id, -1, "block_first_row_index"));
    }

    if (output_template->block_begin_warp_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_warp_index_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_block_begin_warp_index_offset, "device_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset"));
    }

    // 行顺序数组的声明
    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->row_index_before_sort != NULL)
    {
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_row_index_before_sort, "device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"));
    }

    if (output_template->global_warp_block_first_nz_compress == NONE_COMPRESS)
    {
        assert(output_template->global_warp_block_first_nz != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_global_warp_block_first_nz, "device_" + code_of_arr_var_name(dense_block_id, -1, "global_warp_block_first_nz"));
    }

    assert(output_template->val_arr != NULL);
    return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_val_arr, "device_" + code_of_arr_var_name(dense_block_id, -1, "val_arr"));

    assert(output_template->col_index_arr != NULL);
    return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_col_index_arr, "device_" + code_of_arr_var_name(dense_block_id, -1, "col_index_arr"));

    return_str = return_str + "\n";

    if (output_template->row_offset_in_warp_tmp_result_compress == NONE_COMPRESS)
    {
        assert(output_template->row_offset_in_warp_tmp_result != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_row_offset_in_warp_tmp_result, to_string(output_template->size_of_row_offset_in_warp_tmp_result), "device_" + code_of_arr_var_name(dense_block_id, -1, "row_offset_in_warp_tmp_result"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "row_offset_in_warp_tmp_result"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "row_offset_in_warp_tmp_result"), output_template->data_type_of_row_offset_in_warp_tmp_result, to_string(output_template->size_of_row_offset_in_warp_tmp_result), "cudaMemcpyHostToDevice") + "\n";
    }

    if (output_template->block_first_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->block_first_row_index != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_block_first_row_index, to_string(output_template->size_of_block_first_row_index), "device_" + code_of_arr_var_name(dense_block_id, -1, "block_first_row_index"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "block_first_row_index"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "block_first_row_index"), output_template->data_type_of_block_first_row_index, to_string(output_template->size_of_block_first_row_index), "cudaMemcpyHostToDevice") + "\n";
    }

    if (output_template->block_begin_warp_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_warp_index_offset != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_block_begin_warp_index_offset, to_string(output_template->size_of_block_begin_warp_index_offset), "device_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset"), output_template->data_type_of_block_begin_warp_index_offset, to_string(output_template->size_of_block_begin_warp_index_offset), "cudaMemcpyHostToDevice") + "\n";
    }

    // 如果是全局的就直接赋值
    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->global_sort_index == true)
    {
        assert(output_template->local_sort_index == false && output_template->row_index_before_sort != NULL);

        if (force_not_share_global_sort_index == true)
        {
            return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_row_index_before_sort, to_string(output_template->size_of_row_index_before_sort), "device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"));
            // 拷贝
            return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"), output_template->data_type_of_row_index_before_sort, to_string(output_template->size_of_row_index_before_sort), "cudaMemcpyHostToDevice") + "\n";
        }
        else
        {
            return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort") + "=" + "device_global_sort_index;\n";
        }
    }

    // 如果是局部的就拷贝
    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->local_sort_index == true)
    {
        assert(output_template->global_sort_index == false && output_template->row_index_before_sort != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_row_index_before_sort, to_string(output_template->size_of_row_index_before_sort), "device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"), output_template->data_type_of_row_index_before_sort, to_string(output_template->size_of_row_index_before_sort), "cudaMemcpyHostToDevice") + "\n";
    }

    if (output_template->global_warp_block_first_nz_compress == NONE_COMPRESS)
    {
        assert(output_template->global_warp_block_first_nz != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_global_warp_block_first_nz, to_string(output_template->size_of_global_warp_block_first_nz), "device_" + code_of_arr_var_name(dense_block_id, -1, "global_warp_block_first_nz"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "global_warp_block_first_nz"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "global_warp_block_first_nz"), output_template->data_type_of_global_warp_block_first_nz, to_string(output_template->size_of_global_warp_block_first_nz), "cudaMemcpyHostToDevice") + "\n";
    }

    assert(output_template->val_arr != NULL);
    return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_val_arr, to_string(output_template->size_of_val_arr), "device_" + code_of_arr_var_name(dense_block_id, -1, "val_arr"));
    // 拷贝
    return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "val_arr"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "val_arr"), output_template->data_type_of_val_arr, to_string(output_template->size_of_val_arr), "cudaMemcpyHostToDevice") + "\n";

    assert(output_template->col_index_arr != NULL);
    return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_col_index_arr, to_string(output_template->size_of_col_index_arr), "device_" + code_of_arr_var_name(dense_block_id, -1, "col_index_arr"));
    // 拷贝
    return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "col_index_arr"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "col_index_arr"), output_template->data_type_of_col_index_arr, to_string(output_template->size_of_col_index_arr), "cudaMemcpyHostToDevice") + "\n";

    return return_str;
}

// warp在block中的偏移量可以使用线性压缩
bool compress_block_begin_warp_index_offset(shared_memory_total_warp_reduce_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && type == LINEAR_COMPRESS && output_template->block_begin_warp_index_offset != NULL);

    linear_compress_t *compressor = init_linear_compressor(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, output_template->size_of_block_begin_warp_index_offset, need_check);

    if (compressor == NULL)
    {
        return false;
    }

    // 压缩成功
    output_template->block_begin_warp_index_offset_compress_meta = (void *)compressor;
    output_template->block_begin_warp_index_offset_compress = type;

    return true;
}

// 可以使用线性压缩
bool compress_row_offset_in_warp_tmp_result(shared_memory_total_warp_reduce_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && type == LINEAR_COMPRESS && output_template->row_offset_in_warp_tmp_result != NULL);

    linear_compress_t *compressor = init_linear_compressor(output_template->row_offset_in_warp_tmp_result, output_template->data_type_of_row_offset_in_warp_tmp_result, output_template->size_of_row_offset_in_warp_tmp_result, need_check);

    if (compressor == NULL)
    {
        return false;
    }

    // 压缩成功
    output_template->row_offset_in_warp_tmp_result_compress_meta = (void *)compressor;
    output_template->row_offset_in_warp_tmp_result_compress = type;

    return true;
}

// 使用线性压缩，或者循环增量压缩
bool compress_block_first_row_index(shared_memory_total_warp_reduce_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && output_template->block_first_row_index != NULL);
    assert(type == LINEAR_COMPRESS || type == CYCLE_INCREASE_COMPRESS);

    if (type == LINEAR_COMPRESS)
    {
        linear_compress_t *compressor = init_linear_compressor(output_template->block_first_row_index, output_template->data_type_of_block_first_row_index, output_template->size_of_block_first_row_index, need_check);

        if (compressor == NULL)
        {
            return false;
        }

        // 压缩成功
        output_template->block_first_row_index_compress_meta = (void *)compressor;
        output_template->block_first_row_index_compress = type;

        return true;
    }

    if (type == CYCLE_INCREASE_COMPRESS)
    {
        cycle_increase_compress_t *compressor = init_cycle_increase_compressor(output_template->block_first_row_index, output_template->data_type_of_block_first_row_index, output_template->size_of_block_first_row_index, need_check);

        if (compressor == NULL)
        {
            return false;
        }

        // 压缩成功
        output_template->block_first_row_index_compress_meta = (void *)compressor;
        output_template->block_first_row_index_compress = type;

        return true;
    }

    return false;
}

// 每个warp的第一个非零元的索引，可以使用线性压缩
bool compress_global_warp_block_first_nz(shared_memory_total_warp_reduce_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && output_template->global_warp_block_first_nz != NULL);
    assert(type == LINEAR_COMPRESS);

    if (type == LINEAR_COMPRESS)
    {
        linear_compress_t *compressor = init_linear_compressor(output_template->global_warp_block_first_nz, output_template->data_type_of_global_warp_block_first_nz, output_template->size_of_global_warp_block_first_nz, need_check);

        if (compressor == NULL)
        {
            return false;
        }

        // 压缩成功
        output_template->global_warp_block_first_nz_compress_meta = (void *)compressor;
        output_template->global_warp_block_first_nz_compress = type;

        return true;
    }

    return false;
}

bool set_row_reduce_thread_num(shared_memory_total_warp_reduce_template_t *output_template, unsigned long row_reduce_thread_num)
{
    assert(output_template != NULL);
    assert(row_reduce_thread_num == 1 || row_reduce_thread_num == 2 || row_reduce_thread_num == 4 || row_reduce_thread_num == 8 || row_reduce_thread_num == 16 || row_reduce_thread_num == 32);

    if (row_reduce_thread_num == 1)
    {
        output_template->thread_num_of_row_reduce = 1;
    }
    else
    {
        // 小并行，保证每一行线程粒度的块的数量（中间结果的数量），必须大于负责每一行归约的线程的数量
        // 遍历找出每一行线程粒度的块的数量，这个数量必须大于每一行负责归约的块的数量
        // 中间结果数组存储了所有这些信息
        assert(output_template->size_of_row_offset_in_warp_tmp_result == output_template->matrix->dense_row_number + 1);

        for (unsigned long i = 0; i < output_template->size_of_row_offset_in_warp_tmp_result - 1; i++)
        {
            // 一行中中间结果的数量
            unsigned long tmp_result_of_a_row = read_from_array_with_data_type(output_template->row_offset_in_warp_tmp_result, output_template->data_type_of_row_offset_in_warp_tmp_result, i + 1) - read_from_array_with_data_type(output_template->row_offset_in_warp_tmp_result, output_template->data_type_of_row_offset_in_warp_tmp_result, i);
            // 这个数量必须大于每一行的归约线程的数量
            if (tmp_result_of_a_row < row_reduce_thread_num)
            {
                cout << "thread num to reduce a row is more than tmp result number" << endl;
                return false;
            }
        }

        output_template->thread_num_of_row_reduce = row_reduce_thread_num;
    }

    return true;
}

// 尝试所有的压缩
void try_all_compress(shared_memory_total_warp_reduce_template_t *output_template)
{
    assert(output_template != NULL);

    bool is_compressed = false;

    is_compressed = compress_block_begin_warp_index_offset(output_template, true, LINEAR_COMPRESS);

    // 第一个是中间结果行偏移量
    is_compressed = compress_row_offset_in_warp_tmp_result(output_template, true, LINEAR_COMPRESS);

    // block首行索引的压缩
    is_compressed = compress_block_first_row_index(output_template, true, LINEAR_COMPRESS);

    // warp block 的首个非零元索引
    is_compressed = compress_global_warp_block_first_nz(output_template, true, LINEAR_COMPRESS);
}