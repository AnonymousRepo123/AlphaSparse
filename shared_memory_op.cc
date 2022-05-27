#include "shared_memory_op.hpp"

shared_memory_template_t *init_shared_memory_template(code_builder_t *builder, unsigned long dense_block_id)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;
    assert(matrix->block_coor_table.item_arr.size() > dense_block_id);

    // 创建对应模板
    shared_memory_template_t *new_template = new shared_memory_template_t();

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

    // 将reduce相关的索引删掉
    // delete_arr_with_data_type(compressed_block_view->y_write_index[0]->index_arr, compressed_block_view->y_write_index[0]->index_data_type);

    // 首先处理每一线程的全局行索引，将全局行索引搞出来
    // 分别遍历三个层次的索引
    index_of_compress_block_t *block_level_index = compressed_block_view->read_index[2];
    index_of_compress_block_t *warp_level_index = compressed_block_view->read_index[3];
    index_of_compress_block_t *thread_level_index = compressed_block_view->read_index[4];
    assert(block_level_index->level_of_this_index == TBLOCK_LEVEL);
    assert(warp_level_index->level_of_this_index == WRAP_LEVEL);
    assert(thread_level_index->level_of_this_index == THREAD_LEVEL);

    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[0]->max_row_index == block_level_index->max_row_index);
    assert(matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index == block_level_index->max_row_index);

    if (thread_level_index->row_number_of_block_arr != NULL)
    {
        cout << "row num in thread level block must be 1, thread level index shouldn't have this metadata" << endl;
        assert(false);
    }

    // 遍历所有线程块粒度的块所包含的行，如果有相交代表要用原子加来规约显存上的结果
    // 用两个变量分别存储，当前的线程块粒度的块包含的最小行号和最大行号
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
    unsigned long total_row_num = block_level_index->max_row_index - block_level_index->min_row_index + 1;

    vector<unsigned long> new_row_offset_in_thread_tmp_result_vec;
    new_row_offset_in_thread_tmp_result_vec.push_back(0);

    // 用一个数组记录每一行中间结果的数量
    vector<unsigned long> thread_level_result_num_of_each_row(total_row_num);

    // 全部初始化为0
    for (unsigned long i = 0; i < total_row_num; i++)
    {
        thread_level_result_num_of_each_row[i] = 0;
    }

    // 分别遍历三个层次
    // 遍历三个层次的索引
    for (unsigned long index_of_block_level_index = 0; index_of_block_level_index < block_level_index->block_num; index_of_block_level_index++)
    {
        // cout << "index_of_block_level_index:" << index_of_block_level_index << endl;
        // 当前block的首行行号
        unsigned long block_first_row_index = read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, index_of_block_level_index);
        // block中第一个warp号和下一个block的首warp
        unsigned long this_block_first_warp_index = read_from_array_with_data_type(block_level_index->index_arr, block_level_index->index_data_type, index_of_block_level_index);
        unsigned long next_block_first_warp_index = read_from_array_with_data_type(block_level_index->index_arr, block_level_index->index_data_type, index_of_block_level_index + 1);

        // 遍历warp层次
        for (unsigned long index_of_warp_level_index = this_block_first_warp_index; index_of_warp_level_index < next_block_first_warp_index; index_of_warp_level_index++)
        {
            assert(index_of_warp_level_index < warp_level_index->block_num);
            unsigned long warp_first_row_index = read_from_array_with_data_type(warp_level_index->index_of_the_first_row_arr, warp_level_index->data_type_of_index_of_the_first_row_arr, index_of_warp_level_index);
            unsigned long this_warp_first_thread_index = read_from_array_with_data_type(warp_level_index->index_arr, warp_level_index->index_data_type, index_of_warp_level_index);
            unsigned long next_warp_first_thread_index = read_from_array_with_data_type(warp_level_index->index_arr, warp_level_index->index_data_type, index_of_warp_level_index + 1);

            for (unsigned long index_of_thread_level_index = this_warp_first_thread_index; index_of_thread_level_index < next_warp_first_thread_index; index_of_thread_level_index++)
            {
                // assert(index_of_thread_level_index < thread_level_index->block_num);
                if (index_of_thread_level_index >= thread_level_index->block_num)
                {
                    assert(false);
                }
                unsigned long thread_first_row_index = read_from_array_with_data_type(thread_level_index->index_of_the_first_row_arr, thread_level_index->data_type_of_index_of_the_first_row_arr, index_of_thread_level_index);

                // 全局的线程粒度的子块所覆盖的行
                unsigned long global_thread_first_row_index = block_first_row_index + warp_first_row_index + thread_first_row_index;

                assert(global_thread_first_row_index < total_row_num);

                thread_level_result_num_of_each_row[global_thread_first_row_index] = thread_level_result_num_of_each_row[global_thread_first_row_index] + 1;

                // 因为在线程块粒度先进行一次归约，线程块内部肯定是一行一个结果，所以所以本质上判断线程块粒度的块的行的分布来判断对于原子加的要求
            }
        }
    }

    // 遍历每一行结果的数量，从而得出每一行在中间结果中的偏移量
    for (unsigned long row_index = 0; row_index < thread_level_result_num_of_each_row.size(); row_index++)
    {
        new_row_offset_in_thread_tmp_result_vec.push_back(new_row_offset_in_thread_tmp_result_vec[new_row_offset_in_thread_tmp_result_vec.size() - 1] + thread_level_result_num_of_each_row[row_index]);
    }

    assert(new_row_offset_in_thread_tmp_result_vec.size() == total_row_num + 1);

    // 确定数据类型的大小
    new_template->data_type_of_row_offset_in_thread_tmp_result = find_most_suitable_data_type(new_row_offset_in_thread_tmp_result_vec[new_row_offset_in_thread_tmp_result_vec.size() - 1]);
    // 数组的长度
    new_template->size_of_row_offset_in_thread_tmp_result = new_row_offset_in_thread_tmp_result_vec.size();
    // 申请归约数组
    new_template->row_offset_in_thread_tmp_result = malloc_arr(new_template->size_of_row_offset_in_thread_tmp_result, new_template->data_type_of_row_offset_in_thread_tmp_result);
    // 拷贝数组
    copy_unsigned_long_arr_to_others(&(new_row_offset_in_thread_tmp_result_vec[0]), new_template->row_offset_in_thread_tmp_result, new_template->data_type_of_row_offset_in_thread_tmp_result, new_template->size_of_row_offset_in_thread_tmp_result);

    // 拷贝块的首行行号，因为实际大小会多一个，需要重新申请一个数组，最后一位是整个block行的数量，插在最后的CSR索引的大小肯定大于之前的所有数据
    // 考虑到空行，最多也是等于最后一个数据
    assert(total_row_num >= read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, block_level_index->block_num - 1));
    // 创建一个新的数组，包含block_num + 1个元素，最后一个是整个稠密子块的总行号
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

    // // 重新申请数组，比原来大一点点
    // new_template->data_type_of_block_first_row_index = block_level_index->data_type_of_index_of_the_first_row_arr;
    // new_template->block_first_row_index = block_level_index->index_of_the_first_row_arr;
    // // 大小是block级别的块的数量
    // new_template->size_of_block_first_row_index = block_level_index->block_num;

    // 拷贝一些数组，包含block和wrap的主要数据
    // 剩下几个数组直接拷
    new_template->data_type_of_block_begin_warp_index_offset = block_level_index->index_data_type;
    new_template->block_begin_warp_index_offset = block_level_index->index_arr;
    new_template->size_of_block_begin_warp_index_offset = block_level_index->length;

    new_template->data_type_of_warp_begin_thread_index_offset = warp_level_index->index_data_type;
    new_template->warp_begin_thread_index_offset = warp_level_index->index_arr;
    new_template->size_of_warp_begin_thread_index_offset = warp_level_index->length;

    new_template->data_type_of_thread_block_size_in_warp = thread_level_index->data_type_of_coo_block_size_arr;
    new_template->thread_block_size_in_warp = thread_level_index->coo_block_size_arr;
    new_template->size_of_thread_block_size_in_warp = warp_level_index->block_num;

    // block和warp第一个非零元的索引，都按照各自的block size初始化
    new_template->data_type_of_block_nz_begin_offset = block_level_index->data_type_of_coo_begin_index_arr;
    new_template->block_nz_begin_offset = block_level_index->coo_begin_index_arr;
    new_template->size_of_block_nz_begin_offset = block_level_index->block_num;

    new_template->data_type_of_warp_nz_begin_offset = warp_level_index->data_type_of_coo_begin_index_arr;
    new_template->warp_nz_begin_offset = warp_level_index->coo_begin_index_arr;
    new_template->size_of_warp_nz_begin_offset = warp_level_index->block_num;

    void *val_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->padding_val_arr;
    data_type data_type_of_val_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->val_data_type;
    unsigned long size_of_val_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->padding_arr_size;

    // 仅仅经过padding之后的列数组及其数据类型
    void *col_index_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[5]->index_arr;
    assert(col_index_arr_after_padding != NULL);
    data_type data_type_of_col_index_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[5]->index_data_type;
    unsigned long size_of_col_index_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[5]->length;

    assert(size_of_val_arr_after_padding == size_of_col_index_arr_after_padding);

    // 将每个wrap块内部的数据进行交错存储，每个wrap内的线程粒度的块的大小是一样的，大小和padding之后的大小是一样的，数据类型也一样
    new_template->data_type_of_val_arr = data_type_of_val_arr_after_padding;
    new_template->size_of_val_arr = size_of_val_arr_after_padding;
    // 申请值数组
    new_template->val_arr = malloc_arr(new_template->size_of_val_arr, new_template->data_type_of_val_arr);

    // 列号也和padding之后是一样
    new_template->data_type_of_col_index_arr = data_type_of_col_index_arr_after_padding;
    new_template->size_of_col_index_arr = size_of_col_index_arr_after_padding;
    new_template->col_index_arr = malloc_arr(new_template->size_of_col_index_arr, new_template->data_type_of_col_index_arr);

    // 遍历三个层次直到每个非零元，在wrap层次开始重排
    // 遍历三个层次的索引
    for (unsigned long index_of_block_level_index = 0; index_of_block_level_index < block_level_index->block_num; index_of_block_level_index++)
    {
        // cout << "index_of_block_level_index:" << index_of_block_level_index << endl;
        // 当前block的首行行号
        unsigned long block_first_nz_index = read_from_array_with_data_type(block_level_index->coo_begin_index_arr, block_level_index->data_type_of_coo_begin_index_arr, index_of_block_level_index);
        // block中第一个warp号和下一个block的首warp
        unsigned long this_block_first_warp_index = read_from_array_with_data_type(block_level_index->index_arr, block_level_index->index_data_type, index_of_block_level_index);
        unsigned long next_block_first_warp_index = read_from_array_with_data_type(block_level_index->index_arr, block_level_index->index_data_type, index_of_block_level_index + 1);

        // 遍历warp层次
        for (unsigned long index_of_warp_level_index = this_block_first_warp_index; index_of_warp_level_index < next_block_first_warp_index; index_of_warp_level_index++)
        {
            assert(index_of_warp_level_index < warp_level_index->block_num);
            unsigned long warp_first_nz_index = read_from_array_with_data_type(warp_level_index->coo_begin_index_arr, warp_level_index->data_type_of_coo_begin_index_arr, index_of_warp_level_index);

            // 当前wrap的首个非零元的全局索引
            unsigned long global_warp_first_nz_index = block_first_nz_index + warp_first_nz_index;
            // wrap中线程粒度的块的大小
            unsigned long thread_level_block_size_of_this_wrap = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, index_of_warp_level_index);

            unsigned long this_warp_first_thread_index = read_from_array_with_data_type(warp_level_index->index_arr, warp_level_index->index_data_type, index_of_warp_level_index);
            unsigned long next_warp_first_thread_index = read_from_array_with_data_type(warp_level_index->index_arr, warp_level_index->index_data_type, index_of_warp_level_index + 1);

            // wrap中线程粒度的块的数量
            unsigned long thread_level_block_num_of_this_wrap = next_warp_first_thread_index - this_warp_first_thread_index;

            for (unsigned long index_of_thread_level_index = this_warp_first_thread_index; index_of_thread_level_index < next_warp_first_thread_index; index_of_thread_level_index++)
            {
                // thread的wrap内索引
                unsigned long thread_level_block_index_inner_wrap = index_of_thread_level_index - this_warp_first_thread_index;

                // 遍历这个thread中的非零元
                for (unsigned long nz_index_in_thread = 0; nz_index_in_thread < thread_level_block_size_of_this_wrap; nz_index_in_thread++)
                {
                    // 当前非零元在源数组中的位置
                    unsigned long source_index_of_this_nz = global_warp_first_nz_index + thread_level_block_index_inner_wrap * thread_level_block_size_of_this_wrap + nz_index_in_thread;
                    assert(source_index_of_this_nz < size_of_val_arr_after_padding);
                    // 将数据读出来
                    unsigned long col_index = read_from_array_with_data_type(col_index_arr_after_padding, data_type_of_col_index_arr_after_padding, source_index_of_this_nz);
                    double val = read_double_from_array_with_data_type(val_arr_after_padding, data_type_of_val_arr_after_padding, source_index_of_this_nz);

                    // 输出的位置，每个wrap中不同thread中的内容交错起来，
                    unsigned long dest_index_of_this_nz = global_warp_first_nz_index + thread_level_block_index_inner_wrap + thread_level_block_num_of_this_wrap * nz_index_in_thread;
                    assert(dest_index_of_this_nz < size_of_val_arr_after_padding);

                    // 执行输出
                    write_to_array_with_data_type(new_template->col_index_arr, new_template->data_type_of_col_index_arr, dest_index_of_this_nz, col_index);
                    write_double_to_array_with_data_type(new_template->val_arr, new_template->data_type_of_val_arr, dest_index_of_this_nz, val);
                }
            }
        }
    }

    assert(new_template->size_of_col_index_arr == new_template->size_of_val_arr);

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

    assert(new_template->val_arr != NULL);

    // 将数据返回出来
    return new_template;
}

bool is_supported_by_shared_memory_template(sparse_struct_t *matrix, unsigned long dense_block_id)
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

    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[0]->max_row_index == block_level_index->max_row_index);
    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[0]->max_row_index == warp_level_index->max_row_index);
    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[0]->max_row_index == thread_level_index->max_row_index);

    assert(matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index <= block_level_index->max_row_index);

    if (thread_level_index->row_number_of_block_arr != NULL)
    {
        return false;
    }

    // 这个模板不支持compressed block row padding
    if (matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index != compressed_block_view->read_index[0]->max_row_index)
    {
        assert(matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index < compressed_block_view->read_index[0]->max_row_index);
        return false;
    }

    unsigned long total_row_num = block_level_index->max_row_index - block_level_index->min_row_index;

    // 用一个变量，记录是不是有多个TLB处理一行，不存在这种情况会导致模板的检测不通过
    bool is_many_TLB_one_row = false;

    // 前一个TLB对应的全局行号
    bool is_first_TLB = true;
    unsigned long last_TLB_global_row_index = 0;

    // 记录BLB中最大的TLB数量
    unsigned long max_TLB_num_in_BLB = 0;

    // 前一个BLB的首行行号
    bool is_first_BLB = true;
    unsigned long last_BLB_global_row_index = 0;

    // 遍历三个层次的索引，找到BLB中TLB的最大数量、有没有TLB共处理一行的情况、有没有BLB共处理一行的情况、BLB中不同TLB的非零元数量是不是相等
    // 遍历三个层次的索引，计算每一行中间结果的数量，并且计算每一个block线程粒度的块的大小，以及每个块的thead偏移量
    for (unsigned long index_of_block_level_index = 0; index_of_block_level_index < block_level_index->block_num; index_of_block_level_index++)
    {
        // cout << "index_of_block_level_index:" << index_of_block_level_index << endl;
        // 当前block的首行行号
        unsigned long block_first_row_index = read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, index_of_block_level_index);

        if (is_first_BLB == true)
        {
            last_BLB_global_row_index = block_first_row_index;

            is_first_BLB = false;
        }
        else
        {
            if (last_BLB_global_row_index == block_first_row_index)
            {
                // 两个BLB共用一行，检查不能通过
                return false;
            }
            
            last_BLB_global_row_index = block_first_row_index;
        }


        // block中第一个warp号和下一个block的首warp
        unsigned long this_block_first_warp_index = read_from_array_with_data_type(block_level_index->index_arr, block_level_index->index_data_type, index_of_block_level_index);
        unsigned long next_block_first_warp_index = read_from_array_with_data_type(block_level_index->index_arr, block_level_index->index_data_type, index_of_block_level_index + 1);

        // 当前线程块粒度的块的第一个warp粒度的块的线程粒度的块的大小
        assert(this_block_first_warp_index < warp_level_index->block_num && thread_level_index->coo_block_size_arr != NULL);
        unsigned long this_block_first_warp_thread_block_size = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, this_block_first_warp_index);
        assert(this_block_first_warp_thread_block_size != 0);

        // 当前线程块粒度的块第一个线程粒度的块的索引
        assert(this_block_first_warp_index < warp_level_index->length);
        assert(next_block_first_warp_index < warp_level_index->length);
        unsigned long this_block_first_thread_index = read_from_array_with_data_type(warp_level_index->index_arr, warp_level_index->index_data_type, this_block_first_warp_index);
        unsigned long next_block_first_thread_index = read_from_array_with_data_type(warp_level_index->index_arr, warp_level_index->index_data_type, next_block_first_warp_index);
        
        assert(this_block_first_thread_index < thread_level_index->block_num && next_block_first_thread_index <= thread_level_index->block_num);
        assert(next_block_first_thread_index > this_block_first_thread_index);

        unsigned long cur_TLB_num_in_BLB = next_block_first_thread_index - this_block_first_thread_index;
        
        if (max_TLB_num_in_BLB < cur_TLB_num_in_BLB)
        {
            max_TLB_num_in_BLB = cur_TLB_num_in_BLB;
        }

        // 遍历warp层次
        for (unsigned long index_of_warp_level_index = this_block_first_warp_index; index_of_warp_level_index < next_block_first_warp_index; index_of_warp_level_index++)
        {
            assert(index_of_warp_level_index < warp_level_index->block_num);
            unsigned long warp_first_row_index = read_from_array_with_data_type(warp_level_index->index_of_the_first_row_arr, warp_level_index->data_type_of_index_of_the_first_row_arr, index_of_warp_level_index);
            unsigned long this_warp_first_thread_index = read_from_array_with_data_type(warp_level_index->index_arr, warp_level_index->index_data_type, index_of_warp_level_index);
            unsigned long next_warp_first_thread_index = read_from_array_with_data_type(warp_level_index->index_arr, warp_level_index->index_data_type, index_of_warp_level_index + 1);

            // 当前warp的线程粒度的块的大小
            unsigned long this_warp_thread_level_block_size = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, index_of_warp_level_index);

            // 如果WLB和BLB的TLB大小不相等，直接false
            // if (this_block_first_warp_thread_block_size != this_warp_thread_level_block_size)
            // {
            //     return false;
            // }

            for (unsigned long index_of_thread_level_index = this_warp_first_thread_index; index_of_thread_level_index < next_warp_first_thread_index; index_of_thread_level_index++)
            {
                // assert(index_of_thread_level_index < thread_level_index->block_num);
                if (index_of_thread_level_index >= thread_level_index->block_num)
                {
                    assert(false);
                }
                unsigned long thread_first_row_index = read_from_array_with_data_type(thread_level_index->index_of_the_first_row_arr, thread_level_index->data_type_of_index_of_the_first_row_arr, index_of_thread_level_index);

                // 全局的线程粒度的子块所覆盖的行
                unsigned long global_thread_first_row_index = block_first_row_index + warp_first_row_index + thread_first_row_index;

                if (is_first_TLB == true)
                {
                    // 给TLB历史行号赋值
                    last_TLB_global_row_index = global_thread_first_row_index;

                    is_first_TLB = false;
                }
                else
                {
                    // 如果不是第一个TLB，和之前的比较
                    if (last_TLB_global_row_index == global_thread_first_row_index)
                    {
                        is_many_TLB_one_row = true;
                    }

                    last_TLB_global_row_index = global_thread_first_row_index;
                }

                assert(global_thread_first_row_index < total_row_num);
            }
        }
    }

    // 遍历所有线程块粒度的块所包含的行，如果有相交代表要用原子加来规约显存上的结果
    // 用两个变量分别存储，当前的线程块粒度的块包含的最小行号和最大行号
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

    // 如果一个线程处理一行，检查不通过
    if (is_many_TLB_one_row == false)
    {
        return false;
    }

    // 如果BLB中TLB数量大于3950，检查不通过
    if (max_TLB_num_in_BLB > get_config()["SHARED_MEM_TOTAL_SIZE"].as_integer() - 50)
    {
        return false;
    }

    return true;   
}


// 共享内存不溢出，不能一个线程对应一行，一行不能被多个线程块处理。相比warp层次被压缩的模板，不要求所有TLB的大小保持一致，所以需要检查的项会少一点
bool is_supported_by_shared_memory_template(code_builder_t *builder, unsigned long dense_block_id)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;

    return is_supported_by_shared_memory_template(matrix, dense_block_id);
}

void store_template_data(shared_memory_template_t *output_template, string output_dir, bool force_not_share_global_sort_index)
{
    srand(time(0));
    unsigned long matrix_id = rand() + time(0) % 1000;

    // 写这个模板所需要数据的文件夹名称
    output_dir = output_dir + "/" + to_string(matrix_id) + "_" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer());

    // 创建这个文件夹
    system(("mkdir " + output_dir).c_str());

    // 只有不压缩的时候才持久化
    if (output_template->row_offset_in_thread_tmp_result_compress == NONE_COMPRESS)
    {
        assert(output_template->row_offset_in_thread_tmp_result != NULL);
        print_arr_to_file_with_data_type(output_template->row_offset_in_thread_tmp_result, output_template->data_type_of_row_offset_in_thread_tmp_result, output_template->size_of_row_offset_in_thread_tmp_result, output_dir + "/row_offset_in_thread_tmp_result");
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

    if (output_template->warp_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->warp_begin_thread_index_offset != NULL);
        print_arr_to_file_with_data_type(output_template->warp_begin_thread_index_offset, output_template->data_type_of_warp_begin_thread_index_offset, output_template->size_of_warp_begin_thread_index_offset, output_dir + "/warp_begin_thread_index_offset");
    }

    if (output_template->thread_block_size_in_warp_compress == NONE_COMPRESS)
    {
        assert(output_template->thread_block_size_in_warp != NULL);
        print_arr_to_file_with_data_type(output_template->thread_block_size_in_warp, output_template->data_type_of_thread_block_size_in_warp, output_template->size_of_thread_block_size_in_warp, output_dir + "/thread_block_size_in_warp");
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

    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset != NULL);
        print_arr_to_file_with_data_type(output_template->block_nz_begin_offset, output_template->data_type_of_block_nz_begin_offset, output_template->size_of_block_nz_begin_offset, output_dir + "/block_nz_begin_offset");
    }

    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->warp_nz_begin_offset != NULL);
        print_arr_to_file_with_data_type(output_template->warp_nz_begin_offset, output_template->data_type_of_warp_nz_begin_offset, output_template->size_of_warp_nz_begin_offset, output_dir + "/warp_nz_begin_offset");
    }

    // 值
    assert(output_template->val_arr != NULL);
    // cout << "output_template->val_arr:" << output_template->val_arr << endl;
    print_arr_to_file_with_data_type(output_template->val_arr, output_template->data_type_of_val_arr, output_template->size_of_val_arr, output_dir + "/val_arr");

    // 列
    assert(output_template->col_index_arr != NULL);
    print_arr_to_file_with_data_type(output_template->col_index_arr, output_template->data_type_of_col_index_arr, output_template->size_of_col_index_arr, output_dir + "/col_index_arr");

    output_template->hash_of_this_template = matrix_id;
}

string code_of_template_data_struct(shared_memory_template_t *output_template, unsigned long dense_block_id)
{
    // 创建一个数据结构
    string return_str = "typedef struct compressed_dense_block_" + to_string(dense_block_id) + "\n{\n";

    // 对应的位置分别存储行号和块号
    if (output_template->row_offset_in_thread_tmp_result_compress == NONE_COMPRESS)
    {
        assert(output_template->row_offset_in_thread_tmp_result != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_row_offset_in_thread_tmp_result, code_of_arr_var_name(dense_block_id, -1, "row_offset_in_thread_tmp_result"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "row_offset_in_thread_tmp_result") + " = " + to_string(output_template->size_of_row_offset_in_thread_tmp_result) + ";\n";
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

    if (output_template->warp_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->warp_begin_thread_index_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_warp_begin_thread_index_offset, code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset") + " = " + to_string(output_template->size_of_warp_begin_thread_index_offset) + ";\n";
    }

    return_str = return_str + "\n";

    if (output_template->thread_block_size_in_warp_compress == NONE_COMPRESS)
    {
        assert(output_template->thread_block_size_in_warp != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_thread_block_size_in_warp, code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp") + " = " + to_string(output_template->size_of_thread_block_size_in_warp) + ";\n";
    }

    return_str = return_str + "\n";

    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->row_index_before_sort != NULL)
    {
        assert(output_template->row_index_before_sort != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_row_index_before_sort, code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort") + " = " + to_string(output_template->size_of_row_index_before_sort) + ";\n";
    }

    return_str = return_str + "\n";

    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_block_nz_begin_offset, code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset") + " = " + to_string(output_template->size_of_block_nz_begin_offset) + ";\n";
    }

    return_str = return_str + "\n";

    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->warp_nz_begin_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_warp_nz_begin_offset, code_of_arr_var_name(dense_block_id, -1, "warp_nz_begin_offset"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "warp_nz_begin_offset") + " = " + to_string(output_template->size_of_warp_nz_begin_offset) + ";\n";
    }

    return_str = return_str + "\n";
    // cout << "output_template->val_arr:" << output_template->val_arr << endl;
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

string code_of_read_template_data_from_file_func_define(shared_memory_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index)
{
    string return_str = "compressed_dense_block_" + to_string(dense_block_id) + "_t* read_dense_block_" + to_string(dense_block_id) + "_from_file(string file_name_prefix)\n{\n";

    return_str = return_str + "compressed_dense_block_" + to_string(dense_block_id) + "_t *template_data = new " + "compressed_dense_block_" + to_string(dense_block_id) + "_t();\n";

    // 对应的位置分别存储行号和块号
    if (output_template->row_offset_in_thread_tmp_result_compress == NONE_COMPRESS)
    {
        assert(output_template->row_offset_in_thread_tmp_result != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "row_offset_in_thread_tmp_result") + " = (" + code_of_data_type(output_template->data_type_of_row_offset_in_thread_tmp_result) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "row_offset_in_thread_tmp_result") + ", " + convert_data_type_to_string(output_template->data_type_of_row_offset_in_thread_tmp_result) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/row_offset_in_thread_tmp_result\");\n";
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

    if (output_template->warp_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->warp_begin_thread_index_offset != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset") + " = (" + code_of_data_type(output_template->data_type_of_warp_begin_thread_index_offset) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset") + ", " + convert_data_type_to_string(output_template->data_type_of_warp_begin_thread_index_offset) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/warp_begin_thread_index_offset\");\n";
    }

    return_str = return_str + "\n";

    if (output_template->thread_block_size_in_warp_compress == NONE_COMPRESS)
    {
        assert(output_template->thread_block_size_in_warp != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp") + " = (" + code_of_data_type(output_template->data_type_of_thread_block_size_in_warp) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp") + ", " + convert_data_type_to_string(output_template->data_type_of_thread_block_size_in_warp) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/thread_block_size_in_warp\");\n";
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
    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset") + " = (" + code_of_data_type(output_template->data_type_of_block_nz_begin_offset) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset") + ", " + convert_data_type_to_string(output_template->data_type_of_block_nz_begin_offset) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/block_nz_begin_offset\");\n";
    }

    return_str = return_str + "\n";

    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->warp_nz_begin_offset != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "warp_nz_begin_offset") + " = (" + code_of_data_type(output_template->data_type_of_warp_nz_begin_offset) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "warp_nz_begin_offset") + ", " + convert_data_type_to_string(output_template->data_type_of_warp_nz_begin_offset) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/warp_nz_begin_offset\");\n";
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

string code_of_template_kernal(shared_memory_template_t *output_template, unsigned long dense_block_id)
{
    // 内核函数的声明
    string return_str = "__global__ void spmv_" + to_string(dense_block_id) + "(";

    // 用一个变量表明当前形参是不是第一个，如果是第一个就不用点逗号
    bool is_first_param = true;

    // 这里加入形参的声明
    if (output_template->row_offset_in_thread_tmp_result_compress == NONE_COMPRESS)
    {
        assert(output_template->row_offset_in_thread_tmp_result != NULL);
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_row_offset_in_thread_tmp_result, "* row_offset_in_thread_tmp_result");
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

    if (output_template->warp_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }

        assert(output_template->warp_begin_thread_index_offset != NULL);
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_warp_begin_thread_index_offset, "* warp_begin_thread_index_offset");
    }

    if (output_template->thread_block_size_in_warp_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }

        assert(output_template->thread_block_size_in_warp != NULL);
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_thread_block_size_in_warp, "* thread_block_size_in_warp");
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

    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }

        assert(output_template->block_nz_begin_offset != NULL);
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_block_nz_begin_offset, "* block_nz_begin_offset");
    }

    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }

        assert(output_template->warp_nz_begin_offset != NULL);
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_warp_nz_begin_offset, "* warp_nz_begin_offset");
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

    return_str = return_str + "int tid_in_warp = threadIdx.x % 32;\nint bid = blockIdx.x;\nint tid_in_block = threadIdx.x;\nint wid_in_block = (int)(threadIdx.x / 32);\n\n";

    if (!(output_template->tblock_num == output_template->size_of_block_nz_begin_offset))
    {
        return_str = return_str + "int bnum = gridDim.x;\n";
    }

    if (!(output_template->block_begin_warp_index_offset_compress == LINEAR_COMPRESS && ((linear_compress_t *)(output_template->block_begin_warp_index_offset_compress_meta))->coefficient == (output_template->thread_num_in_block / 32)))
    {
        return_str = return_str + "int wnum = blockDim.x / 32;\n";
    }

    // 首行和首列号，如果不是0就保留
    if (output_template->kernal_first_row_index != 0)
    {
        return_str = return_str + "unsigned long kernal_first_row_index = " + to_string(output_template->kernal_first_row_index) + ";\n";
    }

    if (output_template->kernal_first_col_index != 0)
    {
        return_str = return_str + "unsigned long kernal_first_col_index = " + to_string(output_template->kernal_first_col_index) + ";\n";
    }

    // 记录共享内存的使用数量
    unsigned long shared_memory_item_num = 0;

    // 记录block级别和warp级别是否有共享内存
    bool has_block_level_shared_mem = false;
    bool has_warp_level_index = false;

    // 声明元数据共享内存
    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        has_block_level_shared_mem = true;
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_nz_begin_offset) + " block_first_nz_index_shared[1];\n";
        shared_memory_item_num = shared_memory_item_num + 1;
    }

    if (output_template->block_first_row_index_compress == NONE_COMPRESS)
    {
        has_block_level_shared_mem = true;
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_first_row_index) + " this_block_first_row_index_shared[1];\n";
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_first_row_index) + " next_block_first_row_index_shared[1];\n";
        shared_memory_item_num = shared_memory_item_num + 2;
    }

    if (output_template->block_begin_warp_index_offset_compress == NONE_COMPRESS)
    {
        has_block_level_shared_mem = true;
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " this_block_begin_warp_index_offset_shared[1];\n";
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " next_block_begin_warp_index_offset_shared[1];\n";
        shared_memory_item_num = shared_memory_item_num + 2;
    }

    unsigned long max_warp_num_in_block = 0;

    // warp级别的数据，首先获取block中warp的最大数量
    if (output_template->warp_begin_thread_index_offset_compress == NONE_COMPRESS || output_template->warp_nz_begin_offset_compress == NONE_COMPRESS || output_template->thread_block_size_in_warp_compress == NONE_COMPRESS)
    {
        // 遍历block_begin_warp_index_offset
        for (unsigned long block_index = 0; block_index < output_template->size_of_block_begin_warp_index_offset - 1; block_index++)
        {
            unsigned long warp_num_in_block = read_from_array_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, block_index + 1) - read_from_array_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, block_index);
            if (max_warp_num_in_block < warp_num_in_block)
            {
                max_warp_num_in_block = warp_num_in_block;
            }
        }
    }

    // 声明对应元数据的共享内存
    if (output_template->warp_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        assert(max_warp_num_in_block > 0);

        shared_memory_item_num = shared_memory_item_num + max_warp_num_in_block + 1;

        // 查看共享内存是不是够用，不够用说明之前的分块手段不好
        if (shared_memory_item_num + 1 > get_config()["SHARED_MEM_TOTAL_SIZE"].as_integer())
        {
            cout << "shared memory overflow, error: code_of_template_kernal" << endl;
            // assert(false);
        }

        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_warp_begin_thread_index_offset) + " warp_begin_thread_index_offset_shared[" + to_string(max_warp_num_in_block + 1) + "];\n";
    }

    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(max_warp_num_in_block > 0);

        shared_memory_item_num = shared_memory_item_num + max_warp_num_in_block;

        // 查看共享内存是不是够用，不够用说明之前的分块手段不好
        if (shared_memory_item_num + 1 > get_config()["SHARED_MEM_TOTAL_SIZE"].as_integer())
        {
            cout << "shared memory overflow, error: code_of_template_kernal" << endl;
            // assert(false);
        }

        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_warp_nz_begin_offset) + " warp_nz_begin_offset_shared[" + to_string(max_warp_num_in_block) + "];\n";
    }

    if (output_template->thread_num_of_row_reduce != get_config()["HALF_MAX_ROW_REDUCE_THREAD"].as_integer() && output_template->thread_num_of_row_reduce != get_config()["MAX_ROW_REDUCE_THREAD"].as_integer())
    {
        // 如果首个非零元索引没有被压缩，那就用一个线程读出之后广播
        if (output_template->row_offset_in_thread_tmp_result_compress == NONE_COMPRESS)
        {
            return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_row_offset_in_thread_tmp_result) + " row_offset_in_thread_tmp_result_shared[1];\n";
            shared_memory_item_num = shared_memory_item_num + 1;
        }
    }

    if (shared_memory_item_num >= get_config()["SHARED_MEM_TOTAL_SIZE"].as_integer())
    {
        cout << "shared memory overflow, error: code_of_template_kernal" << endl;
        // assert(false);
    }

    if (output_template->thread_block_size_in_warp_compress == NONE_COMPRESS)
    {
        assert(max_warp_num_in_block > 0);

        shared_memory_item_num = shared_memory_item_num + max_warp_num_in_block;

        // 查看共享内存是不是够用，不够用说明之前的分块手段不好
        if (shared_memory_item_num + 1 > get_config()["SHARED_MEM_TOTAL_SIZE"].as_integer())
        {
            cout << "shared memory overflow, error: code_of_template_kernal" << endl;
            // assert(false);
        }

        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_thread_block_size_in_warp) + " thread_block_size_in_warp_shared[" + to_string(max_warp_num_in_block) + "];\n";
    }

    // 存储block内部线程粒度的块的中间结果，找到block中最大的线程粒度的块的数量
    unsigned long max_thread_level_block_num = 0;

    // 打印数组
    // print_arr_to_file_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, output_template->size_of_block_begin_warp_index_offset, "/home/duzhen/spmv_builder/data_source/test_result_0");

    // exit(-1);

    // output_template->size_of_block_nz_begin_offset本质上就是block块的数量
    for (unsigned long block_id = 0; block_id < output_template->size_of_block_nz_begin_offset; block_id++)
    {
        // 获取当前block和下一个block的第一个warp号
        assert(block_id + 1 < output_template->size_of_block_begin_warp_index_offset);
        unsigned long this_block_first_warp_index = read_from_array_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, block_id);
        unsigned long next_block_first_warp_index = read_from_array_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, block_id + 1);

        // 获取两个warp的第一个线程的线程号
        assert(next_block_first_warp_index >= this_block_first_warp_index);

        if (next_block_first_warp_index >= output_template->size_of_warp_begin_thread_index_offset)
        {
            cout << "next_block_first_warp_index:" << next_block_first_warp_index << endl;
            cout << "size_of_warp_begin_thread_index_offset:" << output_template->size_of_warp_begin_thread_index_offset << endl;
            cout << "block_id:" << block_id << endl;
            cout << "this_block_first_warp_index:" << this_block_first_warp_index << endl;
        }

        assert(next_block_first_warp_index < output_template->size_of_warp_begin_thread_index_offset);
        unsigned long this_block_first_thread_index = read_from_array_with_data_type(output_template->warp_begin_thread_index_offset, output_template->data_type_of_warp_begin_thread_index_offset, this_block_first_warp_index);
        unsigned long next_block_first_thread_index = read_from_array_with_data_type(output_template->warp_begin_thread_index_offset, output_template->data_type_of_warp_begin_thread_index_offset, next_block_first_warp_index);
        // 两个相减获得一个当前的线程块最多的线程粒度的块的数量
        unsigned long thread_block_num_in_cur_block = next_block_first_thread_index - this_block_first_thread_index;

        // 对比获得最大块中线程数量
        if (max_thread_level_block_num < thread_block_num_in_cur_block)
        {
            max_thread_level_block_num = thread_block_num_in_cur_block;
        }
    }

    assert(max_thread_level_block_num > 0);

    shared_memory_item_num = shared_memory_item_num + max_thread_level_block_num;

    // 声明中间结果的存储数组
    return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_val_arr) + " thread_tmp_result_inner_block[" + to_string(max_thread_level_block_num) + "];\n\n";

    if (shared_memory_item_num > get_config()["SHARED_MEM_TOTAL_SIZE"].as_integer())
    {
        cout << "shared memory overflow, error: code_of_template_kernal" << endl;
        // assert(false);
    }

    // block层次的遍历，遍历索引的数据类型，根据被分配的block数量和block粒度的块的关系，选择不同的表达
    if (output_template->tblock_num == output_template->size_of_block_nz_begin_offset)
    {
        return_str = return_str + "{\n";
        return_str = return_str + "unsigned int" + " block_level_block_id = bid;\n";
    }
    else
    {
        return_str = return_str + "for(" + "unsigned int" + " block_level_block_id = bid; block_level_block_id < " + to_string(output_template->size_of_block_nz_begin_offset) + "; block_level_block_id = block_level_block_id + bnum)\n{\n";
    }

    // 查看是不是需要block层次的共享内存初始化
    bool need_block_level_shared_init = false;
    bool need_warp_level_shared_init = false;

    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS || output_template->block_begin_warp_index_offset_compress == NONE_COMPRESS || output_template->block_first_row_index_compress == NONE_COMPRESS)
    {
        need_block_level_shared_init = true;
    }

    if (output_template->warp_begin_thread_index_offset_compress == NONE_COMPRESS || output_template->warp_nz_begin_offset_compress == NONE_COMPRESS || output_template->thread_block_size_in_warp_compress == NONE_COMPRESS)
    {
        need_warp_level_shared_init = true;
    }

    bool need_next_block_begin_warp_index_offset;
    // 如果没有warp层次的循环，并且warp没有shared memory需要共享的变量，就不需要block的warp索引上界
    if (output_template->block_begin_warp_index_offset_compress == LINEAR_COMPRESS && ((linear_compress_t *)(output_template->block_begin_warp_index_offset_compress_meta))->coefficient == (output_template->thread_num_in_block / 32) && need_warp_level_shared_init == false)
    {
        // 这里代表没有warp层次的遍历和共享内存初始化
        need_next_block_begin_warp_index_offset = false;
    }
    else
    {
        need_next_block_begin_warp_index_offset = true;
    }

    // 如果没有thread层次的循环，并且（每个线程粒度的块要处理的非零元数量是1 或者 不是1但是线程粒度的索引是线性的、直接用斜率算出来每个warp粒度的块的线程粒度的块的数量）
    bool need_first_thread_index_of_next_warp = true;

    if (output_template->warp_begin_thread_index_offset_compress == LINEAR_COMPRESS && ((linear_compress_t *)output_template->warp_begin_thread_index_offset_compress_meta)->coefficient == 32)
    {
        // 到这里代表不需要thread层次的遍历
        // 判断是否需要线程内部的非零元遍历
        // if (output_template->thread_block_size_in_warp_compress == CONSTANT_COMPRESS && ((constant_compress_t *)output_template->thread_block_size_in_warp_compress_meta)->constant == 1)
        // {
        //     need_first_thread_index_of_next_warp = false;
        // }

        // else
        // {
        //     // 需要线程内部非零元的遍历，但是每个warp中thread的数量可以用常量得出所以也不需要next
        // }
        need_first_thread_index_of_next_warp = false;
    }

    // 查看是否需要next_row的index
    bool need_next_block_first_row_index = true;

    if (output_template->block_first_row_index_compress == LINEAR_COMPRESS && ((linear_compress_t *)output_template->block_first_row_index_compress_meta)->coefficient == output_template->thread_num_in_block)
    {
        need_next_block_first_row_index = false;
    }

    return_str = return_str + code_of_data_type(output_template->data_type_of_block_nz_begin_offset) + " this_block_first_nz_index;\n";
    return_str = return_str + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " this_block_begin_warp_index_offset;\n";

    if (need_next_block_begin_warp_index_offset == true)
    {
        return_str = return_str + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " next_block_begin_warp_index_offset;\n";
    }

    return_str = return_str + code_of_data_type(output_template->data_type_of_block_first_row_index) + " this_block_first_row_index;\n";

    if (need_next_block_first_row_index == true)
    {
        return_str = return_str + code_of_data_type(output_template->data_type_of_block_first_row_index) + " next_block_first_row_index;\n";
    }

    if (need_warp_level_shared_init == true)
    {
        return_str = return_str + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " warp_block_num_in_this_block;\n\n";
    }

    if (output_template->thread_num_of_row_reduce != get_config()["HALF_MAX_ROW_REDUCE_THREAD"].as_integer() && output_template->thread_num_of_row_reduce != get_config()["MAX_ROW_REDUCE_THREAD"].as_integer())
    {
        // 声明首个非零元索引
        return_str = return_str + code_of_data_type(output_template->data_type_of_row_offset_in_thread_tmp_result) + " global_first_tmp_result;\n";
    }

    // 只要有两个的其中一个，就需要加一个同步
    if (need_block_level_shared_init == true || need_warp_level_shared_init == true)
    {
        return_str = return_str + "\n__syncthreads();\n\n";
    }

    // 初始化block级别的数据
    if (need_block_level_shared_init == true)
    {
        return_str = return_str + "if (tid_in_block == 0)\n{\n";

        // 初始化共享内存
        if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
        {
            assert(output_template->block_nz_begin_offset != NULL);
            return_str = return_str + "block_first_nz_index_shared[0] = block_nz_begin_offset[block_level_block_id];\n";
        }

        if (output_template->block_begin_warp_index_offset_compress == NONE_COMPRESS)
        {
            assert(output_template->block_begin_warp_index_offset != NULL);
            return_str = return_str + "this_block_begin_warp_index_offset_shared[0] = block_begin_warp_index_offset[block_level_block_id];\n";
            if (need_next_block_begin_warp_index_offset == true)
            {
                return_str = return_str + "next_block_begin_warp_index_offset_shared[0] = block_begin_warp_index_offset[block_level_block_id + 1];\n";
            }
        }

        if (output_template->block_first_row_index_compress == NONE_COMPRESS)
        {
            assert(output_template->block_first_row_index != NULL);
            return_str = return_str + "this_block_first_row_index_shared[0] = block_first_row_index[block_level_block_id];\n";
            return_str = return_str + "next_block_first_row_index_shared[0] = block_first_row_index[block_level_block_id + 1];\n";
        }

        return_str = return_str + "}\n";
        return_str = return_str + "__syncthreads();\n";
    }

    // 计算block所有的元数据
    // 起始非零元索引，可能是线性压缩
    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset != NULL);
        return_str = return_str + "this_block_first_nz_index = block_first_nz_index_shared[0];\n";
    }
    else if (output_template->block_nz_begin_offset_compress == LINEAR_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset_compress_meta != NULL);
        linear_compress_t *compressor = (linear_compress_t *)output_template->block_nz_begin_offset_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "this_block_first_nz_index", "block_level_block_id") + ";\n";
    }
    else
    {
        cout << "this compress type is not support" << endl;
        assert(false);
    }

    // warp起始索引，可以线性压缩
    if (output_template->block_begin_warp_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_warp_index_offset != NULL);
        return_str = return_str + "this_block_begin_warp_index_offset = this_block_begin_warp_index_offset_shared[0];\n";

        if (need_next_block_begin_warp_index_offset == true)
        {
            return_str = return_str + "next_block_begin_warp_index_offset = next_block_begin_warp_index_offset_shared[0];\n";
        }
    }
    else if (output_template->block_begin_warp_index_offset_compress == LINEAR_COMPRESS)
    {
        assert(output_template->block_begin_warp_index_offset_compress_meta != NULL);
        linear_compress_t *compressor = (linear_compress_t *)output_template->block_begin_warp_index_offset_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "this_block_begin_warp_index_offset", "block_level_block_id") + ";\n";

        if (need_next_block_begin_warp_index_offset == true)
        {
            // 用加法加速下一个block的warp偏移，减少计算量
            return_str = return_str + "next_block_begin_warp_index_offset = this_block_begin_warp_index_offset + " + to_string(compressor->coefficient) + ";\n";
        }
    }
    else
    {
        cout << "this compress type is not support" << endl;
        assert(false);
    }

    // block的首行索引，线性压缩
    if (output_template->block_first_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->block_first_row_index != NULL);
        return_str = return_str + "this_block_first_row_index = this_block_first_row_index_shared[0];\n";

        if (need_next_block_first_row_index == true)
        {
            return_str = return_str + "next_block_first_row_index = next_block_first_row_index_shared[0];\n";
        }
    }
    else if (output_template->block_first_row_index_compress == LINEAR_COMPRESS)
    {
        assert(output_template->block_first_row_index_compress_meta != NULL);
        linear_compress_t *compressor = (linear_compress_t *)output_template->block_first_row_index_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "this_block_first_row_index", "block_level_block_id") + ";\n";

        if (need_next_block_first_row_index == true)
        {
            // 用加法，减少计算量
            return_str = return_str + "next_block_first_row_index = this_block_first_row_index + " + to_string(compressor->coefficient) + ";\n";
        }
    }

    return_str = return_str + "\n";

    // 初始化warp级别的数据
    if (need_warp_level_shared_init == true)
    {
        // warp_block_num_in_this_block主要是用来初始化warp层级的共享内存中的数据
        // block中warp粒度的块的数量，如果被线性压缩，可以直接给一个常值
        if (output_template->block_begin_warp_index_offset_compress == LINEAR_COMPRESS)
        {
            // 用斜率直接赋值
            assert(output_template->block_begin_warp_index_offset_compress_meta != NULL);
            linear_compress_t *compressor = (linear_compress_t *)output_template->block_begin_warp_index_offset_compress_meta;
            return_str = return_str + "warp_block_num_in_this_block = " + to_string(compressor->coefficient) + ";\n";
        }
        else
        {
            return_str = return_str + "warp_block_num_in_this_block = next_block_begin_warp_index_offset - this_block_begin_warp_index_offset;\n";
        }

        return_str = return_str + "for(int i = tid_in_block; i < warp_block_num_in_this_block; i = i + blockDim.x)\n{\n";

        // warp的相对起始非零元
        if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS)
        {
            assert(output_template->warp_nz_begin_offset != NULL);
            return_str = return_str + "warp_nz_begin_offset_shared[i] = warp_nz_begin_offset[this_block_begin_warp_index_offset + i];\n";
        }

        if (output_template->warp_begin_thread_index_offset_compress == NONE_COMPRESS)
        {
            assert(output_template->warp_begin_thread_index_offset != NULL);
            return_str = return_str + "warp_begin_thread_index_offset_shared[i] = warp_begin_thread_index_offset[this_block_begin_warp_index_offset + i];\n";
        }

        if (output_template->thread_block_size_in_warp_compress == NONE_COMPRESS)
        {
            assert(output_template->thread_block_size_in_warp != NULL);
            return_str = return_str + "thread_block_size_in_warp_shared[i] = thread_block_size_in_warp[this_block_begin_warp_index_offset + i];\n";
        }

        return_str = return_str + "}\n\n";

        // thread的偏移额外有一位
        if (output_template->warp_begin_thread_index_offset_compress == NONE_COMPRESS)
        {
            return_str = return_str + "if(tid_in_block == 0)\n{\n";
            return_str = return_str + "warp_begin_thread_index_offset_shared[warp_block_num_in_this_block] = warp_begin_thread_index_offset[this_block_begin_warp_index_offset + warp_block_num_in_this_block];\n";
            return_str = return_str + "}\n";
        }

        // 同步
        return_str = return_str + "__syncthreads();\n\n";
    }

    return_str = return_str + "\n";

    // 计算block的第一个线程粒度的块的索引，可以线性压缩
    if (output_template->warp_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        return_str = return_str + code_of_data_type(output_template->data_type_of_warp_begin_thread_index_offset) + " first_thread_index_of_this_block = warp_begin_thread_index_offset_shared[0];\n";
    }
    else if (output_template->warp_begin_thread_index_offset_compress == LINEAR_COMPRESS)
    {
        assert(output_template->warp_begin_thread_index_offset_compress_meta != NULL);
        linear_compress_t *compressor = (linear_compress_t *)output_template->warp_begin_thread_index_offset_compress_meta;
        // 用全局索引来算对应数据
        return_str = return_str + code_of_data_type(output_template->data_type_of_warp_begin_thread_index_offset) + " " + code_of_arr_read(compressor, "first_thread_index_of_this_block", "this_block_begin_warp_index_offset") + ";\n";
    }
    else
    {
        cout << "compress type is not supported" << endl;
        assert(false);
    }

    return_str = return_str + "\n";

    if (output_template->thread_num_of_row_reduce != get_config()["HALF_MAX_ROW_REDUCE_THREAD"].as_integer() && output_template->thread_num_of_row_reduce != get_config()["MAX_ROW_REDUCE_THREAD"].as_integer())
    {
        // 如果首个非零元索引没有被压缩，那就用一个线程读出之后广播
        if (output_template->row_offset_in_thread_tmp_result_compress == NONE_COMPRESS)
        {
            return_str = return_str + "if (tid_in_block == 0)\n{\n";
            // 从全局内存中将数据读入共享内存
            return_str = return_str + "row_offset_in_thread_tmp_result_shared[0] = row_offset_in_thread_tmp_result[this_block_first_row_index];\n";
            return_str = return_str + "}\n";
            return_str = return_str + "__syncthreads();\n";
        }

        // 将块的首个非零元进行一个赋值
        // 当前块第一个中间结果的位置
        if (output_template->row_offset_in_thread_tmp_result_compress == NONE_COMPRESS)
        {
            assert(output_template->row_offset_in_thread_tmp_result != NULL);
            return_str = return_str + "global_first_tmp_result = row_offset_in_thread_tmp_result_shared[0];\n";
        }
        else if (output_template->row_offset_in_thread_tmp_result_compress == LINEAR_COMPRESS)
        {
            assert(output_template->row_offset_in_thread_tmp_result_compress_meta != NULL);
            linear_compress_t *compressor = (linear_compress_t *)output_template->row_offset_in_thread_tmp_result_compress_meta;
            return_str = return_str + code_of_arr_read(compressor, "global_first_tmp_result", "this_block_first_row_index") + ";\n";
        }
        else
        {
            cout << "compress type is not supported" << endl;
            assert(false);
        }

        return_str = return_str + "\n";
    }

    string var_name_local_warp_level_block_id = "";

    // 遍历warp层次的所有块，如果每个block中warp块的数量一样，并且相等（用了线性压缩斜率和block中被分配的warp数量一致），就触发不同的遍历方式
    assert(output_template->thread_num_in_block % 32 == 0);
    if (output_template->block_begin_warp_index_offset_compress == LINEAR_COMPRESS && ((linear_compress_t *)(output_template->block_begin_warp_index_offset_compress_meta))->coefficient == (output_template->thread_num_in_block / 32))
    {
        // 不用for循环
        return_str = return_str + "{\n";
        // warp_level_block_id = this_block_begin_warp_index_offset + wid_in_block
        // 如果warp级别的索引全是不能压缩的，那就不需要计算全局的warp索引
        if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS && (output_template->thread_block_size_in_warp_compress == NONE_COMPRESS || output_template->thread_block_size_in_warp_compress == CONSTANT_COMPRESS) && output_template->warp_begin_thread_index_offset_compress == NONE_COMPRESS)
        {
            // 全都没有压缩，不需要全局索引
        }
        else
        {
            return_str = return_str + "unsigned int" + " warp_level_block_id = this_block_begin_warp_index_offset + wid_in_block;\n";
        }

        // 用wid_in_block代替所有warp粒度的块在block粒度的块的块内索引
        var_name_local_warp_level_block_id = "wid_in_block";
    }
    else
    {
        // for循环
        return_str = return_str + "for(";
        return_str = return_str + "unsigned int" + " warp_level_block_id = this_block_begin_warp_index_offset + wid_in_block; warp_level_block_id < next_block_begin_warp_index_offset; warp_level_block_id = warp_level_block_id + wnum)\n{\n";
        if (need_warp_level_shared_init == true)
        {
            return_str = return_str + "unsigned int" + " local_warp_level_block_id = warp_level_block_id - this_block_begin_warp_index_offset;\n";
        }
        var_name_local_warp_level_block_id = "local_warp_level_block_id";
    }

    return_str = return_str + "\n";

    // 获取三组数据，第一个非零元索引，第二个线程粒度的块的大小，第三个是thread块的索引
    return_str = return_str + code_of_data_type(output_template->data_type_of_warp_nz_begin_offset) + " local_this_warp_first_nz_index" + ";\n";

    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        return_str = return_str + "local_this_warp_first_nz_index = warp_nz_begin_offset_shared[" + var_name_local_warp_level_block_id + "];\n";
    }
    else if (output_template->warp_nz_begin_offset_compress == LINEAR_COMPRESS)
    {
        // 可以线性压缩
        assert(output_template->warp_nz_begin_offset_compress_meta != NULL);
        linear_compress_t *compressor = (linear_compress_t *)output_template->warp_nz_begin_offset_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "local_this_warp_first_nz_index", "warp_level_block_id") + ";\n";
        ;
    }
    else if (output_template->warp_nz_begin_offset_compress == CYCLE_LINEAR_COMPRESS)
    {
        assert(output_template->warp_nz_begin_offset_compress_meta != NULL);
        cycle_linear_compress_t *compressor = (cycle_linear_compress_t *)output_template->warp_nz_begin_offset_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "local_this_warp_first_nz_index", "warp_level_block_id") + ";\n";
        ;
    }
    else
    {
        cout << "compress type is not supported" << endl;
        assert(false);
    }

    return_str = return_str + "\n";

    return_str = return_str + code_of_data_type(output_template->data_type_of_thread_block_size_in_warp) + " thread_block_size_in_this_warp;\n";

    // 每个warp内的线程粒度的块的大小，可以使用常值压缩和分支压缩
    if (output_template->thread_block_size_in_warp_compress == NONE_COMPRESS)
    {
        return_str = return_str + "thread_block_size_in_this_warp = thread_block_size_in_warp_shared[" + var_name_local_warp_level_block_id + "];\n";
    }
    else if (output_template->thread_block_size_in_warp_compress == CONSTANT_COMPRESS)
    {
        assert(output_template->thread_block_size_in_warp_compress_meta != NULL);
        constant_compress_t *compressor = (constant_compress_t *)output_template->thread_block_size_in_warp_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "thread_block_size_in_this_warp", "warp_level_block_id") + ";\n";
    }
    else if (output_template->thread_block_size_in_warp_compress == BRANCH_COMPRESS)
    {
        assert(output_template->thread_block_size_in_warp_compress_meta != NULL);
        branch_compress_t *compressor = (branch_compress_t *)output_template->thread_block_size_in_warp_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "thread_block_size_in_this_warp", "warp_level_block_id") + ";\n";
    }
    else
    {
        cout << "compress type is not supported" << endl;
        assert(false);
    }

    return_str = return_str + "\n";

    return_str = return_str + code_of_data_type(output_template->data_type_of_warp_begin_thread_index_offset) + " first_thread_index_of_this_warp;\n";

    if (need_first_thread_index_of_next_warp == true)
    {
        return_str = return_str + code_of_data_type(output_template->data_type_of_warp_begin_thread_index_offset) + " first_thread_index_of_next_warp;\n";
    }

    return_str = return_str + "\n";

    // 初始化两个数据
    if (output_template->warp_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        return_str = return_str + "first_thread_index_of_this_warp =  warp_begin_thread_index_offset_shared[" + var_name_local_warp_level_block_id + "];\n";

        if (need_first_thread_index_of_next_warp == true)
        {
            return_str = return_str + "first_thread_index_of_next_warp =  warp_begin_thread_index_offset_shared[" + var_name_local_warp_level_block_id + " + 1];\n";
        }
    }
    else if (output_template->warp_begin_thread_index_offset_compress == LINEAR_COMPRESS)
    {
        assert(output_template->warp_begin_thread_index_offset_compress_meta != NULL);
        linear_compress_t *compressor = (linear_compress_t *)output_template->warp_begin_thread_index_offset_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "first_thread_index_of_this_warp", "warp_level_block_id") + ";\n";

        // 这里使用加法来减少计算量
        if (need_first_thread_index_of_next_warp == true)
        {
            return_str = return_str + "first_thread_index_of_next_warp = first_thread_index_of_this_warp + " + to_string(compressor->coefficient) + ";\n";
        }
    }
    else
    {
        cout << "compress type is not supported" << endl;
        assert(false);
    }

    // 如果每个线程要处理的数据量只有一个非零元，就不需要记录每个warp粒度的块包含的线程粒度块的数量
    if (output_template->thread_block_size_in_warp_compress == CONSTANT_COMPRESS && ((constant_compress_t *)(output_template->thread_block_size_in_warp_compress_meta))->constant == 1)
    {
        // 不需要线程粒度的块内遍历，也不需要一个warp中线程粒度的块的数量
    }
    else
    {
        // 如果有thread首地址的线性压缩，这里直接用斜率直接求的，少一次减法
        if (output_template->warp_begin_thread_index_offset_compress == LINEAR_COMPRESS)
        {
            // 首地址线性压缩
            assert(output_template->warp_begin_thread_index_offset_compress_meta != NULL);
            linear_compress_t *compressor = (linear_compress_t *)output_template->warp_begin_thread_index_offset_compress_meta;
            return_str = return_str + code_of_data_type(output_template->data_type_of_warp_begin_thread_index_offset) + " thread_level_block_num_in_warp = " + to_string(compressor->coefficient) + ";\n";
        }
        else
        {
            // 两个首索引相减，获得warp块中的thread块数量
            assert(need_first_thread_index_of_next_warp == true);
            return_str = return_str + code_of_data_type(output_template->data_type_of_warp_begin_thread_index_offset) + " thread_level_block_num_in_warp = first_thread_index_of_next_warp - first_thread_index_of_this_warp;;\n";
        }
    }

    // thread粒度的块的warp内索引的变量名
    string var_name_of_thread_level_block_index_inner_warp;

    // 线程层次的遍历，在特定时候取消遍历，当只有32个块的时候取消遍历
    if (output_template->warp_begin_thread_index_offset_compress == LINEAR_COMPRESS && ((linear_compress_t *)output_template->warp_begin_thread_index_offset_compress_meta)->coefficient == 32)
    {
        // 这里不需要遍历
        return_str = return_str + "{\n";
        return_str = return_str + "unsigned int" + " thread_level_block_id = first_thread_index_of_this_warp + tid_in_warp;\n";
        var_name_of_thread_level_block_index_inner_warp = "tid_in_warp";
    }
    else
    {
        return_str = return_str + "for(";
        return_str = return_str + "unsigned int" + " thread_level_block_id = first_thread_index_of_this_warp + tid_in_warp; ";
        assert(need_first_thread_index_of_next_warp == true);
        return_str = return_str + "thread_level_block_id < first_thread_index_of_next_warp; thread_level_block_id = thread_level_block_id + 32)\n{\n";
        // 这里要计算warp内索引
        return_str = return_str + code_of_data_type(output_template->data_type_of_warp_begin_thread_index_offset) + " thread_level_block_index_inner_warp = thread_level_block_id - first_thread_index_of_this_warp;\n\n";
        var_name_of_thread_level_block_index_inner_warp = "thread_level_block_index_inner_warp";
    }

    // 声明线程的中间结果
    return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " thread_tmp_result = 0;\n\n";

    return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->size_of_val_arr)) + " global_nz_in_thread_level = this_block_first_nz_index + local_this_warp_first_nz_index + " + var_name_of_thread_level_block_index_inner_warp + ";\n\n";

    string code_of_kernal_col_begin_index = "";

    if (output_template->kernal_first_col_index != 0)
    {
        code_of_kernal_col_begin_index = "kernal_first_col_index + ";
    }

    // 遍历线程粒度的块内索引
    if (output_template->thread_block_size_in_warp_compress == CONSTANT_COMPRESS && ((constant_compress_t *)(output_template->thread_block_size_in_warp_compress_meta))->constant == 1)
    {
        // 不用遍历线程粒度的块内的非零元
        return_str = return_str + "{\n";
        // 执行计算
        return_str = return_str + "thread_tmp_result = val_arr[global_nz_in_thread_level] * __ldg(&(device_x_arr[" + code_of_kernal_col_begin_index + "col_index_arr[global_nz_in_thread_level]]));\n";
    }
    else
    {
        // 遍历线程粒度的块内的所有非零元
        return_str = return_str + "for(";
        return_str = return_str + "unsigned int" + " nz_index_in_thread_level = 0; nz_index_in_thread_level < thread_block_size_in_this_warp; nz_index_in_thread_level++)\n{\n";
        return_str = return_str + "thread_tmp_result = thread_tmp_result + val_arr[global_nz_in_thread_level] * __ldg(&(device_x_arr[" + code_of_kernal_col_begin_index + "col_index_arr[global_nz_in_thread_level]]));\n";
        return_str = return_str + "global_nz_in_thread_level = global_nz_in_thread_level + thread_level_block_num_in_warp;\n";
    }

    return_str = return_str + "}\n";

    // 计算thread块在block内部的索引从而将结果写到全局内存
    return_str = return_str + code_of_data_type(output_template->data_type_of_warp_begin_thread_index_offset) + " thread_level_block_index_inner_block = thread_level_block_id - first_thread_index_of_this_block;\n";

    // cout << "TODO: set thread_tmp_result to shared memory" << endl;
    return_str = return_str + "thread_tmp_result_inner_block[thread_level_block_index_inner_block] = thread_tmp_result;\n";
    // return_str = return_str + "thread_tmp_result_inner_block[thread_level_block_index_inner_block] = thread_tmp_result;\n";

    return_str = return_str + "}\n";
    return_str = return_str + "}\n\n";

    // 归约线程块内的结果
    return_str = return_str + "__syncthreads();\n\n";

    // 如果一个block规约一整行，就不需要偏移量
    // if (output_template->thread_num_of_row_reduce != get_config()["HALF_MAX_ROW_REDUCE_THREAD"].as_integer() && output_template->thread_num_of_row_reduce != get_config()["MAX_ROW_REDUCE_THREAD"].as_integer())
    // {
    //     // 声明block第一个线程粒度的块的中间结果的偏移量
    //     return_str = return_str + code_of_data_type(output_template->data_type_of_row_offset_in_thread_tmp_result) + " global_first_tmp_result;\n";

    //     cout << "TODO: 当前块的第一个结果的偏移量，可以通过共享内存的来广播到所有线程" << endl;

    //     // 当前块第一个中间结果的位置
    //     if (output_template->row_offset_in_thread_tmp_result_compress == NONE_COMPRESS)
    //     {
    //         assert(output_template->row_offset_in_thread_tmp_result != NULL);
    //         return_str = return_str + "global_first_tmp_result = row_offset_in_thread_tmp_result[this_block_first_row_index];\n";
    //     }
    //     else if (output_template->row_offset_in_thread_tmp_result_compress == LINEAR_COMPRESS)
    //     {
    //         assert(output_template->row_offset_in_thread_tmp_result_compress_meta != NULL);
    //         linear_compress_t *compressor = (linear_compress_t *)output_template->row_offset_in_thread_tmp_result_compress_meta;
    //         return_str = return_str + code_of_arr_read(compressor, "global_first_tmp_result", "this_block_first_row_index") + ";\n";
    //     }
    //     else
    //     {
    //         cout << "compress type is not supported" << endl;
    //         assert(false);
    //     }

    //     return_str = return_str + "\n";
    // }

    // 用一个变量存储归约之后的结果
    string reduce_result_var_name = "row_tmp_result";

    // 用一个变量来存储行号的，可能替换为block_level_block_id
    string row_id_var_name = "row_id";

    // 用一个变量来存储，可能替换为tid_in_block
    string tid_in_row_var_name = "tid_in_row";

    // 根据每一行的并行度，使用并行树状规约
    if (output_template->thread_num_of_row_reduce == 1)
    {
        // 如果block内的线程数量和行的数量一致，就不需要遍历了
        if (output_template->block_first_row_index_compress == LINEAR_COMPRESS && ((linear_compress_t *)output_template->block_first_row_index_compress_meta)->coefficient == output_template->thread_num_in_block)
        {
            // 不需要遍历
            return_str = return_str + "{\n";
            return_str = return_str + "unsigned int" + " row_id = this_block_first_row_index + tid_in_block;\n";
        }
        else
        {
            return_str = return_str + "for(";
            assert(need_next_block_first_row_index == true);
            return_str = return_str + "unsigned int" + " row_id = this_block_first_row_index + tid_in_block; row_id < next_block_first_row_index; row_id = row_id + blockDim.x)\n{\n";
        }

        // 声明两个块内行号的变量
        return_str = return_str + code_of_data_type(output_template->data_type_of_row_offset_in_thread_tmp_result) + " this_row_local_first_tmp_result;\n";

        if (need_next_block_first_row_index == true)
        {
            return_str = return_str + code_of_data_type(output_template->data_type_of_row_offset_in_thread_tmp_result) + " next_row_local_first_tmp_result;\n";
        }

        // 为两个块内行号赋值，可以使用线性压缩
        if (output_template->row_offset_in_thread_tmp_result_compress == NONE_COMPRESS)
        {
            assert(output_template->row_offset_in_thread_tmp_result != NULL);
            return_str = return_str + "this_row_local_first_tmp_result = row_offset_in_thread_tmp_result[row_id];\n";
            return_str = return_str + "this_row_local_first_tmp_result = this_row_local_first_tmp_result - global_first_tmp_result;\n";

            return_str = return_str + "\n";

            // if (!(output_template->block_first_row_index_compress == LINEAR_COMPRESS && ((linear_compress_t *)output_template->block_first_row_index_compress_meta)->coefficient == output_template->thread_num_in_block))
            // {
            return_str = return_str + "next_row_local_first_tmp_result = row_offset_in_thread_tmp_result[row_id + 1];\n";
            return_str = return_str + "next_row_local_first_tmp_result = next_row_local_first_tmp_result - global_first_tmp_result;\n";
            // }
        }
        else if (output_template->row_offset_in_thread_tmp_result_compress == LINEAR_COMPRESS)
        {
            assert(output_template->row_offset_in_thread_tmp_result_compress_meta != NULL);
            linear_compress_t *compressor = (linear_compress_t *)output_template->row_offset_in_thread_tmp_result_compress_meta;
            // 根据压缩取row_offset_in_thread_tmp_result，
            return_str = return_str + code_of_arr_read(compressor, "this_row_local_first_tmp_result", "row_id") + ";\n";

            return_str = return_str + "this_row_local_first_tmp_result = this_row_local_first_tmp_result - global_first_tmp_result;\n";

            return_str = return_str + "\n";

            // if (!(output_template->block_first_row_index_compress == LINEAR_COMPRESS && ((linear_compress_t *)output_template->block_first_row_index_compress_meta)->coefficient == output_template->thread_num_in_block))
            // {
            return_str = return_str + "next_row_local_first_tmp_result = this_row_local_first_tmp_result + " + to_string(compressor->coefficient) + ";\n";
            // return_str = return_str + code_of_arr_read(compressor, "next_row_local_first_tmp_result", "(row_id + 1)") + ";\n";
            // return_str = return_str + "this_row_local_first_tmp_result = this_row_local_first_tmp_result - global_first_tmp_result;\n";
            // }
        }
        else
        {
            cout << "compress type is not supported" << endl;
            assert(false);
        }

        // 这里的压缩取消掉
        // 根据一个block线程数量和行数量的关系，确实是否要在归约过程中使用循环的实现
        // if (output_template->block_first_row_index_compress == LINEAR_COMPRESS && ((linear_compress_t *)output_template->block_first_row_index_compress_meta)->coefficient == output_template->thread_num_in_block)
        // {
        //     cout << "TODO: 一个线程对于一行结果的归约，对于一行结果的遍历的最内层循环是否可以压缩，需要考虑" << endl;
        //     // 每一行中间结果的存储
        //     return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " row_tmp_result;\n";
        //     return_str = return_str + "{\n";
        //     // return_str = return_str + code_of_data_type(output_template->data_type_of_row_offset_in_thread_tmp_result) + " temp_result_id = this_row_local_first_tmp_result;\n";
        //     return_str = return_str + "row_tmp_result = thread_tmp_result_inner_block[this_row_local_first_tmp_result];\n";
        // }
        // else
        {
            // 遍历每一行在共享内存中的中间结果
            return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " row_tmp_result = 0;\n";
            return_str = return_str + "for(";
            return_str = return_str + "unsigned int" + " temp_result_id = this_row_local_first_tmp_result; temp_result_id < next_row_local_first_tmp_result; temp_result_id++)\n{\n";
            // 中间结果的累加
            return_str = return_str + "row_tmp_result = row_tmp_result + thread_tmp_result_inner_block[temp_result_id];\n";
        }

        return_str = return_str + "}\n";
    }
    else if (output_template->thread_num_of_row_reduce == get_config()["HALF_MAX_ROW_REDUCE_THREAD"].as_integer())
    {
        reduce_result_var_name = "thread_tmp_result_inner_block[threadIdx.x]";
        row_id_var_name = "block_level_block_id";
        tid_in_row_var_name = "tid_in_block";

        // 针对一行进行归约
        // 先用一个for循环
        return_str = return_str + "for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)\n{\n";
        return_str = return_str + "__syncthreads();\n";
        return_str = return_str + "if (threadIdx.x < stride)\n{\n";

        // 在共享内存中做归约
        return_str = return_str + "thread_tmp_result_inner_block[threadIdx.x] = thread_tmp_result_inner_block[threadIdx.x] + thread_tmp_result_inner_block[threadIdx.x + stride];\n";

        return_str = return_str + "}\n";
    }
    else if (output_template->thread_num_of_row_reduce == get_config()["MAX_ROW_REDUCE_THREAD"].as_integer())
    {
        reduce_result_var_name = "thread_tmp_result_inner_block[threadIdx.x]";
        row_id_var_name = "block_level_block_id";
        tid_in_row_var_name = "tid_in_block";

        // 针对一行进行归约
        // 先用一个for循环
        return_str = return_str + "for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)\n{\n";
        return_str = return_str + "__syncthreads();\n";
        return_str = return_str + "if (threadIdx.x < stride)\n{\n";

        // 在共享内存中做归约
        return_str = return_str + "thread_tmp_result_inner_block[threadIdx.x] = thread_tmp_result_inner_block[threadIdx.x] + thread_tmp_result_inner_block[threadIdx.x + stride];\n";

        return_str = return_str + "}\n";
    }
    else
    {
        assert(output_template->thread_num_of_row_reduce == 2 || output_template->thread_num_of_row_reduce == 4 || output_template->thread_num_of_row_reduce == 8 ||
               output_template->thread_num_of_row_reduce == 16 || output_template->thread_num_of_row_reduce == 32);
        // 从对于行的遍历开始
        return_str = return_str + "unsigned char active_row_reduce_thread_num = " + to_string(output_template->thread_num_of_row_reduce) + ";\n\n";

        // 行内索引
        return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->thread_num_of_row_reduce)) + " tid_in_row = tid_in_block % active_row_reduce_thread_num;\n\n";

        // 如果归约的线程正好够用，就不用for循环（按照一定的步长遍历）
        if (output_template->block_first_row_index_compress == LINEAR_COMPRESS && ((linear_compress_t *)output_template->block_first_row_index_compress_meta)->coefficient == (output_template->thread_num_in_block / output_template->thread_num_of_row_reduce))
        {
            // 不需要for循环
            return_str = return_str + "{\n";

            // 行号
            return_str = return_str + "unsigned int" + " row_id = this_block_first_row_index + tid_in_block / active_row_reduce_thread_num;\n";
        }
        else
        {
            // 需要for循环，首先计算遍历的步长
            return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->thread_num_in_block)) + " step_size = blockDim.x / " + to_string(output_template->thread_num_of_row_reduce) + ";\n";
            // 执行for循环
            return_str = return_str + "for(";
            return_str = return_str + "unsigned int" + " row_id = this_block_first_row_index + tid_in_block / active_row_reduce_thread_num; row_id < next_block_first_row_index; row_id = row_id + step_size)\n{\n";
        }

        // 结果在共享内存中的首地址
        return_str = return_str + code_of_data_type(output_template->data_type_of_row_offset_in_thread_tmp_result) + " this_row_local_first_tmp_result;\n";
        return_str = return_str + code_of_data_type(output_template->data_type_of_row_offset_in_thread_tmp_result) + " next_row_local_first_tmp_result;\n\n";

        if (output_template->row_offset_in_thread_tmp_result_compress == NONE_COMPRESS)
        {
            assert(output_template->row_offset_in_thread_tmp_result != NULL);
            return_str = return_str + "this_row_local_first_tmp_result = row_offset_in_thread_tmp_result[row_id];\n";
            return_str = return_str + "this_row_local_first_tmp_result = this_row_local_first_tmp_result - global_first_tmp_result;\n";

            return_str = return_str + "\n";

            return_str = return_str + "next_row_local_first_tmp_result = row_offset_in_thread_tmp_result[row_id + 1];\n";
            return_str = return_str + "next_row_local_first_tmp_result = next_row_local_first_tmp_result - global_first_tmp_result;\n";
        }
        else if (output_template->row_offset_in_thread_tmp_result_compress == LINEAR_COMPRESS)
        {
            assert(output_template->row_offset_in_thread_tmp_result_compress_meta != NULL);
            linear_compress_t *compressor = (linear_compress_t *)output_template->row_offset_in_thread_tmp_result_compress_meta;
            // 根据压缩取row_offset_in_thread_tmp_result，
            return_str = return_str + code_of_arr_read(compressor, "this_row_local_first_tmp_result", "row_id") + ";\n";
            return_str = return_str + "this_row_local_first_tmp_result = this_row_local_first_tmp_result - global_first_tmp_result;\n";

            return_str = return_str + "\n";
            return_str = return_str + "next_row_local_first_tmp_result = this_row_local_first_tmp_result + " + to_string(compressor->coefficient) + ";\n";
        }
        else
        {
            cout << "compress type is not supported" << endl;
            assert(false);
        }

        // 树状归约的最外层遍历，遍历树状归约的某一层
        return_str = return_str + "for(";
        return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->thread_num_of_row_reduce)) + " cur_active_row_reduce_thread_num = active_row_reduce_thread_num; cur_active_row_reduce_thread_num >= 1; cur_active_row_reduce_thread_num = cur_active_row_reduce_thread_num / 2)\n{\n";

        // 只有活跃的线程要执行归约
        return_str = return_str + "if (tid_in_row < cur_active_row_reduce_thread_num)\n{\n";

        // 执行一层内部的归约
        return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " row_tmp_result = 0;\n";

        return_str = return_str + "for (";

        return_str = return_str + "unsigned int" + " temp_result_id = this_row_local_first_tmp_result + tid_in_row; temp_result_id < next_row_local_first_tmp_result; temp_result_id = temp_result_id + cur_active_row_reduce_thread_num)\n{\n";

        return_str = return_str + "row_tmp_result = row_tmp_result + thread_tmp_result_inner_block[temp_result_id];\n";

        return_str = return_str + "}\n";

        // 将一个线程的结果写回共享内存，并且确定下一层的归约范围
        return_str = return_str + "thread_tmp_result_inner_block[this_row_local_first_tmp_result + tid_in_row] = row_tmp_result;\n";
        return_str = return_str + "next_row_local_first_tmp_result = this_row_local_first_tmp_result + cur_active_row_reduce_thread_num;\n";

        return_str = return_str + "}\n";
        return_str = return_str + "}\n";

        reduce_result_var_name = "thread_tmp_result_inner_block[this_row_local_first_tmp_result]";
    }

    // 如果多个线程归约一行，那么一个行的第一个线程来处理最后写缓存的操作
    if (output_template->thread_num_of_row_reduce != 1)
    {
        return_str = return_str + "if (" + tid_in_row_var_name + " == 0)\n{\n";
    }

    // 将结果写到全局内存中
    string var_name_of_global_row_index = row_id_var_name;

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

        // 有排序，重新找全局行号
        // if (output_template->kernal_first_row_index != 0)
        // {
        //     return_str = return_str + "global_row_index = global_row_index + kernal_first_row_index;\n";
        // }

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

    if (output_template->thread_num_of_row_reduce != 1)
    {
        return_str = return_str + "}\n\n";
    }

    return_str = return_str + "}\n";

    if (need_block_level_shared_init == false)
    {
        return_str = return_str + "__syncthreads();\n";
    }

    return_str = return_str + "}\n";
    return_str = return_str + "}\n";

    return return_str;
}

// 核函数的调用
string code_of_kernal_function_call(shared_memory_template_t *output_template, unsigned long dense_block_id)
{
    assert(output_template != NULL);
    // 线程块的数量和线程的数量不能超标
    assert(output_template->tblock_num <= get_config()["MAX_TBLOCK_NUM"].as_integer() && output_template->thread_num_in_block <= get_config()["MAX_THREAD_NUM_IN_BLOCK"].as_integer());

    string return_str = "spmv_" + to_string(dense_block_id) + "<<<" + to_string(output_template->tblock_num) + ", " + to_string(output_template->thread_num_in_block) + ", 0, stream_arr[" + to_string(dense_block_id) + "]>>>(";

    bool is_first_param = true;

    // 遍历所有的形参
    // 这里加入形参的声明
    if (output_template->row_offset_in_thread_tmp_result_compress == NONE_COMPRESS)
    {
        assert(output_template->row_offset_in_thread_tmp_result != NULL);
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "row_offset_in_thread_tmp_result");
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

    if (output_template->warp_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }
        assert(output_template->warp_begin_thread_index_offset != NULL);
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset");
    }

    if (output_template->thread_block_size_in_warp_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }
        assert(output_template->thread_block_size_in_warp != NULL);
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp");
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

    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }
        assert(output_template->block_nz_begin_offset != NULL);
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset");
    }

    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }
        assert(output_template->warp_nz_begin_offset != NULL);
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "warp_nz_begin_offset");
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

string code_of_write_template_data_to_gpu(shared_memory_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index)
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

    if (output_template->row_offset_in_thread_tmp_result_compress == NONE_COMPRESS)
    {
        assert(output_template->row_offset_in_thread_tmp_result != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_row_offset_in_thread_tmp_result, "device_" + code_of_arr_var_name(dense_block_id, -1, "row_offset_in_thread_tmp_result"));
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

    if (output_template->warp_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->warp_begin_thread_index_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_warp_begin_thread_index_offset, "device_" + code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset"));
    }

    if (output_template->thread_block_size_in_warp_compress == NONE_COMPRESS)
    {
        assert(output_template->thread_block_size_in_warp != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_thread_block_size_in_warp, "device_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp"));
    }

    // 行顺序数组的声明
    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->row_index_before_sort != NULL)
    {
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_row_index_before_sort, "device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"));
    }

    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_block_nz_begin_offset, "device_" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset"));
    }

    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->warp_nz_begin_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_warp_nz_begin_offset, "device_" + code_of_arr_var_name(dense_block_id, -1, "warp_nz_begin_offset"));
    }

    assert(output_template->val_arr != NULL);
    return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_val_arr, "device_" + code_of_arr_var_name(dense_block_id, -1, "val_arr"));

    assert(output_template->col_index_arr != NULL);
    return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_col_index_arr, "device_" + code_of_arr_var_name(dense_block_id, -1, "col_index_arr"));

    return_str = return_str + "\n";

    if (output_template->row_offset_in_thread_tmp_result_compress == NONE_COMPRESS)
    {
        assert(output_template->row_offset_in_thread_tmp_result != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_row_offset_in_thread_tmp_result, to_string(output_template->size_of_row_offset_in_thread_tmp_result), "device_" + code_of_arr_var_name(dense_block_id, -1, "row_offset_in_thread_tmp_result"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "row_offset_in_thread_tmp_result"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "row_offset_in_thread_tmp_result"), output_template->data_type_of_row_offset_in_thread_tmp_result, to_string(output_template->size_of_row_offset_in_thread_tmp_result), "cudaMemcpyHostToDevice") + "\n";
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

    if (output_template->warp_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->warp_begin_thread_index_offset != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_warp_begin_thread_index_offset, to_string(output_template->size_of_warp_begin_thread_index_offset), "device_" + code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset"), output_template->data_type_of_warp_begin_thread_index_offset, to_string(output_template->size_of_warp_begin_thread_index_offset), "cudaMemcpyHostToDevice") + "\n";
    }

    if (output_template->thread_block_size_in_warp_compress == NONE_COMPRESS)
    {
        assert(output_template->thread_block_size_in_warp != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_thread_block_size_in_warp, to_string(output_template->size_of_thread_block_size_in_warp), "device_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp"), output_template->data_type_of_thread_block_size_in_warp, to_string(output_template->size_of_thread_block_size_in_warp), "cudaMemcpyHostToDevice") + "\n";
    }

    // if (output_template->thread_block_size_in_warp_compress == NONE_COMPRESS)
    // {
    //     assert(output_template->thread_block_size_in_warp != NULL);
    //     return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_thread_block_size_in_warp, to_string(output_template->size_of_thread_block_size_in_warp), "device_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp"));
    //     // 拷贝
    //     return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp"), output_template->data_type_of_thread_block_size_in_warp, to_string(output_template->size_of_thread_block_size_in_warp), "cudaMemcpyHostToDevice") + "\n";
    // }

    // 如果是全局的就直接赋值
    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->global_sort_index == true)
    {
        assert(output_template->local_sort_index == false);

        if (force_not_share_global_sort_index == true)
        {
            // 不共享，从disk中读
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

    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_block_nz_begin_offset, to_string(output_template->size_of_block_nz_begin_offset), "device_" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset"), output_template->data_type_of_block_nz_begin_offset, to_string(output_template->size_of_block_nz_begin_offset), "cudaMemcpyHostToDevice") + "\n";
    }

    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->warp_nz_begin_offset != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_warp_nz_begin_offset, to_string(output_template->size_of_warp_nz_begin_offset), "device_" + code_of_arr_var_name(dense_block_id, -1, "warp_nz_begin_offset"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "warp_nz_begin_offset"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "warp_nz_begin_offset"), output_template->data_type_of_warp_nz_begin_offset, to_string(output_template->size_of_warp_nz_begin_offset), "cudaMemcpyHostToDevice") + "\n";
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

bool compress_block_begin_warp_index_offset(shared_memory_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && type == LINEAR_COMPRESS && output_template->block_begin_warp_index_offset != NULL);

    linear_compress_t *compressor = init_linear_compressor(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, output_template->size_of_block_begin_warp_index_offset, need_check);

    if (compressor == NULL)
    {
        return false;
    }

    // 压缩成功，拷贝元数据
    output_template->block_begin_warp_index_offset_compress_meta = (void *)compressor;
    output_template->block_begin_warp_index_offset_compress = type;

    return true;
}

bool compress_warp_begin_thread_index_offset(shared_memory_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && type == LINEAR_COMPRESS && output_template->warp_begin_thread_index_offset != NULL);

    linear_compress_t *compressor = init_linear_compressor(output_template->warp_begin_thread_index_offset, output_template->data_type_of_warp_begin_thread_index_offset, output_template->size_of_warp_begin_thread_index_offset, need_check);

    if (compressor == NULL)
    {
        return false;
    }

    // 压缩成功
    output_template->warp_begin_thread_index_offset_compress_meta = (void *)compressor;
    output_template->warp_begin_thread_index_offset_compress = type;

    return true;
}

bool compress_thread_block_size_in_warp(shared_memory_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && output_template->thread_block_size_in_warp != NULL);
    assert(type == CONSTANT_COMPRESS || type == BRANCH_COMPRESS);

    if (type == CONSTANT_COMPRESS)
    {
        constant_compress_t *compressor = init_constant_compressor(output_template->thread_block_size_in_warp, output_template->data_type_of_thread_block_size_in_warp, output_template->size_of_thread_block_size_in_warp, need_check);

        if (compressor == NULL)
        {
            return false;
        }

        // 压缩成功
        output_template->thread_block_size_in_warp_compress_meta = (void *)compressor;
        output_template->thread_block_size_in_warp_compress = type;
    }

    if (type == BRANCH_COMPRESS)
    {
        branch_compress_t *compressor = init_branch_compressor(output_template->thread_block_size_in_warp, output_template->data_type_of_thread_block_size_in_warp, output_template->size_of_thread_block_size_in_warp, need_check);

        if (compressor == NULL)
        {
            return false;
        }

        // 压缩成功
        output_template->thread_block_size_in_warp_compress_meta = (void *)compressor;
        output_template->thread_block_size_in_warp_compress = type;
    }

    return true;
}

bool compress_block_nz_begin_offset(shared_memory_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && type == LINEAR_COMPRESS && output_template->block_nz_begin_offset != NULL);

    linear_compress_t *compressor = init_linear_compressor(output_template->block_nz_begin_offset, output_template->data_type_of_block_nz_begin_offset, output_template->size_of_block_nz_begin_offset, need_check);

    if (compressor == NULL)
    {
        return false;
    }

    // 压缩成功
    output_template->block_nz_begin_offset_compress_meta = (void *)compressor;
    output_template->block_nz_begin_offset_compress = type;

    return true;
}

bool compress_warp_nz_begin_offset(shared_memory_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && type == CYCLE_LINEAR_COMPRESS && output_template->warp_nz_begin_offset != NULL);

    unsigned long cycle_num;

    // 只有每个block的warp数量相等，才有这里压缩的可能性，首先查看是否压缩过
    if (output_template->block_begin_warp_index_offset_compress == LINEAR_COMPRESS)
    {
        // 周期就是斜率，代表每一个block的warp数量
        linear_compress_t *compressor = (linear_compress_t *)output_template->block_begin_warp_index_offset_compress_meta;
        cycle_num = compressor->coefficient;
    }
    else
    {
        // 没有压缩，直接查看block_begin_warp_index，检查每个block块的warp数量是否相等
        // 首先查看前两位
        unsigned long first_content = read_from_array_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, 0);
        unsigned long second_content = read_from_array_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, 1);

        if (second_content < first_content)
        {
            return false;
        }
        // 两位之间的两位之间的相减就是斜率
        unsigned long tmp_coef = second_content - first_content;

        // 遍历剩下的内容，看看是不是一样的步长递进
        for (unsigned long i = 0; i < output_template->size_of_block_begin_warp_index_offset - 1; i++)
        {
            first_content = read_from_array_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, i);
            second_content = read_from_array_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, i + 1);

            if (second_content < first_content)
            {
                return false;
            }

            if (second_content - first_content != tmp_coef)
            {
                cout << "no cycle:" << i << endl;
                return false;
            }
        }

        // 这里代表存在周期
        cycle_num = tmp_coef;
    }

    // 按照周期压缩
    cycle_linear_compress_t *compressor = init_cycle_linear_compressor(output_template->warp_nz_begin_offset, output_template->data_type_of_warp_nz_begin_offset, output_template->size_of_warp_nz_begin_offset, cycle_num, need_check);

    // 压缩不成功
    if (compressor == NULL)
    {
        return false;
    }

    // 压缩成功，写元数据
    output_template->warp_nz_begin_offset_compress_meta = (void *)compressor;
    output_template->warp_nz_begin_offset_compress = type;

    return true;
}

bool compress_block_first_row_index(shared_memory_template_t *output_template, bool need_check, arr_compress_type type)
{
    // 基本只能线性压缩
    assert(output_template != NULL && type == LINEAR_COMPRESS && output_template->block_first_row_index != NULL);

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

//
bool compress_row_offset_in_thread_tmp_result(shared_memory_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && type == LINEAR_COMPRESS && output_template->row_offset_in_thread_tmp_result != NULL);

    linear_compress_t *compressor = init_linear_compressor(output_template->row_offset_in_thread_tmp_result, output_template->data_type_of_row_offset_in_thread_tmp_result, output_template->size_of_row_offset_in_thread_tmp_result, need_check);

    if (compressor == NULL)
    {
        return false;
    }

    // 压缩成功
    output_template->row_offset_in_thread_tmp_result_compress_meta = (void *)compressor;
    output_template->row_offset_in_thread_tmp_result_compress = type;

    return true;
}

// 尝试所有的压缩
void try_all_compress(shared_memory_template_t *output_template)
{
    assert(output_template != NULL);

    bool is_compressed = false;

    is_compressed = compress_block_begin_warp_index_offset(output_template, true, LINEAR_COMPRESS);

    is_compressed = compress_warp_begin_thread_index_offset(output_template, true, LINEAR_COMPRESS);

    is_compressed = compress_thread_block_size_in_warp(output_template, true, CONSTANT_COMPRESS);

    if (is_compressed == false)
    {
        is_compressed = compress_thread_block_size_in_warp(output_template, true, BRANCH_COMPRESS);
    }

    is_compressed = compress_block_nz_begin_offset(output_template, true, LINEAR_COMPRESS);

    is_compressed = compress_warp_nz_begin_offset(output_template, true, CYCLE_LINEAR_COMPRESS);

    is_compressed = compress_block_first_row_index(output_template, true, LINEAR_COMPRESS);

    is_compressed = compress_row_offset_in_thread_tmp_result(output_template, true, LINEAR_COMPRESS);
}

bool set_row_reduce_thread_num(shared_memory_template_t *output_template, unsigned long row_reduce_thread_num)
{
    assert(output_template != NULL);
    assert(row_reduce_thread_num == 1 || row_reduce_thread_num == 2 || row_reduce_thread_num == 4 || row_reduce_thread_num == 8 || row_reduce_thread_num == 16 || row_reduce_thread_num == 32 || row_reduce_thread_num == get_config()["HALF_MAX_ROW_REDUCE_THREAD"].as_integer() || row_reduce_thread_num == get_config()["MAX_ROW_REDUCE_THREAD"].as_integer());

    if (row_reduce_thread_num == 1)
    {
        output_template->thread_num_of_row_reduce = 1;
    }
    else if (row_reduce_thread_num == get_config()["HALF_MAX_ROW_REDUCE_THREAD"].as_integer())
    {
        // 开启最大并行度，需要满足两个条件：1、一个block负责一行，2、一个线程块粒度的块中线程粒度的块数量和线程块粒度的中的线程数量完全一致（整数倍也行，但是性能不会有提升）
        // 首先检查是不是一个block负责一行，遍历所有block的首行行号
        // 因为有dblock先分为dwarp，然后才有thread，
        // assert(output_template->size_of_row_index_before_sort == output_template->size_of_block_begin_warp_index_offset);
        assert(output_template->size_of_block_first_row_index == output_template->size_of_block_begin_warp_index_offset);

        for (unsigned long i = 0; i < output_template->size_of_block_first_row_index - 1; i++)
        {
            // 查看当前块的行数量
            unsigned long block_row_num = read_from_array_with_data_type(output_template->block_first_row_index, output_template->data_type_of_block_first_row_index, i + 1) - read_from_array_with_data_type(output_template->block_first_row_index, output_template->data_type_of_block_first_row_index, i);
            // 必须是1
            if (block_row_num != 1)
            {
                cout << "block row num is not equal to 1, block_row_num:" << block_row_num << endl;
                return false;
            }

            // 查看当前线程粒度的块的数量，需要和线程块中线程的数量保持完全一致（其实两倍也可以，不做进一步考虑）
            // 当前块的warp号
            unsigned long this_block_first_warp_index = read_from_array_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, i);
            unsigned long next_block_first_warp_index = read_from_array_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, i + 1);

            // 当前块的首个thread号
            unsigned long this_block_first_thread_index = read_from_array_with_data_type(output_template->warp_begin_thread_index_offset, output_template->data_type_of_warp_begin_thread_index_offset, this_block_first_warp_index);
            unsigned long next_block_first_thread_index = read_from_array_with_data_type(output_template->warp_begin_thread_index_offset, output_template->data_type_of_warp_begin_thread_index_offset, next_block_first_warp_index);

            unsigned long block_thread_num = next_block_first_thread_index - this_block_first_thread_index;

            if (block_thread_num != output_template->thread_num_in_block)
            {
                cout << "thread level block num is not equal to blockDim.x" << endl;
                return false;
            }
        }

        output_template->thread_num_of_row_reduce = get_config()["HALF_MAX_ROW_REDUCE_THREAD"].as_integer();
    }
    else if (row_reduce_thread_num == get_config()["MAX_ROW_REDUCE_THREAD"].as_integer())
    {
        assert(output_template->size_of_block_first_row_index == output_template->size_of_block_begin_warp_index_offset);

        for (unsigned long i = 0; i < output_template->size_of_block_first_row_index - 1; i++)
        {
            // 查看当前块的行数量
            unsigned long block_row_num = read_from_array_with_data_type(output_template->block_first_row_index, output_template->data_type_of_block_first_row_index, i + 1) - read_from_array_with_data_type(output_template->block_first_row_index, output_template->data_type_of_block_first_row_index, i);
            // 必须是1
            if (block_row_num != 1)
            {
                cout << "block row num is not equal to 1, block_row_num:" << block_row_num << endl;
                return false;
            }

            // 查看当前线程粒度的块的数量，需要和线程块中线程的数量保持完全一致（其实两倍也可以，不做进一步考虑）
            // 当前块的warp号
            unsigned long this_block_first_warp_index = read_from_array_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, i);
            unsigned long next_block_first_warp_index = read_from_array_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, i + 1);

            // 当前块的首个thread号
            unsigned long this_block_first_thread_index = read_from_array_with_data_type(output_template->warp_begin_thread_index_offset, output_template->data_type_of_warp_begin_thread_index_offset, this_block_first_warp_index);
            unsigned long next_block_first_thread_index = read_from_array_with_data_type(output_template->warp_begin_thread_index_offset, output_template->data_type_of_warp_begin_thread_index_offset, next_block_first_warp_index);

            unsigned long block_thread_num = next_block_first_thread_index - this_block_first_thread_index;

            if (block_thread_num != 2 * output_template->thread_num_in_block)
            {
                cout << "thread level block num is not equal to blockDim.x" << endl;
                return false;
            }
        }

        output_template->thread_num_of_row_reduce = get_config()["MAX_ROW_REDUCE_THREAD"].as_integer();
    }
    else
    {
        // 小并行，保证每一行线程粒度的块的数量（中间结果的数量），必须大于负责每一行归约的线程的数量
        // 遍历找出每一行线程粒度的块的数量，这个数量必须大于每一行负责归约的块的数量
        // 中间结果数组存储了所有这些信息
        assert(output_template->size_of_row_offset_in_thread_tmp_result == output_template->matrix->dense_row_number + 1);

        for (unsigned long i = 0; i < output_template->size_of_row_offset_in_thread_tmp_result - 1; i++)
        {
            // 一行中中间结果的数量
            unsigned long tmp_result_of_a_row = read_from_array_with_data_type(output_template->row_offset_in_thread_tmp_result, output_template->data_type_of_row_offset_in_thread_tmp_result, i + 1) - read_from_array_with_data_type(output_template->row_offset_in_thread_tmp_result, output_template->data_type_of_row_offset_in_thread_tmp_result, i);
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