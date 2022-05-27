#include "direct_atom_op_warp_compress.hpp"
#include <vector>

using namespace std;

direct_atom_template_warp_compress_t *init_direct_atom_template_warp_compress(code_builder_t *builder, unsigned long dense_block_id)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;

    assert(dense_block_id < matrix->block_coor_table.item_arr.size());

    unsigned long kernal_first_row_index = matrix->block_coor_table.item_arr[dense_block_id]->min_dense_row_index;
    unsigned long kernal_first_col_index = matrix->block_coor_table.item_arr[dense_block_id]->min_dense_col_index;

    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr;

    bool need_atom_add = false;

    if (matrix->block_coor_table.item_arr[dense_block_id]->min_dense_col_index == 0 && matrix->block_coor_table.item_arr[dense_block_id]->max_dense_col_index == matrix->dense_col_number - 1)
    {
        // 稠密子块之间没有共享的行
    }
    else
    {
        need_atom_add = true;
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

    assert(block_level_index->max_row_index == compressed_block_view->read_index[0]->max_row_index);
    assert(warp_level_index->max_row_index == compressed_block_view->read_index[0]->max_row_index);
    assert(thread_level_index->max_row_index == compressed_block_view->read_index[0]->max_row_index);

    assert(thread_level_index->coo_block_size_arr != NULL);

    if (thread_level_index->row_number_of_block_arr != NULL)
    {
        cout << "thread_level_index->row_number_of_block_arr must be NULL, row num in thread level block must be 1" << endl;
        assert(false);
    }

    // 每个thread的全局行索引
    vector<unsigned long> global_thread_row_index_vec;
    vector<unsigned long> new_thread_block_size_in_block_vec;
    // 存储每个block的thread偏移量
    vector<unsigned long> new_block_begin_thread_index_offset_vec;

    // cout << 2 << endl;
    // 遍历三个层次的索引
    for (unsigned long index_of_block_level_index = 0; index_of_block_level_index < block_level_index->block_num; index_of_block_level_index++)
    {
        // cout << "index_of_block_level_index:" << index_of_block_level_index << endl;
        // 当前block的首行行号
        unsigned long block_first_row_index = read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, index_of_block_level_index);
        // block中第一个warp号和下一个block的首warp
        unsigned long this_block_first_warp_index = read_from_array_with_data_type(block_level_index->index_arr, block_level_index->index_data_type, index_of_block_level_index);
        unsigned long next_block_first_warp_index = read_from_array_with_data_type(block_level_index->index_arr, block_level_index->index_data_type, index_of_block_level_index + 1);

        // 当前block的第一个warp的TLB大小
        unsigned long TLB_size_of_first_WLB_in_this_BLB = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, this_block_first_warp_index);
        unsigned long thread_begin_offset_in_first_warp = read_from_array_with_data_type(warp_level_index->index_arr, warp_level_index->index_data_type, this_block_first_warp_index);

        // 遍历warp层次
        for (unsigned long index_of_warp_level_index = this_block_first_warp_index; index_of_warp_level_index < next_block_first_warp_index; index_of_warp_level_index++)
        {
            assert(index_of_warp_level_index < warp_level_index->block_num);
            unsigned long warp_first_row_index = read_from_array_with_data_type(warp_level_index->index_of_the_first_row_arr, warp_level_index->data_type_of_index_of_the_first_row_arr, index_of_warp_level_index);
            unsigned long this_warp_first_thread_index = read_from_array_with_data_type(warp_level_index->index_arr, warp_level_index->index_data_type, index_of_warp_level_index);
            unsigned long next_warp_first_thread_index = read_from_array_with_data_type(warp_level_index->index_arr, warp_level_index->index_data_type, index_of_warp_level_index + 1);

            // 当前warp的TLB大小
            unsigned long this_WLB_TLB_size = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, index_of_warp_level_index);

            // 和这个BLB内其他WLB的TLB大小做比较，如果不相等，说明没有办法使用这个模板
            if (this_WLB_TLB_size != TLB_size_of_first_WLB_in_this_BLB)
            {
                cout << "thread level block size is not equal in block " << index_of_block_level_index << endl;
                assert(false);
            }

            for (unsigned long index_of_thread_level_index = this_warp_first_thread_index; index_of_thread_level_index < next_warp_first_thread_index; index_of_thread_level_index++)
            {
                // assert(index_of_thread_level_index < thread_level_index->block_num);
                if (index_of_thread_level_index >= thread_level_index->block_num)
                {
                    cout << "index_of_thread_level_index:" << index_of_thread_level_index << ", "
                         << "thread_level_index->block_num:" << thread_level_index->block_num << ", "
                         << "dense_block_id:" << dense_block_id << endl;
                    assert(false);
                }
                unsigned long thread_first_row_index = read_from_array_with_data_type(thread_level_index->index_of_the_first_row_arr, thread_level_index->data_type_of_index_of_the_first_row_arr, index_of_thread_level_index);
                // 全局的行索引
                unsigned long global_thread_row_index = block_first_row_index + warp_first_row_index + thread_first_row_index;

                // 小于当前块的全局行数量
                assert(global_thread_row_index < (thread_level_index->max_row_index - thread_level_index->min_row_index + 1));

                global_thread_row_index_vec.push_back(global_thread_row_index);

                // 将最后两个值进行比较，如果相等就代表要用原子加
                if (global_thread_row_index_vec.size() >= 2)
                {
                    if (global_thread_row_index_vec[global_thread_row_index_vec.size() - 1] == global_thread_row_index_vec[global_thread_row_index_vec.size() - 2])
                    {
                        // 有行共享
                        need_atom_add = true;
                    }
                }
            }    
        }
        new_thread_block_size_in_block_vec.push_back(TLB_size_of_first_WLB_in_this_BLB);
        new_block_begin_thread_index_offset_vec.push_back(thread_begin_offset_in_first_warp);
    }

    // cout << 3 << endl;
    assert(global_thread_row_index_vec.size() == thread_level_index->block_num);
    new_block_begin_thread_index_offset_vec.push_back(thread_level_index->block_num);
    assert(new_block_begin_thread_index_offset_vec.size() == block_level_index->block_num + 1);
    assert(new_thread_block_size_in_block_vec.size() == block_level_index->block_num);
    // 和warp粒度的块的最后一个索引
    assert(read_from_array_with_data_type(warp_level_index->index_arr, warp_level_index->index_data_type, warp_level_index->block_num) == new_block_begin_thread_index_offset_vec[new_block_begin_thread_index_offset_vec.size() - 1]);

    // 处理block级别的padding
    // 仅仅经过padding之后值数组及其数据类型
    assert(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->padding_val_arr != NULL);
    void *val_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->padding_val_arr;
    data_type data_type_of_val_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->val_data_type;
    unsigned long size_of_val_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->padding_arr_size;

    // 仅仅经过padding之后的列数组及其数据类型
    void *col_index_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[5]->index_arr;
    assert(col_index_arr_after_padding != NULL);
    data_type data_type_of_col_index_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[5]->index_data_type;
    unsigned long size_of_col_index_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[5]->length;

    assert(size_of_val_arr_after_padding == size_of_col_index_arr_after_padding);

    // 新的col数组和val数组
    void *new_col_index_arr = malloc_arr(size_of_col_index_arr_after_padding, data_type_of_col_index_arr_after_padding);
    void *new_val_arr = malloc_arr(size_of_val_arr_after_padding, data_type_of_val_arr_after_padding);

    // 执行val和col的padding，
    for (unsigned long block_index = 0; block_index < block_level_index->block_num; block_index++)
    {
        // 当前块的头部非零元索引
        unsigned long block_begin_nz_index = read_from_array_with_data_type(block_level_index->coo_begin_index_arr, block_level_index->data_type_of_coo_begin_index_arr, block_index);
        // 当前块的thread块的起始位置
        unsigned long block_begin_thread_index = new_block_begin_thread_index_offset_vec[block_index];
        // 下一个块thread块的起始位置
        unsigned long next_block_begin_thread_index = new_block_begin_thread_index_offset_vec[block_index + 1];

        assert(next_block_begin_thread_index >= block_begin_thread_index);

        // 当前线程粒度的块的数量
        unsigned long block_num_of_thread_level_block = next_block_begin_thread_index - block_begin_thread_index;

        // 当前线程粒度的块的大小
        unsigned long block_size_of_thread_level_block = new_thread_block_size_in_block_vec[block_index];

        if (block_index < (block_level_index->block_num - 1))
        {
            assert((block_begin_nz_index + block_num_of_thread_level_block * block_size_of_thread_level_block) == read_from_array_with_data_type(block_level_index->coo_begin_index_arr, block_level_index->data_type_of_coo_begin_index_arr, block_index + 1));
        }

        assert(next_block_begin_thread_index > block_begin_thread_index);
        // 遍历当前线程块粒度的块的所有线程粒度的块
        for (unsigned long thread_index = block_begin_thread_index; thread_index < next_block_begin_thread_index; thread_index++)
        {
            // 线程的块内索引
            unsigned long thread_inner_block = thread_index - block_begin_thread_index;
            // 遍历线程粒度的块的所有非零元
            for (unsigned long nz_index = 0; nz_index < block_size_of_thread_level_block; nz_index++)
            {
                // 当前非零元在源数组中的位置
                unsigned long source_index_of_this_nz = block_begin_nz_index + thread_inner_block * block_size_of_thread_level_block + nz_index;
                assert(source_index_of_this_nz < size_of_val_arr_after_padding);
                // 将val和col从源数组中读出来
                double val = read_double_from_array_with_data_type(val_arr_after_padding, data_type_of_val_arr_after_padding, source_index_of_this_nz);
                unsigned long col_index = read_from_array_with_data_type(col_index_arr_after_padding, data_type_of_col_index_arr_after_padding, source_index_of_this_nz);
                // 当前非零元在目标数组中的位置，按照线程粒度的块的数量交错来存
                unsigned long dest_index_of_this_nz = block_begin_nz_index + thread_inner_block + nz_index * block_num_of_thread_level_block;

                if (dest_index_of_this_nz >= size_of_val_arr_after_padding)
                {
                    cout << "dest_index_of_this_nz:" << dest_index_of_this_nz << endl;
                    cout << "block_begin_nz_index:" << block_begin_nz_index << endl;
                    cout << "thread_inner_block:" << thread_inner_block << endl;
                    cout << "block_num_of_thread_level_block:" << block_num_of_thread_level_block << endl;
                    cout << "nz_index:" << nz_index << endl;
                    cout << "block_index:" << block_index << endl;
                    cout << "size_of_val_arr_after_padding:" << size_of_val_arr_after_padding << endl;
                }
                assert(dest_index_of_this_nz < size_of_val_arr_after_padding);

                // 将内容写入
                write_to_array_with_data_type(new_col_index_arr, data_type_of_col_index_arr_after_padding, dest_index_of_this_nz, col_index);
                write_double_to_array_with_data_type(new_val_arr, data_type_of_val_arr_after_padding, dest_index_of_this_nz, val);
            }
        }
    }

    // 创建一个模板结构体存储所有数据
    direct_atom_template_warp_compress_t *new_template = new direct_atom_template_warp_compress_t();
    new_template->dense_block_index = dense_block_id;
    new_template->matrix = matrix;
    new_template->kernal_first_row_index = kernal_first_row_index;
    new_template->kernal_first_col_index = kernal_first_col_index;

    // 继承之前的原子加策略
    new_template->is_atom_add = need_atom_add;

    // block的第一个thread的偏移量
    new_template->data_type_of_block_begin_thread_index_offset = find_most_suitable_data_type(new_block_begin_thread_index_offset_vec[new_block_begin_thread_index_offset_vec.size() - 1]);
    assert(new_block_begin_thread_index_offset_vec.size() == block_level_index->block_num + 1);
    new_template->size_of_block_begin_thread_index_offset = new_block_begin_thread_index_offset_vec.size();
    new_template->block_begin_thread_index_offset = malloc_arr(new_block_begin_thread_index_offset_vec.size(), new_template->data_type_of_block_begin_thread_index_offset);
    // 拷贝
    copy_unsigned_long_arr_to_others(&(new_block_begin_thread_index_offset_vec[0]), new_template->block_begin_thread_index_offset, new_template->data_type_of_block_begin_thread_index_offset, new_template->size_of_block_begin_thread_index_offset);

    // block首个非零元的偏移
    new_template->data_type_of_block_nz_begin_offset = block_level_index->data_type_of_coo_begin_index_arr;
    new_template->block_nz_begin_offset = block_level_index->coo_begin_index_arr;
    new_template->size_of_block_nz_begin_offset = block_level_index->block_num;

    // 每个block线程粒度的块的大小
    new_template->data_type_of_thread_block_size_in_block = find_most_suitable_data_type(new_thread_block_size_in_block_vec[new_thread_block_size_in_block_vec.size() - 1]);
    assert(new_thread_block_size_in_block_vec.size() == new_template->size_of_block_nz_begin_offset);
    new_template->size_of_thread_block_size_in_block = new_thread_block_size_in_block_vec.size();
    new_template->thread_block_size_in_block = malloc_arr(new_thread_block_size_in_block_vec.size(), new_template->data_type_of_thread_block_size_in_block);
    // 拷贝
    copy_unsigned_long_arr_to_others(&(new_thread_block_size_in_block_vec[0]), new_template->thread_block_size_in_block, new_template->data_type_of_thread_block_size_in_block, new_template->size_of_thread_block_size_in_block);

    
    // 这里处理可能的row padding产生的无效TLB
    if (matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index < compressed_block_view->read_index[0]->max_row_index)
    {
        // 这里代表之前有过row_padding
        // 压缩子块的行数量
        unsigned long compressed_block_row_num = compressed_block_view->read_index[0]->max_row_index - compressed_block_view->read_index[0]->min_row_index + 1;
        unsigned long dense_sub_block_row_num = matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index - matrix->block_coor_table.item_arr[dense_block_id]->min_dense_row_index + 1;

        assert(compressed_block_row_num > dense_sub_block_row_num);

        // 遍历所有的TLB，只要找出TLB的对应行号大于等于dense_sub_block_row_num时，剩下的都是无效的TLB

        // 之前padding过，有无效的TLB，遍历所有的TLB的行号
        for (unsigned long TLB_id = 0; TLB_id < global_thread_row_index_vec.size(); TLB_id++)
        {
            unsigned long cur_TLB_row_index = global_thread_row_index_vec[TLB_id];
            
            if (cur_TLB_row_index >= dense_sub_block_row_num)
            {
                assert(cur_TLB_row_index < compressed_block_row_num);
                
                // 这里代表找到了对应的第一个因为row padding导致的无效TLB
                new_template->effective_TLB_num = TLB_id;
                break;
            }

            // 不可能遍历到最后一个
            assert(TLB_id != global_thread_row_index_vec.size() - 1);
        }
        
        // 遍历剩下的部分
        for (unsigned long TLB_id = new_template->effective_TLB_num; TLB_id < global_thread_row_index_vec.size(); TLB_id++)
        {
            // 剩下的部分，对应的行号必须全部大于压缩子块的有效行索引
            unsigned long cur_TLB_row_index = global_thread_row_index_vec[TLB_id];
            assert(cur_TLB_row_index < compressed_block_row_num);
            assert(cur_TLB_row_index >= dense_sub_block_row_num);
        }
    }
    else
    {
        // 没有padding过，所有的TLB都是有效的
        new_template->effective_TLB_num = global_thread_row_index_vec.size();
    }



    // 排序索引
    // 排序产生的行索引
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

        // 找出原来的索引，因为被padding的行没有参与排序，所以不需要寻找原有的行号
        for (unsigned long row_index_id = 0; row_index_id < new_template->effective_TLB_num; row_index_id++)
        {
            // 当前行号
            unsigned long cur_row_index = global_thread_row_index_vec[row_index_id];

            assert(cur_row_index < new_template->size_of_row_index_before_sort);
            // 排序之前的位置
            unsigned long row_index_before_sort = read_from_array_with_data_type(new_template->row_index_before_sort, new_template->data_type_of_row_index_before_sort, cur_row_index);
            // 重置索引
            global_thread_row_index_vec[row_index_id] = row_index_before_sort;
        }
    }
    else if (matrix->sorted_row_index != NULL)
    {
        cout << "init_direct_atom_template_warp_compress: have global sort" << endl;
        // 在全局范围内有排序
        assert(compressed_block_view->is_sorted == false && matrix->is_sorted == true && builder->sub_block_sort_type_vec[dense_block_id] == GLOBAL_SORT);
        new_template->global_sort_index = true;
        new_template->local_sort_index = false;

        // 拷贝
        new_template->data_type_of_row_index_before_sort = matrix->data_type_of_sorted_row_index;
        new_template->row_index_before_sort = matrix->sorted_row_index;
        new_template->size_of_row_index_before_sort = matrix->dense_row_number;

        // 找出原本的索引，只有有效的TLB才需要原本的行索引
        for (unsigned long row_index_id = 0; row_index_id < new_template->effective_TLB_num; row_index_id++)
        {
            // 当前行号
            unsigned long cur_row_index = global_thread_row_index_vec[row_index_id];

            // 真实行号
            unsigned long matrix_level_row_index = cur_row_index + matrix->block_coor_table.item_arr[dense_block_id]->min_dense_row_index;
            
            assert(matrix_level_row_index < new_template->size_of_row_index_before_sort);
            // 找出之前
            unsigned long row_index_before_sort = read_from_array_with_data_type(new_template->row_index_before_sort, new_template->data_type_of_row_index_before_sort, matrix_level_row_index);

            global_thread_row_index_vec[row_index_id] = row_index_before_sort;
        }
    }

    // thread块的行号
    // 每一行的行号
    // 确定数据类型的大小
    unsigned long max_global_row_index_of_thread_level_block = *max_element(global_thread_row_index_vec.begin(), global_thread_row_index_vec.end());
    new_template->data_type_of_global_row_index_of_thread_level_block = find_most_suitable_data_type(max_global_row_index_of_thread_level_block);
    // 创建对应数组
    new_template->global_row_index_of_thread_level_block = malloc_arr(global_thread_row_index_vec.size(), new_template->data_type_of_global_row_index_of_thread_level_block);
    // 对应数组的长度
    new_template->size_of_global_row_index_of_thread_level_block = global_thread_row_index_vec.size();
    // 拷贝数组
    copy_unsigned_long_arr_to_others(&(global_thread_row_index_vec[0]), new_template->global_row_index_of_thread_level_block, new_template->data_type_of_global_row_index_of_thread_level_block, new_template->size_of_global_row_index_of_thread_level_block);

    // 值
    new_template->data_type_of_val_arr = data_type_of_val_arr_after_padding;
    new_template->val_arr = new_val_arr;
    new_template->size_of_val_arr = size_of_val_arr_after_padding;

    // 列
    new_template->data_type_of_col_index_arr = data_type_of_col_index_arr_after_padding;
    new_template->col_index_arr = new_col_index_arr;
    new_template->size_of_col_index_arr = size_of_col_index_arr_after_padding;

    return new_template;
}


direct_atom_template_warp_compress_t *init_direct_atom_template_warp_compress(direct_atom_template_t *old_template)
{
    cout << "init_direct_atom_template_warp_compress: old API, is not supported" << endl;
    assert(false);

    assert(old_template != NULL);
    vector<unsigned long> new_thread_block_size_in_block_vec;
    // 存储每个block的thread偏移量
    vector<unsigned long> new_block_begin_thread_index_offset_vec;

    // 遍历所有block，查看所有warp内的threadsize是不是相等
    assert(old_template->block_begin_warp_index_offset != NULL);
    for (unsigned long block_index = 0; block_index < (old_template->size_of_block_begin_warp_index_offset - 1); block_index++)
    {
        // 起始和结束的warp
        unsigned long warp_begin_index_in_block = read_from_array_with_data_type(old_template->block_begin_warp_index_offset, old_template->data_type_of_block_begin_warp_index_offset, block_index);
        unsigned long warp_begin_index_in_next_block = read_from_array_with_data_type(old_template->block_begin_warp_index_offset, old_template->data_type_of_block_begin_warp_index_offset, block_index + 1);
        // 一个block中第一个warp的线程粒度的块的大小
        assert(warp_begin_index_in_block < old_template->size_of_thread_block_size_in_warp);
        unsigned long thread_level_size_in_first_warp = read_from_array_with_data_type(old_template->thread_block_size_in_warp, old_template->data_type_of_thread_block_size_in_warp, warp_begin_index_in_block);
        // 一个block中第一个warp的第一个thread的索引
        unsigned long thread_begin_offset_in_first_warp = read_from_array_with_data_type(old_template->warp_begin_thread_index_offset, old_template->data_type_of_warp_begin_thread_index_offset, warp_begin_index_in_block);
        // 遍历所有的warp粒度的块，查看warp内thread的大小是不是相等
        for (unsigned long warp_index = warp_begin_index_in_block; warp_index < warp_begin_index_in_next_block; warp_index++)
        {
            // 获取当前warp的线程粒度的块的大小
            assert(warp_index < old_template->size_of_thread_block_size_in_warp);
            unsigned long thread_level_size_in_this_warp = read_from_array_with_data_type(old_template->thread_block_size_in_warp, old_template->data_type_of_thread_block_size_in_warp, warp_index);
            if (thread_level_size_in_this_warp != thread_level_size_in_first_warp)
            {
                cout << "can not compress in block " << block_index << " because thread level block size is not the same" << endl;
                assert(false);
            }
        }
        // 通过了校验，写new_thread_block_size_in_block_vec
        new_thread_block_size_in_block_vec.push_back(thread_level_size_in_first_warp);
        new_block_begin_thread_index_offset_vec.push_back(thread_begin_offset_in_first_warp);
    }

    assert(new_thread_block_size_in_block_vec.size() == (old_template->size_of_block_begin_warp_index_offset - 1));

    // thread偏移使用的是CSR的方式，要将thread的总数量放在最后
    new_block_begin_thread_index_offset_vec.push_back(read_from_array_with_data_type(old_template->warp_begin_thread_index_offset, old_template->data_type_of_warp_begin_thread_index_offset, old_template->size_of_warp_begin_thread_index_offset - 1));

    // 仅仅经过padding之后值数组及其数据类型
    assert(old_template->matrix->block_coor_table.item_arr[old_template->dense_block_index]->compressed_block_ptr->padding_val_arr != NULL);
    void *val_arr_after_padding = old_template->matrix->block_coor_table.item_arr[old_template->dense_block_index]->compressed_block_ptr->padding_val_arr;
    data_type data_type_of_val_arr_after_padding = old_template->matrix->block_coor_table.item_arr[old_template->dense_block_index]->compressed_block_ptr->val_data_type;
    unsigned long size_of_val_arr_after_padding = old_template->matrix->block_coor_table.item_arr[old_template->dense_block_index]->compressed_block_ptr->padding_arr_size;

    // 仅仅经过padding之后的列数组及其数据类型
    void *col_index_arr_after_padding = old_template->matrix->block_coor_table.item_arr[old_template->dense_block_index]->compressed_block_ptr->read_index[5]->index_arr;
    assert(col_index_arr_after_padding != NULL);
    data_type data_type_of_col_index_arr_after_padding = old_template->matrix->block_coor_table.item_arr[old_template->dense_block_index]->compressed_block_ptr->read_index[5]->index_data_type;
    unsigned long size_of_col_index_arr_after_padding = old_template->matrix->block_coor_table.item_arr[old_template->dense_block_index]->compressed_block_ptr->read_index[5]->length;

    assert(size_of_val_arr_after_padding == size_of_col_index_arr_after_padding);
    assert(size_of_col_index_arr_after_padding == old_template->size_of_val_arr);
    assert(old_template->size_of_col_index_arr == old_template->size_of_val_arr);

    // 新的col数组和val数组
    void *new_col_index_arr = malloc_arr(size_of_col_index_arr_after_padding, data_type_of_col_index_arr_after_padding);
    void *new_val_arr = malloc_arr(size_of_val_arr_after_padding, data_type_of_val_arr_after_padding);

    // 这里做一个交错存储，每一个thread中的元素交错起来，每个block内分别交错自己的内容
    // 遍历每一个block
    for (unsigned long block_index = 0; block_index < (old_template->size_of_block_begin_warp_index_offset - 1); block_index++)
    {
        // 当前块的头部非零元索引
        unsigned long block_begin_nz_index = read_from_array_with_data_type(old_template->block_nz_begin_offset, old_template->data_type_of_block_nz_begin_offset, block_index);
        // 当前块的thread块的起始位置
        unsigned long block_begin_thread_index = new_block_begin_thread_index_offset_vec[block_index];
        // 下一个块thread块的起始位置
        unsigned long next_block_begin_thread_index = new_block_begin_thread_index_offset_vec[block_index + 1];

        assert(next_block_begin_thread_index >= block_begin_thread_index);

        // 当前线程粒度的块的数量
        unsigned long block_num_of_thread_level_block = next_block_begin_thread_index - block_begin_thread_index;

        // 当前线程粒度的块的大小
        unsigned long block_size_of_thread_level_block = new_thread_block_size_in_block_vec[block_index];

        if (block_index < (old_template->size_of_block_begin_warp_index_offset - 2))
        {
            assert((block_begin_nz_index + block_num_of_thread_level_block * block_size_of_thread_level_block) == read_from_array_with_data_type(old_template->block_nz_begin_offset, old_template->data_type_of_block_nz_begin_offset, block_index + 1));
        }

        assert(next_block_begin_thread_index > block_begin_thread_index);
        // 遍历当前线程块粒度的块的所有线程粒度的块
        for (unsigned long thread_index = block_begin_thread_index; thread_index < next_block_begin_thread_index; thread_index++)
        {
            // 线程的块内索引
            unsigned long thread_inner_block = thread_index - block_begin_thread_index;
            // 遍历线程粒度的块的所有非零元
            for (unsigned long nz_index = 0; nz_index < block_size_of_thread_level_block; nz_index++)
            {
                // 当前非零元在源数组中的位置
                unsigned long source_index_of_this_nz = block_begin_nz_index + thread_inner_block * block_size_of_thread_level_block + nz_index;
                assert(source_index_of_this_nz < size_of_val_arr_after_padding);
                // 将val和col从源数组中读出来
                double val = read_double_from_array_with_data_type(val_arr_after_padding, data_type_of_val_arr_after_padding, source_index_of_this_nz);
                unsigned long col_index = read_from_array_with_data_type(col_index_arr_after_padding, data_type_of_col_index_arr_after_padding, source_index_of_this_nz);
                // 当前非零元在目标数组中的位置，按照线程粒度的块的数量交错来存
                unsigned long dest_index_of_this_nz = block_begin_nz_index + thread_inner_block + nz_index * block_num_of_thread_level_block;

                if (dest_index_of_this_nz >= size_of_val_arr_after_padding)
                {
                    cout << "dest_index_of_this_nz:" << dest_index_of_this_nz << endl;
                    cout << "block_begin_nz_index:" << block_begin_nz_index << endl;
                    cout << "thread_inner_block:" << thread_inner_block << endl;
                    cout << "block_num_of_thread_level_block:" << block_num_of_thread_level_block << endl;
                    cout << "nz_index:" << nz_index << endl;
                    cout << "block_index:" << block_index << endl;
                    cout << "size_of_val_arr_after_padding:" << size_of_val_arr_after_padding << endl;
                }
                assert(dest_index_of_this_nz < size_of_val_arr_after_padding);

                // 将内容写入
                write_to_array_with_data_type(new_col_index_arr, data_type_of_col_index_arr_after_padding, dest_index_of_this_nz, col_index);
                write_double_to_array_with_data_type(new_val_arr, data_type_of_val_arr_after_padding, dest_index_of_this_nz, val);
            }
        }
    }

    // 析构warp层次的一系列元数据
    delete_arr_with_data_type(old_template->warp_begin_thread_index_offset, old_template->data_type_of_warp_begin_thread_index_offset);
    delete_arr_with_data_type(old_template->warp_nz_begin_offset, old_template->data_type_of_warp_nz_begin_offset);
    delete_arr_with_data_type(old_template->thread_block_size_in_warp, old_template->data_type_of_thread_block_size_in_warp);
    // block的warp首索引
    delete_arr_with_data_type(old_template->block_begin_warp_index_offset, old_template->data_type_of_block_begin_warp_index_offset);
    old_template->warp_begin_thread_index_offset = NULL;
    old_template->warp_nz_begin_offset = NULL;
    old_template->thread_block_size_in_warp = NULL;
    old_template->block_begin_warp_index_offset = NULL;

    // 创建一个模板结构体存储所有数据
    direct_atom_template_warp_compress_t *new_template = new direct_atom_template_warp_compress_t();
    new_template->dense_block_index = old_template->dense_block_index;
    new_template->matrix = old_template->matrix;
    new_template->kernal_first_row_index = old_template->kernal_first_row_index;
    new_template->kernal_first_col_index = old_template->kernal_first_col_index;

    // 继承之前的原子加策略
    new_template->is_atom_add = old_template->is_atom_add;

    // 每个线程块的行号的拷贝
    new_template->global_row_index_of_thread_level_block = old_template->global_row_index_of_thread_level_block;
    new_template->data_type_of_global_row_index_of_thread_level_block = old_template->data_type_of_global_row_index_of_thread_level_block;
    new_template->size_of_global_row_index_of_thread_level_block = old_template->size_of_global_row_index_of_thread_level_block;

    // block的第一个thread的偏移量
    new_template->data_type_of_block_begin_thread_index_offset = find_most_suitable_data_type(new_block_begin_thread_index_offset_vec[new_block_begin_thread_index_offset_vec.size() - 1]);
    assert(new_block_begin_thread_index_offset_vec.size() == old_template->size_of_block_begin_warp_index_offset);
    new_template->size_of_block_begin_thread_index_offset = new_block_begin_thread_index_offset_vec.size();
    new_template->block_begin_thread_index_offset = malloc_arr(new_block_begin_thread_index_offset_vec.size(), new_template->data_type_of_block_begin_thread_index_offset);
    // 拷贝
    copy_unsigned_long_arr_to_others(&(new_block_begin_thread_index_offset_vec[0]), new_template->block_begin_thread_index_offset, new_template->data_type_of_block_begin_thread_index_offset, new_template->size_of_block_begin_thread_index_offset);

    // block首个非零元的偏移
    new_template->data_type_of_block_nz_begin_offset = old_template->data_type_of_block_nz_begin_offset;
    new_template->size_of_block_nz_begin_offset = old_template->size_of_block_nz_begin_offset;
    assert(new_template->size_of_block_nz_begin_offset && new_template->size_of_block_begin_thread_index_offset - 1);
    new_template->block_nz_begin_offset = old_template->block_nz_begin_offset;

    // 每个block线程的数量
    new_template->data_type_of_thread_block_size_in_block = find_most_suitable_data_type(new_thread_block_size_in_block_vec[new_thread_block_size_in_block_vec.size() - 1]);
    assert(new_thread_block_size_in_block_vec.size() == old_template->size_of_block_nz_begin_offset);
    new_template->size_of_thread_block_size_in_block = new_thread_block_size_in_block_vec.size();
    new_template->thread_block_size_in_block = malloc_arr(new_thread_block_size_in_block_vec.size(), new_template->data_type_of_thread_block_size_in_block);
    // 拷贝
    copy_unsigned_long_arr_to_others(&(new_thread_block_size_in_block_vec[0]), new_template->thread_block_size_in_block, new_template->data_type_of_thread_block_size_in_block, new_template->size_of_thread_block_size_in_block);

    // 排序元数据的拷贝
    new_template->global_sort_index = old_template->global_sort_index;
    new_template->local_sort_index = old_template->local_sort_index;
    new_template->row_index_before_sort = old_template->row_index_before_sort;
    new_template->data_type_of_row_index_before_sort = old_template->data_type_of_row_index_before_sort;
    new_template->size_of_row_index_before_sort = old_template->size_of_row_index_before_sort;

    // 值数组
    new_template->data_type_of_val_arr = old_template->data_type_of_val_arr;
    new_template->size_of_val_arr = old_template->size_of_val_arr;
    new_template->val_arr = new_val_arr;

    // 列数组
    new_template->data_type_of_col_index_arr = old_template->data_type_of_col_index_arr;
    new_template->size_of_col_index_arr = old_template->size_of_col_index_arr;
    new_template->col_index_arr = new_col_index_arr;

    new_template->tblock_num = old_template->tblock_num;
    new_template->thread_num_in_block = old_template->thread_num_in_block;

    // 将旧数组删掉
    delete old_template;

    return new_template;
}

bool is_supported_by_direct_atom_template_warp_compress(sparse_struct_t* matrix, unsigned long dense_block_id)
{
    assert(dense_block_id < matrix->block_coor_table.item_arr.size());

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

    // 同一个block的TLB大小要保持一致
    for (unsigned long index_of_block_level_index = 0; index_of_block_level_index < block_level_index->block_num; index_of_block_level_index++)
    {
        // block中第一个warp号和下一个block的首warp
        unsigned long this_block_first_warp_index = read_from_array_with_data_type(block_level_index->index_arr, block_level_index->index_data_type, index_of_block_level_index);
        unsigned long next_block_first_warp_index = read_from_array_with_data_type(block_level_index->index_arr, block_level_index->index_data_type, index_of_block_level_index + 1);

        // 当前block的第一个warp的TLB大小
        unsigned long TLB_size_of_first_WLB_in_this_BLB = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, this_block_first_warp_index);

        // 遍历warp层次
        for (unsigned long index_of_warp_level_index = this_block_first_warp_index; index_of_warp_level_index < next_block_first_warp_index; index_of_warp_level_index++)
        {
            assert(index_of_warp_level_index < warp_level_index->block_num);

            // 当前warp的TLB大小
            unsigned long this_WLB_TLB_size = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, index_of_warp_level_index);

            // 和这个BLB内其他WLB的TLB大小做比较，如果不相等，说明没有办法使用这个模板
            if (this_WLB_TLB_size != TLB_size_of_first_WLB_in_this_BLB)
            {
                return false;
            } 
        }
    }

    return true;
}

bool is_supported_by_direct_atom_template_warp_compress(code_builder_t *builder, unsigned long dense_block_id)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;

    return is_supported_by_direct_atom_template_warp_compress(matrix, dense_block_id);
}

void store_template_data(direct_atom_template_warp_compress_t *output_template, string output_dir, bool force_not_share_global_sort_index)
{
    srand(time(0));
    unsigned long matrix_id = rand() + time(0) % 1000;

    // 写这个模板所需要数据的文件夹名称
    output_dir = output_dir + "/" + to_string(matrix_id) + "_" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer());

    // 创建这个文件夹
    system(("mkdir " + output_dir).c_str());

    // 只有不压缩的时候才持久化
    if (output_template->global_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_thread_level_block != NULL);
        print_arr_to_file_with_data_type(output_template->global_row_index_of_thread_level_block, output_template->data_type_of_global_row_index_of_thread_level_block, output_template->size_of_global_row_index_of_thread_level_block, output_dir + "/global_row_index_of_thread_level_block");
    }

    if (output_template->block_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_thread_index_offset != NULL);
        print_arr_to_file_with_data_type(output_template->block_begin_thread_index_offset, output_template->data_type_of_block_begin_thread_index_offset, output_template->size_of_block_begin_thread_index_offset, output_dir + "/block_begin_thread_index_offset");
    }

    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset != NULL);
        print_arr_to_file_with_data_type(output_template->block_nz_begin_offset, output_template->data_type_of_block_nz_begin_offset, output_template->size_of_block_nz_begin_offset, output_dir + "/block_nz_begin_offset");
    }

    if (output_template->thread_block_size_in_block_compress == NONE_COMPRESS)
    {
        assert(output_template->thread_block_size_in_block != NULL);
        print_arr_to_file_with_data_type(output_template->thread_block_size_in_block, output_template->data_type_of_thread_block_size_in_block, output_template->size_of_thread_block_size_in_block, output_dir + "/thread_block_size_in_block");
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

    // 值
    assert(output_template->val_arr != NULL);
    print_arr_to_file_with_data_type(output_template->val_arr, output_template->data_type_of_val_arr, output_template->size_of_val_arr, output_dir + "/val_arr");

    // 列
    assert(output_template->col_index_arr != NULL);
    print_arr_to_file_with_data_type(output_template->col_index_arr, output_template->data_type_of_col_index_arr, output_template->size_of_col_index_arr, output_dir + "/col_index_arr");

    output_template->hash_of_this_template = matrix_id;
}

string code_of_template_data_struct(direct_atom_template_warp_compress_t *output_template, unsigned long dense_block_id)
{
    // 创建一个数据结构
    string return_str = "typedef struct compressed_dense_block_" + to_string(dense_block_id) + "\n{\n";

    // 对应的位置分别存储行号和块号
    if (output_template->global_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_thread_level_block != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_global_row_index_of_thread_level_block, code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_thread_level_block"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_thread_level_block") + " = " + to_string(output_template->size_of_global_row_index_of_thread_level_block) + ";\n";
    }

    return_str = return_str + "\n";

    if (output_template->block_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_thread_index_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_block_begin_thread_index_offset, code_of_arr_var_name(dense_block_id, -1, "block_begin_thread_index_offset"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_thread_index_offset") + " = " + to_string(output_template->size_of_block_begin_thread_index_offset) + ";\n";
    }

    return_str = return_str + "\n";

    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_block_nz_begin_offset, code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset") + " = " + to_string(output_template->size_of_block_nz_begin_offset) + ";\n";
    }

    return_str = return_str + "\n";

    if (output_template->thread_block_size_in_block_compress == NONE_COMPRESS)
    {
        assert(output_template->thread_block_size_in_block != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_thread_block_size_in_block, code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_block"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_block") + " = " + to_string(output_template->size_of_thread_block_size_in_block) + ";\n";
    }

    return_str = return_str + "\n";

    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->row_index_before_sort != NULL)
    {
        assert(output_template->row_index_before_sort != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_row_index_before_sort, code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort") + " = " + to_string(output_template->size_of_row_index_before_sort) + ";\n";
    }

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

// 读取所有的内容的函数
string code_of_read_template_data_from_file_func_define(direct_atom_template_warp_compress_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index)
{
    string return_str = "compressed_dense_block_" + to_string(dense_block_id) + "_t* read_dense_block_" + to_string(dense_block_id) + "_from_file(string file_name_prefix)\n{\n";

    return_str = return_str + "compressed_dense_block_" + to_string(dense_block_id) + "_t *template_data = new " + "compressed_dense_block_" + to_string(dense_block_id) + "_t();\n";

    if (output_template->global_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_thread_level_block != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_thread_level_block") + " = (" + code_of_data_type(output_template->data_type_of_global_row_index_of_thread_level_block) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_thread_level_block") + ", " + convert_data_type_to_string(output_template->data_type_of_global_row_index_of_thread_level_block) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/global_row_index_of_thread_level_block\");\n";
    }

    return_str = return_str + "\n";

    if (output_template->block_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_thread_index_offset != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "block_begin_thread_index_offset") + " = (" + code_of_data_type(output_template->data_type_of_block_begin_thread_index_offset) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_thread_index_offset") + ", " + convert_data_type_to_string(output_template->data_type_of_block_begin_thread_index_offset) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/block_begin_thread_index_offset\");\n";
    }

    return_str = return_str + "\n";

    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset") + " = (" + code_of_data_type(output_template->data_type_of_block_nz_begin_offset) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset") + ", " + convert_data_type_to_string(output_template->data_type_of_block_nz_begin_offset) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/block_nz_begin_offset\");\n";
    }

    if (output_template->thread_block_size_in_block_compress == NONE_COMPRESS)
    {
        assert(output_template->thread_block_size_in_block != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_block") + " = (" + code_of_data_type(output_template->data_type_of_thread_block_size_in_block) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_block") + ", " + convert_data_type_to_string(output_template->data_type_of_thread_block_size_in_block) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/thread_block_size_in_block\");\n";
    }

    return_str = return_str + "\n";

    if (output_template->thread_block_size_in_block_compress == NONE_COMPRESS)
    {
        assert(output_template->thread_block_size_in_block != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_block") + " = (" + code_of_data_type(output_template->data_type_of_thread_block_size_in_block) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_block") + ", " + convert_data_type_to_string(output_template->data_type_of_thread_block_size_in_block) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/thread_block_size_in_block\");\n";
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

string code_of_write_template_data_to_gpu(direct_atom_template_warp_compress_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index)
{
    // 读到对应结构体中的代码
    // 存储结构体的名字
    string template_data_name = "dense_block_" + to_string(dense_block_id) + "_template_data";

    string return_str = "compressed_dense_block_" + to_string(dense_block_id) + "_t *" + template_data_name + " = read_dense_block_" + to_string(dense_block_id) + "_from_file(" + "\"" + string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source/" + to_string(output_template->hash_of_this_template) + "_" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer()) + "\");\n\n";

    // 全局排序的数组取一个特殊的名字，并且只处理一次，剩下的从这里拷贝即可
    // 如果是不是共享的这里就不需要
    if (output_template->global_sort_index == true && force_not_share_global_sort_index == false)
    {
        if (output_template->dense_block_index == 0)
        {
            return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_row_index_before_sort, "device_global_sort_index");
            // 申请、拷贝、一气呵成
            return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_row_index_before_sort, to_string(output_template->size_of_row_index_before_sort), "device_global_sort_index");
            return_str = return_str + code_line_of_cuda_memcpy("device_global_sort_index", template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"), output_template->data_type_of_row_index_before_sort, to_string(output_template->size_of_row_index_before_sort), "cudaMemcpyHostToDevice") + "\n";
        }
    }

    if (output_template->global_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_thread_level_block != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_global_row_index_of_thread_level_block, "device_" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_thread_level_block"));
    }

    if (output_template->block_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_thread_index_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_block_begin_thread_index_offset, "device_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_thread_index_offset"));
    }

    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_block_nz_begin_offset, "device_" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset"));
    }

    if (output_template->thread_block_size_in_block_compress == NONE_COMPRESS)
    {
        assert(output_template->thread_block_size_in_block != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_thread_block_size_in_block, "device_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_block"));
    }

    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->row_index_before_sort != NULL)
    {
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_row_index_before_sort, "device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"));
    }

    assert(output_template->val_arr != NULL);
    return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_val_arr, "device_" + code_of_arr_var_name(dense_block_id, -1, "val_arr"));

    assert(output_template->col_index_arr != NULL);
    return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_col_index_arr, "device_" + code_of_arr_var_name(dense_block_id, -1, "col_index_arr"));

    return_str = return_str + "\n";

    // 申请数组的代码
    if (output_template->global_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_thread_level_block != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_global_row_index_of_thread_level_block, to_string(output_template->size_of_global_row_index_of_thread_level_block), "device_" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_thread_level_block"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_thread_level_block"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_thread_level_block"), output_template->data_type_of_global_row_index_of_thread_level_block, to_string(output_template->size_of_global_row_index_of_thread_level_block), "cudaMemcpyHostToDevice") + "\n";
    }

    if (output_template->block_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_thread_index_offset != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_block_begin_thread_index_offset, to_string(output_template->size_of_block_begin_thread_index_offset), "device_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_thread_index_offset"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_thread_index_offset"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "block_begin_thread_index_offset"), output_template->data_type_of_block_begin_thread_index_offset, to_string(output_template->size_of_block_begin_thread_index_offset), "cudaMemcpyHostToDevice") + "\n";
    }

    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_block_nz_begin_offset, to_string(output_template->size_of_block_nz_begin_offset), "device_" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset"), output_template->data_type_of_block_nz_begin_offset, to_string(output_template->size_of_block_nz_begin_offset), "cudaMemcpyHostToDevice") + "\n";
    }

    if (output_template->thread_block_size_in_block_compress == NONE_COMPRESS)
    {
        assert(output_template->thread_block_size_in_block != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_thread_block_size_in_block, to_string(output_template->size_of_thread_block_size_in_block), "device_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_block"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_block"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_block"), output_template->data_type_of_thread_block_size_in_block, to_string(output_template->size_of_thread_block_size_in_block), "cudaMemcpyHostToDevice") + "\n";
    }

    // 如果是局部的就拷贝
    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->local_sort_index == true)
    {
        assert(output_template->global_sort_index == false && output_template->row_index_before_sort != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_row_index_before_sort, to_string(output_template->size_of_row_index_before_sort), "device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"), output_template->data_type_of_row_index_before_sort, to_string(output_template->size_of_row_index_before_sort), "cudaMemcpyHostToDevice") + "\n";
    }

    // 如果是全局的就直接赋值
    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->global_sort_index == true)
    {
        assert(output_template->local_sort_index == false);
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

string code_of_kernal_function_call(direct_atom_template_warp_compress_t *output_template, unsigned long dense_block_id)
{
    assert(output_template != NULL);
    // 线程块的数量和线程的数量不能超标
    assert(output_template->tblock_num <= get_config()["MAX_TBLOCK_NUM"].as_integer() && output_template->thread_num_in_block <= get_config()["MAX_THREAD_NUM_IN_BLOCK"].as_integer());

    string return_str = "spmv_" + to_string(dense_block_id) + "<<<" + to_string(output_template->tblock_num) + ", " + to_string(output_template->thread_num_in_block) + ", 0, stream_arr[" + to_string(dense_block_id) + "]>>>(";

    bool is_first_param = true;

    // 遍历所有的形参
    // 这里加入形参的声明
    if (output_template->global_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_thread_level_block != NULL);
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_thread_level_block");
        is_first_param = false;
    }

    if (output_template->block_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }
        assert(output_template->block_begin_thread_index_offset != NULL);
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_thread_index_offset");
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

    if (output_template->thread_block_size_in_block_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }
        assert(output_template->thread_block_size_in_block != NULL);
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_block");
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

string code_of_template_kernal(direct_atom_template_warp_compress_t *output_template, unsigned long dense_block_id)
{
    // 内核函数的声明
    string return_str = "__global__ void spmv_" + to_string(dense_block_id) + "(";

    // 用一个变量表明当前形参是不是第一个，如果是第一个就不用点逗号
    bool is_first_param = true;

    // 这里加入形参的声明
    if (output_template->global_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_thread_level_block != NULL);
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_global_row_index_of_thread_level_block, "* global_row_index_of_thread_level_block");
        is_first_param = false;
    }

    if (output_template->block_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }

        assert(output_template->block_begin_thread_index_offset != NULL);
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_block_begin_thread_index_offset, "* block_begin_thread_index_offset");
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

    if (output_template->thread_block_size_in_block_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }

        assert(output_template->thread_block_size_in_block != NULL);
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_thread_block_size_in_block, "* thread_block_size_in_block");
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

    return_str = return_str + "int bid = blockIdx.x;\nint tid_in_block = threadIdx.x;\n";

    if (output_template->size_of_thread_block_size_in_block > output_template->tblock_num)
    {
        return_str = return_str + "int bnum = gridDim.x;\n";
    }

    if (!(output_template->block_begin_thread_index_offset_compress == LINEAR_COMPRESS && ((linear_compress_t *)(output_template->block_begin_thread_index_offset_compress_meta))->coefficient == output_template->thread_num_in_block))
    {
        return_str = return_str + "int thread_num_in_block = blockDim.x;\n";
    }

    if (output_template->kernal_first_col_index != 0)
    {
        return_str = return_str + "unsigned long kernal_first_col_index = " + to_string(output_template->kernal_first_col_index) + ";\n";
    }

    if (output_template->kernal_first_row_index != 0)
    {
        return_str = return_str + "unsigned long kernal_first_row_index = " + to_string(output_template->kernal_first_row_index) + ";\n";
    }

    // 判断是否需要
    bool need_first_thread_index_of_next_block = true;
    if (output_template->block_begin_thread_index_offset_compress == LINEAR_COMPRESS && ((linear_compress_t *)(output_template->block_begin_thread_index_offset_compress_meta))->coefficient == output_template->thread_num_in_block)
    {
        need_first_thread_index_of_next_block = false;
    }

    // 根据是否压缩决定是否申请共享内存
    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset != NULL);
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_nz_begin_offset) + " block_first_nz_index_shared[1];\n";
    }

    if (output_template->block_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_thread_index_offset != NULL);
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_begin_thread_index_offset) + " first_thread_index_of_this_block_shared[1];\n";

        if (need_first_thread_index_of_next_block == true)
        {
            return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_begin_thread_index_offset) + " first_thread_index_of_next_block_shared[1];\n";
        }
    }

    if (output_template->thread_block_size_in_block_compress == NONE_COMPRESS)
    {
        assert(output_template->thread_block_size_in_block != NULL);
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_thread_block_size_in_block) + " thread_block_size_in_block_shared[1];\n";
    }

    // 分配的线程块数量决定第一层循环的写法
    if (output_template->size_of_thread_block_size_in_block > output_template->tblock_num)
    {
        // for循环的方法
        return_str = return_str + "for(";
        return_str = return_str + "unsigned int" + " block_level_block_id = bid; block_level_block_id < ";
        return_str = return_str + to_string(output_template->size_of_thread_block_size_in_block) + "; block_level_block_id = block_level_block_id + bnum)\n{\n";
    }
    else if (output_template->size_of_thread_block_size_in_block == output_template->tblock_num)
    {
        // 不加任何方法
        return_str = return_str + "{\n";
        return_str = return_str + "unsigned int" + " block_level_block_id = bid;\n";
    }
    else if (output_template->size_of_thread_block_size_in_block < output_template->tblock_num)
    {
        return_str = return_str + "if(";
        return_str = return_str + "bid < " + to_string(output_template->size_of_thread_block_size_in_block) + ")\n{\n";
        return_str = return_str + "unsigned int" + " block_level_block_id = bid;\n";
    }
    else
    {
        cout << "error" << endl;
        assert(false);
    }

    return_str = return_str + "\n";

    // 当前block的几个需要的元数据
    return_str = return_str + code_of_data_type(output_template->data_type_of_block_nz_begin_offset) + " block_first_nz_of_this_block;\n";
    return_str = return_str + code_of_data_type(output_template->data_type_of_block_begin_thread_index_offset) + " first_thread_index_of_this_block;\n";

    if (need_first_thread_index_of_next_block == true)
    {
        return_str = return_str + code_of_data_type(output_template->data_type_of_block_begin_thread_index_offset) + " first_thread_index_of_next_block;\n";
    }

    return_str = return_str + code_of_data_type(output_template->data_type_of_thread_block_size_in_block) + " thread_block_size_of_this_block;\n";

    return_str = return_str + "\n";

    // 只要有一个没有压缩就需要这个
    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS || output_template->block_begin_thread_index_offset_compress == NONE_COMPRESS || output_template->thread_block_size_in_block_compress == NONE_COMPRESS)
    {
        return_str = return_str + "__syncthreads();\n\n";
        // 0号线程读所有元数据
        return_str = return_str + "if(tid_in_block == 0)\n{\n";

        if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
        {
            assert(output_template->block_nz_begin_offset != NULL);
            return_str = return_str + "block_first_nz_index_shared[0] = block_nz_begin_offset[block_level_block_id];\n";
        }

        if (output_template->block_begin_thread_index_offset_compress == NONE_COMPRESS)
        {
            assert(output_template->block_begin_thread_index_offset != NULL);
            return_str = return_str + "first_thread_index_of_this_block_shared[0] = block_begin_thread_index_offset[block_level_block_id];\n";

            if (need_first_thread_index_of_next_block == true)
            {
                return_str = return_str + "first_thread_index_of_next_block_shared[0] = block_begin_thread_index_offset[block_level_block_id + 1];\n";
            }
        }

        if (output_template->thread_block_size_in_block_compress == NONE_COMPRESS)
        {
            assert(output_template->thread_block_size_in_block != NULL);
            return_str = return_str + "thread_block_size_in_block_shared[0] = thread_block_size_in_block[block_level_block_id];\n";
        }

        return_str = return_str + "}\n\n";

        return_str = return_str + "__syncthreads();\n\n";
    }

    // 根据压缩情况给这些元数据赋值
    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        return_str = return_str + "block_first_nz_of_this_block = block_first_nz_index_shared[0];\n";
    }
    else if (output_template->block_nz_begin_offset_compress == LINEAR_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset_compress_meta != NULL);
        linear_compress_t *compressor = (linear_compress_t *)output_template->block_nz_begin_offset_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "block_first_nz_of_this_block", "block_level_block_id") + ";\n";
    }
    else
    {
        cout << "compress type is not supported" << endl;
        assert(false);
    }

    if (output_template->block_begin_thread_index_offset_compress == NONE_COMPRESS)
    {
        return_str = return_str + "first_thread_index_of_this_block = first_thread_index_of_this_block_shared[0];\n";

        if (need_first_thread_index_of_next_block == true)
        {
            return_str = return_str + "first_thread_index_of_next_block = first_thread_index_of_next_block_shared[0];\n";
        }
    }
    else if (output_template->block_begin_thread_index_offset_compress == LINEAR_COMPRESS)
    {
        assert(output_template->block_begin_thread_index_offset_compress_meta != NULL);
        linear_compress_t *compressor = (linear_compress_t *)output_template->block_begin_thread_index_offset_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "first_thread_index_of_this_block", "block_level_block_id") + ";\n";

        // 这里直接使用加法，减少计算量
        if (need_first_thread_index_of_next_block == true)
        {
            return_str = return_str + "first_thread_index_of_next_block = first_thread_index_of_this_block + " + to_string(compressor->coefficient) + ";\n";
        }
    }
    else
    {
        cout << "compress type is not supported" << endl;
        assert(false);
    }

    if (output_template->thread_block_size_in_block_compress == NONE_COMPRESS)
    {
        return_str = return_str + "thread_block_size_of_this_block = thread_block_size_in_block_shared[0];\n";
    }
    else if (output_template->thread_block_size_in_block_compress == BRANCH_COMPRESS)
    {
        assert(output_template->thread_block_size_in_block_compress_meta != NULL);
        branch_compress_t *compressor = (branch_compress_t *)output_template->thread_block_size_in_block_compress_meta;
        return_str = return_str + "\n" + code_of_arr_read(compressor, "thread_block_size_of_this_block", "block_level_block_id") + "\n";
    }
    else if (output_template->thread_block_size_in_block_compress == CONSTANT_COMPRESS)
    {
        assert(output_template->thread_block_size_in_block_compress_meta != NULL);
        constant_compress_t *compressor = (constant_compress_t *)output_template->thread_block_size_in_block_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "thread_block_size_of_this_block", "block_level_block_id") + ";\n";
    }
    else
    {
        cout << "compress type is not supported" << endl;
        assert(false);
    }

    return_str = return_str + "\n";

    if (output_template->block_begin_thread_index_offset_compress == LINEAR_COMPRESS)
    {
        assert(output_template->block_begin_thread_index_offset_compress_meta != NULL);
        linear_compress_t *compressor = (linear_compress_t *)output_template->block_begin_thread_index_offset_compress_meta;
        return_str = return_str + code_of_data_type(output_template->data_type_of_block_begin_thread_index_offset) + " thread_level_block_num_of_block = " + to_string(compressor->coefficient) + ";\n";
    }
    else
    {
        // 线程块粒度的块的数量，如果是线性压缩就直接给一个常值
        assert(need_first_thread_index_of_next_block == true);
        return_str = return_str + code_of_data_type(output_template->data_type_of_block_begin_thread_index_offset) + " thread_level_block_num_of_block = first_thread_index_of_next_block - first_thread_index_of_this_block;\n";
    }

    return_str = return_str + "\n";

    // 根据线程数量和线程块数量之间的关系，使用不同的遍历方法，如果每个块中线程的数量相等，并且等于被分配的线程数量的时候，就不需要for循环
    if (output_template->block_begin_thread_index_offset_compress == LINEAR_COMPRESS && ((linear_compress_t *)(output_template->block_begin_thread_index_offset_compress_meta))->coefficient == output_template->thread_num_in_block)
    {
        return_str = return_str + "unsigned int" + " thread_level_block_id = first_thread_index_of_this_block + tid_in_block;\n";
        
        if (output_template->effective_TLB_num != output_template->size_of_global_row_index_of_thread_level_block)
        {
            assert(output_template->effective_TLB_num < output_template->size_of_global_row_index_of_thread_level_block);
            // 这里代表了有一部分TLB是无效的
            return_str = return_str + "if (thread_level_block_id < " + to_string(output_template->effective_TLB_num) + ")\n{\n";
        }
        else
        {
            return_str = return_str + "{\n";
        }

        return_str = return_str + "unsigned int" + " thread_level_block_first_nz_index = block_first_nz_of_this_block + tid_in_block;\n";
        return_str = return_str + "\n";
    }
    else
    {
        assert(need_first_thread_index_of_next_block == true);
        return_str = return_str + "for(";
        return_str = return_str + "unsigned int" + " thread_level_block_id = first_thread_index_of_this_block + tid_in_block;";
        // 因为只有一部分TLB是有效的如果TLB是无效的就不需要进一步处理
        return_str = return_str + "thread_level_block_id < first_thread_index_of_next_block && thread_level_block_id < " + to_string(output_template->effective_TLB_num) + "; thread_level_block_id = thread_level_block_id + thread_num_in_block)\n{\n";
        return_str = return_str + "\n";
        return_str = return_str + "unsigned int" + " thread_level_block_first_nz_index = block_first_nz_of_this_block + thread_level_block_id - first_thread_index_of_this_block;\n";
        return_str = return_str + "\n";
    }

    return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " thread_block_tmp_result = 0;\n";
    return_str = return_str + "\n";

    // 根据每个线程非零元数量的大小来选择不同的实现
    if (output_template->thread_block_size_in_block_compress == CONSTANT_COMPRESS && ((constant_compress_t *)(output_template->thread_block_size_in_block_compress_meta))->constant == 1)
    {
        // 算列号的时候需要加上一个kernal的列偏移量
        if (output_template->kernal_first_col_index == 0)
        {
            // 不带列偏移量
            return_str = return_str + "thread_block_tmp_result = val_arr[thread_level_block_first_nz_index] * device_x_arr[col_index_arr[thread_level_block_first_nz_index]];\n";
        }
        else
        {
            // 带列偏移量
            return_str = return_str + "thread_block_tmp_result = val_arr[thread_level_block_first_nz_index] * device_x_arr[kernal_first_col_index + col_index_arr[thread_level_block_first_nz_index]];\n";
        }
    }
    else
    {
        // 一开始在外面申请，全局global的大小，
        return_str = return_str + "unsigned int" + " global_nz_index = thread_level_block_first_nz_index;\n";

        return_str = return_str + "for(";
        return_str = return_str + "unsigned int" + " inner_thread_nz_level_id = 0; inner_thread_nz_level_id < thread_block_size_of_this_block; inner_thread_nz_level_id++)\n{\n";

        // 算列号的时候需要加上一个kernal的列偏移量
        if (output_template->kernal_first_col_index == 0)
        {
            // 不带列偏移量
            return_str = return_str + "thread_block_tmp_result = thread_block_tmp_result + val_arr[global_nz_index] * __ldg(&(device_x_arr[col_index_arr[global_nz_index]]));\n";
        }
        else
        {
            // 带列偏移量
            return_str = return_str + "thread_block_tmp_result = thread_block_tmp_result + val_arr[global_nz_index] * __ldg(&(device_x_arr[kernal_first_col_index + col_index_arr[global_nz_index]]));\n";
        }

        // 增加全局非零元索引的自增
        return_str = return_str + "global_nz_index = global_nz_index + thread_level_block_num_of_block;\n";

        return_str = return_str + "}\n";
    }

    return_str = return_str + "\n";

    // 归约
    // 获取局部的行号
    return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->matrix->dense_row_number)) + " global_row_index;\n";

    return_str = return_str + "\n";

    // 每个线程粒度的行号
    if (output_template->global_row_index_compress == NONE_COMPRESS)
    {
        return_str = return_str + "global_row_index = global_row_index_of_thread_level_block[thread_level_block_id];\n";
    }
    else if (output_template->global_row_index_compress == LINEAR_COMPRESS)
    {
        assert(output_template->global_row_index_compress_meta != NULL);

        linear_compress_t *compressor = (linear_compress_t *)output_template->global_row_index_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "global_row_index", "thread_level_block_id") + ";\n";
    }
    else if (output_template->global_row_index_compress == CYCLE_INCREASE_COMPRESS)
    {
        assert(output_template->global_row_index_compress_meta != NULL);
        cycle_increase_compress_t *compressor = (cycle_increase_compress_t *)output_template->global_row_index_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "global_row_index", "thread_level_block_id") + ";\n";
    }
    else
    {
        cout << "compress type is not supported,global_row_index_compress" << endl;
        assert(false);
    }

    // 根据排序的情况来修改为真实的行号
    // 局部排序
    if (output_template->local_sort_index == true)
    {
        assert(output_template->global_sort_index == false);
        // 获取真实的行索引
        // if (output_template->kernal_first_row_index != 0)
        // {
        //     return_str = return_str + "global_row_index = row_index_before_sort[global_row_index] + kernal_first_row_index;\n";
        // }
        // else
        // {
        //     return_str = return_str + "global_row_index = row_index_before_sort[global_row_index];\n";
        // }

        if (output_template->kernal_first_row_index != 0)
        {
            return_str = return_str + "global_row_index = global_row_index + kernal_first_row_index;\n";
        }
        else
        {
            // return_str = return_str + "global_row_index = global_row_index;\n";
        }
    }

    // 全局排序
    if (output_template->global_sort_index == true)
    {
        // assert(output_template->local_sort_index == false);

        // if (output_template->kernal_first_row_index != 0)
        // {
        //     return_str = return_str + "global_row_index = row_index_before_sort[global_row_index + kernal_first_row_index];\n";
        // }
        // else
        // {
        //     return_str = return_str + "global_row_index = row_index_before_sort[global_row_index];\n";
        // }
    }

    // 如果不排序
    if (output_template->local_sort_index == false && output_template->global_sort_index == false)
    {
        if (output_template->kernal_first_row_index != 0)
        {
            return_str = return_str + "global_row_index = global_row_index + kernal_first_row_index;\n";
        }
        else
        {
            // return_str = return_str + "global_row_index = global_row_index;\n";
        }
    }

    // 原子加
    if (output_template->is_atom_add == true)
    {
        // 原子加
        return_str = return_str + "atomicAdd(&(device_y_arr[global_row_index]), thread_block_tmp_result);\n";
    }
    else
    {
        // 赋值
        return_str = return_str + "device_y_arr[global_row_index] = thread_block_tmp_result;\n";
    }

    return_str = return_str + "\n}\n";
    return_str = return_str + "\n}\n";
    return_str = return_str + "\n}\n";

    return return_str;
}

bool compress_global_row_index_of_thread_level_block(direct_atom_template_warp_compress_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && output_template->global_row_index_of_thread_level_block != NULL);
    assert(type == CYCLE_INCREASE_COMPRESS || type == LINEAR_COMPRESS);

    if (type == CYCLE_INCREASE_COMPRESS)
    {
        cycle_increase_compress_t *compressor = init_cycle_increase_compressor(output_template->global_row_index_of_thread_level_block, output_template->data_type_of_global_row_index_of_thread_level_block, output_template->size_of_global_row_index_of_thread_level_block, need_check);

        if (compressor == NULL)
        {
            return false;
        }

        // 压缩成功
        // 压缩成功，拷贝元数据
        output_template->global_row_index_compress_meta = (void *)compressor;
        output_template->global_row_index_compress = type;

        return true;
    }

    if (type == LINEAR_COMPRESS)
    {
        linear_compress_t *compressor = init_linear_compressor(output_template->global_row_index_of_thread_level_block, output_template->data_type_of_global_row_index_of_thread_level_block, output_template->size_of_global_row_index_of_thread_level_block, need_check);

        // 查看能否成功压缩
        if (compressor == NULL)
        {
            // 不成功
            return false;
        }

        // 压缩成功，拷贝元数据
        output_template->global_row_index_compress_meta = (void *)compressor;
        output_template->global_row_index_compress = type;

        return true;
    }

    return false;
}

bool compress_block_begin_thread_index_offset(direct_atom_template_warp_compress_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && type == LINEAR_COMPRESS && output_template->block_begin_thread_index_offset != NULL);

    linear_compress_t *compressor = init_linear_compressor(output_template->block_begin_thread_index_offset, output_template->data_type_of_block_begin_thread_index_offset, output_template->size_of_block_begin_thread_index_offset, need_check);

    if (compressor == NULL)
    {
        return false;
    }

    // 压缩成功，拷贝元数据
    output_template->block_begin_thread_index_offset_compress_meta = (void *)compressor;
    output_template->block_begin_thread_index_offset_compress = type;

    return true;
}

bool compress_thread_block_size_in_block(direct_atom_template_warp_compress_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && output_template->thread_block_size_in_block != NULL);
    assert(type == CONSTANT_COMPRESS || type == BRANCH_COMPRESS);

    if (type == CONSTANT_COMPRESS)
    {
        constant_compress_t *compressor = init_constant_compressor(output_template->thread_block_size_in_block, output_template->data_type_of_thread_block_size_in_block, output_template->size_of_thread_block_size_in_block, need_check);

        if (compressor == NULL)
        {
            return false;
        }

        // 压缩成功
        output_template->thread_block_size_in_block_compress_meta = (void *)compressor;
        output_template->thread_block_size_in_block_compress = type;
    }

    if (type == BRANCH_COMPRESS)
    {
        branch_compress_t *compressor = init_branch_compressor(output_template->thread_block_size_in_block, output_template->data_type_of_thread_block_size_in_block, output_template->size_of_thread_block_size_in_block, need_check);

        if (compressor == NULL)
        {
            return false;
        }

        // 压缩成功
        output_template->thread_block_size_in_block_compress_meta = (void *)compressor;
        output_template->thread_block_size_in_block_compress = type;
    }

    return true;
}

bool compress_block_nz_begin_offset(direct_atom_template_warp_compress_t *output_template, bool need_check, arr_compress_type type)
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

void try_all_compress(direct_atom_template_warp_compress_t *output_template)
{
    bool is_compressed = false;

    // 压缩compress_global_row_index_of_thread_level_block
    is_compressed = compress_global_row_index_of_thread_level_block(output_template, true, LINEAR_COMPRESS);

    // 如果没法被压缩，就试试其他压缩方式
    if (is_compressed == false)
    {
        is_compressed = compress_global_row_index_of_thread_level_block(output_template, true, CYCLE_INCREASE_COMPRESS);
    }

    is_compressed = compress_block_begin_thread_index_offset(output_template, true, LINEAR_COMPRESS);

    // 压缩线程粒度的块的大小
    is_compressed = compress_thread_block_size_in_block(output_template, true, CONSTANT_COMPRESS);

    if (is_compressed == false)
    {
        is_compressed = compress_thread_block_size_in_block(output_template, true, BRANCH_COMPRESS);
    }

    is_compressed = compress_block_nz_begin_offset(output_template, true, LINEAR_COMPRESS);
}