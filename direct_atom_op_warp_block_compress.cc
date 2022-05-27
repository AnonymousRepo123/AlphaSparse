#include "direct_atom_op_warp_block_compress.hpp"
#include <ctime>
#include <vector>

using namespace std;

// 条件是要求所有warp内的线程粒度的块的数量相等，直接生成一个压缩之后的模板
direct_atom_template_warp_block_compress_t *init_direct_atom_template_warp_block_compress(code_builder_t *builder, unsigned long dense_block_id)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;

    // 首先判断所有warp的中的TLB非零元数量是不是相等，只有相等才能进一步生成模板
    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr;
    
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

    bool need_atom_add = false;

    // 遍历thread的首行地址和所占的行的数量，用来判断输入的矩阵是不是适合这个模板，并且需不需要原子加
    if (matrix->block_coor_table.item_arr[dense_block_id]->min_dense_col_index == 0 && matrix->block_coor_table.item_arr[dense_block_id]->max_dense_col_index == matrix->dense_col_number - 1)
    {
        // 稠密子块之间没有共享的行
    }
    else
    {
        need_atom_add = true;
    }
    
    assert(thread_level_index->coo_block_size_arr != NULL);

    if (thread_level_index->row_number_of_block_arr != NULL)
    {
        cout << "init_direct_atom_template_warp_block_compress: thread_level_index->row_number_of_block_arr must be NULL, row num in thread level block must be 1" << endl;
        assert(false);
    }

    assert(thread_level_index->index_of_the_first_row_arr != NULL);

    // 每个thread的全局行索引
    vector<unsigned long> global_thread_row_index_vec;

    // 遍历thread层次的块大小
    for (unsigned long WLB_id = 0; WLB_id < warp_level_index->block_num - 1; WLB_id++)
    {
        // 当前warp的TLB非零元数量和之后的做比较
        unsigned long cur_WLB_thread_block_size = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, WLB_id);
        unsigned long next_WLB_thread_block_size = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, WLB_id + 1);

        if (cur_WLB_thread_block_size != next_WLB_thread_block_size)
        {
            cout << "init_direct_atom_template_warp_block_compress: can not compress in block " << WLB_id << " because thread level block size is not the same" << endl;
            assert(false);
        }
    }

    // 遍历三个层次，获得每个TLB的全局行号
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
    }

    assert(global_thread_row_index_vec.size() == thread_level_index->block_num);

    // 以全局为粒度，生成一个规模巨大的padding
    void *val_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->padding_val_arr;
    assert(val_arr_after_padding != NULL);
    data_type data_type_of_val_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->val_data_type;
    unsigned long size_of_val_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->padding_arr_size;

    void *col_index_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[5]->index_arr;
    assert(col_index_arr_after_padding != NULL);
    data_type data_type_of_col_index_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[5]->index_data_type;
    unsigned long size_of_col_index_arr_after_padding = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[5]->length;

    assert(size_of_val_arr_after_padding == size_of_col_index_arr_after_padding);

    // 执行padding，在全局范围内执行一个巨大padding操作，为此需要申请两个新的数组
    void *new_col_index_arr = malloc_arr(size_of_col_index_arr_after_padding, data_type_of_col_index_arr_after_padding);
    void *new_val_arr = malloc_arr(size_of_val_arr_after_padding, data_type_of_val_arr_after_padding);

    // 数组的总大小是线程粒度的块的大小的整数倍
    unsigned long TLB_nnz = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, 0);
    assert(size_of_val_arr_after_padding % TLB_nnz == 0);

    // 一共有的块的数量
    unsigned long total_thread_level_block_num = size_of_val_arr_after_padding / TLB_nnz;

    for (unsigned long i = 0; i < size_of_val_arr_after_padding; i++)
    {
        // 获取当前非零元的thread粒度块的编号
        unsigned long thread_level_block_index = i / TLB_nnz;
        // 非零元的线程粒度的块内的索引
        unsigned long index_inner_thread_level_block = i % TLB_nnz;
        // 获取输出位置
        unsigned long dest_index_of_this_nz = thread_level_block_index + index_inner_thread_level_block * total_thread_level_block_num;

        // 将数据读出来
        unsigned long col_index = read_from_array_with_data_type(col_index_arr_after_padding, data_type_of_col_index_arr_after_padding, i);
        double val = read_double_from_array_with_data_type(val_arr_after_padding, data_type_of_val_arr_after_padding, i);

        // 将数据写到对应的目标位置
        write_to_array_with_data_type(new_col_index_arr, data_type_of_col_index_arr_after_padding, dest_index_of_this_nz, col_index);
        write_double_to_array_with_data_type(new_val_arr, data_type_of_val_arr_after_padding, dest_index_of_this_nz, val);
    }

    direct_atom_template_warp_block_compress_t *new_template = new direct_atom_template_warp_block_compress_t();

    new_template->dense_block_index = dense_block_id;
    new_template->matrix = matrix;

    new_template->kernal_first_row_index = matrix->block_coor_table.item_arr[dense_block_id]->min_dense_row_index;
    new_template->kernal_first_col_index = matrix->block_coor_table.item_arr[dense_block_id]->min_dense_col_index;

    if (matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index < compressed_block_view->read_index[0]->max_row_index)
    {
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
     

    // 是否需要原子性操作
    new_template->is_atom_add = need_atom_add;
    
    new_template->thread_block_size_in_block = TLB_nnz;

    // 排序产生的行索引
    // 最后给出排序索引类型和具体的数组
    // 直接将排序数组和行索引数组合并在一起。直接在global_thread_row_index_vec存储排序之前的位置
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
            unsigned long row_index_before_sort = read_from_array_with_data_type(new_template->row_index_before_sort, new_template->data_type_of_row_index_before_sort, cur_row_index);
            // 重置索引
            global_thread_row_index_vec[row_index_id] = row_index_before_sort;
        }
    }
    else if (matrix->sorted_row_index != NULL)
    {
        cout << "init_direct_atom_template_warp_block_compress: have global sort" << endl;
        // 在全局范围内有排序
        assert(compressed_block_view->is_sorted == false && matrix->is_sorted == true && builder->sub_block_sort_type_vec[dense_block_id] == GLOBAL_SORT);
        new_template->global_sort_index = true;
        new_template->local_sort_index = false;

        // 拷贝
        new_template->data_type_of_row_index_before_sort = matrix->data_type_of_sorted_row_index;
        new_template->row_index_before_sort = matrix->sorted_row_index;
        new_template->size_of_row_index_before_sort = matrix->dense_row_number;

        // 找出原本的索引
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

    // 每一行的行号
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

    // 返回当前模板
    return new_template;
}

// 在自动调优过程中不允许被使用，会给内存回收造成混乱
direct_atom_template_warp_block_compress_t *init_direct_atom_template_warp_block_compress(direct_atom_template_t *old_template)
{
    cout << "init_direct_atom_template_warp_block_compress: is not supported, old API" << endl;
    assert(false);
    
    // 检查所有warp的线程粒度的块是否相等
    assert(old_template != NULL);

    // 遍历所有block，查看所有warp内的threadsize是不是相等
    assert(old_template->thread_block_size_in_warp != NULL);
    unsigned long thread_level_size_in_first_warp = read_from_array_with_data_type(old_template->thread_block_size_in_warp, old_template->data_type_of_thread_block_size_in_warp, 0);
    for (unsigned long warp_index = 0; warp_index < old_template->size_of_thread_block_size_in_warp; warp_index++)
    {
        unsigned long thread_level_size_in_this_warp = read_from_array_with_data_type(old_template->thread_block_size_in_warp, old_template->data_type_of_thread_block_size_in_warp, warp_index);

        if (thread_level_size_in_this_warp != thread_level_size_in_first_warp)
        {
            cout << "can not compress in block " << warp_index << " because thread level block size is not the same" << endl;
            assert(false);
        }
    }

    // 重新执行一个规模巨大的padding，以全局为粒度
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

    // 每个线程负责的非零元数量是完全一致的
    // 这里做一个交错存储，每一个thread中的元素交错
    assert(old_template->size_of_val_arr % thread_level_size_in_first_warp == 0);

    // 一共有的块的数量
    unsigned long total_thread_level_block_num = old_template->size_of_val_arr / thread_level_size_in_first_warp;

    for (unsigned long i = 0; i < old_template->size_of_val_arr; i++)
    {
        // 获取当前非零元的thread粒度块的编号
        unsigned long thread_level_block_index = i / thread_level_size_in_first_warp;
        // 非零元的线程粒度的块内的索引
        unsigned long index_inner_thread_level_block = i % thread_level_size_in_first_warp;
        // 获取输出位置
        unsigned long dest_index_of_this_nz = thread_level_block_index + index_inner_thread_level_block * total_thread_level_block_num;

        // 将数据读出来
        unsigned long col_index = read_from_array_with_data_type(col_index_arr_after_padding, data_type_of_col_index_arr_after_padding, i);
        double val = read_double_from_array_with_data_type(val_arr_after_padding, data_type_of_val_arr_after_padding, i);

        // 将数据写到对应的目标位置
        write_to_array_with_data_type(new_col_index_arr, data_type_of_col_index_arr_after_padding, dest_index_of_this_nz, col_index);
        write_double_to_array_with_data_type(new_val_arr, data_type_of_val_arr_after_padding, dest_index_of_this_nz, val);
    }

    // 析构block层次和warp层次的元数据，列索引和值索引不用不用考虑，不是拷贝出来的
    delete_arr_with_data_type(old_template->block_begin_warp_index_offset, old_template->data_type_of_block_begin_warp_index_offset);
    delete_arr_with_data_type(old_template->block_nz_begin_offset, old_template->data_type_of_block_nz_begin_offset);
    delete_arr_with_data_type(old_template->warp_begin_thread_index_offset, old_template->data_type_of_warp_begin_thread_index_offset);
    delete_arr_with_data_type(old_template->warp_nz_begin_offset, old_template->data_type_of_warp_nz_begin_offset);
    delete_arr_with_data_type(old_template->thread_block_size_in_warp, old_template->data_type_of_thread_block_size_in_warp);
    old_template->block_begin_warp_index_offset = NULL;
    old_template->block_nz_begin_offset = NULL;
    old_template->warp_begin_thread_index_offset = NULL;
    old_template->warp_nz_begin_offset = NULL;
    old_template->thread_block_size_in_warp = NULL;

    // 申请新的模板结构体
    direct_atom_template_warp_block_compress_t *new_template = new direct_atom_template_warp_block_compress_t();
    new_template->dense_block_index = old_template->dense_block_index;
    new_template->matrix = old_template->matrix;
    new_template->kernal_first_row_index = old_template->kernal_first_row_index;
    new_template->kernal_first_col_index = old_template->kernal_first_col_index;

    new_template->is_atom_add = old_template->is_atom_add;

    new_template->global_row_index_of_thread_level_block = old_template->global_row_index_of_thread_level_block;
    new_template->data_type_of_global_row_index_of_thread_level_block = old_template->data_type_of_global_row_index_of_thread_level_block;
    new_template->size_of_global_row_index_of_thread_level_block = old_template->size_of_global_row_index_of_thread_level_block;

    new_template->thread_block_size_in_block = thread_level_size_in_first_warp;

    new_template->global_sort_index = old_template->global_sort_index;
    new_template->local_sort_index = old_template->local_sort_index;
    new_template->row_index_before_sort = old_template->row_index_before_sort;
    new_template->data_type_of_row_index_before_sort = old_template->data_type_of_row_index_before_sort;
    new_template->size_of_row_index_before_sort = old_template->size_of_row_index_before_sort;

    new_template->val_arr = new_val_arr;
    new_template->data_type_of_val_arr = data_type_of_val_arr_after_padding;
    new_template->size_of_val_arr = size_of_val_arr_after_padding;

    new_template->col_index_arr = new_col_index_arr;
    new_template->data_type_of_col_index_arr = data_type_of_col_index_arr_after_padding;
    new_template->size_of_col_index_arr = size_of_col_index_arr_after_padding;

    new_template->global_row_index_compress = old_template->global_row_index_compress;
    new_template->global_row_index_compress_meta = old_template->global_row_index_compress_meta;

    new_template->row_index_before_sort_compress = old_template->row_index_before_sort_compress;
    new_template->row_index_before_sort_compress_meta = old_template->row_index_before_sort_compress_meta;

    new_template->tblock_num = old_template->tblock_num;
    new_template->thread_num_in_block = old_template->thread_num_in_block;

    return new_template;
}

bool is_supported_by_direct_atom_template_warp_block_compress(sparse_struct_t* matrix, unsigned long dense_block_id)
{
    assert(matrix != NULL);

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

    // 所有TLB的非零元数量全部相同
    for (unsigned long WLB_id = 0; WLB_id < warp_level_index->block_num - 1; WLB_id++)
    {
        // 当前warp的TLB非零元数量和之后的做比较
        unsigned long cur_WLB_thread_block_size = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, WLB_id);
        unsigned long next_WLB_thread_block_size = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, WLB_id + 1);

        if (cur_WLB_thread_block_size != next_WLB_thread_block_size)
        {
            return false;
        }
    }

    return true;
}

bool is_supported_by_direct_atom_template_warp_block_compress(code_builder_t *builder, unsigned long dense_block_id)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;

    return is_supported_by_direct_atom_template_warp_block_compress(matrix, dense_block_id);
}

// 打印数组中的内容
void store_template_data(direct_atom_template_warp_block_compress_t *output_template, string output_dir, bool force_not_share_global_sort_index)
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

string code_of_template_data_struct(direct_atom_template_warp_block_compress_t *output_template, unsigned long dense_block_id)
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

string code_of_read_template_data_from_file_func_define(direct_atom_template_warp_block_compress_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index)
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

string code_of_write_template_data_to_gpu(direct_atom_template_warp_block_compress_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index)
{
    // 读到对应结构体中的代码
    // 存储结构体的名字
    string template_data_name = "dense_block_" + to_string(dense_block_id) + "_template_data";

    string return_str = "compressed_dense_block_" + to_string(dense_block_id) + "_t *" + template_data_name + " = read_dense_block_" + to_string(dense_block_id) + "_from_file(" + "\"" + string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source/" + to_string(output_template->hash_of_this_template) + "_" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer()) + "\");\n\n";

    // 全局排序的数组取一个特殊的名字，并且只处理一次，剩下的从这里拷贝即可，但是如果不共享就不需要这个过程
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

    // 如果是局部的就拷贝
    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->local_sort_index == true)
    {
        assert(output_template->global_sort_index == false && output_template->row_index_before_sort != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_row_index_before_sort, to_string(output_template->size_of_row_index_before_sort), "device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"), output_template->data_type_of_row_index_before_sort, to_string(output_template->size_of_row_index_before_sort), "cudaMemcpyHostToDevice") + "\n";
    }

    // 如果是全局的就直接赋值，如果不共享还是要从全局内存中老实获取
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

string code_of_kernal_function_call(direct_atom_template_warp_block_compress_t *output_template, unsigned long dense_block_id)
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

string code_of_template_kernal(direct_atom_template_warp_block_compress_t *output_template, unsigned long dense_block_id)
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

    return_str = return_str + "int global_tid = blockDim.x * blockIdx.x + threadIdx.x;\n";

    if (output_template->kernal_first_row_index != 0)
    {
        return_str = return_str + "int kernal_first_row_index = " + to_string(output_template->kernal_first_row_index) + ";\n";
    }

    if (output_template->kernal_first_col_index != 0)
    {
        return_str = return_str + "int kernal_first_col_index = " + to_string(output_template->kernal_first_col_index) + ";\n";
    }

    
    return_str = return_str + "int total_thread_num = blockDim.x * gridDim.x;\n";
    

    // 存储thread粒度的块号的变量名
    string thread_block_id_var_name;

    // 如果thread粒度的块的数量和实际被分配的线程的数量不一样，就有for循环和直接算的两个分支
    // 因为有padding的存在，所以只需要遍历有效的行号即可
    if (output_template->effective_TLB_num == output_template->tblock_num * output_template->thread_num_in_block)
    {
        thread_block_id_var_name = "global_tid";
        return_str = return_str + "{\n";
    }
    else
    {
        // for循环的写法
        thread_block_id_var_name = "thread_level_block_id";
        return_str = return_str + "for(" + "unsigned int" + " thread_level_block_id = global_tid; ";
        return_str = return_str + "thread_level_block_id < " + to_string(output_template->effective_TLB_num) + "; ";
        return_str = return_str + "thread_level_block_id = thread_level_block_id + total_thread_num)\n{\n";
    }

    // 遍历线程粒度的块的内部，根据线程粒度的块的大小处理得到不同的代码
    if (output_template->thread_block_size_in_block == 1)
    {
        return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " thread_block_tmp_result;\n";
        return_str = return_str + "{\n";
        // 根据是否有列索引执行不同的计算
        if (output_template->kernal_first_col_index == 0)
        {
            return_str = return_str + "thread_block_tmp_result = val_arr[" + thread_block_id_var_name + "] * device_x_arr[col_index_arr[" + thread_block_id_var_name + "]];\n";
        }
        else
        {
            return_str = return_str + "thread_block_tmp_result = val_arr[" + thread_block_id_var_name + "] * device_x_arr[kernal_first_col_index + col_index_arr[" + thread_block_id_var_name + "]];\n";
        }
    }
    else
    {
        // 开始加
        return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " thread_block_tmp_result = 0;\n";

        // 声明全局偏移量
        return_str = return_str + "unsigned int" + " global_nz_index = " + thread_block_id_var_name + ";\n";

        // 开始for循环
        return_str = return_str + "for(" + "unsigned int" + " nz_index_inner_thread_level_block = 0; nz_index_inner_thread_level_block < ";
        return_str = return_str + to_string(output_template->thread_block_size_in_block) + "; nz_index_inner_thread_level_block++)\n{\n";

        if (output_template->kernal_first_col_index == 0)
        {
            return_str = return_str + "thread_block_tmp_result = thread_block_tmp_result + val_arr[" + "global_nz_index" + "] * __ldg(&(device_x_arr[col_index_arr[global_nz_index]]));\n";
        }
        else
        {
            return_str = return_str + "thread_block_tmp_result = thread_block_tmp_result + val_arr[" + "global_nz_index" + "] * __ldg(&(device_x_arr[kernal_first_col_index + col_index_arr[global_nz_index]]));\n";
        }

        // 加上偏移量，数据类型由非零元数量决定
        return_str = return_str + "global_nz_index = global_nz_index + " + to_string(output_template->size_of_global_row_index_of_thread_level_block) + ";\n";
    }

    return_str = return_str + "}\n";

    // 归约
    // 获取局部的行号
    return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->matrix->dense_row_number)) + " global_row_index;\n";

    return_str = return_str + "\n";

    // 每个线程粒度的行号，这里已经得到了排序之后的结果。所以只需要在相对排序和不排序的时候加一个偏移量即可
    if (output_template->global_row_index_compress == NONE_COMPRESS)
    {
        return_str = return_str + "global_row_index = global_row_index_of_thread_level_block[" + thread_block_id_var_name + "];\n";
    }
    else if (output_template->global_row_index_compress == LINEAR_COMPRESS)
    {
        assert(output_template->global_row_index_compress_meta != NULL);

        linear_compress_t *compressor = (linear_compress_t *)output_template->global_row_index_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "global_row_index", thread_block_id_var_name) + ";\n";
    }
    else if (output_template->global_row_index_compress == CYCLE_INCREASE_COMPRESS)
    {
        assert(output_template->global_row_index_compress_meta != NULL);
        cycle_increase_compress_t *compressor = (cycle_increase_compress_t *)output_template->global_row_index_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "global_row_index", thread_block_id_var_name) + ";\n";
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
            // 这里省略掉了
            // return_str = return_str + "global_row_index = global_row_index;\n";
        }
    }

    // 全局排序
    if (output_template->global_sort_index == true)
    {
        assert(output_template->local_sort_index == false);

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
            // 这里省略掉了
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

    return_str = return_str + "}\n";
    return_str = return_str + "}\n";

    return return_str;
}

bool compress_global_row_index_of_thread_level_block(direct_atom_template_warp_block_compress_t *output_template, bool need_check, arr_compress_type type)
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

void try_all_compress(direct_atom_template_warp_block_compress_t *output_template)
{
    bool is_compressed = false;

    is_compressed = compress_global_row_index_of_thread_level_block(output_template, true, LINEAR_COMPRESS);

    if (is_compressed == false)
    {
        is_compressed = compress_global_row_index_of_thread_level_block(output_template, true, CYCLE_INCREASE_COMPRESS);
    }
}