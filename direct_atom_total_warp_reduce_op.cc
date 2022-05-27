#include "direct_atom_total_warp_reduce_op.hpp"

direct_atom_total_warp_reduce_template_t *init_direct_atom_total_warp_reduce_template(code_builder_t *builder, unsigned long dense_block_id)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;
    assert(matrix->block_coor_table.item_arr.size() > dense_block_id);

    direct_atom_total_warp_reduce_template_t *new_template = new direct_atom_total_warp_reduce_template_t();

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

    assert(compressed_block_view->read_index[0]->max_row_index == block_level_index->max_row_index);
    assert(compressed_block_view->read_index[0]->max_row_index == warp_level_index->max_row_index);
    assert(compressed_block_view->read_index[0]->max_row_index == thread_level_index->max_row_index);

    if (thread_level_index->row_number_of_block_arr != NULL)
    {
        cout << "row num in thread level block must be 1, thread level index shouldn't have this metadata" << endl;
        assert(false);
    }

    // 当前稠密子块的行数量，这个值也是块最大行号+1
    unsigned long total_row_num = block_level_index->max_row_index - block_level_index->min_row_index + 1;

    // 每个warp粒度的块的全局行号
    vector<unsigned long> global_row_index_of_warp_level_block_vec(warp_level_index->block_num);
    // 每个warp粒度的块的首个非零元的索引，使用CSR的索引，一共是warp块的数量+1，这一点和其他的模板不一样
    vector<unsigned long> global_warp_block_first_nz_vec(warp_level_index->block_num + 1);

    // 遍历block和warp两个层次，计算行号、非零元的起始索引
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

            global_row_index_of_warp_level_block_vec[index_of_warp_level_index] = global_warp_row_index;
            global_warp_block_first_nz_vec[index_of_warp_level_index] = global_warp_first_nz_index;

            // 行号重合，说明要使用原子加
            if (index_of_warp_level_index >= 1)
            {
                if (global_row_index_of_warp_level_block_vec[index_of_warp_level_index] == global_row_index_of_warp_level_block_vec[index_of_warp_level_index - 1])
                {
                    new_template->is_atom_add = true;
                }
            }

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

    assert(matrix->block_coor_table.item_arr[dense_block_id]->min_dense_row_index == compressed_block_view->read_index[0]->min_row_index);

    if (matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index != compressed_block_view->read_index[0]->max_row_index)
    {
        assert(matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index < compressed_block_view->read_index[0]->max_row_index);

        // 压缩子块和稠密子块的行数量
        unsigned long compressed_block_row_num = compressed_block_view->read_index[0]->max_row_index - compressed_block_view->read_index[0]->min_row_index + 1;
        unsigned long dense_sub_block_row_num = matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index - matrix->block_coor_table.item_arr[dense_block_id]->min_dense_row_index + 1;

        // 因为有压缩视图的padding，所以并不是所有的WLB都是有效的，需要遍历所有的WLB，找出有效WLB的数量
        for (unsigned long WLB_id = 0; WLB_id < global_row_index_of_warp_level_block_vec.size(); WLB_id++)
        {
            unsigned long cur_WLB_row_index = global_row_index_of_warp_level_block_vec[WLB_id];

            if (cur_WLB_row_index >= dense_sub_block_row_num)
            {
                assert(cur_WLB_row_index < compressed_block_row_num);
                // 第一个无效的WLB
                new_template->effective_WLB_num = WLB_id;
                break;
            }

            assert(WLB_id != global_row_index_of_warp_level_block_vec.size() - 1);
        }

        for (unsigned long WLB_id = new_template->effective_WLB_num; WLB_id < global_row_index_of_warp_level_block_vec.size(); WLB_id++)
        {
            unsigned long cur_WLB_row_index = global_row_index_of_warp_level_block_vec[WLB_id];
            assert(cur_WLB_row_index < compressed_block_row_num);
            assert(cur_WLB_row_index >= dense_sub_block_row_num);
        }
    }
    else
    {
        new_template->effective_WLB_num = global_row_index_of_warp_level_block_vec.size();
    }

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

        // 找出原来的索引，因为被padding的行没有参与排序，所以不需要寻找原有的行号
        for (unsigned long row_index_id = 0; row_index_id < new_template->effective_WLB_num; row_index_id++)
        {
            // 当前行号
            unsigned long cur_row_index = global_row_index_of_warp_level_block_vec[row_index_id];

            assert(cur_row_index < new_template->size_of_row_index_before_sort);
            // 排序之前的位置
            unsigned long row_index_before_sort = read_from_array_with_data_type(new_template->row_index_before_sort, new_template->data_type_of_row_index_before_sort, cur_row_index);
            // 重置索引
            global_row_index_of_warp_level_block_vec[row_index_id] = row_index_before_sort;
        }
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

        // 找出原本的索引，因为有row padding，只有一部分WLB是有效的
        for (unsigned long row_index_id = 0; row_index_id < new_template->effective_WLB_num; row_index_id++)
        {
            // 当前行号
            unsigned long cur_row_index = global_row_index_of_warp_level_block_vec[row_index_id];

            // 真实行号
            unsigned long matrix_level_row_index = cur_row_index + matrix->block_coor_table.item_arr[dense_block_id]->min_dense_row_index;
            
            assert(matrix_level_row_index < new_template->size_of_row_index_before_sort);
            // 找出之前
            unsigned long row_index_before_sort = read_from_array_with_data_type(new_template->row_index_before_sort, new_template->data_type_of_row_index_before_sort, matrix_level_row_index);

            global_row_index_of_warp_level_block_vec[row_index_id] = row_index_before_sort;
        }
    }

    // 将全局行号拷贝到模板中
    unsigned long max_global_row_index_of_thread_level_block = *max_element(global_row_index_of_warp_level_block_vec.begin(), global_row_index_of_warp_level_block_vec.end());
    new_template->data_type_of_global_row_index_of_warp_level_block = find_most_suitable_data_type(max_global_row_index_of_thread_level_block);
    new_template->size_of_global_row_index_of_warp_level_block = global_row_index_of_warp_level_block_vec.size();
    new_template->global_row_index_of_warp_level_block = malloc_arr(new_template->size_of_global_row_index_of_warp_level_block, new_template->data_type_of_global_row_index_of_warp_level_block);
    copy_unsigned_long_arr_to_others(&(global_row_index_of_warp_level_block_vec[0]), new_template->global_row_index_of_warp_level_block, new_template->data_type_of_global_row_index_of_warp_level_block, new_template->size_of_global_row_index_of_warp_level_block);

    // 每个warp粒度的块的第一个非零元的索引。不需要block索引，采用的CSR的索引方法
    new_template->size_of_global_warp_nz_begin_offset = global_warp_block_first_nz_vec.size();
    new_template->data_type_of_global_warp_nz_begin_offset = find_most_suitable_data_type(matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->padding_arr_size + 1);
    new_template->global_warp_nz_begin_offset = malloc_arr(new_template->size_of_global_warp_nz_begin_offset, new_template->data_type_of_global_warp_nz_begin_offset);
    // 将数据拷贝进来，从global_warp_block_first_nz_vec中拷贝进来，数量比warp粒度的块的数量多一个。
    for (unsigned long i = 0; i < global_warp_block_first_nz_vec.size(); i++)
    {
        unsigned long nz_index = global_warp_block_first_nz_vec[i];
        write_to_array_with_data_type(new_template->global_warp_nz_begin_offset, new_template->data_type_of_global_warp_nz_begin_offset, i, nz_index);
    }

    // 列索引的值直接做一个拷贝，可以拷贝交错存储的版本
    // 值
    new_template->data_type_of_val_arr = compressed_block_view->val_data_type;
    new_template->val_arr = compressed_block_view->staggered_padding_val_arr;
    new_template->size_of_val_arr = compressed_block_view->staggered_padding_val_arr_size;

    // 这两个数组的大小和warpnz的最后一个非零元大小相同
    assert(new_template->size_of_val_arr == read_from_array_with_data_type(new_template->global_warp_nz_begin_offset, new_template->data_type_of_global_warp_nz_begin_offset, new_template->size_of_global_warp_nz_begin_offset - 1));

    // 列
    new_template->data_type_of_col_index_arr = compressed_block_view->read_index[6]->index_data_type;
    new_template->col_index_arr = compressed_block_view->read_index[6]->index_arr;
    new_template->size_of_col_index_arr = compressed_block_view->read_index[6]->length;

    assert(new_template->size_of_val_arr == new_template->size_of_col_index_arr);

    return new_template;
}

bool is_supported_by_direct_atom_total_warp_reduce_template(sparse_struct_t *matrix, unsigned long dense_block_id)
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

    // 遍历所有warp级别的块，让每个块的的行数量都是1
    for (unsigned long WLB_id = 0; WLB_id < warp_level_index->block_num; WLB_id++)
    {
        unsigned long cur_WLB_row_num = read_from_array_with_data_type(warp_level_index->row_number_of_block_arr, warp_level_index->data_type_of_row_number_of_block_arr, WLB_id);

        if (cur_WLB_row_num != 1)
        {
            return false;
        }
    }

    return true;
}

// 保证每个WLB的内容不能跨两行，TLB的所有特征会被忽略。
bool is_supported_by_direct_atom_total_warp_reduce_template(code_builder_t *builder, unsigned long dense_block_id)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;

    return is_supported_by_direct_atom_total_warp_reduce_template(matrix, dense_block_id);
}

void store_template_data(direct_atom_total_warp_reduce_template_t *output_template, string output_dir, bool force_not_share_global_sort_index)
{
    srand(time(0));
    unsigned long matrix_id = rand() + time(0) % 1000;

    // 写这个模板所需要数据的文件夹名称
    output_dir = output_dir + "/" + to_string(matrix_id) + "_" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer());

    // 创建这个文件夹
    system(("mkdir " + output_dir).c_str());

    if (output_template->global_row_index_of_warp_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_warp_level_block != NULL);
        print_arr_to_file_with_data_type(output_template->global_row_index_of_warp_level_block, output_template->data_type_of_global_row_index_of_warp_level_block, output_template->size_of_global_row_index_of_warp_level_block, output_dir + "/global_row_index_of_warp_level_block");
    }

    if (output_template->global_warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->global_warp_nz_begin_offset != NULL);
        print_arr_to_file_with_data_type(output_template->global_warp_nz_begin_offset, output_template->data_type_of_global_warp_nz_begin_offset, output_template->size_of_global_warp_nz_begin_offset, output_dir + "/global_warp_nz_begin_offset");
    }

    // if (output_template->global_warp_nz_begin_offset_compress == NONE_COMPRESS)
    // {
    //     assert(output_template->global_warp_nz_begin_offset != NULL);
    //     print_arr_to_file_with_data_type(output_template->global_warp_nz_begin_offset, output_template->data_type_of_global_warp_nz_begin_offset, output_template->size_of_global_warp_nz_begin_offset, output_dir + "/global_warp_nz_begin_offset");
    // }

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

string code_of_template_data_struct(direct_atom_total_warp_reduce_template_t *output_template, unsigned long dense_block_id)
{
    // 创建一个数据结构
    string return_str = "typedef struct compressed_dense_block_" + to_string(dense_block_id) + "\n{\n";

    if (output_template->global_row_index_of_warp_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_warp_level_block != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_global_row_index_of_warp_level_block, code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_warp_level_block"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_warp_level_block") + " = " + to_string(output_template->size_of_global_row_index_of_warp_level_block) + ";\n";
    }

    return_str = return_str + "\n";

    if (output_template->global_warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->global_warp_nz_begin_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_global_warp_nz_begin_offset, code_of_arr_var_name(dense_block_id, -1, "global_warp_nz_begin_offset"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "global_warp_nz_begin_offset") + " = " + to_string(output_template->size_of_global_warp_nz_begin_offset) + ";\n";
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

string code_of_read_template_data_from_file_func_define(direct_atom_total_warp_reduce_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index)
{
    string return_str = "compressed_dense_block_" + to_string(dense_block_id) + "_t* read_dense_block_" + to_string(dense_block_id) + "_from_file(string file_name_prefix)\n{\n";

    return_str = return_str + "compressed_dense_block_" + to_string(dense_block_id) + "_t *template_data = new " + "compressed_dense_block_" + to_string(dense_block_id) + "_t();\n";

    if (output_template->global_row_index_of_warp_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_warp_level_block != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_warp_level_block") + " = (" + code_of_data_type(output_template->data_type_of_global_row_index_of_warp_level_block) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_warp_level_block") + ", " + convert_data_type_to_string(output_template->data_type_of_global_row_index_of_warp_level_block) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/global_row_index_of_warp_level_block\");\n";
    }

    return_str = return_str + "\n";

    if (output_template->global_warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->global_warp_nz_begin_offset != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "global_warp_nz_begin_offset") + " = (" + code_of_data_type(output_template->data_type_of_global_warp_nz_begin_offset) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "global_warp_nz_begin_offset") + ", " + convert_data_type_to_string(output_template->data_type_of_global_warp_nz_begin_offset) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/global_warp_nz_begin_offset\");\n";
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

string code_of_write_template_data_to_gpu(direct_atom_total_warp_reduce_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index)
{
    // 读到对应结构体中的代码
    // 存储结构体的名字
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

    if (output_template->global_row_index_of_warp_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_warp_level_block != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_global_row_index_of_warp_level_block, "device_" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_warp_level_block"));
    }

    if (output_template->global_warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->global_warp_nz_begin_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_global_warp_nz_begin_offset, "device_" + code_of_arr_var_name(dense_block_id, -1, "global_warp_nz_begin_offset"));
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
    if (output_template->global_row_index_of_warp_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_warp_level_block != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_global_row_index_of_warp_level_block, to_string(output_template->size_of_global_row_index_of_warp_level_block), "device_" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_warp_level_block"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_warp_level_block"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_warp_level_block"), output_template->data_type_of_global_row_index_of_warp_level_block, to_string(output_template->size_of_global_row_index_of_warp_level_block), "cudaMemcpyHostToDevice") + "\n";
    }

    if (output_template->global_warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->global_warp_nz_begin_offset != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_global_warp_nz_begin_offset, to_string(output_template->size_of_global_warp_nz_begin_offset), "device_" + code_of_arr_var_name(dense_block_id, -1, "global_warp_nz_begin_offset"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "global_warp_nz_begin_offset"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "global_warp_nz_begin_offset"), output_template->data_type_of_global_warp_nz_begin_offset, to_string(output_template->size_of_global_warp_nz_begin_offset), "cudaMemcpyHostToDevice") + "\n";
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

string code_of_kernal_function_call(direct_atom_total_warp_reduce_template_t *output_template, unsigned long dense_block_id)
{
    assert(output_template != NULL);
    // 线程块的数量和线程的数量不能超标
    assert(output_template->tblock_num <= get_config()["MAX_TBLOCK_NUM"].as_integer() && output_template->thread_num_in_block <= get_config()["MAX_THREAD_NUM_IN_BLOCK"].as_integer());

    string return_str = "spmv_" + to_string(dense_block_id) + "<<<" + to_string(output_template->tblock_num) + ", " + to_string(output_template->thread_num_in_block) + ", 0, stream_arr[" + to_string(dense_block_id) + "]>>>(";

    bool is_first_param = true;

    // 遍历所有的形参
    // 这里加入形参的声明
    if (output_template->global_row_index_of_warp_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_warp_level_block != NULL);
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_warp_level_block");
        is_first_param = false;
    }

    if (output_template->global_warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }
        assert(output_template->global_warp_nz_begin_offset != NULL);
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "global_warp_nz_begin_offset");
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

string code_of_template_kernal(direct_atom_total_warp_reduce_template_t *output_template, unsigned long dense_block_id)
{
    // 内核函数的声明
    string return_str = "__global__ void spmv_" + to_string(dense_block_id) + "(";

    // 用一个变量表明当前形参是不是第一个，如果是第一个就不用点逗号
    bool is_first_param = true;

    // 这里加入形参的声明
    if (output_template->global_row_index_of_warp_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_warp_level_block != NULL);
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_global_row_index_of_warp_level_block, "* global_row_index_of_warp_level_block");
        is_first_param = false;
    }

    if (output_template->global_warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }

        assert(output_template->global_warp_nz_begin_offset != NULL);
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_global_warp_nz_begin_offset, "* global_warp_nz_begin_offset");
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

    // 查看每个warp负责的非零元数量是不是32，从而决定是不是需要下一个WLB的首个非零元索引
    bool need_next_warp_block_first_nz = true;

    if (output_template->global_warp_nz_begin_offset_compress == LINEAR_COMPRESS)
    {
        assert(output_template->global_warp_nz_begin_offset_compress_meta != NULL);
        linear_compress_t *compressor = (linear_compress_t *)output_template->global_warp_nz_begin_offset_compress_meta;

        if (compressor->coefficient == 32)
        {
            need_next_warp_block_first_nz = false;
        }
    }

    return_str = return_str + "int tid_in_warp = threadIdx.x % 32;\n";

    // 如果warp的数量和WLB的数量一致，不需要wum
    if ((output_template->thread_num_in_block * output_template->tblock_num) / 32 == output_template->size_of_global_row_index_of_warp_level_block)
    {
    }
    else
    {
        return_str = return_str + "int wum = (gridDim.x * blockDim.x) / 32;\n";
    }

    return_str = return_str + "int wid = (blockDim.x * blockIdx.x + threadIdx.x) / 32;\n";

    return_str = return_str + "\n";

    if (output_template->kernal_first_row_index != 0)
    {
        return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->kernal_first_row_index + 1)) + " kernal_first_row_index = " + to_string(output_template->kernal_first_row_index) + ";\n";
    }

    if (output_template->kernal_first_col_index != 0)
    {
        return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->kernal_first_col_index + 1)) + " kernal_first_col_index = " + to_string(output_template->kernal_first_col_index) + ";\n";
    }

    return_str = return_str + "\n";

    if ((output_template->thread_num_in_block * output_template->tblock_num) / 32 == output_template->size_of_global_row_index_of_warp_level_block)
    {
        return_str = return_str + "unsigned int warp_level_block_id = wid;\n";
        // 不采用for循环，将对应的循环压缩
        // 如果出现了row padding，那就只处理有效的warp就行了
        if (output_template->effective_WLB_num != output_template->size_of_global_row_index_of_warp_level_block)
        {
            assert(output_template->effective_WLB_num < output_template->size_of_global_row_index_of_warp_level_block);
            return_str = return_str + "if(warp_level_block_id < " + to_string(output_template->effective_WLB_num) + ")\n{\n";
        }
        else
        {
            return_str = return_str + "{\n";   
        }
    }
    else
    {
        return_str = return_str + "for (unsigned int warp_level_block_id = wid; warp_level_block_id < " + to_string(output_template->effective_WLB_num) + "; warp_level_block_id = warp_level_block_id + wum)";
        return_str = return_str + "\n{\n";
    }

    // 声明需要的非零元索引
    return_str = return_str + code_of_data_type(output_template->data_type_of_global_warp_nz_begin_offset) + " this_warp_block_first_nz;\n";

    if (need_next_warp_block_first_nz == true)
    {
        return_str = return_str + code_of_data_type(output_template->data_type_of_global_warp_nz_begin_offset) + " next_warp_block_first_nz;\n";
    }

    return_str = return_str + "\n";

    if (output_template->global_warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->global_warp_nz_begin_offset != NULL && output_template->global_warp_nz_begin_offset_compress_meta == NULL);
        // 没有压缩就要从共享内存中取
        return_str = return_str + "if (tid_in_warp == 0)\n{\n";

        return_str = return_str + "this_warp_block_first_nz = global_warp_nz_begin_offset[warp_level_block_id];\n";

        if (need_next_warp_block_first_nz == true)
        {
            return_str = return_str + "next_warp_block_first_nz = global_warp_nz_begin_offset[warp_level_block_id + 1];\n";
        }

        return_str = return_str + "}\n";
    }

    return_str = return_str + "\n";

    // 执行一次赋值
    if (output_template->global_warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        return_str = return_str + "this_warp_block_first_nz = __shfl_sync(0xFFFFFFFF, this_warp_block_first_nz, 0, 32);\n";

        if (need_next_warp_block_first_nz == true)
        {
            return_str = return_str + "next_warp_block_first_nz = __shfl_sync(0xFFFFFFFF, next_warp_block_first_nz, 0, 32);\n";
        }
    }
    else if (output_template->global_warp_nz_begin_offset_compress == LINEAR_COMPRESS)
    {
        assert(output_template->global_warp_nz_begin_offset_compress_meta != NULL);

        linear_compress_t *compressor = (linear_compress_t *)output_template->global_warp_nz_begin_offset_compress_meta;

        return_str = return_str + code_of_arr_read(compressor, "this_warp_block_first_nz", "warp_level_block_id") + ";\n";

        if (need_next_warp_block_first_nz == true)
        {
            return_str = return_str + "next_warp_block_first_nz = this_warp_block_first_nz + " + to_string(compressor->coefficient) + ";\n";
        }
    }
    else
    {
        cout << "compress type is not supported in this template" << endl;
        assert(false);
    }

    return_str = return_str + "\n";

    // 查看是不是需要一个列号偏移
    string code_of_kernal_first_col_index_add = "";

    if (output_template->kernal_first_col_index != 0)
    {
        code_of_kernal_first_col_index_add = "kernal_first_col_index + ";
    }

    // 用一个变量存储一个线程的中间结果
    return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " result_tmp_result = 0;\n";

    // 根据是否压缩决定是不是有这一层循环
    if (need_next_warp_block_first_nz == true)
    {
        // 使用for循环遍历每一个非零元
        return_str = return_str + "for (unsigned int global_nz_index = this_warp_block_first_nz + tid_in_warp; global_nz_index < next_warp_block_first_nz; global_nz_index = global_nz_index + 32)\n{\n";

        return_str = return_str + "result_tmp_result = result_tmp_result + val_arr[global_nz_index] * __ldg(&(device_x_arr[" + code_of_kernal_first_col_index_add + "col_index_arr[global_nz_index]]));\n";

        return_str = return_str + "}\n";
    }
    else
    {
        return_str = return_str + "unsigned int global_nz_index = this_warp_block_first_nz + tid_in_warp;\n";
        return_str = return_str + "{\n";

        // 直接给线程的中间结果赋值
        return_str = return_str + "result_tmp_result = val_arr[global_nz_index] * __ldg(&(device_x_arr[" + code_of_kernal_first_col_index_add + "col_index_arr[global_nz_index]]));\n";

        return_str = return_str + "}\n";
    }

    return_str = return_str + "\n";

    // warp内部的归约，将归约结果放到warp第一个线程的result_tmp_result变量中
    return_str = return_str + "for (int offset = 16; offset > 0; offset = offset / 2)\n{\n";

    return_str = return_str + "result_tmp_result = result_tmp_result + __shfl_down_sync(0xFFFFFFFF, result_tmp_result, offset);\n";

    return_str = return_str + "}\n";

    // warp之间的归约，warp的第一个线程直接写显存
    return_str = return_str + "if (tid_in_warp == 0)\n{\n";

    // 获取局部的行号
    return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->matrix->dense_row_number)) + " global_row_index;\n";

    return_str = return_str + "\n";

    // 每个线程粒度的行号
    if (output_template->global_row_index_of_warp_level_block_compress == NONE_COMPRESS)
    {
        return_str = return_str + "global_row_index = global_row_index_of_warp_level_block[warp_level_block_id];\n";
    }
    else if (output_template->global_row_index_of_warp_level_block_compress == LINEAR_COMPRESS)
    {
        assert(output_template->global_row_index_of_warp_level_block_compress_meta != NULL);

        linear_compress_t *compressor = (linear_compress_t *)output_template->global_row_index_of_warp_level_block_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "global_row_index", "warp_level_block_id") + ";\n";
    }
    else if (output_template->global_row_index_of_warp_level_block_compress == CYCLE_INCREASE_COMPRESS)
    {
        assert(output_template->global_row_index_of_warp_level_block_compress_meta != NULL);
        cycle_increase_compress_t *compressor = (cycle_increase_compress_t *)output_template->global_row_index_of_warp_level_block_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "global_row_index", "warp_level_block_id") + ";\n";
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
        return_str = return_str + "atomicAdd(&(device_y_arr[global_row_index]), result_tmp_result);\n";
    }
    else
    {
        // 赋值
        return_str = return_str + "device_y_arr[global_row_index] = result_tmp_result;\n";
    }

    return_str = return_str + "}\n";

    // warp的循环
    return_str = return_str + "\n}\n";
    // kernal的级别
    return_str = return_str + "\n}\n";

    return return_str;
}

bool compress_global_row_index_of_warp_level_block(direct_atom_total_warp_reduce_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && output_template->global_row_index_of_warp_level_block != NULL);
    assert(type == CYCLE_INCREASE_COMPRESS || type == LINEAR_COMPRESS);

    if (type == CYCLE_INCREASE_COMPRESS)
    {
        cycle_increase_compress_t *compressor = init_cycle_increase_compressor(output_template->global_row_index_of_warp_level_block, output_template->data_type_of_global_row_index_of_warp_level_block, output_template->size_of_global_row_index_of_warp_level_block, need_check);

        if (compressor == NULL)
        {
            return false;
        }

        // 压缩成功
        // 压缩成功，拷贝元数据
        output_template->global_row_index_of_warp_level_block_compress_meta = (void *)compressor;
        output_template->global_row_index_of_warp_level_block_compress = type;

        return true;
    }

    if (type == LINEAR_COMPRESS)
    {
        linear_compress_t *compressor = init_linear_compressor(output_template->global_row_index_of_warp_level_block, output_template->data_type_of_global_row_index_of_warp_level_block, output_template->size_of_global_row_index_of_warp_level_block, need_check);

        // 查看能否成功压缩
        if (compressor == NULL)
        {
            // 不成功
            return false;
        }

        // 压缩成功，拷贝元数据
        output_template->global_row_index_of_warp_level_block_compress_meta = (void *)compressor;
        output_template->global_row_index_of_warp_level_block_compress = type;

        return true;
    }

    return false;
}

bool compress_global_warp_nz_begin_offset(direct_atom_total_warp_reduce_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && type == LINEAR_COMPRESS && output_template->global_warp_nz_begin_offset != NULL);

    linear_compress_t *compressor = init_linear_compressor(output_template->global_warp_nz_begin_offset, output_template->data_type_of_global_warp_nz_begin_offset, output_template->size_of_global_warp_nz_begin_offset, need_check);

    if (compressor == NULL)
    {
        return false;
    }

    // 压缩成功
    output_template->global_warp_nz_begin_offset_compress_meta = (void *)compressor;
    output_template->global_warp_nz_begin_offset_compress = type;

    return true;
}

void try_all_compress(direct_atom_total_warp_reduce_template_t *output_template)
{
    bool is_compressed = false;

    is_compressed = compress_global_row_index_of_warp_level_block(output_template, true, LINEAR_COMPRESS);

    if (is_compressed == false)
    {
        is_compressed = compress_global_row_index_of_warp_level_block(output_template, true, CYCLE_INCREASE_COMPRESS);
    }

    is_compressed == compress_global_warp_nz_begin_offset(output_template, true, LINEAR_COMPRESS);
}
