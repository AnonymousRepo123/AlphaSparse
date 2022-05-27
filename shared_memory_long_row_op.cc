#include "shared_memory_long_row_op.hpp"

shared_memory_long_row_template_t *init_shared_memory_long_row_template(code_builder_t *builder, unsigned long dense_block_id)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;

    shared_memory_long_row_template_t *new_template = new shared_memory_long_row_template_t();

    if (matrix->block_coor_table.item_arr[dense_block_id]->min_dense_col_index == 0 && matrix->block_coor_table.item_arr[dense_block_id]->max_dense_col_index == matrix->dense_col_number - 1)
    {
        // 稠密子块之间没有共享的行
    }
    else
    {
        new_template->is_atom_add = true;
    }

    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr;

    // 三个级别的索引
    index_of_compress_block_t *block_level_index = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[2];
    index_of_compress_block_t *warp_level_index = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[3];
    index_of_compress_block_t *thread_level_index = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[4];

    assert(block_level_index != NULL && warp_level_index != NULL && thread_level_index != NULL);

    assert(block_level_index->index_of_the_first_row_arr != NULL && block_level_index->row_number_of_block_arr != NULL);

    //
    assert(block_level_index->level_of_this_index == TBLOCK_LEVEL);
    assert(warp_level_index->level_of_this_index == WRAP_LEVEL);
    assert(thread_level_index->level_of_this_index == THREAD_LEVEL);

    if (thread_level_index->row_number_of_block_arr != NULL)
    {
        cout << "thread_level_index->row_number_of_block_arr must be NULL, row num in thread level block must be 1" << endl;
        assert(false);
    }

    // 存储每一个线程块粒度的块的行号
    // vector<unsigned long> block_row_index_vec;

    // 首先进行检查，查看每个线程粒度的块的首行的行号，一共是block_num个
    // 除此之外每个块的非零元数量需要是32的倍数，用以支持warp reduce
    for (unsigned long i = 0; i < block_level_index->block_num; i++)
    {
        unsigned long this_block_first_row = read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, i);
        unsigned long this_block_row_num = read_from_array_with_data_type(block_level_index->row_number_of_block_arr, block_level_index->data_type_of_row_number_of_block_arr, i);

        if (this_block_row_num != 1)
        {
            cout << "in this template, row num of block must be 1" << endl;
            assert(false);
        }

        // 下一个块的首行行号
        if (i != block_level_index->block_num - 1)
        {
            unsigned long next_block_first_row = read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, i + 1);
            // 如果相等就用原子加
            if (this_block_first_row == next_block_first_row)
            {
                new_template->is_atom_add = true;
            }
        }

        // 记录这个块所在的行号
        // block_row_index_vec.push_back(this_block_first_row);

        // 这里记录每一个块非零元的起始位置
        unsigned long this_block_nz_begin_index = read_from_array_with_data_type(block_level_index->coo_begin_index_arr, block_level_index->data_type_of_coo_begin_index_arr, i);
        unsigned long next_block_nz_begin_index = read_from_array_with_data_type(block_level_index->coo_begin_index_arr, block_level_index->data_type_of_coo_begin_index_arr, i + 1);

        if ((next_block_nz_begin_index - this_block_nz_begin_index) % 32 != 0)
        {
            cout << "in this template, nnz in block is not a multiply of 32" << endl;
            assert(false);
        }

        // 最后一位的行索引和非零元索引要满足要求
        if (i == block_level_index->block_num - 1)
        {
            assert(next_block_nz_begin_index == compressed_block_view->padding_arr_size);
        }
    }

    // 遍历所有warp中thread粒度的块的大小，必须保证是1
    assert(thread_level_index->coo_block_size_arr != NULL);
    for (unsigned long i = 0; i < warp_level_index->block_num; i++)
    {
        unsigned long thread_block_size = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, i);
        // 如果线程粒度的块不是1也不行
        if (thread_block_size != 1)
        {
            cout << "in this tmeplate, nnz in thread level block must be 1" << endl;
            assert(false);
        }
    }

    // 初始化当前的模板的所有数据，所有的数组都不用重新处理，直接用之前的指针即可
    new_template->dense_block_index = dense_block_id;
    new_template->matrix = matrix;
    new_template->kernal_first_row_index = matrix->block_coor_table.item_arr[dense_block_id]->min_dense_row_index;
    new_template->kernal_first_col_index = matrix->block_coor_table.item_arr[dense_block_id]->min_dense_col_index;

    // 可以将row_index_of_block_level_block和sort索引合并，让row_index_of_block直接存储排序之后的索引
    vector<unsigned long> global_row_index_of_BLB;

    // 将数据拷贝出来
    for (unsigned long i = 0; i < block_level_index->block_num; i++)
    {
        global_row_index_of_BLB.push_back(read_from_array_with_data_type(block_level_index->index_of_the_first_row_arr, block_level_index->data_type_of_index_of_the_first_row_arr, i));
    }
    
    // 排序
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

        // 找出原来的索引，
        for (unsigned long row_index_id = 0; row_index_id < global_row_index_of_BLB.size(); row_index_id++)
        {
            // 当前行号
            unsigned long cur_row_index = global_row_index_of_BLB[row_index_id];

            assert(cur_row_index < new_template->size_of_row_index_before_sort);
            unsigned long row_index_before_sort = read_from_array_with_data_type(new_template->row_index_before_sort, new_template->data_type_of_row_index_before_sort, cur_row_index);
            // 重置索引
            global_row_index_of_BLB[row_index_id] = row_index_before_sort;
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

        // 找出原本的索引
        for (unsigned long row_index_id = 0; row_index_id < global_row_index_of_BLB.size(); row_index_id++)
        {
            // 当前行号
            unsigned long cur_row_index = global_row_index_of_BLB[row_index_id];

            // 真实行号
            unsigned long matrix_level_row_index = cur_row_index + matrix->block_coor_table.item_arr[dense_block_id]->min_dense_row_index;

            assert(matrix_level_row_index < new_template->size_of_row_index_before_sort);
            // 找出之前
            unsigned long row_index_before_sort = read_from_array_with_data_type(new_template->row_index_before_sort, new_template->data_type_of_row_index_before_sort, matrix_level_row_index);

            global_row_index_of_BLB[row_index_id] = row_index_before_sort;
        }
    }

    // cout << 0 << endl;
    // 找出最大值
    unsigned long max_global_row_index_of_BLB = *max_element(global_row_index_of_BLB.begin(), global_row_index_of_BLB.end());
    // cout << 1 << endl;
    // 一个块一个首行索引
    new_template->data_type_of_row_index_of_block_level_block = find_most_suitable_data_type(max_global_row_index_of_BLB);
    // cout << 2 << endl;
    new_template->size_of_row_index_of_block_level_block = global_row_index_of_BLB.size();
    assert(global_row_index_of_BLB.size() == block_level_index->block_num);
    new_template->row_index_of_block_level_block = malloc_arr(new_template->size_of_row_index_of_block_level_block, new_template->data_type_of_row_index_of_block_level_block);

    copy_unsigned_long_arr_to_others(&(global_row_index_of_BLB[0]), new_template->row_index_of_block_level_block, new_template->data_type_of_row_index_of_block_level_block, new_template->size_of_row_index_of_block_level_block);

    

    new_template->block_nz_begin_offset = block_level_index->coo_begin_index_arr;
    new_template->size_of_block_nz_begin_offset = block_level_index->block_num + 1;
    new_template->data_type_of_block_nz_begin_offset = block_level_index->data_type_of_coo_begin_index_arr;

    // 值
    new_template->data_type_of_val_arr = compressed_block_view->val_data_type;
    new_template->val_arr = compressed_block_view->staggered_padding_val_arr;
    new_template->size_of_val_arr = compressed_block_view->staggered_padding_val_arr_size;

    // 这两个数组的大小和blocknz的最后一个非零元大小相同
    assert(new_template->size_of_val_arr == read_from_array_with_data_type(new_template->block_nz_begin_offset, new_template->data_type_of_block_nz_begin_offset, new_template->size_of_block_nz_begin_offset - 1));

    // 列
    new_template->data_type_of_col_index_arr = compressed_block_view->read_index[6]->index_data_type;
    new_template->col_index_arr = compressed_block_view->read_index[6]->index_arr;
    new_template->size_of_col_index_arr = compressed_block_view->read_index[6]->length;

    assert(new_template->size_of_val_arr == new_template->size_of_col_index_arr);

    return new_template;
}

bool is_supported_by_shared_memory_long_row_template(sparse_struct_t *matrix, unsigned long dense_block_id)
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

    // 遍历每一个BLB，行的数量为1
    for (unsigned long BLB_id = 0; BLB_id < block_level_index->block_num; BLB_id++)
    {
        unsigned long cur_BLB_row_number = read_from_array_with_data_type(block_level_index->row_number_of_block_arr, block_level_index->data_type_of_row_number_of_block_arr, BLB_id);

        if (cur_BLB_row_number != 1)
        {
            return false;
        }
    }

    return true;
}

// 一个线程块负责一行，WLB和TLB的分块信息会被全部忽视掉
bool is_supported_by_shared_memory_long_row_template(code_builder_t *builder, unsigned long dense_block_id)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;

    return is_supported_by_shared_memory_long_row_template(matrix, dense_block_id);
}

void store_template_data(shared_memory_long_row_template_t *output_template, string output_dir,  bool force_not_share_global_sort_index)
{
    srand(time(0));
    unsigned long matrix_id = rand() + time(0) % 1000;

    // 写这个模板所需要数据的文件夹名称
    output_dir = output_dir + "/" + to_string(matrix_id) + "_" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer());

    // 创建这个文件夹
    system(("mkdir " + output_dir).c_str());

    if (output_template->row_index_of_block_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->row_index_of_block_level_block != NULL);
        print_arr_to_file_with_data_type(output_template->row_index_of_block_level_block, output_template->data_type_of_row_index_of_block_level_block, output_template->size_of_row_index_of_block_level_block, output_dir + "/row_index_of_block_level_block");
    }

    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset != NULL);
        print_arr_to_file_with_data_type(output_template->block_nz_begin_offset, output_template->data_type_of_block_nz_begin_offset, output_template->size_of_block_nz_begin_offset, output_dir + "/block_nz_begin_offset");
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

string code_of_template_data_struct(shared_memory_long_row_template_t *output_template, unsigned long dense_block_id)
{
    // 创建一个数据结构
    string return_str = "typedef struct compressed_dense_block_" + to_string(dense_block_id) + "\n{\n";

    // 对应的位置分别存储行号和块号
    if (output_template->row_index_of_block_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->row_index_of_block_level_block != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_row_index_of_block_level_block, code_of_arr_var_name(dense_block_id, -1, "row_index_of_block_level_block"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "row_index_of_block_level_block") + " = " + to_string(output_template->size_of_row_index_of_block_level_block) + ";\n";
    }

    return_str = return_str + "\n";

    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_block_nz_begin_offset, code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset") + " = " + to_string(output_template->size_of_block_nz_begin_offset) + ";\n";
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

string code_of_read_template_data_from_file_func_define(shared_memory_long_row_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index)
{
    string return_str = "compressed_dense_block_" + to_string(dense_block_id) + "_t* read_dense_block_" + to_string(dense_block_id) + "_from_file(string file_name_prefix)\n{\n";

    return_str = return_str + "compressed_dense_block_" + to_string(dense_block_id) + "_t *template_data = new " + "compressed_dense_block_" + to_string(dense_block_id) + "_t();\n";

    // 对应的位置分别存储行号和块号
    if (output_template->row_index_of_block_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->row_index_of_block_level_block != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "row_index_of_block_level_block") + " = (" + code_of_data_type(output_template->data_type_of_row_index_of_block_level_block) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "row_index_of_block_level_block") + ", " + convert_data_type_to_string(output_template->data_type_of_row_index_of_block_level_block) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/row_index_of_block_level_block\");\n";
    }

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

string code_of_template_kernal(shared_memory_long_row_template_t *output_template, unsigned long dense_block_id)
{
    // 内核函数的声明
    string return_str = "__global__ void spmv_" + to_string(dense_block_id) + "(";

    // 用一个变量表明当前形参是不是第一个，如果是第一个就不用点逗号
    bool is_first_param = true;

    // 这里加入形参的声明
    if (output_template->row_index_of_block_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->row_index_of_block_level_block != NULL);
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_row_index_of_block_level_block, "* row_index_of_block_level_block");
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

    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        // 块首非零元
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

    // 计算资源编号
    return_str = return_str + "int bid = blockIdx.x;\n";
    return_str = return_str + "int tid_in_block = threadIdx.x;\n";
    return_str = return_str + "int wid_in_block = threadIdx.x / 32;\n";

    // 起始列号和行号
    if (output_template->kernal_first_row_index != 0)
    {
        return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->kernal_first_row_index + 1)) + " kernal_first_row_index = " + to_string(output_template->kernal_first_row_index) + ";\n";
    }

    if (output_template->kernal_first_col_index != 0)
    {
        return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->kernal_first_row_index + 1)) + " kernal_first_row_index = " + to_string(output_template->kernal_first_col_index) + ";\n";
    }

    return_str = return_str + "\n";

    // 计算warp的数量
    assert(output_template->thread_num_in_block % 32 == 0);

    unsigned long warp_num = output_template->thread_num_in_block / 32;

    return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_val_arr) + " thread_tmp_result_inner_block[" + to_string(warp_num) + "];\n";

    // 根据压缩的情况确定nz的记录是否被需要，可以使用线性压缩
    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_nz_begin_offset) + " this_block_first_nz_index_shared[1];\n";
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_nz_begin_offset) + " next_block_first_nz_index_shared[1];\n";
    }

    return_str = return_str + "\n";

    // 最外层的遍历for循环，可能需要也可能不需要
    // 如果线程块粒度的块的数量和线程块的数量一致
    // cout << "output_template->tblock_num:" << output_template->tblock_num << " output_template->size_of_row_index_of_block_level_block:" << output_template->size_of_row_index_of_block_level_block << endl;
    if (output_template->tblock_num == output_template->size_of_row_index_of_block_level_block)
    {
        // 不需要for循环
        return_str = return_str + "{\n";
        return_str = return_str + "unsigned int block_level_block_id = bid;\n";
    }
    else
    {
        // 需要for循环
        return_str = return_str + "for(";
        return_str = return_str + "unsigned int block_level_block_id = bid; block_level_block_id < " + to_string(output_template->size_of_row_index_of_block_level_block) + "; block_level_block_id = block_level_block_id + gridDim.x)\n{\n";
    }

    return_str = return_str + code_of_data_type(output_template->data_type_of_block_nz_begin_offset) + " this_block_first_nz_index;\n";
    return_str = return_str + code_of_data_type(output_template->data_type_of_block_nz_begin_offset) + " next_block_first_nz_index;\n";

    return_str = return_str + "\n";

    // 一开始必带一个全局同步
    return_str = return_str + "__syncthreads();\n\n";

    // 如果快的首行非零元索引没有压缩，就需要初始化
    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        return_str = return_str + "if (tid_in_block == 0)\n{\n";

        return_str = return_str + "this_block_first_nz_index_shared[0] = block_nz_begin_offset[block_level_block_id];\n";
        return_str = return_str + "next_block_first_nz_index_shared[0] = block_nz_begin_offset[block_level_block_id + 1];\n";

        return_str = return_str + "}\n\n";

        return_str = return_str + "__syncthreads();\n\n";

        return_str = return_str + "this_block_first_nz_index = this_block_first_nz_index_shared[0];\n";
        return_str = return_str + "next_block_first_nz_index = next_block_first_nz_index_shared[0];\n\n";
    }
    else if (output_template->block_nz_begin_offset_compress == LINEAR_COMPRESS)
    {
        linear_compress_t *compressor = (linear_compress_t *)output_template->block_nz_begin_offset_compress_meta;
        assert(compressor != NULL);
        return_str = return_str + code_of_arr_read(compressor, "this_block_first_nz_index", "block_level_block_id") + ";\n";
        return_str = return_str + "next_block_first_nz_index = this_block_first_nz_index + " + to_string(compressor->coefficient) + ";\n";
        // return_str = return_str + code_of_arr_read(compressor, "next_block_first_nz_index", "(block_level_block_id + 1)") + ";\n";
    }
    else
    {
        cout << "this compress type is not support in this template" << endl;
        assert(false);
    }

    return_str = return_str + "\n";

    return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " result_tmp_result = 0;\n";

    return_str = return_str + "for (unsigned int global_nz_index = this_block_first_nz_index + tid_in_block; global_nz_index < next_block_first_nz_index; global_nz_index = global_nz_index + blockDim.x)\n{\n";

    // 根据是否有初始的列偏移来计算一行的结果
    if (output_template->kernal_first_col_index == 0)
    {
        return_str = return_str + "result_tmp_result = result_tmp_result + val_arr[global_nz_index] * __ldg(&(device_x_arr[col_index_arr[global_nz_index]]));\n";
    }
    else
    {
        return_str = return_str + "result_tmp_result = result_tmp_result + val_arr[global_nz_index] * __ldg(&(device_x_arr[kernal_first_col_index + col_index_arr[global_nz_index]]));\n";
    }

    return_str = return_str + "}\n";

    // 每个warp规约自己的结果
    return_str = return_str + "for (int offset = 16; offset > 0; offset = offset / 2)\n{\n";

    return_str = return_str + "result_tmp_result = result_tmp_result + __shfl_down_sync(0xFFFFFFFF, result_tmp_result, offset);\n";

    return_str = return_str + "}\n";

    // 线程束第一个线程吧结果硅规约给共享内存
    return_str = return_str + "thread_tmp_result_inner_block[wid_in_block] = result_tmp_result;\n";

    return_str = return_str + "__syncthreads();\n";

    // 归约中间结果
    return_str = return_str + "if (tid_in_block == 0)\n{\n";

    return_str = return_str + "result_tmp_result = 0;\n";

    return_str = return_str + "for (unsigned long i = 0; i < " + to_string(warp_num) + "; i++)\n{\n";

    return_str = return_str + "result_tmp_result = result_tmp_result + thread_tmp_result_inner_block[i];\n";

    return_str = return_str + "}\n";

    // 全局行号
    return_str = return_str + "unsigned long global_row_index;\n";

    // 通过block id来获取block所对应的行号，可以是循环压缩
    if (output_template->row_index_of_block_level_block_compress == NONE_COMPRESS)
    {
        return_str = return_str + "global_row_index = row_index_of_block_level_block[block_level_block_id];\n\n";
    }
    else if (output_template->row_index_of_block_level_block_compress == LINEAR_COMPRESS)
    {
        linear_compress_t *compressor = (linear_compress_t *)output_template->row_index_of_block_level_block_compress_meta;
        assert(compressor != NULL);
        return_str = return_str + code_of_arr_read(compressor, "global_row_index", "block_level_block_id") + ";\n";
    }
    else if (output_template->row_index_of_block_level_block_compress == CYCLE_INCREASE_COMPRESS)
    {
        cycle_increase_compress_t *compressor = (cycle_increase_compress_t *)output_template->row_index_of_block_level_block_compress_meta;
        assert(compressor != NULL);
        return_str = return_str + code_of_arr_read(compressor, "global_row_index", "block_level_block_id") + ";\n";
    }
    else
    {
        cout << "this compress type is not support in this template" << endl;
        assert(false);
    }

    // 将结果写到全局内存中
    string reduce_result_var_name = "result_tmp_result";
    string var_name_of_global_row_index = "global_row_index";

    

    // 如果排序就需要新的变量
    if (output_template->local_sort_index == false && output_template->global_sort_index == false)
    {
        // 没有排序，直接加上对应的起始行号
        if (output_template->kernal_first_row_index != 0)
        {
            return_str = return_str + "global_row_index = global_row_index + kernal_first_row_index;\n";
        }
    }
    else
    {
        // return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->matrix->dense_row_number)) + " global_row_index = " + var_name_of_global_row_index + ";\n";
        // 子块内部的排序的
        if (output_template->local_sort_index == true)
        {
            assert(output_template->global_sort_index == false);
            // 获取真实的行索引
            if (output_template->kernal_first_row_index != 0)
            {
                return_str = return_str + "global_row_index = global_row_index + kernal_first_row_index;\n";
            }
        }

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
    return_str = return_str + "}\n";
    return_str = return_str + "}\n";

    return return_str;
}

string code_of_kernal_function_call(shared_memory_long_row_template_t *output_template, unsigned long dense_block_id)
{
    assert(output_template != NULL);
    // 线程块的数量和线程的数量不能超标
    assert(output_template->tblock_num <= get_config()["MAX_TBLOCK_NUM"].as_integer() && output_template->thread_num_in_block <= get_config()["MAX_THREAD_NUM_IN_BLOCK"].as_integer());

    string return_str = "spmv_" + to_string(dense_block_id) + "<<<" + to_string(output_template->tblock_num) + ", " + to_string(output_template->thread_num_in_block) + ", 0, stream_arr[" + to_string(dense_block_id) + "]>>>(";

    bool is_first_param = true;

    if (output_template->row_index_of_block_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->row_index_of_block_level_block != NULL);
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_of_block_level_block");
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

    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        // 块首非零元
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

string code_of_write_template_data_to_gpu(shared_memory_long_row_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index)
{
    // 读到对应结构体中的代码
    // 存储结构体的名字
    string template_data_name = "dense_block_" + to_string(dense_block_id) + "_template_data";

    string return_str = "compressed_dense_block_" + to_string(dense_block_id) + "_t *" + template_data_name + " = read_dense_block_" + to_string(dense_block_id) + "_from_file(" + "\"" + string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source/" + to_string(output_template->hash_of_this_template) + "_" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer()) + "\");\n\n";

    // 全局排序的数组取一个特殊的名字，并且只处理一次，剩下的从这里拷贝即可
    // 如果不共享，这段逻辑就完全不用出现
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

    if (output_template->row_index_of_block_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->row_index_of_block_level_block != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_row_index_of_block_level_block, "device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_of_block_level_block"));
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

    assert(output_template->val_arr != NULL);
    return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_val_arr, "device_" + code_of_arr_var_name(dense_block_id, -1, "val_arr"));

    assert(output_template->col_index_arr != NULL);
    return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_col_index_arr, "device_" + code_of_arr_var_name(dense_block_id, -1, "col_index_arr"));

    return_str = return_str + "\n";

    // 申请数组的代码
    if (output_template->row_index_of_block_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->row_index_of_block_level_block != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_row_index_of_block_level_block, to_string(output_template->size_of_row_index_of_block_level_block), "device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_of_block_level_block"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_of_block_level_block"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "row_index_of_block_level_block"), output_template->data_type_of_row_index_of_block_level_block, to_string(output_template->size_of_row_index_of_block_level_block), "cudaMemcpyHostToDevice") + "\n";
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

    // 如果是局部的就拷贝
    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->local_sort_index == true)
    {
        assert(output_template->global_sort_index == false && output_template->row_index_before_sort != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_row_index_before_sort, to_string(output_template->size_of_row_index_before_sort), "device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"), output_template->data_type_of_row_index_before_sort, to_string(output_template->size_of_row_index_before_sort), "cudaMemcpyHostToDevice") + "\n";
    }

    // 块与warp的第一个非零元的索引
    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_block_nz_begin_offset, to_string(output_template->size_of_block_nz_begin_offset), "device_" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "block_nz_begin_offset"), output_template->data_type_of_block_nz_begin_offset, to_string(output_template->size_of_block_nz_begin_offset), "cudaMemcpyHostToDevice") + "\n";
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

// 压缩每个线程粒度的子块的全局行号，一般使用线性压缩
bool compress_row_index_of_block_level_block(shared_memory_long_row_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && output_template->row_index_of_block_level_block != NULL);
    assert(type == CYCLE_INCREASE_COMPRESS || type == LINEAR_COMPRESS);

    if (type == CYCLE_INCREASE_COMPRESS)
    {
        // cout << "3" << endl;
        cycle_increase_compress_t *compressor = init_cycle_increase_compressor(output_template->row_index_of_block_level_block, output_template->data_type_of_row_index_of_block_level_block, output_template->size_of_row_index_of_block_level_block, need_check);
        // cout << "4" << endl;
        if (compressor == NULL)
        {
            return false;
        }

        // 压缩成功
        // 压缩成功，拷贝元数据
        output_template->row_index_of_block_level_block_compress_meta = (void *)compressor;
        output_template->row_index_of_block_level_block_compress = type;

        return true;
    }

    if (type == LINEAR_COMPRESS)
    {
        linear_compress_t *compressor = init_linear_compressor(output_template->row_index_of_block_level_block, output_template->data_type_of_row_index_of_block_level_block, output_template->size_of_row_index_of_block_level_block, need_check);

        // 查看能否成功压缩
        if (compressor == NULL)
        {
            // 不成功
            return false;
        }

        // 压缩成功，拷贝元数据
        output_template->row_index_of_block_level_block_compress_meta = (void *)compressor;
        output_template->row_index_of_block_level_block_compress = type;

        return true;
    }

    return false;
}

bool compress_block_nz_begin_offset(shared_memory_long_row_template_t *output_template, bool need_check, arr_compress_type type)
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

void try_all_compress(shared_memory_long_row_template_t *output_template)
{
    assert(output_template != NULL);

    bool is_compressed = false;

    is_compressed = compress_row_index_of_block_level_block(output_template, true, LINEAR_COMPRESS);

    if (is_compressed == false)
    {
        is_compressed = compress_row_index_of_block_level_block(output_template, true, CYCLE_INCREASE_COMPRESS);
    }

    is_compressed = compress_block_nz_begin_offset(output_template, true, LINEAR_COMPRESS);
}
