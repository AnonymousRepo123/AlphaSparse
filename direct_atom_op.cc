#include "direct_atom_op.hpp"
#include <assert.h>
#include "config.hpp"

direct_atom_template_t *init_direct_atom_template(code_builder_t *builder, unsigned long dense_block_id)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;

    // 初始化atom必备的几个数组
    direct_atom_template_t *new_template = new direct_atom_template_t();

    new_template->dense_block_index = dense_block_id;

    assert(dense_block_id < matrix->block_coor_table.item_arr.size());

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

    // 暂时不支持在密集子块的padding
    assert(matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index == block_level_index->max_row_index);
    assert(matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index == warp_level_index->max_row_index);
    assert(matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index == thread_level_index->max_row_index);

    assert(thread_level_index->coo_block_size_arr != NULL);

    if (thread_level_index->row_number_of_block_arr != NULL)
    {
        cout << "thread_level_index->row_number_of_block_arr must be NULL, row num in thread level block must be 1" << endl;
        assert(false);
    }

    // 每个thread的全局行索引
    vector<unsigned long> global_thread_row_index_vec;

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
                        new_template->is_atom_add = true;
                    }
                }
            }
        }
    }

    // cout << 3 << endl;

    assert(global_thread_row_index_vec.size() == thread_level_index->block_num);

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

    // 值
    new_template->data_type_of_val_arr = compressed_block_view->val_data_type;
    new_template->val_arr = compressed_block_view->staggered_padding_val_arr;
    new_template->size_of_val_arr = compressed_block_view->staggered_padding_val_arr_size;

    // 列
    new_template->data_type_of_col_index_arr = compressed_block_view->read_index[6]->index_data_type;
    new_template->col_index_arr = compressed_block_view->read_index[6]->index_arr;
    new_template->size_of_col_index_arr = compressed_block_view->read_index[6]->length;

    assert(new_template->size_of_val_arr == new_template->size_of_col_index_arr);

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

        // 找出原来的索引
        for (unsigned long row_index_id = 0; row_index_id < global_thread_row_index_vec.size(); row_index_id++)
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
        cout << "init_direct_atom_template: have global sort" << endl;
        // 在全局范围内有排序
        assert(compressed_block_view->is_sorted == false && matrix->is_sorted == true && builder->sub_block_sort_type_vec[dense_block_id] == GLOBAL_SORT);
        new_template->global_sort_index = true;
        new_template->local_sort_index = false;

        // 拷贝
        new_template->data_type_of_row_index_before_sort = matrix->data_type_of_sorted_row_index;
        new_template->row_index_before_sort = matrix->sorted_row_index;
        new_template->size_of_row_index_before_sort = matrix->dense_row_number;

        // 找出原本的索引
        for (unsigned long row_index_id = 0; row_index_id < global_thread_row_index_vec.size(); row_index_id++)
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

    // 确定数据类型的大小
    unsigned long max_global_row_index_of_thread_level_block = *max_element(global_thread_row_index_vec.begin(), global_thread_row_index_vec.end());
    new_template->data_type_of_global_row_index_of_thread_level_block = find_most_suitable_data_type(max_global_row_index_of_thread_level_block);
    // 创建对应数组
    new_template->global_row_index_of_thread_level_block = malloc_arr(global_thread_row_index_vec.size(), new_template->data_type_of_global_row_index_of_thread_level_block);
    // 对应数组的长度
    new_template->size_of_global_row_index_of_thread_level_block = global_thread_row_index_vec.size();
    // 拷贝数组
    copy_unsigned_long_arr_to_others(&(global_thread_row_index_vec[0]), new_template->global_row_index_of_thread_level_block, new_template->data_type_of_global_row_index_of_thread_level_block, new_template->size_of_global_row_index_of_thread_level_block);

    // 初始化为不使用压缩，压缩也需要元数据
    new_template->global_row_index_compress = NONE_COMPRESS;
    new_template->block_begin_warp_index_compress = NONE_COMPRESS;
    new_template->warp_begin_thread_index_compress = NONE_COMPRESS;
    new_template->thread_block_size_compress = NONE_COMPRESS;
    new_template->row_index_before_sort_compress = NONE_COMPRESS;

    new_template->warp_nz_begin_offset_compress = NONE_COMPRESS;
    new_template->block_nz_begin_offset_compress = NONE_COMPRESS;

    return new_template;
}

bool is_supported_by_direct_atom_template(sparse_struct_t *matrix, unsigned long dense_block_id)
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

    return true;
}

bool is_supported_by_direct_atom_template(code_builder_t *builder, unsigned long dense_block_id)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;

    // 在压缩视图下padding之后不支持
    if (matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index != matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[0]->max_row_index)
    {
        return false;
    }

    if (matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index != matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[1]->max_row_index)
    {
        return false;
    }

    if (matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index != matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[2]->max_row_index)
    {
        return false;
    }

    if (matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index != matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr->read_index[3]->max_row_index)
    {
        return false;
    }

    return is_supported_by_direct_atom_template(matrix, dense_block_id);
}

void store_template_data(direct_atom_template_t *output_template, string output_dir, bool force_not_share_global_sort_index)
{
    srand(time(0));
    unsigned long matrix_id = rand() + time(0) % 1000;

    // 写这个模板所需要数据的文件夹名称
    output_dir = output_dir + "/" + to_string(matrix_id) + "_" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer());

    // 创建这个文件夹
    system(("mkdir " + output_dir).c_str());

    // 要写到硬盘中的数据
    // void* global_row_index_of_thread_level_block = NULL;
    // void* block_begin_warp_index_offset = NULL;
    // void* warp_begin_thread_index_offset = NULL;
    // void* thread_block_size_in_warp = NULL;
    // void* row_index_before_sort = NULL;

    // 只有不压缩的时候才持久化
    if (output_template->global_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_thread_level_block != NULL);
        print_arr_to_file_with_data_type(output_template->global_row_index_of_thread_level_block, output_template->data_type_of_global_row_index_of_thread_level_block, output_template->size_of_global_row_index_of_thread_level_block, output_dir + "/global_row_index_of_thread_level_block");
    }

    if (output_template->block_begin_warp_index_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_warp_index_offset != NULL);
        print_arr_to_file_with_data_type(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, output_template->size_of_block_begin_warp_index_offset, output_dir + "/block_begin_warp_index_offset");
    }

    if (output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
    {
        assert(output_template->warp_begin_thread_index_offset != NULL);
        print_arr_to_file_with_data_type(output_template->warp_begin_thread_index_offset, output_template->data_type_of_warp_begin_thread_index_offset, output_template->size_of_warp_begin_thread_index_offset, output_dir + "/warp_begin_thread_index_offset");
    }

    if (output_template->thread_block_size_compress == NONE_COMPRESS)
    {
        // 这和数组一定有
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
    print_arr_to_file_with_data_type(output_template->val_arr, output_template->data_type_of_val_arr, output_template->size_of_val_arr, output_dir + "/val_arr");

    // 列
    assert(output_template->col_index_arr != NULL);
    print_arr_to_file_with_data_type(output_template->col_index_arr, output_template->data_type_of_col_index_arr, output_template->size_of_col_index_arr, output_dir + "/col_index_arr");

    output_template->hash_of_this_template = matrix_id;
}

string code_of_template_data_struct(direct_atom_template_t *output_template, unsigned long dense_block_id)
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

    if (output_template->block_begin_warp_index_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_warp_index_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_block_begin_warp_index_offset, code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset") + " = " + to_string(output_template->size_of_block_begin_warp_index_offset) + ";\n";
    }

    return_str = return_str + "\n";

    if (output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
    {
        assert(output_template->warp_begin_thread_index_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_warp_begin_thread_index_offset, code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset") + " = " + to_string(output_template->size_of_warp_begin_thread_index_offset) + ";\n";
    }

    return_str = return_str + "\n";

    if (output_template->thread_block_size_compress == NONE_COMPRESS)
    {
        // 这和数组一定有
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

// 设计一个函数来输出一个读取所有内容的函数声明
string code_of_read_template_data_from_file_func_define(direct_atom_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index)
{
    string return_str = "compressed_dense_block_" + to_string(dense_block_id) + "_t* read_dense_block_" + to_string(dense_block_id) + "_from_file(string file_name_prefix)\n{\n";

    return_str = return_str + "compressed_dense_block_" + to_string(dense_block_id) + "_t *template_data = new " + "compressed_dense_block_" + to_string(dense_block_id) + "_t();\n";

    // 5个数组
    // 对应的位置分别存储行号和块号
    if (output_template->global_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_thread_level_block != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_thread_level_block") + " = (" + code_of_data_type(output_template->data_type_of_global_row_index_of_thread_level_block) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_thread_level_block") + ", " + convert_data_type_to_string(output_template->data_type_of_global_row_index_of_thread_level_block) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/global_row_index_of_thread_level_block\");\n";
    }
    
    return_str = return_str + "\n";

    if (output_template->block_begin_warp_index_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_warp_index_offset != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset") + " = (" + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset") + ", " + convert_data_type_to_string(output_template->data_type_of_block_begin_warp_index_offset) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/block_begin_warp_index_offset\");\n";
    }

    return_str = return_str + "\n";

    if (output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
    {
        assert(output_template->warp_begin_thread_index_offset != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset") + " = (" + code_of_data_type(output_template->data_type_of_warp_begin_thread_index_offset) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset") + ", " + convert_data_type_to_string(output_template->data_type_of_warp_begin_thread_index_offset) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/warp_begin_thread_index_offset\");\n";
    }

    return_str = return_str + "\n";

    if (output_template->thread_block_size_compress == NONE_COMPRESS)
    {
        // 这和数组一定有
        assert(output_template->thread_block_size_in_warp != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp") + " = (" + code_of_data_type(output_template->data_type_of_thread_block_size_in_warp) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp") + ", " + convert_data_type_to_string(output_template->data_type_of_thread_block_size_in_warp) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/thread_block_size_in_warp\");\n";
    }

    return_str = return_str + "\n";

    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->row_index_before_sort != NULL)
    {
        // 如果有全局的排序索引，只有0号块或者强制存储的部分需要存储
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

string code_of_template_kernal(direct_atom_template_t *output_template, unsigned long dense_block_id)
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

    if (output_template->block_begin_warp_index_compress == NONE_COMPRESS)
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

    if (output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
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

    if (output_template->thread_block_size_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }
        // 这和数组一定有
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

    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        // warp首相对非零元
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

    // 计算
    return_str = return_str + "int tid_in_warp = threadIdx.x % 32;\n";
    return_str = return_str + "int bid = blockIdx.x;\n";

    // tid_in_block需要接受一个判断，判断是不是需要的
    bool need_tid_in_block = false;
    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS || output_template->block_begin_warp_index_compress == NONE_COMPRESS)
    {
        need_tid_in_block = true;
    }

    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS || output_template->thread_block_size_compress == NONE_COMPRESS || output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
    {
        need_tid_in_block = true;
    }

    if (need_tid_in_block == true)
    {
        return_str = return_str + "int tid_in_block = threadIdx.x;\n";
    }

    return_str = return_str + "int wid_in_block = (int)(threadIdx.x / 32);\n";

    if (output_template->kernal_first_row_index != 0)
    {
        return_str = return_str + "unsigned long kernal_first_row_index = " + to_string(output_template->kernal_first_row_index) + ";\n";
    }

    if (output_template->kernal_first_col_index != 0)
    {
        return_str = return_str + "unsigned long kernal_first_col_index = " + to_string(output_template->kernal_first_col_index) + ";\n";
    }

    if (!(output_template->tblock_num == (output_template->size_of_block_begin_warp_index_offset - 1)))
    {
        return_str = return_str + "int bnum = gridDim.x;\n";
    }

    if (!(output_template->block_begin_warp_index_compress == LINEAR_COMPRESS && ((linear_compress_t *)(output_template->block_begin_warp_index_compress_meta))->coefficient == ((output_template->thread_num_in_block) / 32)))
    {
        return_str = return_str + "int wnum = blockDim.x / 32;\n\n";
    }

    // 当前block共享内存的用量
    unsigned long shared_memory_item_num = 0;

    // 如果block首非零元索引是压缩过的，那就不用共享内存
    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_nz_begin_offset) + " block_first_nz_index_shared[1];\n";
        shared_memory_item_num = shared_memory_item_num + 1;
    }
    else
    {
        // cout << "compress type is not supported" << endl;
        // exit(-1);
    }

    // 如果block首warp索引没有压缩，那就用共享内存
    if (output_template->block_begin_warp_index_compress == NONE_COMPRESS)
    {
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " first_warp_index_of_this_block_shared[1];\n";
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " first_warp_index_of_next_block_shared[1];\n";
        shared_memory_item_num = shared_memory_item_num + 2;
    }
    else
    {
    }

    // 查看block内部warp最多是多少
    unsigned long max_warp_num_in_block = 0;

    // 只要存在未压缩的warp级别元数据，就需要计算block最大的warp数量
    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS || output_template->thread_block_size_compress == NONE_COMPRESS || output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
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

    // 没有压缩的warp第一个非零元的相对索引
    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        assert(max_warp_num_in_block > 0);

        shared_memory_item_num = shared_memory_item_num + max_warp_num_in_block;

        // 查看共享内存是不是够用，不够用说明之前的分块手段不好
        if (shared_memory_item_num + 1 > get_config()["SHARED_MEM_TOTAL_SIZE"].as_integer())
        {
            cout << "code_of_template_kernal: shared memory overflow, error: code_of_template_kernal" << endl;
            // exit(-1);
        }

        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_warp_nz_begin_offset) + " warp_first_nz_index_shared[" + to_string(max_warp_num_in_block) + "];\n";
    }
    else
    {
        // cout << "compress type is not supported" << endl;
        // exit(-1);
    }

    // warp第一个线程粒度的块的索引没压缩就放到共享内存中
    if (output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
    {
        assert(max_warp_num_in_block > 0);
        shared_memory_item_num = shared_memory_item_num + max_warp_num_in_block + 1;
        if (shared_memory_item_num + 1 > get_config()["SHARED_MEM_TOTAL_SIZE"].as_integer())
        {
            cout << "code_of_template_kernal: shared memory overflow, error: code_of_template_kernal" << endl;
            // exit(-1);
        }
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_warp_begin_thread_index_offset) + " warp_begin_thread_index_shared[" + to_string(max_warp_num_in_block + 1) + "];\n";
    }
    else
    {
    }

    // 每个warp内的线程粒度的块的大小
    if (output_template->thread_block_size_compress == NONE_COMPRESS)
    {
        assert(max_warp_num_in_block > 0);
        shared_memory_item_num = shared_memory_item_num + max_warp_num_in_block;
        if (shared_memory_item_num + 1 > get_config()["SHARED_MEM_TOTAL_SIZE"].as_integer())
        {
            cout << "code_of_template_kernal: shared memory overflow, error: code_of_template_kernal" << endl;
            // exit(-1);
        }
        return_str = return_str + "__shared__ " + code_of_data_type(output_template->data_type_of_thread_block_size_in_warp) + " thread_block_size_in_warp_shared[" + to_string(max_warp_num_in_block) + "];\n";
    }
    else
    {
    }

    // 遍历block级别的块
    // 如果线程块数量和实际block块数量相同，去掉第一层for循环
    if (output_template->tblock_num == (output_template->size_of_block_begin_warp_index_offset - 1))
    {
        // 不需要第一个for循环
        return_str = return_str + "{\n";
        return_str = return_str + "int block_level_block_id = bid;\n";
    }
    else
    {
        return_str = return_str + "for(int block_level_block_id = bid; block_level_block_id < " + to_string(output_template->size_of_block_begin_warp_index_offset - 1) + "; block_level_block_id = block_level_block_id + bnum)\n{\n";
    }

    return_str = return_str + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " first_warp_index_of_this_block;\n";
    return_str = return_str + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " first_warp_index_of_next_block;\n";

    // 只有不压缩的时候才有
    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS || output_template->thread_block_size_compress == NONE_COMPRESS || output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
    {
        return_str = return_str + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " warp_block_num_in_this_block;\n\n";
    }
    else
    {
        // cout << "compress type is not supported" << endl;
        // exit(-1);
    }

    // block的起始warp索引和下一个block起始索引的赋值，根据是否压缩，选择不同的赋值方式
    // if (output_template->block_begin_warp_index_compress == NONE_COMPRESS)
    // {
    //     return_str = return_str + "first_warp_index_of_this_block = block_begin_warp_index_offset[block_level_block_id];\n";
    // }
    // else if (output_template->block_begin_warp_index_compress == LINEAR_COMPRESS)
    // {
    //     // block层次可能有线性压缩，生成对应的代码
    //     assert(output_template->block_begin_warp_index_compress_meta != NULL);
    //     linear_compress_t *compressor = (linear_compress_t *)output_template->block_begin_warp_index_compress_meta;
    //     return_str = return_str + code_of_arr_read(compressor, "first_warp_index_of_next_block", "block_level_block_id") + ";\n";
    // }
    // else
    // {
    //     cout << "compress type is not supported,block_begin_warp_index_compress" << endl;
    //     exit(-1);
    // }

    // if (output_template->block_begin_warp_index_compress == NONE_COMPRESS)
    // {
    //     return_str = return_str + "first_warp_index_of_next_block = block_begin_warp_index_offset[block_level_block_id + 1];\n";
    // }
    // else if (output_template->block_begin_warp_index_compress == LINEAR_COMPRESS)
    // {
    //     // block层次可能有线性压缩，生成对应的代码
    //     assert(output_template->block_begin_warp_index_compress_meta != NULL);
    //     linear_compress_t *compressor = (linear_compress_t *)output_template->block_begin_warp_index_compress_meta;
    //     return_str = return_str + code_of_arr_read(compressor, "first_warp_index_of_next_block", "(block_level_block_id + 1)") + ";\n";
    // }
    // else
    // {
    //     cout << "compress type is not supported,block_begin_warp_index_compress" << endl;
    //     exit(-1);
    // }

    // // 只有warp_nz_begin_offset_compress不存在的时候才需要每个block的warp数量
    // if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS)
    // {
    //     return_str = return_str + "warp_block_num_in_this_block = first_warp_index_of_next_block - first_warp_index_of_this_block;\n\n";
    // }
    // else
    // {
    // }

    // 只要有一个不压缩，就需要在一开始有一个全局同步
    if (output_template->block_begin_warp_index_compress == NONE_COMPRESS || output_template->block_nz_begin_offset_compress == NONE_COMPRESS || output_template->warp_nz_begin_offset_compress == NONE_COMPRESS ||
        output_template->thread_block_size_compress == NONE_COMPRESS || output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
    {
        return_str = return_str + "__syncthreads();\n\n";
    }

    // 初始化共享内存
    // block层次所有的三个数组
    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS || output_template->block_begin_warp_index_compress == NONE_COMPRESS)
    {
        return_str = return_str + "if(tid_in_block == 0)\n{\n";

        if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
        {
            return_str = return_str + "block_first_nz_index_shared[0] = block_nz_begin_offset[block_level_block_id];\n";
        }

        if (output_template->block_begin_warp_index_compress == NONE_COMPRESS)
        {
            return_str = return_str + "first_warp_index_of_this_block_shared[0] = block_begin_warp_index_offset[block_level_block_id];\n";
            return_str = return_str + "first_warp_index_of_next_block_shared[0] = block_begin_warp_index_offset[block_level_block_id + 1];\n";
        }

        return_str = return_str + "}\n\n";
        return_str = return_str + "__syncthreads();\n\n";
    }

    // 获取块级别元数据，主要是warp的索引
    // block的起始warp索引和下一个block起始索引的赋值，根据是否压缩，选择不同的赋值方式
    if (output_template->block_begin_warp_index_compress == NONE_COMPRESS)
    {
        // 没压缩就从
        return_str = return_str + "first_warp_index_of_this_block = first_warp_index_of_this_block_shared[0];\n";
        return_str = return_str + "first_warp_index_of_next_block = first_warp_index_of_next_block_shared[0];\n";
    }
    else if (output_template->block_begin_warp_index_compress == LINEAR_COMPRESS)
    {
        // block层次可能有线性压缩，生成对应的代码
        assert(output_template->block_begin_warp_index_compress_meta != NULL);
        linear_compress_t *compressor = (linear_compress_t *)output_template->block_begin_warp_index_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "first_warp_index_of_this_block", "block_level_block_id") + ";\n";

        // 第二个直接加上斜率，减少计算量
        return_str = return_str + "first_warp_index_of_next_block = first_warp_index_of_this_block + " + to_string(compressor->coefficient) + ";\n";
    }
    else
    {
        cout << "compress type is not supported,block_begin_warp_index_compress" << endl;
        assert(false);
    }

    // 只有warp级别的几个元数据都没有压缩的时候才需要这个变量
    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS || output_template->thread_block_size_compress == NONE_COMPRESS || output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
    {
        return_str = return_str + "warp_block_num_in_this_block = first_warp_index_of_next_block - first_warp_index_of_this_block;\n\n";

        // 初始化warp级别的元数据
        return_str = return_str + "for(int i = tid_in_block; i < warp_block_num_in_this_block; i = i + blockDim.x)\n{\n";

        if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS)
        {
            return_str = return_str + "warp_first_nz_index_shared[i] = warp_nz_begin_offset[first_warp_index_of_this_block + i];\n";
        }

        if (output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
        {
            return_str = return_str + "warp_begin_thread_index_shared[i] = warp_begin_thread_index_offset[first_warp_index_of_this_block + i];\n";
        }

        if (output_template->thread_block_size_compress == NONE_COMPRESS)
        {
            return_str = return_str + "thread_block_size_in_warp_shared[i] = thread_block_size_in_warp[first_warp_index_of_this_block + i];\n";
        }

        return_str = return_str + "}\n\n";

        // warp_begin_thread_index_shared多一位要赋值
        if (output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
        {
            return_str = return_str + "if(tid_in_block == 0)\n{\n";
            return_str = return_str + "warp_begin_thread_index_shared[warp_block_num_in_this_block] = warp_begin_thread_index_offset[first_warp_index_of_this_block + warp_block_num_in_this_block];\n";
            return_str = return_str + "}\n";
        }

        return_str = return_str + "__syncthreads();\n\n";
    }
    else
    {
    }

    // 声明block的第一个非零元全局索引
    return_str = return_str + code_of_data_type(output_template->data_type_of_block_nz_begin_offset) + " this_block_first_nz_index;\n";

    // 如果block首非零元没有压缩，那就从共享内存读
    if (output_template->block_nz_begin_offset_compress == NONE_COMPRESS)
    {
        return_str = return_str + "this_block_first_nz_index = block_first_nz_index_shared[0];\n\n";
    }
    else if (output_template->block_nz_begin_offset_compress == LINEAR_COMPRESS)
    {
        assert(output_template->block_nz_begin_offset_compress_meta != NULL);
        linear_compress_t *compressor = (linear_compress_t *)output_template->block_nz_begin_offset_compress_meta;
        // 用全局tblock号来获得全局的tblock非零元首行索引
        return_str = return_str + code_of_arr_read(compressor, "this_block_first_nz_index", "block_level_block_id") + ";\n";
    }
    else
    {
        cout << "compress type is not supported,block_nz_begin_offset_compress" << endl;
        assert(false);
    }

    if (output_template->block_begin_warp_index_compress == LINEAR_COMPRESS)
    {
        assert(output_template->block_begin_warp_index_compress_meta != NULL);
        assert((output_template->thread_num_in_block) % 32 == 0);
    }

    if (output_template->block_begin_warp_index_compress == LINEAR_COMPRESS && ((linear_compress_t *)(output_template->block_begin_warp_index_compress_meta))->coefficient == ((output_template->thread_num_in_block) / 32))
    {
        // 如果每个block的warp数量相等，并且等于线程块中分warp数量，那就不用for循环，直接从初值还是算即可
        return_str = return_str + "{\n";
        return_str = return_str + "unsigned int" + " warp_level_block_id = first_warp_index_of_this_block + wid_in_block;\n";
    }
    else
    {
        // warp层次的遍历
        return_str = return_str + "for(" + "unsigned int";
        return_str = return_str + " warp_level_block_id = first_warp_index_of_this_block + wid_in_block; warp_level_block_id < first_warp_index_of_next_block; warp_level_block_id = warp_level_block_id + wnum)\n{\n";
    }

    // 一个block内warp相对索引，数据类型是block第一个warp索引这个数组的数据类型，当warp级别的元数据有未压缩的时候才需要
    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS || output_template->thread_block_size_compress == NONE_COMPRESS || output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
    {
        return_str = return_str + code_of_data_type(output_template->data_type_of_block_begin_warp_index_offset) + " local_warp_level_block_id = warp_level_block_id - first_warp_index_of_this_block;\n";
    }

    return_str = return_str + code_of_data_type(output_template->data_type_of_warp_nz_begin_offset) + " local_this_warp_first_nz_index;\n";

    // 根据warp nz的压缩情况来决定warp_first_nz_index的读取方法，可能采用的是线性周期压缩
    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        return_str = return_str + "local_this_warp_first_nz_index = warp_first_nz_index_shared[local_warp_level_block_id];\n";
    }
    else if (output_template->warp_nz_begin_offset_compress == CYCLE_LINEAR_COMPRESS)
    {
        assert(output_template->warp_nz_begin_offset_compress_meta != NULL);
        // 压缩之后使用全局索引
        cycle_linear_compress_t *compressor = (cycle_linear_compress_t *)output_template->warp_nz_begin_offset_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "local_this_warp_first_nz_index", "warp_level_block_id") + ";\n";
    }
    else
    {
        cout << "compress type is not supported,warp_nz_begin_offset_compress" << endl;
        assert(false);
    }

    // warp内部的每一个线程子块的大小
    return_str = return_str + code_of_data_type(output_template->data_type_of_thread_block_size_in_warp) + " thread_block_size_in_this_warp;\n";

    // 根据是否压缩来决定赋值方式
    if (output_template->thread_block_size_compress == NONE_COMPRESS)
    {
        return_str = return_str + "thread_block_size_in_this_warp = thread_block_size_in_warp_shared[local_warp_level_block_id];\n";
    }
    else if (output_template->thread_block_size_compress == CONSTANT_COMPRESS)
    {
        assert(output_template->thread_block_size_compress_meta != NULL);
        constant_compress_t *compressor = (constant_compress_t *)output_template->thread_block_size_compress_meta;
        // 拼接常量的代码，对于压缩之后代码而言，使用的是全局索引
        return_str = return_str + code_of_arr_read(compressor, "thread_block_size_in_this_warp", "warp_level_block_id") + ";\n";
    }
    else if (output_template->thread_block_size_compress == BRANCH_COMPRESS)
    {
        assert(output_template->thread_block_size_compress_meta != NULL);
        branch_compress_t *compressor = (branch_compress_t *)output_template->thread_block_size_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "thread_block_size_in_this_warp", "warp_level_block_id") + "\n";
    }
    else
    {
        cout << "compress type is not supported,thread_block_size_compress" << endl;
        assert(false);
    }

    return_str = return_str + "\n";

    // 一个warp内thread的上界和下界，可以线性压缩
    return_str = return_str + code_of_data_type(output_template->data_type_of_warp_begin_thread_index_offset) + " first_thread_index_of_this_warp;\n";

    if (output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
    {
        return_str = return_str + "first_thread_index_of_this_warp =  warp_begin_thread_index_shared[local_warp_level_block_id];\n";
    }
    else if (output_template->warp_begin_thread_index_compress == LINEAR_COMPRESS)
    {
        assert(output_template->warp_begin_thread_index_compress_meta != NULL);
        linear_compress_t *compressor = (linear_compress_t *)output_template->warp_begin_thread_index_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "first_thread_index_of_this_warp", "warp_level_block_id") + ";\n";
    }
    else
    {
        cout << "compress type is not supported,warp_begin_thread_index_compress" << endl;
        assert(false);
    }

    if (output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
    {
        return_str = return_str + code_of_data_type(output_template->data_type_of_warp_begin_thread_index_offset) + " first_thread_index_of_next_warp;\n";
        return_str = return_str + "first_thread_index_of_next_warp =  warp_begin_thread_index_shared[local_warp_level_block_id + 1];\n";
    }
    else if (output_template->warp_begin_thread_index_compress == LINEAR_COMPRESS)
    {
        // 如果是线性压缩，不需要thread编号的上界
        // assert(output_template->warp_begin_thread_index_compress_meta != NULL);
        // linear_compress_t *compressor = (linear_compress_t *)output_template->warp_begin_thread_index_compress_meta;
        // return_str = return_str + code_of_arr_read(compressor, "first_thread_index_of_next_warp", "(warp_level_block_id + 1)") + ";\n";
    }
    else
    {
        cout << "compress type is not supported,warp_begin_thread_index_compress" << endl;
        assert(false);
    }

    return_str = return_str + "\n";

    // 声明warp中的组数量只有在没有压缩的时候才需要
    // 遍历所有的warp，获取最多的组数量
    unsigned long max_group_num = 0;
    for (unsigned long i = 0; i < output_template->size_of_warp_begin_thread_index_offset - 1; i++)
    {
        unsigned long first_thread_index_of_this_warp = read_from_array_with_data_type(output_template->warp_begin_thread_index_offset, output_template->data_type_of_warp_begin_thread_index_offset, i);
        unsigned long first_thread_index_of_next_warp = read_from_array_with_data_type(output_template->warp_begin_thread_index_offset, output_template->data_type_of_warp_begin_thread_index_offset, i + 1);
        assert((first_thread_index_of_next_warp - first_thread_index_of_this_warp) % 32 == 0);
        if (max_group_num < (first_thread_index_of_next_warp - first_thread_index_of_this_warp) / 32)
        {
            max_group_num = (first_thread_index_of_next_warp - first_thread_index_of_this_warp) / 32;
        }
    }

    // 如果存在group这个层次，group相关元数据的计算
    // 这里要推测出可能的数据类型
    if (output_template->warp_begin_thread_index_compress == LINEAR_COMPRESS && ((linear_compress_t *)(output_template->warp_begin_thread_index_compress_meta))->coefficient == 32)
    {
        // 有压缩，不需要group层次的元数据
    }
    else
    {
        // 如果是线性压缩，一个warp中group的数量就一定是一个定值
        if (output_template->warp_begin_thread_index_compress == LINEAR_COMPRESS)
        {
            assert(output_template->warp_begin_thread_index_compress_meta != NULL);
            linear_compress_t *compressor = (linear_compress_t *)output_template->warp_begin_thread_index_compress_meta;
            assert(compressor->coefficient % 32 == 0);
            // 直接用常值优化
            return_str = return_str + code_of_data_type(find_most_suitable_data_type((unsigned long)(compressor->coefficient / 32))) + " group_num_in_this_warp = " + to_string((unsigned long)(compressor->coefficient / 32)) + ";\n";
        }
        else
        {
            return_str = return_str + code_of_data_type(find_most_suitable_data_type(max_group_num)) + " group_num_in_this_warp = (first_thread_index_of_next_warp - first_thread_index_of_this_warp) / 32;\n";
        }
    }

    // 如果一个warp里面只有一个组，就直接省略group层次的遍历，可以减少大量元数据的计算
    //
    if (output_template->warp_begin_thread_index_compress == LINEAR_COMPRESS && ((linear_compress_t *)(output_template->warp_begin_thread_index_compress_meta))->coefficient == 32)
    {
        assert(((linear_compress_t *)(output_template->warp_begin_thread_index_compress_meta))->intercept == 0);

        return_str = return_str + "{\n";

        // 获取组的第一个非零元的索引

        return_str = return_str + "unsigned int" + " thread_block_group_first_nz = this_block_first_nz_index + local_this_warp_first_nz_index;\n\n";

        // 线程粒度的块的全局索引
        return_str = return_str + "unsigned int" + " global_thread_block_index = tid_in_warp + first_thread_index_of_this_warp;\n\n";
    }
    else
    {
        // 全局线程块号
        return_str = return_str + "unsigned int" + " global_thread_block_index = tid_in_warp + first_thread_index_of_this_warp;\n";

        // 获取组的第一个非零元的索引
        // 如果thread的大小是一个常数，这里执行一个预计算
        return_str = return_str + "unsigned int" + " thread_block_group_first_nz = this_block_first_nz_index + local_this_warp_first_nz_index;\n";

        // group层次的遍历
        // 遍历所有的组
        return_str = return_str + "for(" + "unsigned int";
        return_str = return_str + " thread_level_block_group_id = 0; thread_level_block_group_id < group_num_in_this_warp; thread_level_block_group_id = thread_level_block_group_id + 1)\n{\n";
    }

    // 用来存储thread对应结果的数值的变量
    return_str = return_str + "double thread_block_tmp_result = 0;\n\n";

    // 如果一个线程只主处理一个非零元，这层循环完全不需要
    if (output_template->thread_block_size_compress == CONSTANT_COMPRESS && ((constant_compress_t *)(output_template->thread_block_size_compress_meta))->constant == 1)
    {
        return_str = return_str + "{\n";
        return_str = return_str + "unsigned int" + " global_nz_index = thread_block_group_first_nz + tid_in_warp;\n";
    }
    else
    {
        // 声明全局非零元的大小
        return_str = return_str + "unsigned int" + " global_nz_index = thread_block_group_first_nz + tid_in_warp;\n";

        // 用来计算，遍历一个线程粒度的块
        return_str = return_str + "for(" + "unsigned int" + " nz_index_in_thread = 0; nz_index_in_thread < thread_block_size_in_this_warp; nz_index_in_thread = nz_index_in_thread + 1)\n{\n";
    }

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

    // 如果没有被常量1压缩，这里自增获得下一次遍历的全局非零元索引
    if (output_template->thread_block_size_compress == CONSTANT_COMPRESS && ((constant_compress_t *)(output_template->thread_block_size_compress_meta))->constant == 1)
    {
        // 常量压缩只有一次循环，不需要考虑非零元索引的自增
    }
    else
    {
        // 声明全局非零元的大小
        return_str = return_str + "global_nz_index = global_nz_index + " + "32" + ";\n";
    }

    return_str = return_str + "}\n\n";

    // 获取局部的行号
    return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->matrix->dense_row_number)) + " global_row_index;\n";

    // 每个线程粒度的行号
    if (output_template->global_row_index_compress == NONE_COMPRESS)
    {
        return_str = return_str + "global_row_index = global_row_index_of_thread_level_block[global_thread_block_index];\n";
    }
    else if (output_template->global_row_index_compress == LINEAR_COMPRESS)
    {
        assert(output_template->global_row_index_compress_meta != NULL);
        linear_compress_t *compressor = (linear_compress_t *)output_template->global_row_index_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "global_row_index", "global_thread_block_index") + ";\n";
    }
    else if (output_template->global_row_index_compress == CYCLE_INCREASE_COMPRESS)
    {
        assert(output_template->global_row_index_compress_meta != NULL);
        cycle_increase_compress_t *compressor = (cycle_increase_compress_t *)output_template->global_row_index_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "global_row_index", "global_thread_block_index") + ";\n";
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

    // 对于完全不压缩的情况，这里自增一下thread_block_group_first_nz
    // 这里是group层次被压缩的情况
    if (output_template->warp_begin_thread_index_compress == LINEAR_COMPRESS && ((linear_compress_t *)(output_template->warp_begin_thread_index_compress_meta))->coefficient == 32)
    {
        // 如果不压缩
    }
    else
    {
        return_str = return_str + "thread_block_group_first_nz = thread_block_group_first_nz + ";

        // 根据thread_block_size_in_this_warp是否压缩填入对应的值
        if (output_template->thread_block_size_compress == CONSTANT_COMPRESS)
        {
            assert(output_template->thread_block_size_compress_meta != NULL);
            constant_compress_t *compressor = (constant_compress_t *)output_template->thread_block_size_compress_meta;
            return_str = return_str + to_string(32 * compressor->constant) + ";\n\n";
        }
        else
        {
            // 其他的压缩方式thread_block_size_in_this_warp不是一个常值，需要在运行时计算
            return_str = return_str + "32 * thread_block_size_in_this_warp;\n\n";
        }

        return_str = return_str + "global_thread_block_index = global_thread_block_index + 32;\n\n";
    }

    return_str = return_str + "}\n";
    return_str = return_str + "}\n";
    return_str = return_str + "}\n";
    return_str = return_str + "}\n";

    return return_str;
}

// 模板对应的核函数调用
string code_of_kernal_function_call(direct_atom_template_t *output_template, unsigned long dense_block_id)
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

    if (output_template->block_begin_warp_index_compress == NONE_COMPRESS)
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

    if (output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
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

    if (output_template->thread_block_size_compress == NONE_COMPRESS)
    {
        if (is_first_param == false)
        {
            return_str = return_str + ", ";
        }
        else
        {
            is_first_param = false;
        }
        // 这和数组一定有
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

    if (output_template->warp_nz_begin_offset_compress == NONE_COMPRESS)
    {
        // warp首相对非零元
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

string code_of_write_template_data_to_gpu(direct_atom_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index)
{
    // 读到对应结构体中的代码
    // 存储结构体的名字
    string template_data_name = "dense_block_" + to_string(dense_block_id) + "_template_data";

    string return_str = "compressed_dense_block_" + to_string(dense_block_id) + "_t *" + template_data_name + " = read_dense_block_" + to_string(dense_block_id) + "_from_file(" + "\"" + string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source/" + to_string(output_template->hash_of_this_template) + "_" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer()) + "\");\n\n";

    // 如果是global的排序索引是共享的，全局排序的数组取一个特殊的名字，并且只处理一次，剩下的从这里拷贝即可
    // 但是如果不是共享的，那就不需要由一个子块先把数据取出来，这段逻辑是不需要的
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

    // 声明对应的GPU指针
    // 要写到硬盘中的数据
    // void* global_row_index_of_thread_level_block = NULL;
    // void* block_begin_warp_index_offset = NULL;
    // void* warp_begin_thread_index_offset = NULL;
    // void* thread_block_size_in_warp = NULL;
    // void* row_index_before_sort = NULL;

    if (output_template->global_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_thread_level_block != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_global_row_index_of_thread_level_block, "device_" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_thread_level_block"));
    }

    if (output_template->block_begin_warp_index_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_warp_index_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_block_begin_warp_index_offset, "device_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset"));
    }

    if (output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
    {
        assert(output_template->warp_begin_thread_index_offset != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_warp_begin_thread_index_offset, "device_" + code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset"));
    }

    if (output_template->thread_block_size_compress == NONE_COMPRESS)
    {
        // 这和数组一定有
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
    // 申请数组的代码
    if (output_template->global_row_index_compress == NONE_COMPRESS)
    {
        assert(output_template->global_row_index_of_thread_level_block != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_global_row_index_of_thread_level_block, to_string(output_template->size_of_global_row_index_of_thread_level_block), "device_" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_thread_level_block"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_thread_level_block"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "global_row_index_of_thread_level_block"), output_template->data_type_of_global_row_index_of_thread_level_block, to_string(output_template->size_of_global_row_index_of_thread_level_block), "cudaMemcpyHostToDevice") + "\n";
    }

    if (output_template->block_begin_warp_index_compress == NONE_COMPRESS)
    {
        assert(output_template->block_begin_warp_index_offset != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_block_begin_warp_index_offset, to_string(output_template->size_of_block_begin_warp_index_offset), "device_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "block_begin_warp_index_offset"), output_template->data_type_of_block_begin_warp_index_offset, to_string(output_template->size_of_block_begin_warp_index_offset), "cudaMemcpyHostToDevice") + "\n";
    }

    if (output_template->warp_begin_thread_index_compress == NONE_COMPRESS)
    {
        assert(output_template->warp_begin_thread_index_offset != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_warp_begin_thread_index_offset, to_string(output_template->size_of_warp_begin_thread_index_offset), "device_" + code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "warp_begin_thread_index_offset"), output_template->data_type_of_warp_begin_thread_index_offset, to_string(output_template->size_of_warp_begin_thread_index_offset), "cudaMemcpyHostToDevice") + "\n";
    }

    if (output_template->thread_block_size_compress == NONE_COMPRESS)
    {
        // 这和数组一定有
        assert(output_template->thread_block_size_in_warp != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_thread_block_size_in_warp, to_string(output_template->size_of_thread_block_size_in_warp), "device_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "thread_block_size_in_warp"), output_template->data_type_of_thread_block_size_in_warp, to_string(output_template->size_of_thread_block_size_in_warp), "cudaMemcpyHostToDevice") + "\n";
    }

    // 如果是全局的就直接赋值
    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->global_sort_index == true)
    {
        assert(output_template->local_sort_index == false);
        assert(output_template->row_index_before_sort != NULL);
        if (force_not_share_global_sort_index == true)
        {
            return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_row_index_before_sort, to_string(output_template->size_of_row_index_before_sort), "device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"));
            // 拷贝
            return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"), output_template->data_type_of_row_index_before_sort, to_string(output_template->size_of_row_index_before_sort), "cudaMemcpyHostToDevice") + "\n";
        }
        else
        {
            // 如果全局行排序索引是共享的，就用下面这行代码
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



bool compress_global_row_index_of_thread_level_block(direct_atom_template_t *output_template, bool need_check, arr_compress_type type)
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

bool compress_block_begin_warp_index_offset(direct_atom_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && type == LINEAR_COMPRESS && output_template->block_begin_warp_index_offset != NULL);

    linear_compress_t *compressor = init_linear_compressor(output_template->block_begin_warp_index_offset, output_template->data_type_of_block_begin_warp_index_offset, output_template->size_of_block_begin_warp_index_offset, need_check);

    if (compressor == NULL)
    {
        return false;
    }

    // 压缩成功，拷贝元数据
    output_template->block_begin_warp_index_compress_meta = (void *)compressor;
    output_template->block_begin_warp_index_compress = type;

    return true;
}

bool compress_warp_begin_thread_index_offset(direct_atom_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && type == LINEAR_COMPRESS && output_template->warp_begin_thread_index_offset != NULL);

    linear_compress_t *compressor = init_linear_compressor(output_template->warp_begin_thread_index_offset, output_template->data_type_of_warp_begin_thread_index_offset, output_template->size_of_warp_begin_thread_index_offset, need_check);

    if (compressor == NULL)
    {
        return false;
    }

    // 压缩成功
    output_template->warp_begin_thread_index_compress_meta = (void *)compressor;
    output_template->warp_begin_thread_index_compress = type;

    return true;
}

// 可以常量压缩和分支压缩
bool compress_thread_block_size_in_warp(direct_atom_template_t *output_template, bool need_check, arr_compress_type type)
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
        output_template->thread_block_size_compress_meta = (void *)compressor;
        output_template->thread_block_size_compress = type;
    }

    if (type == BRANCH_COMPRESS)
    {
        branch_compress_t *compressor = init_branch_compressor(output_template->thread_block_size_in_warp, output_template->data_type_of_thread_block_size_in_warp, output_template->size_of_thread_block_size_in_warp, need_check);

        if (compressor == NULL)
        {
            return false;
        }

        // 压缩成功
        output_template->thread_block_size_compress_meta = (void *)compressor;
        output_template->thread_block_size_compress = type;
    }

    return true;
}

bool compress_block_nz_begin_offset(direct_atom_template_t *output_template, bool need_check, arr_compress_type type)
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

bool compress_warp_nz_begin_offset(direct_atom_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && type == CYCLE_LINEAR_COMPRESS && output_template->warp_nz_begin_offset != NULL);

    unsigned long cycle_num;

    // 只有每个block的warp数量相等，才有这里压缩的可能性，首先查看是否压缩过
    if (output_template->block_begin_warp_index_compress == LINEAR_COMPRESS)
    {
        // 周期就是斜率，代表每一个block的warp数量
        linear_compress_t *compressor = (linear_compress_t *)output_template->block_begin_warp_index_compress_meta;
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

void try_all_compress(direct_atom_template_t *output_template)
{
    bool is_compressed = false;

    is_compressed = compress_global_row_index_of_thread_level_block(output_template, true, LINEAR_COMPRESS);

    if (is_compressed == false)
    {
        is_compressed = compress_global_row_index_of_thread_level_block(output_template, true, CYCLE_INCREASE_COMPRESS);
    }

    is_compressed = compress_block_begin_warp_index_offset(output_template, true, LINEAR_COMPRESS);

    is_compressed = compress_warp_begin_thread_index_offset(output_template, true, LINEAR_COMPRESS);

    is_compressed = compress_thread_block_size_in_warp(output_template, true, CONSTANT_COMPRESS);

    if (is_compressed == false)
    {
        is_compressed = compress_thread_block_size_in_warp(output_template, true, BRANCH_COMPRESS);
    }

    is_compressed = compress_block_nz_begin_offset(output_template, true, LINEAR_COMPRESS);

    is_compressed = compress_warp_nz_begin_offset(output_template, true, CYCLE_LINEAR_COMPRESS);
}
