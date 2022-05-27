#include "unaligned_warp_reduce_same_TLB_size_op.hpp"
#include <string>
#include <sstream>

unaligned_warp_reduce_same_TLB_size_template_t *init_unaligned_warp_reduce_same_TLB_size_template(code_builder_t *builder, unsigned long dense_block_id)
{
    assert(builder != NULL);

    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;
    assert(matrix->block_coor_table.item_arr.size() > dense_block_id);

    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr;
    assert(compressed_block_view != NULL);

    // 检查，warp和block层次都只有一个块，也就是放弃了分块
    index_of_compress_block_t *global_row_index = compressed_block_view->read_index[0];
    index_of_compress_block_t *global_col_index = compressed_block_view->read_index[1];
    index_of_compress_block_t *block_level_index = compressed_block_view->read_index[2];
    index_of_compress_block_t *warp_level_index = compressed_block_view->read_index[3];
    index_of_compress_block_t *thread_level_index = compressed_block_view->read_index[4];
    assert(global_row_index->type_of_index == ROW_INDEX);
    assert(global_col_index->type_of_index == COL_INDEX);
    assert(block_level_index->level_of_this_index == TBLOCK_LEVEL);
    assert(warp_level_index->level_of_this_index == WRAP_LEVEL);
    assert(thread_level_index->level_of_this_index == THREAD_LEVEL);

    assert(global_row_index->max_row_index == matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index);
    assert(global_row_index->max_row_index == block_level_index->max_row_index);
    assert(block_level_index->max_row_index == thread_level_index->max_row_index);

    // 线程粒度的块有每个TLB所占行号的记录
    assert(thread_level_index->row_number_of_block_arr != NULL);

    assert(block_level_index->block_num == 1 && warp_level_index->block_num == 1);

    // 线程粒度的块的大小，
    unsigned long global_TLB_size = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, 0);

    // 传入矩阵的已有COO格式，得到经过去除空行、padding到32*TLB_size大小矩阵。
    // COO矩阵的三个索引
    vector<unsigned long> dest_row_index_vec;
    vector<unsigned long> dest_col_index_vec;
    vector<double> dest_val_vec;

    assert(compressed_block_view->read_index[0]->type_of_index == ROW_INDEX);
    assert(compressed_block_view->read_index[1]->type_of_index == COL_INDEX);
    // 重构COO的三个矩阵
    fill_empty_and_padding_to_align_warp(compressed_block_view->read_index[0]->index_arr, compressed_block_view->read_index[1]->index_arr, compressed_block_view->val_arr,
                                         compressed_block_view->read_index[0]->index_data_type, compressed_block_view->read_index[1]->index_data_type, compressed_block_view->val_data_type,
                                         compressed_block_view->size, dest_row_index_vec, dest_col_index_vec, dest_val_vec, global_TLB_size);

    // TLB的数量
    assert(dest_col_index_vec.size() % global_TLB_size == 0);
    unsigned long TLB_num = dest_col_index_vec.size() / global_TLB_size;
    assert(TLB_num % 32 == 0);
    unsigned long WLB_num = TLB_num / 32;

    // 得到加和的起始位置的bool flag
    vector<vector<bool>> sum_begin_bool_flag = get_sum_begin_bool_flag_of_each_thread(dest_row_index_vec, global_TLB_size);
    assert(sum_begin_bool_flag.size() == TLB_num);

    // 得到WLB的首行索引
    vector<unsigned long> warp_level_block_first_row_vec = get_first_global_row_index_of_each_warp(dest_row_index_vec, global_TLB_size);
    assert(warp_level_block_first_row_vec.size() == WLB_num);

    // 归约偏移量的最大值
    unsigned long max_tmp_result_reduce_offset = 0;
    
    vector<unsigned long> tmp_result_reduce_offset_vec = get_tmp_result_reduce_offset_vec(sum_begin_bool_flag, &max_tmp_result_reduce_offset);
    assert(tmp_result_reduce_offset_vec.size() == TLB_num);

    // TLB的归约偏移量最大值
    unsigned long max_relative_reduce_row_of_thread_level_block = 0;
    
    vector<unsigned long> first_relative_reduce_row_of_thread_level_block_vec = get_first_relative_reduce_row_of_thread_level_block_vec(dest_row_index_vec, warp_level_block_first_row_vec, sum_begin_bool_flag, global_TLB_size, &max_relative_reduce_row_of_thread_level_block);
    assert(first_relative_reduce_row_of_thread_level_block_vec.size() == TLB_num);

    // store_bool_flag_of_sum_begin_to_file(sum_begin_bool_flag, "/home/duzhen/spmv_builder/data_source/test_result_4");
    // print_arr_to_file_with_data_type(&(first_relative_reduce_row_of_thread_level_block_vec[0]), UNSIGNED_LONG, first_relative_reduce_row_of_thread_level_block_vec.size(), "/home/duzhen/spmv_builder/data_source/test_result_5");
    // print_arr_to_file_with_data_type(&(tmp_result_reduce_offset_vec[0]), UNSIGNED_LONG, tmp_result_reduce_offset_vec.size(), "/home/duzhen/spmv_builder/data_source/test_result_3");
    // cout << "max_relative_reduce_row_of_thread_level_block:" << max_relative_reduce_row_of_thread_level_block << endl;
    // cout << "max_tmp_result_reduce_offset:" << max_tmp_result_reduce_offset << endl;

    // 查看三个元数据所占的bit
    int bit_num_of_relative_reduce_row_of_thread_level_block = get_max_bit_num_of_meta_data(max_relative_reduce_row_of_thread_level_block);
    int bit_num_of_tmp_result_reduce_offset = get_max_bit_num_of_meta_data(max_tmp_result_reduce_offset);
    int bit_num_of_sum_begin_flag = global_TLB_size;

    // 对于最大是0的元数据，也需要占用一个bit
    assert(bit_num_of_sum_begin_flag != 0);

    if (bit_num_of_relative_reduce_row_of_thread_level_block == 0)
    {
        bit_num_of_relative_reduce_row_of_thread_level_block = 1;
    }

    if (bit_num_of_tmp_result_reduce_offset == 0)
    {
        bit_num_of_tmp_result_reduce_offset = 1;
    }

    // 这三个加起来要小于64，要不就没有办法使用这个模板
    if (bit_num_of_relative_reduce_row_of_thread_level_block + bit_num_of_tmp_result_reduce_offset + bit_num_of_sum_begin_flag > 64)
    {
        cout << "too large meta data size:" << bit_num_of_relative_reduce_row_of_thread_level_block + bit_num_of_tmp_result_reduce_offset + bit_num_of_sum_begin_flag << ", is not supported" << endl;
        assert(false);
    }

    // cout << max_tmp_result_reduce_offset << endl;
    // cout << bit_num_of_tmp_result_reduce_offset << endl;
    // cout << convert_meta_data_to_bit_flag_string(bit_num_of_tmp_result_reduce_offset) << endl;
    // cout << convert_meta_data_to_bit_flag_string((unsigned long)combine_meta_data_to_unsigned_int(sum_begin_bool_flag[1], tmp_result_reduce_offset_vec[1], first_relative_reduce_row_of_thread_level_block_vec[1], bit_num_of_sum_begin_flag, bit_num_of_tmp_result_reduce_offset, bit_num_of_relative_reduce_row_of_thread_level_block)) << endl;
    // exit(-1);
    
    // 申请模板的实例，创建一个模板
    unaligned_warp_reduce_same_TLB_size_template_t* output_template = new unaligned_warp_reduce_same_TLB_size_template_t();
    
    output_template->dense_block_index = dense_block_id;
    output_template->matrix = matrix;
    output_template->kernal_first_row_index = matrix->block_coor_table.item_arr[dense_block_id]->min_dense_row_index;
    output_template->kernal_first_col_index = matrix->block_coor_table.item_arr[dense_block_id]->min_dense_col_index;

    // 找出最合适的数据结构
    data_type combine_meta_data_type = find_most_suitable_data_type_by_bit_num(bit_num_of_relative_reduce_row_of_thread_level_block + bit_num_of_tmp_result_reduce_offset + bit_num_of_sum_begin_flag);
    // 数据结构对应的bit占用
    int combine_meta_bit_num = bit_num_of_data_type(combine_meta_data_type);

    output_template->bit_num_of_thread_level_combine_meta = combine_meta_bit_num;
    output_template->bit_num_of_sum_begin_bit_flag = bit_num_of_sum_begin_flag;
    output_template->bit_num_of_first_relative_reduce_row_of_thread_level_block = bit_num_of_relative_reduce_row_of_thread_level_block;
    output_template->bit_num_of_tmp_result_reduce_offset_of_thread_level_block = bit_num_of_tmp_result_reduce_offset;

    // cout << max_relative_reduce_row_of_thread_level_block << " , " << max_tmp_result_reduce_offset << endl;

    // 如果出现过列分块，那就全部强制原子加
    if (matrix->block_coor_table.item_arr[dense_block_id]->min_dense_col_index == 0 && matrix->block_coor_table.item_arr[dense_block_id]->max_dense_col_index == matrix->dense_col_number - 1)
    {
        // 稠密子块之间没有共享的行
    }
    else
    {
        output_template->is_all_force_atom_add = true;
    }

    // 全局的TLB的大小
    output_template->global_thread_level_block_size = global_TLB_size;

    // 用一个数组来存储我们warp的首行索引
    output_template->data_type_of_global_first_row_index_of_warp_level_block = global_row_index->index_data_type;
    output_template->size_of_global_first_row_index_of_warp_level_block = warp_level_block_first_row_vec.size();
    // 申请一个数组来存储首行索引
    output_template->global_first_row_index_of_warp_level_block = malloc_arr(output_template->size_of_global_first_row_index_of_warp_level_block, output_template->data_type_of_global_first_row_index_of_warp_level_block);
    copy_unsigned_long_arr_to_others(&(warp_level_block_first_row_vec[0]), output_template->global_first_row_index_of_warp_level_block, output_template->data_type_of_global_first_row_index_of_warp_level_block, output_template->size_of_global_first_row_index_of_warp_level_block);

    // 加和起始位置二维bool global
    output_template->sum_bool_flag_of_sum_begin = sum_begin_bool_flag;

    // TLB首个归约位置的相对行号
    output_template->data_type_of_first_relative_reduce_row_of_thread_level_block = find_most_suitable_data_type(max_relative_reduce_row_of_thread_level_block);
    output_template->size_of_first_relative_reduce_row_of_thread_level_block = first_relative_reduce_row_of_thread_level_block_vec.size();
    output_template->first_relative_reduce_row_of_thread_level_block = malloc_arr(output_template->size_of_first_relative_reduce_row_of_thread_level_block, output_template->data_type_of_first_relative_reduce_row_of_thread_level_block);
    // 拷贝
    copy_unsigned_long_arr_to_others(&(first_relative_reduce_row_of_thread_level_block_vec[0]), output_template->first_relative_reduce_row_of_thread_level_block, output_template->data_type_of_first_relative_reduce_row_of_thread_level_block, output_template->size_of_first_relative_reduce_row_of_thread_level_block);

    // TLB之间的归约偏移量
    output_template->data_type_of_tmp_result_reduce_offset_of_thread_level_block = find_most_suitable_data_type(max_tmp_result_reduce_offset);
    output_template->size_of_tmp_result_reduce_offset_of_thread_level_block = tmp_result_reduce_offset_vec.size();
    output_template->tmp_result_reduce_offset_of_thread_level_block = malloc_arr(output_template->size_of_tmp_result_reduce_offset_of_thread_level_block, output_template->data_type_of_tmp_result_reduce_offset_of_thread_level_block);
    // 拷贝
    copy_unsigned_long_arr_to_others(&(tmp_result_reduce_offset_vec[0]), output_template->tmp_result_reduce_offset_of_thread_level_block, output_template->data_type_of_tmp_result_reduce_offset_of_thread_level_block, output_template->size_of_tmp_result_reduce_offset_of_thread_level_block);
    
    // 三个数组的非零元数量完全相等
    assert(output_template->size_of_first_relative_reduce_row_of_thread_level_block == output_template->size_of_tmp_result_reduce_offset_of_thread_level_block);
    assert(output_template->size_of_tmp_result_reduce_offset_of_thread_level_block == output_template->sum_bool_flag_of_sum_begin.size());
    assert(output_template->size_of_tmp_result_reduce_offset_of_thread_level_block == TLB_num);

    // 申请合并之后的元数据空间
    output_template->data_type_of_combine_meta_of_thread_level_block = combine_meta_data_type;
    output_template->size_of_combine_meta_of_thread_level_block = TLB_num;
    output_template->combine_meta_of_thread_level_block = malloc_arr(output_template->size_of_combine_meta_of_thread_level_block, output_template->data_type_of_combine_meta_of_thread_level_block);

    // 执行合并，将TLB粒度的所有元数据合并起来
    for (unsigned long TLB_id = 0; TLB_id < TLB_num; TLB_id++)
    {
        vector<bool> bool_flag = output_template->sum_bool_flag_of_sum_begin[TLB_id];
        unsigned long row_offset = read_from_array_with_data_type(output_template->first_relative_reduce_row_of_thread_level_block, output_template->data_type_of_first_relative_reduce_row_of_thread_level_block, TLB_id);
        unsigned long reduce_offset = read_from_array_with_data_type(output_template->tmp_result_reduce_offset_of_thread_level_block, output_template->data_type_of_tmp_result_reduce_offset_of_thread_level_block, TLB_id);
        
        if (combine_meta_data_type == UNSIGNED_LONG)
        {
            unsigned long combine_meta = combine_meta_data_to_unsigned_long(bool_flag, reduce_offset, row_offset, bit_num_of_sum_begin_flag, bit_num_of_tmp_result_reduce_offset, bit_num_of_relative_reduce_row_of_thread_level_block);
            write_to_array_with_data_type(output_template->combine_meta_of_thread_level_block, combine_meta_data_type, TLB_id, combine_meta);
            continue;
        }

        if (combine_meta_data_type == UNSIGNED_INT)
        {
            unsigned int combine_meta = combine_meta_data_to_unsigned_int(bool_flag, reduce_offset, row_offset, bit_num_of_sum_begin_flag, bit_num_of_tmp_result_reduce_offset, bit_num_of_relative_reduce_row_of_thread_level_block);
            write_to_array_with_data_type(output_template->combine_meta_of_thread_level_block, combine_meta_data_type, TLB_id, combine_meta);
            continue;
        }

        if (combine_meta_data_type == UNSIGNED_SHORT)
        {
            unsigned short combine_meta = combine_meta_data_to_unsigned_short(bool_flag, reduce_offset, row_offset, bit_num_of_sum_begin_flag, bit_num_of_tmp_result_reduce_offset, bit_num_of_relative_reduce_row_of_thread_level_block);
            write_to_array_with_data_type(output_template->combine_meta_of_thread_level_block, combine_meta_data_type, TLB_id, combine_meta);
            continue;
        }

        if (combine_meta_data_type == UNSIGNED_CHAR)
        {
            unsigned char combine_meta = combine_meta_data_to_unsigned_char(bool_flag, reduce_offset, row_offset, bit_num_of_sum_begin_flag, bit_num_of_tmp_result_reduce_offset, bit_num_of_relative_reduce_row_of_thread_level_block);
            write_to_array_with_data_type(output_template->combine_meta_of_thread_level_block, combine_meta_data_type, TLB_id, combine_meta);
            continue;
        }

        cout << "combine meta data type is not supported" << endl;
        assert(false);
    }

    // 将合并之后处理的结果放到文件中
    // write_combine_meta_data_to_file(output_template->combine_meta_of_thread_level_block, output_template->data_type_of_combine_meta_of_thread_level_block, output_template->size_of_combine_meta_of_thread_level_block, "/home/duzhen/spmv_builder/data_source/test_result_3");

    // exit(-1);

    // cout << convert_meta_data_to_bit_flag_string_with_data_type(read_from_array_with_data_type(output_template->combine_meta_of_thread_level_block, output_template->data_type_of_combine_meta_of_thread_level_block, 2), combine_meta_data_type) << endl;
    // 接下里处理排序产生的数组
    // 最后给出排序索引类型和具体的数组
    if (compressed_block_view->y_write_index.size() > 0)
    {
        // 在子块内排序了
        assert(compressed_block_view->is_sorted == true && builder->sub_block_sort_type_vec[dense_block_id] == SUB_BLOCK_SORT && matrix->is_sorted == false);
        output_template->global_sort_index = false;
        output_template->local_sort_index = true;

        // 拷贝
        output_template->data_type_of_row_index_before_sort = compressed_block_view->y_write_index[0]->index_data_type;
        output_template->row_index_before_sort = compressed_block_view->y_write_index[0]->index_arr;
        output_template->size_of_row_index_before_sort = compressed_block_view->y_write_index[0]->length;
    }
    else if (matrix->sorted_row_index != NULL)
    {
        cout << "have global sort" << endl;
        // 在全局范围内有排序
        assert(compressed_block_view->is_sorted == false && matrix->is_sorted == true && builder->sub_block_sort_type_vec[dense_block_id] == GLOBAL_SORT);
        output_template->global_sort_index = true;
        output_template->local_sort_index = false;

        // 拷贝
        output_template->data_type_of_row_index_before_sort = matrix->data_type_of_sorted_row_index;
        output_template->row_index_before_sort = matrix->sorted_row_index;
        output_template->size_of_row_index_before_sort = matrix->dense_row_number;
    }

    // 执行交错存储，先申请两个空间
    output_template->data_type_of_col_index_arr = global_col_index->index_data_type;
    output_template->size_of_col_index_arr = dest_col_index_vec.size();
    output_template->col_index_arr = malloc_arr(output_template->size_of_col_index_arr, output_template->data_type_of_col_index_arr);
    
    output_template->data_type_of_val_arr = matrix->val_data_type;
    output_template->size_of_val_arr = dest_val_vec.size();
    output_template->val_arr = malloc_arr(output_template->size_of_val_arr, output_template->data_type_of_val_arr);

    // 遍历所有padding之后的非零元
    assert(dest_col_index_vec.size() == dest_val_vec.size());
    
    for (unsigned long nz_id = 0; nz_id < dest_val_vec.size(); nz_id++)
    {
        // 查看当前非零元在线程中的索引
        unsigned long cur_col_index = dest_col_index_vec[nz_id];
        double cur_val = dest_val_vec[nz_id];

        // 当前非零元所在的TLB
        unsigned long TLB_id_of_nz = nz_id / global_TLB_size;
        // 当前非零元所在的WLB
        unsigned long WLB_id_of_nz = TLB_id_of_nz / 32;

        // 当前非零元所在的TLB内的相对索引
        unsigned long nz_id_in_TLB = nz_id % global_TLB_size;
        // 当前TLB在WLB中的索引
        unsigned long TLB_id_in_WLB = TLB_id_of_nz % 32;

        // 当前非零元的目标位置
        unsigned long dest_nz_id = WLB_id_of_nz * (32 * global_TLB_size) + TLB_id_in_WLB + nz_id_in_TLB * 32;
        assert(dest_nz_id < dest_val_vec.size());
        
        // 将非零元写到对应位置
        write_to_array_with_data_type(output_template->col_index_arr, output_template->data_type_of_col_index_arr, dest_nz_id, cur_col_index);
        write_double_to_array_with_data_type(output_template->val_arr, output_template->data_type_of_val_arr, dest_nz_id, cur_val);
    }

    return output_template;
}

bool is_supported_by_unaligned_warp_reduce_same_TLB_size_template(sparse_struct_t* matrix, unsigned long dense_block_id)
{
    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[dense_block_id]->compressed_block_ptr;
    assert(compressed_block_view != NULL);

    // 检查，warp和block层次都只有一个块，也就是放弃了分块
    index_of_compress_block_t *global_row_index = compressed_block_view->read_index[0];
    index_of_compress_block_t *global_col_index = compressed_block_view->read_index[1];
    index_of_compress_block_t *block_level_index = compressed_block_view->read_index[2];
    index_of_compress_block_t *warp_level_index = compressed_block_view->read_index[3];
    index_of_compress_block_t *thread_level_index = compressed_block_view->read_index[4];
    assert(global_row_index->type_of_index == ROW_INDEX);
    assert(global_col_index->type_of_index == COL_INDEX);
    assert(block_level_index->level_of_this_index == TBLOCK_LEVEL);
    assert(warp_level_index->level_of_this_index == WRAP_LEVEL);
    assert(thread_level_index->level_of_this_index == THREAD_LEVEL);

    assert(global_row_index->max_row_index == matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index);
    assert(global_row_index->max_row_index == block_level_index->max_row_index);
    assert(block_level_index->max_row_index == thread_level_index->max_row_index);

    if (global_row_index->max_row_index != matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index)
    {
        assert(global_row_index->max_row_index > matrix->block_coor_table.item_arr[dense_block_id]->max_dense_row_index);
        return false;
    }

    // 如果thread的压缩类型不是其他就false
    if (thread_level_index->index_compressed_type != NO_INDEX)
    {
        return false;
    }

    // 如果没有TLB所占行的数量的记录，就false
    if (thread_level_index->row_number_of_block_arr == NULL)
    {
        return false;
    }

    if (block_level_index->block_num != 1)
    {
        return false;
    }

    if (warp_level_index->block_num != 1)
    {
        return false;
    }

    unsigned long global_TLB_size = read_from_array_with_data_type(thread_level_index->coo_block_size_arr, thread_level_index->data_type_of_coo_block_size_arr, 0);

    // padding之后算元数据，然后看看元数据超不超
    // 传入矩阵的已有COO格式，得到经过去除空行、padding到32*TLB_size大小矩阵。
    // COO矩阵的三个索引
    vector<unsigned long> dest_row_index_vec;
    vector<unsigned long> dest_col_index_vec;
    vector<double> dest_val_vec;

    assert(compressed_block_view->read_index[0]->type_of_index == ROW_INDEX);
    assert(compressed_block_view->read_index[1]->type_of_index == COL_INDEX);
    // 重构COO的三个矩阵
    fill_empty_and_padding_to_align_warp(compressed_block_view->read_index[0]->index_arr, compressed_block_view->read_index[1]->index_arr, compressed_block_view->val_arr,
                                         compressed_block_view->read_index[0]->index_data_type, compressed_block_view->read_index[1]->index_data_type, compressed_block_view->val_data_type,
                                         compressed_block_view->size, dest_row_index_vec, dest_col_index_vec, dest_val_vec, global_TLB_size);

    // TLB的数量
    assert(dest_col_index_vec.size() % global_TLB_size == 0);
    unsigned long TLB_num = dest_col_index_vec.size() / global_TLB_size;
    assert(TLB_num % 32 == 0);
    unsigned long WLB_num = TLB_num / 32;

    // 得到加和的起始位置的bool flag
    vector<vector<bool>> sum_begin_bool_flag = get_sum_begin_bool_flag_of_each_thread(dest_row_index_vec, global_TLB_size);
    assert(sum_begin_bool_flag.size() == TLB_num);

    // 得到WLB的首行索引
    vector<unsigned long> warp_level_block_first_row_vec = get_first_global_row_index_of_each_warp(dest_row_index_vec, global_TLB_size);
    assert(warp_level_block_first_row_vec.size() == WLB_num);

    // 归约偏移量的最大值
    unsigned long max_tmp_result_reduce_offset = 0;
    
    vector<unsigned long> tmp_result_reduce_offset_vec = get_tmp_result_reduce_offset_vec(sum_begin_bool_flag, &max_tmp_result_reduce_offset);
    assert(tmp_result_reduce_offset_vec.size() == TLB_num);

    // TLB的归约偏移量最大值
    unsigned long max_relative_reduce_row_of_thread_level_block = 0;
    
    vector<unsigned long> first_relative_reduce_row_of_thread_level_block_vec = get_first_relative_reduce_row_of_thread_level_block_vec(dest_row_index_vec, warp_level_block_first_row_vec, sum_begin_bool_flag, global_TLB_size, &max_relative_reduce_row_of_thread_level_block);
    assert(first_relative_reduce_row_of_thread_level_block_vec.size() == TLB_num);

    // 查看三个元数据所占的bit
    int bit_num_of_relative_reduce_row_of_thread_level_block = get_max_bit_num_of_meta_data(max_relative_reduce_row_of_thread_level_block);
    int bit_num_of_tmp_result_reduce_offset = get_max_bit_num_of_meta_data(max_tmp_result_reduce_offset);
    int bit_num_of_sum_begin_flag = global_TLB_size;

    // 对于最大是0的元数据，也需要占用一个bit
    assert(bit_num_of_sum_begin_flag != 0);

    if (bit_num_of_relative_reduce_row_of_thread_level_block == 0)
    {
        bit_num_of_relative_reduce_row_of_thread_level_block = 1;
    }

    if (bit_num_of_tmp_result_reduce_offset == 0)
    {
        bit_num_of_tmp_result_reduce_offset = 1;
    }

    // 这三个加起来要小于64，要不就没有办法使用这个模板
    if (bit_num_of_relative_reduce_row_of_thread_level_block + bit_num_of_tmp_result_reduce_offset + bit_num_of_sum_begin_flag > 64)
    {
        // cout << "too large meta data size:" << bit_num_of_relative_reduce_row_of_thread_level_block + bit_num_of_tmp_result_reduce_offset + bit_num_of_sum_begin_flag << ", is not supported" << endl;
        return false;
    }

    // 

    return true;
}

bool is_supported_by_unaligned_warp_reduce_same_TLB_size_template(code_builder_t* builder, unsigned long dense_block_id)
{
    assert(builder != NULL);

    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;
    assert(matrix->block_coor_table.item_arr.size() > dense_block_id);

    return is_supported_by_unaligned_warp_reduce_same_TLB_size_template(matrix, dense_block_id);
}

void fill_empty_and_padding_to_align_warp(void *source_row_index_arr, void *source_col_index_arr, void *source_val_arr, data_type data_type_of_row_index_arr,
                                          data_type data_type_of_col_index_arr, data_type data_type_of_val_arr, unsigned long nnz, vector<unsigned long> &dest_row_index_vec,
                                          vector<unsigned long> &dest_col_index_vec, vector<double> &dest_val_vec, unsigned long global_thread_level_block_size)
{
    assert(data_type_of_row_index_arr == UNSIGNED_CHAR || data_type_of_row_index_arr == UNSIGNED_SHORT || data_type_of_row_index_arr == UNSIGNED_INT || data_type_of_row_index_arr == UNSIGNED_LONG);
    assert(data_type_of_col_index_arr == UNSIGNED_CHAR || data_type_of_col_index_arr == UNSIGNED_SHORT || data_type_of_col_index_arr == UNSIGNED_INT || data_type_of_col_index_arr == UNSIGNED_LONG);
    assert(data_type_of_val_arr == DOUBLE || data_type_of_val_arr == FLOAT);
    assert(source_row_index_arr != NULL && source_col_index_arr != NULL && source_val_arr != NULL);
    assert(dest_row_index_vec.size() == 0 && dest_col_index_vec.size() == 0 && dest_val_vec.size() == 0);

    // 上一行的行号
    unsigned long last_row_index = 0;

    // 首先先执行空行的补齐，将已经没有空行的结果放在对应的几个vector中
    for (unsigned long nz_id = 0; nz_id < nnz; nz_id++)
    {
        // 读出当前行的行号
        unsigned long cur_row_index = read_from_array_with_data_type(source_row_index_arr, data_type_of_row_index_arr, nz_id);
        unsigned long cur_col_index = read_from_array_with_data_type(source_col_index_arr, data_type_of_col_index_arr, nz_id);
        double cur_val = read_double_from_array_with_data_type(source_val_arr, data_type_of_val_arr, nz_id);

        // 如果一上来就是空行，特殊处理，因为需要从0行开始补
        if (nz_id == 0 && cur_row_index > 0)
        {
            for (unsigned long row_id = 0; row_id < cur_row_index; row_id++)
            {
                dest_row_index_vec.push_back(row_id);
                dest_col_index_vec.push_back(0);
                dest_val_vec.push_back(0);
            }
        }
        else if (cur_row_index > last_row_index + 1)
        {
            // 如果是在中途出现的空行
            for (unsigned long row_id = last_row_index + 1; row_id < cur_row_index; row_id++)
            {
                dest_row_index_vec.push_back(row_id);
                dest_col_index_vec.push_back(0);
                dest_val_vec.push_back(0);
            }
        }

        // 空行补完了写本体
        dest_row_index_vec.push_back(cur_row_index);
        dest_col_index_vec.push_back(cur_col_index);
        dest_val_vec.push_back(cur_val);

        last_row_index = cur_row_index;
    }

    // 空行补完了，执行padding到对齐的大小
    nnz = dest_row_index_vec.size();

    unsigned warp_level_block_nnz = 32 * global_thread_level_block_size;

    unsigned long target_nnz = nnz;

    if (nnz % warp_level_block_nnz != 0)
    {
        target_nnz = (nnz / warp_level_block_nnz + 1) * warp_level_block_nnz;
    }

    assert(target_nnz >= nnz);

    // 需要增加的非零元
    unsigned long added_nnz = target_nnz - nnz;

    for (unsigned long added_nz_id = 0; added_nz_id < added_nnz; added_nz_id++)
    {
        // 新加的非零元坐标和矩阵的最后一个非零元坐标是一样的
        dest_row_index_vec.push_back(dest_row_index_vec[dest_row_index_vec.size() - 1]);
        dest_col_index_vec.push_back(dest_col_index_vec[dest_col_index_vec.size() - 1]);
        dest_val_vec.push_back(0);
    }

    // 最后一下检查
    assert(dest_col_index_vec.size() == dest_row_index_vec.size() && dest_row_index_vec.size() == dest_val_vec.size());
    assert(dest_row_index_vec.size() % (32 * global_thread_level_block_size) == 0);   
}

void store_template_data(unaligned_warp_reduce_same_TLB_size_template_t *output_template, string output_dir, bool force_not_share_global_sort_index)
{
    assert(output_template != NULL);
    
    srand(time(0));
    unsigned long matrix_id = rand() + time(0) % 1000;

    // 写这个模板所需要数据的文件夹名称
    output_dir = output_dir + "/" + to_string(matrix_id) + "_" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer());

    // 创建这个文件夹
    system(("mkdir " + output_dir).c_str());

    // 不压缩
    if (output_template->global_first_row_index_of_warp_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->global_first_row_index_of_warp_level_block != NULL);
        // 压缩完之后写
        print_arr_to_file_with_data_type(output_template->global_first_row_index_of_warp_level_block, output_template->data_type_of_global_first_row_index_of_warp_level_block, output_template->size_of_global_first_row_index_of_warp_level_block, output_dir + "/global_first_row_index_of_warp_level_block");
    }

    // 经过压缩的线程粒度的元数据
    assert(output_template->combine_meta_of_thread_level_block != NULL);
    print_arr_to_file_with_data_type(output_template->combine_meta_of_thread_level_block, output_template->data_type_of_combine_meta_of_thread_level_block, output_template->size_of_combine_meta_of_thread_level_block, output_dir + "/combine_meta_of_thread_level_block");

    // 排序相关的数据
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

string code_of_template_data_struct(unaligned_warp_reduce_same_TLB_size_template_t *output_template, unsigned long dense_block_id)
{
    assert(output_template != NULL);
    // 创建一个数据结构
    string return_str = "typedef struct compressed_dense_block_" + to_string(dense_block_id) + "\n{\n";

    if (output_template->global_first_row_index_of_warp_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->global_first_row_index_of_warp_level_block != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_global_first_row_index_of_warp_level_block, code_of_arr_var_name(dense_block_id, -1, "global_first_row_index_of_warp_level_block"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "global_first_row_index_of_warp_level_block") + " = " + to_string(output_template->size_of_global_first_row_index_of_warp_level_block) + ";\n";
    }

    return_str = return_str + "\n";

    assert(output_template->combine_meta_of_thread_level_block != NULL);
    return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_combine_meta_of_thread_level_block, code_of_arr_var_name(dense_block_id, -1, "combine_meta_of_thread_level_block"));
    return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "combine_meta_of_thread_level_block") + " = " + to_string(output_template->size_of_combine_meta_of_thread_level_block) + ";\n";

    return_str = return_str + "\n";

    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->row_index_before_sort != NULL)
    {
        assert(output_template->row_index_before_sort != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_row_index_before_sort, code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"));
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort") + " = " + to_string(output_template->size_of_row_index_before_sort) + ";\n";
    }

    // 值和列索引
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

string code_of_read_template_data_from_file_func_define(unaligned_warp_reduce_same_TLB_size_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index)
{
    assert(output_template != NULL);
    
    string return_str = "compressed_dense_block_" + to_string(dense_block_id) + "_t* read_dense_block_" + to_string(dense_block_id) + "_from_file(string file_name_prefix)\n{\n";

    return_str = return_str + "compressed_dense_block_" + to_string(dense_block_id) + "_t *template_data = new " + "compressed_dense_block_" + to_string(dense_block_id) + "_t();\n";

    // warp首行行号
    if (output_template->global_first_row_index_of_warp_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->global_first_row_index_of_warp_level_block != NULL);
        return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "global_first_row_index_of_warp_level_block") + " = (" + code_of_data_type(output_template->data_type_of_global_first_row_index_of_warp_level_block) + " *)";
        return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "global_first_row_index_of_warp_level_block") + ", " + convert_data_type_to_string(output_template->data_type_of_global_first_row_index_of_warp_level_block) + ", ";
        // 要读的文件名
        return_str = return_str + "file_name_prefix + \"/global_first_row_index_of_warp_level_block\");\n";
    }

    return_str = return_str + "\n";

    assert(output_template->combine_meta_of_thread_level_block != NULL);
    return_str = return_str + "template_data->" + code_of_arr_var_name(dense_block_id, -1, "combine_meta_of_thread_level_block") + " = (" + code_of_data_type(output_template->data_type_of_combine_meta_of_thread_level_block) + " *)";
    return_str = return_str + "read_arr_from_file_with_data_type(template_data->size_of_" + code_of_arr_var_name(dense_block_id, -1, "combine_meta_of_thread_level_block") + ", " + convert_data_type_to_string(output_template->data_type_of_combine_meta_of_thread_level_block) + ", ";
    // 要读的文件名
    return_str = return_str + "file_name_prefix + \"/combine_meta_of_thread_level_block\");\n";

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

// 将模板的数据从CPU拷贝到GPU
string code_of_write_template_data_to_gpu(unaligned_warp_reduce_same_TLB_size_template_t *output_template, unsigned long dense_block_id, bool force_not_share_global_sort_index)
{
    assert(output_template != NULL);
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

    if (output_template->global_first_row_index_of_warp_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->global_first_row_index_of_warp_level_block != NULL);
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_global_first_row_index_of_warp_level_block, "device_" + code_of_arr_var_name(dense_block_id, -1, "global_first_row_index_of_warp_level_block"));
    }

    // 行顺序数组的声明
    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->row_index_before_sort != NULL)
    {
        return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_row_index_before_sort, "device_" + code_of_arr_var_name(dense_block_id, -1, "row_index_before_sort"));
    }

    assert(output_template->combine_meta_of_thread_level_block != NULL);
    return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_combine_meta_of_thread_level_block, "device_" + code_of_arr_var_name(dense_block_id, -1, "combine_meta_of_thread_level_block"));

    assert(output_template->val_arr != NULL);
    return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_val_arr, "device_" + code_of_arr_var_name(dense_block_id, -1, "val_arr"));

    assert(output_template->col_index_arr != NULL);
    return_str = return_str + code_line_of_pointer_define(output_template->data_type_of_col_index_arr, "device_" + code_of_arr_var_name(dense_block_id, -1, "col_index_arr"));

    return_str = return_str + "\n";

    // 申请数组的代码
    if (output_template->global_first_row_index_of_warp_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->global_first_row_index_of_warp_level_block != NULL);
        return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_global_first_row_index_of_warp_level_block, to_string(output_template->size_of_global_first_row_index_of_warp_level_block), "device_" + code_of_arr_var_name(dense_block_id, -1, "global_first_row_index_of_warp_level_block"));
        // 拷贝
        return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "global_first_row_index_of_warp_level_block"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "global_first_row_index_of_warp_level_block"), output_template->data_type_of_global_first_row_index_of_warp_level_block, to_string(output_template->size_of_global_first_row_index_of_warp_level_block), "cudaMemcpyHostToDevice") + "\n";
    }

    assert(output_template->combine_meta_of_thread_level_block != NULL);
    return_str = return_str + code_line_of_cuda_malloc(output_template->data_type_of_combine_meta_of_thread_level_block, to_string(output_template->size_of_combine_meta_of_thread_level_block), "device_" + code_of_arr_var_name(dense_block_id, -1, "combine_meta_of_thread_level_block"));
    // 拷贝
    return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(dense_block_id, -1, "combine_meta_of_thread_level_block"), template_data_name + "->" + code_of_arr_var_name(dense_block_id, -1, "combine_meta_of_thread_level_block"), output_template->data_type_of_combine_meta_of_thread_level_block, to_string(output_template->size_of_combine_meta_of_thread_level_block), "cudaMemcpyHostToDevice") + "\n";

    // 如果是全局的就直接赋值
    if (output_template->row_index_before_sort_compress == NONE_COMPRESS && output_template->global_sort_index == true)
    {
        assert(output_template->local_sort_index == false);

        // 如果不分享就从内disk中读
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

string code_of_template_kernal(unaligned_warp_reduce_same_TLB_size_template_t *output_template, unsigned long dense_block_id)
{
    assert(output_template != NULL);

    // 内核函数的声明
    string return_str = "__global__ void spmv_" + to_string(dense_block_id) + "(";

    // 用一个变量表明当前形参是不是第一个，如果是第一个就不用点逗号
    bool is_first_param = true;

    if (output_template->global_first_row_index_of_warp_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->global_first_row_index_of_warp_level_block != NULL);
        return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_global_first_row_index_of_warp_level_block, "* global_first_row_index_of_warp_level_block");
        is_first_param = false;
    }

    if (is_first_param == false)
    {
        return_str = return_str + ", ";
    }
    else
    {
        is_first_param = false;
    }

    assert(output_template->combine_meta_of_thread_level_block != NULL);
    return_str = return_str + code_of_a_formal_param_declare(output_template->data_type_of_combine_meta_of_thread_level_block, "* combine_meta_of_thread_level_block");

    // 排序相关的数组
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

    // 获取对应计算资源ID
    return_str = return_str + "int global_tid = blockDim.x * blockIdx.x + threadIdx.x;\n";
    return_str = return_str + "int warp_id = global_tid / 32;\n";
    return_str = return_str + "int tid_in_warp = global_tid % 32;\n";
    return_str = return_str + "int warp_num = blockDim.x * gridDim.x / 32;\n\n";

    // 首行和首列号，如果不是0就保留
    if (output_template->kernal_first_row_index != 0)
    {
        return_str = return_str + "unsigned long kernal_first_row_index = " + to_string(output_template->kernal_first_row_index) + ";\n";
    }

    if (output_template->kernal_first_col_index != 0)
    {
        return_str = return_str + "unsigned long kernal_first_col_index = " + to_string(output_template->kernal_first_col_index) + ";\n";
    }

    // 查看warp的数量
    assert((output_template->tblock_num * output_template->thread_num_in_block) % 32 == 0);

    assert((output_template->size_of_col_index_arr % (output_template->size_of_combine_meta_of_thread_level_block * output_template->global_thread_level_block_size)) == 0);

    int warp_num = (output_template->tblock_num * output_template->thread_num_in_block) / 32;

    int WLB_num = output_template->size_of_col_index_arr / (output_template->global_thread_level_block_size * 32);
    assert(WLB_num == output_template->size_of_global_first_row_index_of_warp_level_block);

    // cout << "warp_num:" << warp_num << endl;
    // cout << "WLB_num:" << WLB_num << endl;

    // warp_num的数量更少，那就使用for循环
    if (warp_num < WLB_num)
    {
        return_str = return_str + "for (unsigned int WLB_id = warp_id; WLB_id < " + to_string(WLB_num) + "; WLB_id = WLB_id + warp_num)\n{\n";
    }
    else
    {
        // warp的数量更多，用if语句
        return_str = return_str + "unsigned int WLB_id = warp_id;\n";
        return_str = return_str + "if (WLB_id < "+ to_string(WLB_num) +")\n{\n";
    }

    // 获取首行行号，根据压缩和不压缩分为两种情况
    return_str = return_str + "unsigned int WLB_first_row;\n";
    if (output_template->global_first_row_index_of_warp_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->global_first_row_index_of_warp_level_block != NULL);
        return_str = return_str + "WLB_first_row = global_first_row_index_of_warp_level_block[WLB_id];\n";
    }
    else if (output_template->global_first_row_index_of_warp_level_block_compress == LINEAR_COMPRESS)
    {
        assert(output_template->global_first_row_index_of_warp_level_block_compress_meta != NULL);
        linear_compress_t* compressor = (linear_compress_t *)output_template->global_first_row_index_of_warp_level_block_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "WLB_first_row", "WLB_id") + ";\n";
    }
    else if (output_template->global_first_row_index_of_warp_level_block_compress == CYCLE_INCREASE_COMPRESS)
    {
        assert(output_template->global_first_row_index_of_warp_level_block_compress_meta != NULL);
        cycle_increase_compress_t* compressor = (cycle_increase_compress_t *)output_template->global_first_row_index_of_warp_level_block_compress_meta;
        return_str = return_str + code_of_arr_read(compressor, "WLB_first_row", "WLB_id") + ";\n";
    }
    else
    {
        cout << "compress type is not supported in this template" << endl;
        assert(false);
    }    

    // 当前线程负责的TLB号
    return_str = return_str + "unsigned int global_TLB_id = WLB_id * 32 + tid_in_warp;\n";

    return_str = return_str + "\n";

    // 获取当前线程的元数据
    return_str = return_str + code_of_data_type(output_template->data_type_of_combine_meta_of_thread_level_block) + " combine_meta = combine_meta_of_thread_level_block[global_TLB_id];\n";

    return_str = return_str + "\n";
    
    // cout << output_template->bit_num_of_thread_level_combine_meta << endl;
    // cout << output_template->bit_num_of_tmp_result_reduce_offset_of_thread_level_block << endl;
    // cout << output_template->bit_num_of_first_relative_reduce_row_of_thread_level_block << endl;
    // cout << output_template->bit_num_of_sum_begin_bit_flag << endl;

    // 获取当前线程第一个归约的行位置，需要分两行来移位，防止编译器把这一行优化掉，导致高位没有丢掉
    return_str = return_str + code_of_data_type(output_template->data_type_of_combine_meta_of_thread_level_block) + " TLB_first_reduce_row = combine_meta << " + to_string(output_template->bit_num_of_thread_level_combine_meta - output_template->bit_num_of_first_relative_reduce_row_of_thread_level_block) + ";\n";
    return_str = return_str + "TLB_first_reduce_row = TLB_first_reduce_row >> " + to_string(output_template->bit_num_of_thread_level_combine_meta - output_template->bit_num_of_first_relative_reduce_row_of_thread_level_block) + ";\n";

    return_str = return_str + "\n";

    // TLB向全局内存写中间结果的行偏移量
    return_str = return_str + "unsigned int y_offset = TLB_first_reduce_row;\n";

    return_str = return_str + "\n";

    // 每个线程行身体提前归约的偏移量，分两行归约，防止编译器搞一些优化
    return_str = return_str + code_of_data_type(output_template->data_type_of_combine_meta_of_thread_level_block) + " reduce_offset = combine_meta << " + to_string(output_template->bit_num_of_thread_level_combine_meta - output_template->bit_num_of_first_relative_reduce_row_of_thread_level_block - output_template->bit_num_of_tmp_result_reduce_offset_of_thread_level_block) + ";\n";
    return_str = return_str + "reduce_offset = reduce_offset >> " + to_string(output_template->bit_num_of_thread_level_combine_meta - output_template->bit_num_of_tmp_result_reduce_offset_of_thread_level_block) + ";\n";

    return_str = return_str + "\n";

    // 用一个变量记录之前是否出现过sum起始点
    return_str = return_str + "bool through_sum_begin_bit = false;\n";

    // 行的头部、行的身体和加和的中间结果
    return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " row_head_tmp_result = 0;\n";
    return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " row_other_tmp_result = 0;\n";
    return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " sum_tmp = 0;\n";

    // 当前非零元的全局索引
    return_str = return_str + "unsigned int global_nz_index = WLB_id * " + to_string(output_template->global_thread_level_block_size * 32) + " + tid_in_warp;\n";

    return_str = return_str + "\n";

    // 遍历线程的所有非零元
    return_str = return_str + "for (unsigned int TLB_nz_id = 0; TLB_nz_id < " + to_string(output_template->global_thread_level_block_size) + "; TLB_nz_id++)\n{\n";

    // 获取当前非零元加和起始标记，偏移量的计算依赖于合并之后的元数据大小
    return_str = return_str + "bool cur_sum_begin_bit = (combine_meta >> (" + to_string(output_template->bit_num_of_thread_level_combine_meta - 1) + " - TLB_nz_id)) & 0x1;\n";

    // 如果当前非零元是第一个行首非零元，那么之前的结果要存起来，用来跨线程归约的行其他部分中间结果
    return_str = return_str + "if (cur_sum_begin_bit == true)\n{\n";

    // 如果之前已经出现的加和起始位置，那么就需要向显存中写数据了
    return_str = return_str + "if (through_sum_begin_bit == true)\n{\n";

    // 需要向内存的对应位置写结果，这里计算真实行号
    // 根据是不是排序来选择对应的输入方式，假设最大行号不超过
    return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->matrix->dense_row_number)) + " global_row_index;\n";

    // 首先变成当前行号
    return_str = return_str + "global_row_index = WLB_first_row + y_offset;\n";

    // 如果有排序，就要得到排序之前行号
    if (output_template->local_sort_index == true)
    {
        assert(output_template->global_sort_index == false);

        // 根据压缩的情况来找出排序前的行号
        if (output_template->row_index_before_sort_compress == NONE_COMPRESS)
        {
            assert(output_template->row_index_before_sort != NULL);

            return_str = return_str + "global_row_index = row_index_before_sort[global_row_index];\n";
        }
        else if (output_template->row_index_before_sort_compress == LINEAR_COMPRESS)
        {
            // 线性压缩
            assert(output_template->row_index_before_sort_compress_meta != NULL);
            linear_compress_t* compressor = (linear_compress_t *)output_template->row_index_before_sort_compress_meta;
            return_str = return_str + code_of_arr_read(compressor, "global_row_index", "global_row_index") + ";\n";
        }
        else
        {
            cout << "compress type is not supported in this template" << endl;
            assert(false);
        }
        
        if (output_template->kernal_first_row_index != 0)
        {
            return_str = return_str + "global_row_index = global_row_index + kernal_first_row_index;\n";
        }
    }

    if (output_template->global_sort_index == true)
    {
        assert(output_template->local_sort_index == false);

        if (output_template->kernal_first_row_index != 0)
        {
            return_str = return_str + "global_row_index = global_row_index + kernal_first_row_index;\n";
        }

        // 根据压缩来，去原矩阵的值
        if (output_template->row_index_before_sort_compress == NONE_COMPRESS)
        {
            assert(output_template->row_index_before_sort != NULL);
            return_str = return_str + "global_row_index = row_index_before_sort[global_row_index];\n";
        }
        else if (output_template->row_index_before_sort_compress == LINEAR_COMPRESS)
        {
            assert(output_template->row_index_before_sort_compress_meta != NULL);
            linear_compress_t* compressor = (linear_compress_t *) output_template->row_index_before_sort_compress_meta;
            return_str = return_str + code_of_arr_read(compressor, "global_row_index", "global_row_index") + ";\n";
        }
        else
        {
            cout << "compress type is not supported in this template" << endl;
            assert(false);
        }
    }

    // 完全没有排序，就加一个子矩阵偏移量
    if (output_template->global_sort_index == false && output_template->local_sort_index == false && output_template->kernal_first_row_index != 0)
    {
        return_str = return_str + "global_row_index = global_row_index + kernal_first_row_index;\n";
    }

    // 如果全局都需要原子加，那就不需要额外的判断，直接使用原子加
    if (output_template->is_all_force_atom_add == false)
    {
        // 如果是WLB第一个TLB的第一个写回位置，就要用原子加写内存
        return_str = return_str + "if (tid_in_warp == 0 && y_offset == TLB_first_reduce_row)\n{\n";

        // 执行原子加
        return_str = return_str + "atomicAdd(&(device_y_arr[global_row_index]), sum_tmp);\n";

        return_str = return_str + "\n}\nelse\n{\n";

        return_str = return_str + "device_y_arr[global_row_index] = sum_tmp;\n";

        return_str = return_str + "\n}\n";
    }
    else
    {
        // 不带条件全部使用原子加
        return_str = return_str + "atomicAdd(&(device_y_arr[global_row_index]), sum_tmp);\n";
    }

    return_str = return_str + "\n}\nelse\n{\n";

    // 第一个行首非零元，记录行身体的中间结果
    return_str = return_str + "row_other_tmp_result = sum_tmp;\n";

    return_str = return_str + "\n}\n";
    
    return_str = return_str + "\n}\n";

    // 更新线程的行偏移量
    return_str = return_str + "y_offset = y_offset + (through_sum_begin_bit & cur_sum_begin_bit);\n";
    
    return_str = return_str + "\n";

    return_str = return_str + "through_sum_begin_bit = through_sum_begin_bit || cur_sum_begin_bit;\n";

    return_str = return_str + "\n";

    return_str = return_str + "sum_tmp = cur_sum_begin_bit ? 0 : sum_tmp;\n";

    return_str = return_str + "\n";

    if (output_template->kernal_first_col_index != 0)
    {
        return_str = return_str + "sum_tmp = sum_tmp + val_arr[global_nz_index] * __ldg(&(device_x_arr[kernal_first_col_index + col_index_arr[global_nz_index]]));\n";
    }
    else
    {
        // 执行单个非零元的加和操作
        return_str = return_str + "sum_tmp = sum_tmp + val_arr[global_nz_index] * __ldg(&(device_x_arr[col_index_arr[global_nz_index]]));\n";
    }

    // 非零元索引
    return_str = return_str + "global_nz_index = global_nz_index + 32;\n";

    return_str = return_str + "\n}\n";

    // 之后执行线程间归约，首先初始化所有线程的行身中间结果
    // 对于出现过加和起始位置的，那么行身结果应该已经被赋值了，对于没有出现过加和起始位置的，那么行身的结果就在加和的当前中间结果中
    return_str = return_str + "row_other_tmp_result = through_sum_begin_bit ? row_other_tmp_result : sum_tmp;\n";

    // 所有的线程末尾都会加一个行脑袋
    return_str = return_str + "row_head_tmp_result = sum_tmp;\n";

    // 行身体的结果向前一个线程挪一个
    return_str = return_str + "row_other_tmp_result = __shfl_down_sync(0xFFFFFFFF, row_other_tmp_result, 1);\n";

    // 挪完之后位最后一个线程的行身体结果赋值为0，其他保持不变
    return_str = return_str + "row_other_tmp_result = (tid_in_warp == 31) ? 0 : row_other_tmp_result;\n";

    return_str = return_str + "\n";

    // 然后执行一个扫描加，在每个线程中存储之前所有线程的中间结果
    return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " scan_tmp_sum = row_other_tmp_result;\n";

    // 从其他线程取数据，然后做加和操作
    return_str = return_str + code_of_data_type(output_template->data_type_of_val_arr) + " scan_tmp_sum_from_other_thread = __shfl_up_sync(0xFFFFFFFF, scan_tmp_sum, 1);\n";
    return_str = return_str + "scan_tmp_sum = (tid_in_warp >= 1) ? scan_tmp_sum_from_other_thread + scan_tmp_sum : scan_tmp_sum;\n";
    return_str = return_str + "scan_tmp_sum_from_other_thread = __shfl_up_sync(0xFFFFFFFF, scan_tmp_sum, 2);\n";
    return_str = return_str + "scan_tmp_sum = (tid_in_warp >= 2) ? scan_tmp_sum_from_other_thread + scan_tmp_sum : scan_tmp_sum;\n";
    return_str = return_str + "scan_tmp_sum_from_other_thread = __shfl_up_sync(0xFFFFFFFF, scan_tmp_sum, 4);\n";
    return_str = return_str + "scan_tmp_sum = (tid_in_warp >= 4) ? scan_tmp_sum_from_other_thread + scan_tmp_sum : scan_tmp_sum;\n";
    return_str = return_str + "scan_tmp_sum_from_other_thread = __shfl_up_sync(0xFFFFFFFF, scan_tmp_sum, 8);\n";
    return_str = return_str + "scan_tmp_sum = (tid_in_warp >= 8) ? scan_tmp_sum_from_other_thread + scan_tmp_sum : scan_tmp_sum;\n";
    return_str = return_str + "scan_tmp_sum_from_other_thread = __shfl_up_sync(0xFFFFFFFF, scan_tmp_sum, 16);\n";
    return_str = return_str + "scan_tmp_sum = (tid_in_warp >= 16) ? scan_tmp_sum_from_other_thread + scan_tmp_sum : scan_tmp_sum;\n";
    return_str = return_str + "\n";
    return_str = return_str + "scan_tmp_sum_from_other_thread = __shfl_down_sync(0xFFFFFFFF, scan_tmp_sum, reduce_offset);\n";
    return_str = return_str + "row_other_tmp_result = scan_tmp_sum_from_other_thread - scan_tmp_sum + row_other_tmp_result;\n";

    // 只有出现过加和起始flag的线程才有行脑袋的中间结果
    return_str = return_str + "\n";

    return_str = return_str + "row_head_tmp_result = through_sum_begin_bit ? row_head_tmp_result + row_other_tmp_result : 0;\n";

    return_str = return_str + "\n";

    // 对于有起始加标记的行来说，才执行写会
    return_str = return_str + "if (through_sum_begin_bit == true)\n{\n";

    // 获取全局的行号
    // 需要向内存的对应位置写结果，这里计算真实行号
    // 根据是不是排序来选择对应的输入方式，假设最大行号不超过
    return_str = return_str + code_of_data_type(find_most_suitable_data_type(output_template->matrix->dense_row_number)) + " global_row_index;\n";

    // 首先变成当前行号
    return_str = return_str + "global_row_index = WLB_first_row + y_offset;\n";

    // 如果有排序，就要得到排序之前行号
    if (output_template->local_sort_index == true)
    {
        assert(output_template->global_sort_index == false);

        // 根据压缩的情况来找出排序前的行号
        if (output_template->row_index_before_sort_compress == NONE_COMPRESS)
        {
            assert(output_template->row_index_before_sort != NULL);

            return_str = return_str + "global_row_index = row_index_before_sort[global_row_index];\n";
        }
        else if (output_template->row_index_before_sort_compress == LINEAR_COMPRESS)
        {
            // 线性压缩
            assert(output_template->row_index_before_sort_compress_meta != NULL);
            linear_compress_t* compressor = (linear_compress_t *)output_template->row_index_before_sort_compress_meta;
            return_str = return_str + code_of_arr_read(compressor, "global_row_index", "global_row_index") + ";\n";
        }
        else
        {
            cout << "compress type is not supported in this template" << endl;
            assert(false);
        }
        
        if (output_template->kernal_first_row_index != 0)
        {
            return_str = return_str + "global_row_index = global_row_index + kernal_first_row_index;\n";
        }
    }

    if (output_template->global_sort_index == true)
    {
        assert(output_template->local_sort_index == false);

        if (output_template->kernal_first_row_index != 0)
        {
            return_str = return_str + "global_row_index = global_row_index + kernal_first_row_index;\n";
        }

        // 根据压缩来，去原矩阵的值
        if (output_template->row_index_before_sort_compress == NONE_COMPRESS)
        {
            assert(output_template->row_index_before_sort != NULL);
            return_str = return_str + "global_row_index = row_index_before_sort[global_row_index];\n";
        }
        else if (output_template->row_index_before_sort_compress == LINEAR_COMPRESS)
        {
            assert(output_template->row_index_before_sort_compress_meta != NULL);
            linear_compress_t* compressor = (linear_compress_t *) output_template->row_index_before_sort_compress_meta;
            return_str = return_str + code_of_arr_read(compressor, "global_row_index", "global_row_index") + ";\n";
        }
        else
        {
            cout << "compress type is not supported in this template" << endl;
            assert(false);
        }
    }

    // 完全没有排序，就加一个子矩阵偏移量
    if (output_template->global_sort_index == false && output_template->local_sort_index == false && output_template->kernal_first_row_index != 0)
    {
        return_str = return_str + "global_row_index = global_row_index + kernal_first_row_index;\n";
    }

    // 如果全局都需要原子加，那就不需要额外的判断，直接使用原子加
    if (output_template->is_all_force_atom_add == false)
    {
        // 如果是WLB第一个TLB的第一个写回位置，就要用原子加写内存
        return_str = return_str + "if ((tid_in_warp == 0 && y_offset == TLB_first_reduce_row) || (tid_in_warp + reduce_offset == 31))\n{\n";

        // 执行原子加
        return_str = return_str + "atomicAdd(&(device_y_arr[global_row_index]), sum_tmp);\n";

        return_str = return_str + "\n}\nelse\n{\n";

        return_str = return_str + "device_y_arr[global_row_index] = sum_tmp;\n";

        return_str = return_str + "\n}\n";
    }
    else
    {
        // 不带条件全部使用原子加
        return_str = return_str + "atomicAdd(&(device_y_arr[global_row_index]), sum_tmp);\n";
    }

    return_str = return_str + "\n}\n";
    return_str = return_str + "\n}\n";
    return_str = return_str + "\n}\n";

    return return_str;
}

// 对于核函数的调用
string code_of_kernal_function_call(unaligned_warp_reduce_same_TLB_size_template_t *output_template, unsigned long dense_block_id)
{
    assert(output_template != NULL);
    // 线程块的数量和线程的数量不能超标
    assert(output_template->tblock_num <= get_config()["MAX_TBLOCK_NUM"].as_integer() && output_template->thread_num_in_block <= get_config()["MAX_THREAD_NUM_IN_BLOCK"].as_integer());

    string return_str = "spmv_" + to_string(dense_block_id) + "<<<" + to_string(output_template->tblock_num) + ", " + to_string(output_template->thread_num_in_block) + ", 0, stream_arr[" + to_string(dense_block_id) + "]>>>(";

    bool is_first_param = true;

    // 加入形参
    if (output_template->global_first_row_index_of_warp_level_block_compress == NONE_COMPRESS)
    {
        assert(output_template->global_first_row_index_of_warp_level_block != NULL);
        return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "global_first_row_index_of_warp_level_block");
        is_first_param = false;
    }

    if (is_first_param == false)
    {
        return_str = return_str + ", ";
    }
    else
    {
        is_first_param = false;
    }
    assert(output_template->combine_meta_of_thread_level_block != NULL);
    return_str = return_str + "device_" + code_of_arr_var_name(dense_block_id, -1, "combine_meta_of_thread_level_block");

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

vector<vector<bool>> get_sum_begin_bool_flag_of_each_thread(vector<unsigned long> row_index_vec, int nnz_in_thread)
{
    assert(row_index_vec.size() % (nnz_in_thread * 32) == 0);

    // 一共的线程粒度的块的数量
    unsigned long TLB_num = row_index_vec.size() / nnz_in_thread;

    // 每个线程粒度的块都有一个bool标记
    vector<vector<bool>> sum_begin_bool_flag_of_each_thread(TLB_num);

    for (unsigned long TLB_id = 0; TLB_id < TLB_num; TLB_id++)
    {
        // 判断是不是WLB的第一个块
        bool is_first_TLB_of_WLB = false;

        if (TLB_id % 32 == 0)
        {
            is_first_TLB_of_WLB = true;
        }

        // 遍历TLB的每一个非零元
        for (unsigned long TLB_nz_id = 0; TLB_nz_id < nnz_in_thread; TLB_nz_id++)
        {
            // TLB_nz_id是0，并且TLB是WLB的第一个TLB
            if (TLB_nz_id == 0 && is_first_TLB_of_WLB == true)
            {
                sum_begin_bool_flag_of_each_thread[TLB_id].push_back(true);
            }
            else
            {
                // 非零元对应的全局行号
                unsigned long global_nz_index = TLB_id * nnz_in_thread + TLB_nz_id;
                assert(global_nz_index < row_index_vec.size());
                // 当前非零元行号对应的行号
                unsigned long row_index = row_index_vec[global_nz_index];

                assert(global_nz_index > 0);
                unsigned long previous_row_index = row_index_vec[global_nz_index - 1];

                // 如果前后的行索引一样，就输入false
                if (row_index == previous_row_index)
                {
                    sum_begin_bool_flag_of_each_thread[TLB_id].push_back(false);
                }
                else
                {
                    // 如果和之前的行索引不一样，那就可能是换行了
                    sum_begin_bool_flag_of_each_thread[TLB_id].push_back(true);
                }
            }
        }

        assert(sum_begin_bool_flag_of_each_thread[TLB_id].size() == nnz_in_thread);
    }

    return sum_begin_bool_flag_of_each_thread;
}

// WLB的首行行号
vector<unsigned long> get_first_global_row_index_of_each_warp(vector<unsigned long> row_index_vec, int nnz_in_thread)
{
    assert(row_index_vec.size() % (32 * nnz_in_thread) == 0);

    unsigned int WLB_num = row_index_vec.size() / (32 * nnz_in_thread);

    unsigned int WLB_nz_num = 32 * nnz_in_thread;

    vector<unsigned long> WLB_first_global_row_index;

    // 遍历每一个WLB，查看WLB第一个row index，这个row index是全局行号
    for (unsigned long WLB_id = 0; WLB_id < WLB_num; WLB_id++)
    {
        // WLB第一个非零元的行号
        unsigned int WLB_first_nz_index = WLB_id * WLB_nz_num;

        // 找到对应的行号
        unsigned int WLB_first_row_index = row_index_vec[WLB_first_nz_index];

        WLB_first_global_row_index.push_back(WLB_first_row_index);
    }

    assert(WLB_first_global_row_index.size() == WLB_num);

    return WLB_first_global_row_index;
}


vector<unsigned long> get_tmp_result_reduce_offset_vec(vector<vector<bool>> begin_sum_bool_flag, unsigned long* max_tmp_result_reduce_offset)
{
    assert(begin_sum_bool_flag.size() > 0);
    // 线程的数量是32的倍数
    assert(begin_sum_bool_flag.size() % 32 == 0);
    assert(max_tmp_result_reduce_offset != NULL);

    *max_tmp_result_reduce_offset = 0;

    // 第一个块肯定有T
    {
        bool has_T_in_first_thread = false;

        for (unsigned long i = 0; i < begin_sum_bool_flag[0].size(); i++)
        {
            has_T_in_first_thread = has_T_in_first_thread || begin_sum_bool_flag[0][i];
        }

        assert(has_T_in_first_thread == true);
    }

    // 上一个有T的线程id
    unsigned long last_has_T_TLB_id = 0;
    // 累计全F的行的数量
    unsigned long TLB_num_with_all_F = 0;

    vector<unsigned long> reduce_offset_vec(begin_sum_bool_flag.size());

    // 遍历所有的线程
    for (unsigned long TLB_id = 1; TLB_id < begin_sum_bool_flag.size(); TLB_id++)
    {
        bool has_T = false;
        // 首先判断是不是全是F
        for (unsigned long nz_id_in_TLB = 0; nz_id_in_TLB < begin_sum_bool_flag[TLB_id].size(); nz_id_in_TLB++)
        {
            has_T = has_T || begin_sum_bool_flag[TLB_id][nz_id_in_TLB];
        }

        // 如果没有T就累加了一下
        if (has_T == false)
        {
            TLB_num_with_all_F++;
        }
        else
        {
            // 到了有T的，记录归约的位置
            assert(last_has_T_TLB_id < begin_sum_bool_flag.size());

            if (*max_tmp_result_reduce_offset < TLB_num_with_all_F)
            {
                *max_tmp_result_reduce_offset = TLB_num_with_all_F;
            }

            reduce_offset_vec[last_has_T_TLB_id] = TLB_num_with_all_F;
            TLB_num_with_all_F = 0;
            last_has_T_TLB_id = TLB_id;
        }
    }

    // 最后一个TLB的
    assert(last_has_T_TLB_id < begin_sum_bool_flag.size());
    
    if (*max_tmp_result_reduce_offset < TLB_num_with_all_F)
    {
        *max_tmp_result_reduce_offset = TLB_num_with_all_F;
    }

    reduce_offset_vec[last_has_T_TLB_id] = TLB_num_with_all_F;

    // 返回对应的归约偏移量
    return reduce_offset_vec;
}

vector<unsigned long> get_first_relative_reduce_row_of_thread_level_block_vec(vector<unsigned long> row_index_vec, vector<unsigned long> WLB_first_global_row_index, vector<vector<bool>> row_begin_bool_flag, int nnz_in_thread, unsigned long *max_first_relative_reduce_row)
{
    // 查看每一个线程第一个行首元素的相对行号，主要是当前TLB之前的所有TLB的
    assert(row_index_vec.size() % (32 * nnz_in_thread) == 0);
    assert(max_first_relative_reduce_row != NULL);

    *max_first_relative_reduce_row = 0;

    // TLB的数量
    unsigned long TLB_num = row_index_vec.size() / nnz_in_thread;

    vector<unsigned long> TLB_first_relative_row_index;

    for (unsigned long TLB_id = 0; TLB_id < TLB_num; TLB_id++)
    {
        // 当前TLB所在的WLB号
        unsigned long WLB_id = TLB_id / 32;

        // 当前WLB的首行行号
        unsigned long WLB_first_row_index = WLB_first_global_row_index[WLB_id];

        assert(row_begin_bool_flag[TLB_id].size() == nnz_in_thread);

        bool has_T_in_TLB = false;

        // 遍历TLB中的所有数据，找出第一个T的行号
        for (unsigned long nz_id_in_TLB = 0; nz_id_in_TLB < nnz_in_thread; nz_id_in_TLB++)
        {
            if (row_begin_bool_flag[TLB_id][nz_id_in_TLB] == true)
            {
                has_T_in_TLB = true;
                unsigned long TLB_global_first_row_index = row_index_vec[TLB_id * nnz_in_thread + nz_id_in_TLB];

                assert(TLB_global_first_row_index >= WLB_first_row_index);
                unsigned long TLB_relative_first_row_index = TLB_global_first_row_index - WLB_first_row_index;
                
                if (*max_first_relative_reduce_row < TLB_relative_first_row_index)
                {
                    *max_first_relative_reduce_row = TLB_relative_first_row_index;
                }
                
                // 写到对应位置
                TLB_first_relative_row_index.push_back(TLB_relative_first_row_index);
                break;
            }
        }

        // 如果全是F，就继承之前的行号
        if (has_T_in_TLB == false)
        {
            assert(TLB_first_relative_row_index.size() > 0);

            if (*max_first_relative_reduce_row < TLB_first_relative_row_index[TLB_first_relative_row_index.size() - 1])
            {
                *max_first_relative_reduce_row = TLB_first_relative_row_index[TLB_first_relative_row_index.size() - 1];
            }
            
            TLB_first_relative_row_index.push_back(TLB_first_relative_row_index[TLB_first_relative_row_index.size() - 1]);
        }
    }

    return TLB_first_relative_row_index;
}

void store_bool_flag_of_sum_begin_to_file(vector<vector<bool>> bool_vec, string input_file)
{
    // 向文件中写文件
    ofstream arrWrite(input_file, ios::out | ios::trunc);

    cout << "file_name:" << input_file << endl;

    // 打印
    for (unsigned int i = 0; i < bool_vec.size(); i++)
    {
        for (unsigned int j = 0; j < bool_vec[i].size(); j++)
        {
            arrWrite << bool_vec[i][j] << " , ";
        }
        arrWrite << endl;
    }

    arrWrite.close();
}

int get_max_bit_num_of_meta_data(unsigned long max_meta_data)
{
    // 一直右移直到0
    int bit_num = 0;

    while (max_meta_data != 0)
    {
        max_meta_data = max_meta_data >> 1;
        bit_num++;
    }

    return bit_num;
}

// 将三组数据合并到一起，产生某种类型的数据
// 将bool bit、归约偏移、相对行号合并到一个数据中，高位是bool flag产生的bit flag。低位是紧挨着的归约偏移和相对行号
// 高位|bit_flag|可能存在的空白|归约偏移|相对行号|低位
template<typename T>
T combine_meta_data(vector<bool> sum_begin_bool_flag, T reduce_offset, T reduce_relative_row_index, int bit_num_of_sum_begin_flag, int bit_num_of_reduce_offset, int bit_num_of_row_index)
{
    int combine_data_size = sizeof(T) * 8;

    if (bit_num_of_sum_begin_flag + bit_num_of_reduce_offset + bit_num_of_row_index > combine_data_size)
    {
        cout << "unsigned long cannot content all metadata" << endl;
        assert(false);
    }
    
    T return_combine_meta_data = 0;

    // 空白部分的大小
    int bit_num_empty_bit = combine_data_size - bit_num_of_sum_begin_flag - bit_num_of_reduce_offset - bit_num_of_row_index;

    assert(bit_num_empty_bit >= 0);

    assert(sum_begin_bool_flag.size() == bit_num_of_sum_begin_flag);

    // 将sum begin的bit存起来，放在低位，从高位开始存
    T compressed_num_of_sum_begin_flag = 0;
    for (int i = 0; i < sum_begin_bool_flag.size(); i++)
    {
        bool cur_bool = sum_begin_bool_flag[i];

        if (cur_bool == true)
        {
            T temp_bool_bit = 1 << (bit_num_of_sum_begin_flag - 1 - i);
            compressed_num_of_sum_begin_flag = compressed_num_of_sum_begin_flag | temp_bool_bit;
        }
    }

    // compressed_num_of_sum_begin_flag中存的是bit flag处理完之后的结果
    return_combine_meta_data = return_combine_meta_data | compressed_num_of_sum_begin_flag;

    // 将空白部分和归约偏移量所占的bit空出来，然后将归约偏移量写进去
    return_combine_meta_data = return_combine_meta_data << (bit_num_empty_bit + bit_num_of_reduce_offset);

    // 写归约偏移
    return_combine_meta_data = return_combine_meta_data | reduce_offset;

    // 再挪出一个空间存储相对行号
    return_combine_meta_data = return_combine_meta_data << bit_num_of_row_index;

    // 存储相对行号
    return_combine_meta_data = return_combine_meta_data | reduce_relative_row_index;

    // 压缩合并之后的结果
    return return_combine_meta_data;
}

unsigned long combine_meta_data_to_unsigned_long(vector<bool> sum_begin_bool_flag, unsigned long reduce_offset, unsigned long reduce_relative_row_index, int bit_num_of_sum_begin_flag, int bit_num_of_reduce_offset, int bit_num_of_row_index)
{
    unsigned long return_combine_meta;
    return_combine_meta = combine_meta_data(sum_begin_bool_flag, (unsigned long)reduce_offset, (unsigned long) reduce_relative_row_index, bit_num_of_sum_begin_flag, bit_num_of_reduce_offset, bit_num_of_row_index);
    return return_combine_meta;
}

unsigned int combine_meta_data_to_unsigned_int(vector<bool> sum_begin_bool_flag, unsigned long reduce_offset, unsigned long reduce_relative_row_index, int bit_num_of_sum_begin_flag, int bit_num_of_reduce_offset, int bit_num_of_row_index)
{
    unsigned int return_combine_meta;
    return_combine_meta = combine_meta_data(sum_begin_bool_flag, (unsigned int)reduce_offset, (unsigned int)reduce_relative_row_index, bit_num_of_sum_begin_flag, bit_num_of_reduce_offset, bit_num_of_row_index);
    return return_combine_meta;
}

unsigned short combine_meta_data_to_unsigned_short(vector<bool> sum_begin_bool_flag, unsigned long reduce_offset, unsigned long reduce_relative_row_index, int bit_num_of_sum_begin_flag, int bit_num_of_reduce_offset, int bit_num_of_row_index)
{
    unsigned short return_combine_meta;
    return_combine_meta = combine_meta_data(sum_begin_bool_flag, (unsigned short)reduce_offset, (unsigned short)reduce_relative_row_index, bit_num_of_sum_begin_flag, bit_num_of_reduce_offset, bit_num_of_row_index);
    return return_combine_meta;
}

unsigned char combine_meta_data_to_unsigned_char(vector<bool> sum_begin_bool_flag, unsigned long reduce_offset, unsigned long reduce_relative_row_index, int bit_num_of_sum_begin_flag, int bit_num_of_reduce_offset, int bit_num_of_row_index)
{
    unsigned char return_combine_meta;
    return_combine_meta = combine_meta_data(sum_begin_bool_flag, (unsigned char) reduce_offset, (unsigned char)reduce_relative_row_index, bit_num_of_sum_begin_flag, bit_num_of_reduce_offset, bit_num_of_row_index);
    return return_combine_meta;
}

template<typename T>
string convert_meta_data_to_bit_flag_string(T compress_bit)
{
    // 字符串流
    stringstream ss;
    
    // 压缩之后元数据的bit数量
    int combine_meta_bit_num = sizeof(T) * 8;

    // 遍历每一个bit
    for (int bit_id = 0; bit_id < combine_meta_bit_num; bit_id++)
    {
        unsigned int bit = (compress_bit >> (combine_meta_bit_num - 1 - bit_id)) & 0x1;

        assert(bit <= 1);

        ss << bit;

        if ((bit_id + 1) % 4 == 0)
        {
            ss << " ";
        }
    }

    return ss.str();
}

// 按照数据类型的大小将合并之后的数据转化为string
template<typename T>
string convert_meta_data_to_bit_flag_string_with_data_type(T compress_bit, data_type type)
{
    if (type == UNSIGNED_LONG)
    {
        return convert_meta_data_to_bit_flag_string((unsigned long)compress_bit);
    }

    if (type == UNSIGNED_INT)
    {
        return convert_meta_data_to_bit_flag_string((unsigned int)compress_bit);
    }

    if (type == UNSIGNED_SHORT)
    {
        return convert_meta_data_to_bit_flag_string((unsigned short)compress_bit);
    }

    if (type == UNSIGNED_CHAR)
    {
        return convert_meta_data_to_bit_flag_string((unsigned char)compress_bit);
    }

    cout << "convert_meta_data_to_bit_flag_string_with_data_type: data type is not supported" << endl;
    assert(false);
}

bool compress_global_first_row_index_of_warp_level_block(unaligned_warp_reduce_same_TLB_size_template_t *output_template, bool need_check, arr_compress_type type)
{   
    assert(output_template != NULL && (type == LINEAR_COMPRESS || type == CYCLE_INCREASE_COMPRESS) && output_template->global_first_row_index_of_warp_level_block != NULL);

    if (type == LINEAR_COMPRESS)
    {
        linear_compress_t *compressor = init_linear_compressor(output_template->global_first_row_index_of_warp_level_block, output_template->data_type_of_global_first_row_index_of_warp_level_block, output_template->size_of_global_first_row_index_of_warp_level_block, need_check);

        if (compressor == NULL)
        {
            return false;
        }

        // 压缩成功
        output_template->global_first_row_index_of_warp_level_block_compress_meta = (void *)compressor;
        output_template->global_first_row_index_of_warp_level_block_compress = type;

        return true;
    }

    if (type == CYCLE_INCREASE_COMPRESS)
    {
        cycle_increase_compress_t *compressor = init_cycle_increase_compressor(output_template->global_first_row_index_of_warp_level_block, output_template->data_type_of_global_first_row_index_of_warp_level_block, output_template->size_of_global_first_row_index_of_warp_level_block, need_check);

        if (compressor == NULL)
        {
            return false;
        }

        // 压缩成功
        output_template->global_first_row_index_of_warp_level_block_compress_meta = (void *)compressor;
        output_template->global_first_row_index_of_warp_level_block_compress = type;

        return true;
    }

    return false;
}

// 可以使用线性排序
bool compress_row_index_before_sort(unaligned_warp_reduce_same_TLB_size_template_t *output_template, bool need_check, arr_compress_type type)
{
    assert(output_template != NULL && type == LINEAR_COMPRESS && output_template->row_index_before_sort != NULL);

    // 使用线性压缩
    linear_compress_t *compressor = init_linear_compressor(output_template->row_index_before_sort, output_template->data_type_of_row_index_before_sort, output_template->size_of_row_index_before_sort, need_check);

    if (compressor == NULL)
    {
        return false;
    }

    output_template->row_index_before_sort_compress_meta = (void *)compressor;
    output_template->row_index_before_sort_compress = type;

    return true;
}

void try_all_compress(unaligned_warp_reduce_same_TLB_size_template_t *output_template)
{
    assert(output_template != NULL);
    
    bool is_compressed = false;

    // 压缩warp行首索引，分别尝试两种方式
    is_compressed = compress_global_first_row_index_of_warp_level_block(output_template, true, LINEAR_COMPRESS);

    if (is_compressed == false)
    {
        is_compressed = compress_global_first_row_index_of_warp_level_block(output_template, true, CYCLE_INCREASE_COMPRESS);
    }

    // 暂时先不压缩排序的数组
    // is_compressed = compress_row_index_before_sort(output_template, true, LINEAR_COMPRESS);
}

void write_combine_meta_data_to_file(void* combine_meta_arr, data_type type, unsigned long length, string file_name)
{
    assert(combine_meta_arr != NULL);
    assert(type == UNSIGNED_LONG || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_CHAR);

    ofstream arrWrite(file_name, ios::out | ios::trunc);

    cout << "file_name:" << file_name << endl;

    for (unsigned long combine_meta_id = 0; combine_meta_id < length; combine_meta_id++)
    {
        // 将bit转化为一个字符串
        if (type == UNSIGNED_LONG)
        {
            unsigned long combine_meta = read_from_array_with_data_type(combine_meta_arr, type, combine_meta_id);

            string combine_meta_str = convert_meta_data_to_bit_flag_string_with_data_type(combine_meta, type);

            arrWrite << combine_meta_str << endl;

            continue;
        }

        if (type == UNSIGNED_INT)
        {
            unsigned int combine_meta = read_from_array_with_data_type(combine_meta_arr, type, combine_meta_id);

            string combine_meta_str = convert_meta_data_to_bit_flag_string_with_data_type(combine_meta, type);

            arrWrite << combine_meta_str << endl;

            continue;
        }

        if (type == UNSIGNED_SHORT)
        {
            unsigned short combine_meta = read_from_array_with_data_type(combine_meta_arr, type, combine_meta_id);
            
            string combine_meta_str = convert_meta_data_to_bit_flag_string_with_data_type(combine_meta, type);

            arrWrite << combine_meta_str << endl;

            continue;
        }

        if (type == UNSIGNED_CHAR)
        {
            unsigned char combine_meta = read_from_array_with_data_type(combine_meta_arr, type, combine_meta_id);
            
            string combine_meta_str = convert_meta_data_to_bit_flag_string_with_data_type(combine_meta, type);

            arrWrite << combine_meta_str << endl;

            continue;
        }
    }

    arrWrite.close();
}