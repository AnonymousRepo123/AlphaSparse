#include "op_manager.hpp"
#include <cassert>
#include <iostream>
#include <algorithm>
#include <string.h>
#include "config.hpp"
#include "memory_garbage_manager.hpp"

vector<sparse_struct_t *> long_short_row_decomposition(sparse_struct_t *matrix_struct, vector<unsigned int> row_nnz_low_bound_of_sub_matrix)
{
    // 矩阵还没有执行任何操作
    assert(matrix_struct != NULL);
    assert(matrix_struct->is_compressed == false && matrix_struct->is_blocked == false && matrix_struct->is_sorted == false);

    // 原非零元数量和新非零元数量是一致的
    assert(matrix_struct->nnz == matrix_struct->origin_nnz);

    // 三个数组都有东西
    assert(matrix_struct->coo_row_index_cache != NULL && matrix_struct->coo_col_index_cache != NULL && matrix_struct->coo_value_cache != NULL);

    // 检查行非零元下界
    assert(row_nnz_low_bound_of_sub_matrix.size() > 1 && row_nnz_low_bound_of_sub_matrix[0] == 0);

    // 获取整个矩阵的行非零元数量
    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(matrix_struct->coo_row_index_cache, UNSIGNED_LONG, 0, matrix_struct->dense_row_number - 1, 0, matrix_struct->nnz - 1);

    assert(nnz_of_each_row.size() == matrix_struct->dense_row_number);

    // 用一个数组存储每一行所属于的子矩阵索引
    vector<int> sub_matrix_id_of_a_row(matrix_struct->dense_row_number);

    // 遍历所有的行的非零元，将行索引登记到对应的桶中
    for (unsigned long row_id = 0; row_id < nnz_of_each_row.size(); row_id++)
    {
        // 当前行的非零元数量
        unsigned long row_nnz = nnz_of_each_row[row_id];

        // 从高的比较到底的，比到第一个比下界大的，就是当前行所在的子矩阵
        for (unsigned long sub_matrix_id = row_nnz_low_bound_of_sub_matrix.size() - 1; sub_matrix_id >= 0; sub_matrix_id--)
        {
            // 当前下界
            unsigned long cur_low_bound = row_nnz_low_bound_of_sub_matrix[sub_matrix_id];
            // 如果当前行非零元数量大于等于下界，那就属于当前子块
            if (cur_low_bound <= row_nnz)
            {
                sub_matrix_id_of_a_row[row_id] = sub_matrix_id;
                break;
            }
        }

        // cout << "row_id" << row_id << endl;
    }

    vector<sparse_struct_t *> return_matrix_vec;

    // 遍历所有的行索引集合
    for (unsigned long sub_matrix_id = 0; sub_matrix_id < row_nnz_low_bound_of_sub_matrix.size(); sub_matrix_id++)
    {
        // 用三个矩阵，分别存储一个子矩阵的三个索引
        vector<unsigned long> row_index_vec;
        vector<unsigned long> col_index_vec;
        vector<float> val_float_vec;
        vector<double> val_double_vec;


        // 便利原矩阵中所有的非零元，放到新的索引中。以此构建新的子矩阵
        for (unsigned long nz_id = 0; nz_id < matrix_struct->nnz; nz_id++)
        {
            // 值和坐标
            unsigned long row_index = matrix_struct->coo_row_index_cache[nz_id];
            unsigned long col_index = matrix_struct->coo_col_index_cache[nz_id];

            double val = read_double_from_array_with_data_type(matrix_struct->coo_value_cache, matrix_struct->val_data_type, nz_id);

            // 当前行属于的子块
            int sub_matrix_id_of_cur_row = sub_matrix_id_of_a_row[row_index];

            // 写到当前的缓存中
            if (sub_matrix_id_of_cur_row == sub_matrix_id)
            {
                row_index_vec.push_back(row_index);
                col_index_vec.push_back(col_index);

                if (matrix_struct->val_data_type == FLOAT)
                {
                    val_float_vec.push_back(val);
                }
                else
                {
                    val_double_vec.push_back(val);
                }
            }
        }

        // 只要缓存里面有东西，就建立一个子矩阵
        if (row_index_vec.size() == 0)
        {
            continue;
        }

        sparse_struct_t* new_matrix = init_sparse_struct_by_coo_vector(row_index_vec, col_index_vec, val_float_vec, val_double_vec, matrix_struct->val_data_type, matrix_struct->dense_col_number - 1, matrix_struct->dense_row_number - 1);

        assert(new_matrix != NULL);
        
        return_matrix_vec.push_back(new_matrix);
    }

    assert(return_matrix_vec.size() > 1);

    unsigned long total_nnz = 0;
    
    for (unsigned long sub_matrix_id = 0; sub_matrix_id < return_matrix_vec.size(); sub_matrix_id++)
    {
        total_nnz = total_nnz + return_matrix_vec[sub_matrix_id]->nnz;
    }

    assert(total_nnz == matrix_struct->nnz);

    memory_garbage_manager_t mem_manager;

    delete_sparse_struct_t(&mem_manager, matrix_struct);


    return return_matrix_vec;
}

void fixed_len_row_div(sparse_struct_t *matrix_struct, dense_block_table_item_t *sub_block, int len)
{
    assert((matrix_struct->block_coor_table.item_arr.size() == 0 && sub_block == NULL) ||
           (matrix_struct->block_coor_table.item_arr.size() >= 0 && sub_block != NULL));

    // 如果没有多个快，直接切
    if (sub_block == NULL)
    {
        // 从初始块中切出来多个块，假设行号从0开始
        // 用一个for遍历所有的行
        int i;
        // 一个条带的行范围在i到i+len-1之间
        // 当前分出的子块索引
        int cur_sub_block_index = 0;
        cout << "begin compress row index" << endl;

        // 将要切的部分每一行在coo格式中的偏移量算出
        vector<unsigned long> coo_offset_arr = find_coo_row_index_range(matrix_struct, 0, matrix_struct->nnz - 1);

        // 计算实际的最小行号和最大行号
        unsigned long real_min_row_index = matrix_struct->coo_row_index_cache[0];
        unsigned long real_max_row_index = matrix_struct->coo_row_index_cache[matrix_struct->nnz - 1];

        // 最大行号和最小行号之间满足关系
        assert((real_max_row_index - real_min_row_index + 1) == (coo_offset_arr.size() - 1));

        cout << "finish compress row index" << endl;

        for (i = 0; i < matrix_struct->dense_row_number; i = i + len)
        {

            // 这里首先要加一个判断，当前行条带和实际存在的行取值范围有没有交集，没有就直接continue
            // 主要是判断一个行条带的行号的上界和下界是不是至少有一个在当前子块的实际行号取值范围内
            // 并且逻辑行号要和实际的行编号下界减一下才能才能行
            dense_block_table_item_t *new_item = new dense_block_table_item_t();

            // 初始化范围
            new_item->min_dense_row_index = i;

            // 不能超过矩阵上界
            if ((i + len - 1) >= (matrix_struct->dense_row_number - 1))
            {
                new_item->max_dense_row_index = matrix_struct->dense_row_number - 1;
            }
            else
            {
                new_item->max_dense_row_index = i + len - 1;
            }

            // 看看重合情况
            if (new_item->max_dense_row_index < real_min_row_index || new_item->min_dense_row_index > real_max_row_index)
            {
                // 完全不重合，说明条带全是空的
                cout << "find empty row group" << endl;
                delete new_item;
                continue;
            }

            new_item->min_dense_col_index = 0;
            new_item->max_dense_col_index = matrix_struct->dense_col_number - 1;

            // 首先和最小的实际index减一下，然后再决定其coo偏移量
            unsigned long row_different = new_item->min_dense_row_index - real_min_row_index;

            if (row_different < 0)
            {
                // 这个下界还没有够到子块最小行号，那就从最小行号的偏移量开始算
                new_item->begin_coo_index = coo_offset_arr[0];
            }
            else
            {
                // 反之用最小行号的偏移量来找到coo偏移量
                new_item->begin_coo_index = coo_offset_arr[row_different];
            }

            row_different = new_item->max_dense_row_index - real_min_row_index;

            if (row_different >= coo_offset_arr.size() - 1)
            {
                cout << "TODO: has empty line in parent block bottom" << endl;
            }

            // CSR最后一位和行是不对应的
            assert(row_different < coo_offset_arr.size() - 1);

            // 最大值行号要和这个块拥有的真实最大行号比较，如果大于最大真实行号，那就按照最大真实行号的偏移量来处理
            if (new_item->max_dense_row_index > real_max_row_index)
            {
                // 按照最后一个真实行的偏移量来
                new_item->end_coo_index = coo_offset_arr[coo_offset_arr.size() - 1] - 1;
            }
            else
            {
                new_item->end_coo_index = coo_offset_arr[row_different + 1] - 1;
            }

            // 如果行条带的没有非零元就跳过
            if (new_item->begin_coo_index > new_item->end_coo_index)
            {
                cout << "find empty row group" << endl;
                delete new_item;
                continue;
            }

            // 当前块的坐标
            new_item->block_coordinate.push_back(cur_sub_block_index);
            cur_sub_block_index++;

            // 没有压缩之前还没有指针
            new_item->compressed_block_ptr = NULL;

            // 子块的元数据加入到表中
            matrix_struct->block_coor_table.item_arr.push_back(new_item);
        }
    }
    else
    {
        // 对某一个子块进一步分块，实际的最大最小索引和逻辑的最大最小索引，考虑到空行，并且考虑到CSR压缩处理不了头部和尾部的空行
        unsigned long begin_row_index = sub_block->min_dense_row_index;
        unsigned long end_row_index = sub_block->max_dense_row_index;
        unsigned long begin_col_index = sub_block->min_dense_col_index;
        unsigned long end_col_index = sub_block->max_dense_col_index;

        unsigned long begin_coo_index = sub_block->begin_coo_index;
        unsigned long end_coo_index = sub_block->end_coo_index;

        // cout << begin_coo_index << " " << end_coo_index << endl;

        // 搜出这个子块中所有行在coo格式中的起始位置
        vector<unsigned long> coo_offset_arr = find_coo_row_index_range(matrix_struct, begin_coo_index, end_coo_index);

        // 计算实际的最小行号和最大行号
        unsigned long real_min_row_index = matrix_struct->coo_row_index_cache[begin_coo_index];
        unsigned long real_max_row_index = matrix_struct->coo_row_index_cache[end_coo_index];

        // 最大行号和最小行号之间满足关系
        // cout << real_max_row_index - real_min_row_index + 1 << " " << coo_offset_arr.size() - 1 << endl;

        assert((real_max_row_index - real_min_row_index + 1) == (coo_offset_arr.size() - 1));
        // int q;
        // for(q = 0; q < coo_offset_arr.size(); q++){
        //     cout << coo_offset_arr[q] << ",";
        // }

        // cout << endl;

        vector<int> basic_coor(sub_block->block_coordinate);

        // 在表格中搜出这个元素、删除这个元素
        auto iter = std::remove(matrix_struct->block_coor_table.item_arr.begin(),
                                matrix_struct->block_coor_table.item_arr.end(), sub_block);
        matrix_struct->block_coor_table.item_arr.erase(iter, matrix_struct->block_coor_table.item_arr.end());

        // 用for循环产生新的元素，插入到表格中
        int cur_sub_block_index = 0;

        int i;
        for (i = begin_row_index; i <= end_row_index; i = i + len)
        {
            // 这里首先要加一个判断，当前行条带和实际存在的行取值范围有没有交集，没有就直接continue
            // 主要是判断一个行条带的行号的上界和下界是不是至少有一个在当前子块的实际行号取值范围内，在列中处理了空列，行中也要处理空行的情况
            dense_block_table_item_t *new_item = new dense_block_table_item_t();

            // 初始化范围
            new_item->min_dense_row_index = i;

            // 不能超过矩阵上界
            if ((i + len - 1) >= end_row_index)
            {
                new_item->max_dense_row_index = end_row_index;
            }
            else
            {
                new_item->max_dense_row_index = i + len - 1;
            }

            // 看看重合情况
            if (new_item->max_dense_row_index < real_min_row_index || new_item->min_dense_row_index > real_max_row_index)
            {
                // 完全不重合，说明条带全是空的
                cout << "find empty row group" << endl;
                delete new_item;
                continue;
            }

            new_item->min_dense_col_index = begin_col_index;
            new_item->max_dense_col_index = end_col_index;

            // 首先和最小的实际index减一下，然后再决定其coo偏移量
            unsigned long row_different = new_item->min_dense_row_index - real_min_row_index;

            if (row_different < 0)
            {
                // 这个下界还没有够到子块最小行号，那就从最小行号的偏移量开始算
                new_item->begin_coo_index = coo_offset_arr[0];
            }
            else
            {
                // 反之用最小行号的偏移量来找到coo偏移量
                new_item->begin_coo_index = coo_offset_arr[row_different];
            }

            row_different = new_item->max_dense_row_index - real_min_row_index;

            // 最大值行号要和这个块拥有的真实最大行号比较，如果大于最大真实行号，那就按照最大真实行号的偏移量来处理
            if (new_item->max_dense_row_index > real_max_row_index)
            {
                // 按照最后一个真实行的偏移量来
                new_item->end_coo_index = coo_offset_arr[coo_offset_arr.size() - 1] - 1;
            }
            else
            {
                new_item->end_coo_index = coo_offset_arr[row_different + 1] - 1;
            }

            // cout << new_item->begin_coo_index << " " << new_item->end_coo_index << endl;

            // 如果行条带的没有非零元就跳过
            if (new_item->begin_coo_index > new_item->end_coo_index)
            {
                cout << "find empty row group:" << cur_sub_block_index << endl;
                delete new_item;
                continue;
            }

            new_item->block_coordinate.insert(new_item->block_coordinate.end(), basic_coor.begin(), basic_coor.end());
            new_item->block_coordinate.push_back(cur_sub_block_index);
            cur_sub_block_index++;

            // 没有压缩之前还没有指针
            new_item->compressed_block_ptr = NULL;

            // 子块的元数据加入到表中
            matrix_struct->block_coor_table.item_arr.push_back(new_item);
        }

        // 析构这个元素
        delete sub_block;
    }

    matrix_struct->is_blocked = true;
}

void var_len_row_div(sparse_struct_t *matrix_struct, dense_block_table_item_t *sub_block, vector<unsigned long> block_first_row_csr_index)
{
    assert((matrix_struct->block_coor_table.item_arr.size() == 0 && sub_block == NULL) ||
           (matrix_struct->block_coor_table.item_arr.size() >= 0 && sub_block != NULL));

    // 如果没有多个快，直接切
    if (sub_block == NULL)
    {
        // 直接切，block_first_row_csr_index要满足一些条件
        assert(block_first_row_csr_index.size() > 2);

        assert(block_first_row_csr_index[0] == 0 && block_first_row_csr_index[block_first_row_csr_index.size() - 1] == matrix_struct->dense_row_number);

        // 从初始块中切出来多个块，假设行号从0开始
        // 用一个for遍历所有的行
        int i;
        // 一个条带的行范围在i到i+len-1之间
        // 当前分出的子块索引
        int cur_sub_block_index = 0;
        cout << "begin compress row index" << endl;

        // 将要切的部分每一行在coo格式中的偏移量算出
        vector<unsigned long> coo_offset_arr = find_coo_row_index_range(matrix_struct, 0, matrix_struct->nnz - 1);

        // 计算实际的最小行号和最大行号，因为存在空行，所以实际的行索引和范围和真实的行索引范围有出入
        unsigned long real_min_row_index = matrix_struct->coo_row_index_cache[0];
        unsigned long real_max_row_index = matrix_struct->coo_row_index_cache[matrix_struct->nnz - 1];

        // cout << "real_min_row_index:" << real_min_row_index << " real_max_row_index:" << real_max_row_index << " coo_offset_arr.size():" << coo_offset_arr.size() << "matrix_struct->dense_row_number:" << matrix_struct->dense_row_number << endl;

        // 将coo_offset_arr输出到文件
        // print_arr_to_file_with_data_type(&(coo_offset_arr[0]), UNSIGNED_LONG, coo_offset_arr.size(), "/home/duzhen/spmv_builder/data_source/test_result_3");

        // exit(-1);

        // 最大行号和最小行号之间满足关系，用两种方式分别计算行数量
        assert((real_max_row_index - real_min_row_index + 1) == (coo_offset_arr.size() - 1));

        cout << "finish compress row index" << endl;

        // row条带的索引
        unsigned long row_band_index = 0;

        // 每一个行条带的宽度
        unsigned long len = 0;

        // 遍历每一个宽度的条带
        for (i = 0; i < matrix_struct->dense_row_number; i = i + len)
        {
            // 下一个行条带的宽度
            assert(block_first_row_csr_index[row_band_index + 1] - block_first_row_csr_index[row_band_index] >= 0);
            len = block_first_row_csr_index[row_band_index + 1] - block_first_row_csr_index[row_band_index];
            // cout << "len:" << len << endl;
            row_band_index++;
            // 这里首先要加一个判断，当前行条带和实际存在的行取值范围有没有交集，没有就直接continue
            // 主要是判断一个行条带的行号的上界和下界是不是至少有一个在当前子块的实际行号取值范围内
            // 并且逻辑行号要和实际的行编号下界减一下才能才能行
            dense_block_table_item_t *new_item = new dense_block_table_item_t();

            // 初始化范围
            new_item->min_dense_row_index = i;

            // 不能超过矩阵上界
            if ((i + len - 1) >= (matrix_struct->dense_row_number - 1))
            {
                new_item->max_dense_row_index = matrix_struct->dense_row_number - 1;
            }
            else
            {
                new_item->max_dense_row_index = i + len - 1;
            }

            // 看看重合情况
            if (new_item->max_dense_row_index < real_min_row_index || new_item->min_dense_row_index > real_max_row_index)
            {
                // 完全不重合，说明条带全是空的
                cout << "find empty row group" << endl;
                delete new_item;
                continue;
            }

            // 因为子块的指针是NULL，所以之前没有排过，
            new_item->min_dense_col_index = 0;
            new_item->max_dense_col_index = matrix_struct->dense_col_number - 1;

            // 首先和最小的实际index减一下，然后再决定其coo偏移量，这里考虑的是一开始有偏移量的情况，
            long row_different = new_item->min_dense_row_index - real_min_row_index;

            if (new_item->min_dense_row_index < real_min_row_index)
            {
                // 这个下界还没有够到子块最小行号，那就从最小行号的偏移量开始算
                new_item->begin_coo_index = coo_offset_arr[0];
            }
            else
            {
                // 反之用最小行号的偏移量来找到coo偏移量
                new_item->begin_coo_index = coo_offset_arr[row_different];
            }

            // 当前子块最后一行的行号
            row_different = new_item->max_dense_row_index - real_min_row_index;

            // CSR最后一位和行是不对应的
            if (row_different >= coo_offset_arr.size() - 1)
            {
                // 这里说明被划分的块的末尾存在一些空行，需要用实际的存在
                cout << "has empty line in parent block bottom" << endl;
                cout << "row_different:" << row_different << " coo_offset_arr.size() - 1:" << coo_offset_arr.size() - 1 << endl;

                // 之后row_different不会再被用到
                row_different = real_max_row_index - real_min_row_index;
            }

            assert(row_different < coo_offset_arr.size() - 1);

            // 最大值行号要和这个块拥有的真实最大行号比较，如果大于最大真实行号，那就按照最大真实行号的偏移量来处理
            if (new_item->max_dense_row_index > real_max_row_index)
            {
                // 按照最后一个真实行的偏移量来
                new_item->end_coo_index = coo_offset_arr[coo_offset_arr.size() - 1] - 1;
            }
            else
            {
                new_item->end_coo_index = coo_offset_arr[row_different + 1] - 1;
            }

            // 如果行条带的没有非零元就跳过
            if (new_item->begin_coo_index > new_item->end_coo_index)
            {
                cout << "find empty row group" << endl;
                delete new_item;
                continue;
            }

            // 当前块的坐标
            new_item->block_coordinate.push_back(cur_sub_block_index);
            cur_sub_block_index++;

            // 没有压缩之前还没有指针
            new_item->compressed_block_ptr = NULL;

            // 子块的元数据加入到表中
            matrix_struct->block_coor_table.item_arr.push_back(new_item);
        }

        // cout << "i:" << i << " matrix_struct->dense_row_number" << matrix_struct->dense_row_number << endl;
        // assert(i == matrix_struct->dense_row_number);
    }
    else
    {
        cout << "further div is not support" << endl;
        assert(false);
    }

    matrix_struct->is_blocked = true;
}

// 类似于CSR压缩的函数
vector<unsigned long> find_coo_row_index_range(sparse_struct_t *matrix_struct, unsigned long find_coo_begin,
                                               unsigned long find_coo_end)
{
    vector<unsigned long> return_coo_index;
    unsigned long *row_index_arr = matrix_struct->coo_row_index_cache;
    unsigned long *col_index_arr = matrix_struct->coo_col_index_cache;

    void *value_arr = matrix_struct->coo_value_cache;

    unsigned long last_row_index;

    int i;
    for (i = find_coo_begin; i <= find_coo_end; i++)
    {
        // 第一个单独处理
        if (i == find_coo_begin)
        {
            return_coo_index.push_back(i);
            last_row_index = row_index_arr[i];
        }
        else
        {
            // 如果行索引的值变化了，就记录偏移量，根据值变化了多少可能需要记录多个偏移量来处理空行的情况
            unsigned long index_different = row_index_arr[i] - last_row_index;

            if (index_different != 0)
            {
                // 到新的行了，添加多个偏移量，index_different中包含了空行的数量
                int j;
                for (j = 0; j < index_different; j++)
                {
                    return_coo_index.push_back(i);
                }

                last_row_index = row_index_arr[i];
            }
        }
    }

    // 给return_coo_index最后再添加一个表示下边界的元素
    return_coo_index.push_back(find_coo_end + 1);

    // 最后返回压缩的索引
    return return_coo_index;
}

vector<coo_element_double_t> find_all_col_double_element(sparse_struct_t *matrix_struct, unsigned long find_coo_begin,
                                                         unsigned long find_coo_end, unsigned long col_index_begin,
                                                         unsigned long col_index_end)
{
    assert(find_coo_begin <= find_coo_end);
    assert(col_index_begin <= col_index_end);
    assert(matrix_struct->val_data_type == DOUBLE);

    vector<coo_element_double_t> element_arr;

    int i;
    for (i = find_coo_begin; i <= find_coo_end; i++)
    {
        if (matrix_struct->coo_col_index_cache[i] >= col_index_begin && matrix_struct->coo_col_index_cache[i] <= col_index_end)
        {
            element_arr.push_back({matrix_struct->coo_row_index_cache[i], matrix_struct->coo_col_index_cache[i], ((double *)(matrix_struct->coo_value_cache))[i]});
        }
    }

    return element_arr;
}

void find_all_col_double_element(sparse_struct_t *matrix_struct, unsigned long find_coo_begin,
                                 unsigned long find_coo_end, unsigned long col_index_begin,
                                 unsigned long col_index_end, vector<unsigned long> *output_row_arr,
                                 vector<unsigned long> *output_col_arr, vector<double> *double_var_arr)
{
    assert(find_coo_begin <= find_coo_end);
    assert(col_index_begin <= col_index_end);
    assert(matrix_struct->val_data_type == DOUBLE);
    assert(output_row_arr != NULL && output_col_arr != NULL && double_var_arr != NULL);

    int i;
    for (i = find_coo_begin; i <= find_coo_end; i++)
    {
        if (matrix_struct->coo_col_index_cache[i] >= col_index_begin && matrix_struct->coo_col_index_cache[i] <= col_index_end)
        {
            // 满足要求
            output_row_arr->push_back(matrix_struct->coo_row_index_cache[i]);
            output_col_arr->push_back(matrix_struct->coo_col_index_cache[i]);
            double_var_arr->push_back(((double *)(matrix_struct->coo_value_cache))[i]);
        }
    }
}

void find_all_col_float_element(sparse_struct_t *matrix_struct, unsigned long find_coo_begin,
                                unsigned long find_coo_end, unsigned long col_index_begin,
                                unsigned long col_index_end, vector<unsigned long> *output_row_arr,
                                vector<unsigned long> *output_col_arr, vector<float> *float_var_arr)
{
    assert(find_coo_begin <= find_coo_end);
    assert(col_index_begin <= col_index_end);
    assert(matrix_struct->val_data_type == FLOAT);
    assert(output_row_arr != NULL && output_col_arr != NULL && float_var_arr != NULL);

    int i;
    for (i = find_coo_begin; i <= find_coo_end; i++)
    {
        if (matrix_struct->coo_col_index_cache[i] >= col_index_begin && matrix_struct->coo_col_index_cache[i] <= col_index_end)
        {
            // 满足要求
            output_row_arr->push_back(matrix_struct->coo_row_index_cache[i]);
            output_col_arr->push_back(matrix_struct->coo_col_index_cache[i]);
            float_var_arr->push_back(((float *)(matrix_struct->coo_value_cache))[i]);
        }
    }
}

vector<coo_element_float_t> find_all_col_float_element(sparse_struct_t *matrix_struct, unsigned long find_coo_begin,
                                                       unsigned long find_coo_end, unsigned long col_index_begin,
                                                       unsigned long col_index_end)
{
    assert(find_coo_begin <= find_coo_end);
    assert(col_index_begin <= col_index_end);
    assert(matrix_struct->val_data_type == FLOAT);

    vector<coo_element_float_t> element_arr;

    int i;
    for (i = find_coo_begin; i <= find_coo_end; i++)
    {
        if (matrix_struct->coo_col_index_cache[i] >= col_index_begin && matrix_struct->coo_col_index_cache[i] <= col_index_end)
        {
            element_arr.push_back({matrix_struct->coo_row_index_cache[i], matrix_struct->coo_col_index_cache[i], ((float *)(matrix_struct->coo_value_cache))[i]});
        }
    }

    return element_arr;
}

// 定长列分块操作
void fixed_len_col_div(sparse_struct_t *matrix_struct, dense_block_table_item_t *sub_block, int len)
{
    // 一上来的分块操作和后来的是不一样的
    assert((matrix_struct->block_coor_table.item_arr.size() == 0 && sub_block == NULL) ||
           (matrix_struct->block_coor_table.item_arr.size() >= 0 && sub_block != NULL));

    if (sub_block == NULL)
    {
        // 按照len分块
        int i;

        int cur_sub_block_index = 0;
        unsigned long cur_coo_index = 0;

        // 用一个数组存储新的coo数据
        vector<unsigned long> new_row_index;
        vector<unsigned long> new_col_index;
        vector<double> new_double_var;
        vector<float> new_float_var;

        // 连续多次搜索范围内列号满足要求的非零元
        for (i = 0; i < matrix_struct->dense_col_number; i = i + len)
        {
            dense_block_table_item_t *new_item = new dense_block_table_item_t();
            new_item->is_col_blocked = true;

            // 初始化范围
            new_item->min_dense_col_index = i;

            // 不超过矩阵上界的列索引范围
            if ((i + len - 1) >= (matrix_struct->dense_col_number - 1))
            {
                new_item->max_dense_col_index = matrix_struct->dense_col_number - 1;
            }
            else
            {
                new_item->max_dense_col_index = i + len - 1;
            }

            // 行索引范围就按照最大的来
            new_item->min_dense_row_index = 0;
            new_item->max_dense_row_index = matrix_struct->dense_row_number - 1;

            if (matrix_struct->val_data_type == DOUBLE)
            {
                // 搜出对应列块的非零元
                vector<unsigned long> output_row_arr;
                vector<unsigned long> output_col_arr;
                vector<double> output_double_arr;

                // 在一定范围内找到对应列的非零元
                find_all_col_double_element(matrix_struct, 0, matrix_struct->nnz - 1, new_item->min_dense_col_index,
                                            new_item->max_dense_col_index, &output_row_arr, &output_col_arr, &output_double_arr);

                // 如果不存在，就直接跳过
                if (output_row_arr.size() == 0)
                {
                    cout << "find empy col group:" << cur_sub_block_index << endl;
                    delete new_item;
                    continue;
                }

                assert(output_row_arr.size() == output_col_arr.size() && output_double_arr.size() == output_row_arr.size());
                // 起始位置是
                new_item->begin_coo_index = cur_coo_index;
                new_item->end_coo_index = new_item->begin_coo_index + output_row_arr.size() - 1;
                cur_coo_index = cur_coo_index + output_row_arr.size();

                // 坐标
                new_item->block_coordinate.push_back(cur_sub_block_index);
                cur_sub_block_index++;

                // 将找到的非零元拷贝到数组中
                new_row_index.insert(new_row_index.end(), output_row_arr.begin(), output_row_arr.end());
                new_col_index.insert(new_col_index.end(), output_col_arr.begin(), output_col_arr.end());
                new_double_var.insert(new_double_var.end(), output_double_arr.begin(), output_double_arr.end());

                // 没有压缩之前还没有指针
                new_item->compressed_block_ptr = NULL;

                // 子块的元数据加入到表中
                matrix_struct->block_coor_table.item_arr.push_back(new_item);
            }
            else
            {
                // 搜出对应列块的非零元
                vector<unsigned long> output_row_arr;
                vector<unsigned long> output_col_arr;
                vector<float> output_float_arr;

                // 在一定范围内找到对应列的非零元
                find_all_col_float_element(matrix_struct, 0, matrix_struct->nnz - 1, new_item->min_dense_col_index,
                                           new_item->max_dense_col_index, &output_row_arr, &output_col_arr, &output_float_arr);

                // 如果不存在，就直接跳过
                if (output_row_arr.size() == 0)
                {
                    cout << "find empy col group:" << cur_sub_block_index << endl;
                    delete new_item;
                    continue;
                }

                assert(output_row_arr.size() == output_col_arr.size() && output_float_arr.size() == output_row_arr.size());
                // 起始位置是
                new_item->begin_coo_index = cur_coo_index;
                new_item->end_coo_index = new_item->begin_coo_index + output_row_arr.size() - 1;
                cur_coo_index = cur_coo_index + output_row_arr.size();

                // 坐标
                new_item->block_coordinate.push_back(cur_sub_block_index);
                cur_sub_block_index++;

                // 将找到的非零元拷贝到数组中
                new_row_index.insert(new_row_index.end(), output_row_arr.begin(), output_row_arr.end());
                new_col_index.insert(new_col_index.end(), output_col_arr.begin(), output_col_arr.end());
                new_float_var.insert(new_float_var.end(), output_float_arr.begin(), output_float_arr.end());

                // 没有压缩之前还没有指针
                new_item->compressed_block_ptr = NULL;

                // 子块的元数据加入到表中
                matrix_struct->block_coor_table.item_arr.push_back(new_item);
            }
        }

        // 新的coo数据覆盖老的
        assert(new_row_index.size() == matrix_struct->nnz && new_col_index.size() == matrix_struct->nnz);

        memcpy(matrix_struct->coo_row_index_cache, &new_row_index[0], sizeof(unsigned long) * new_row_index.size());
        memcpy(matrix_struct->coo_col_index_cache, &new_col_index[0], sizeof(unsigned long) * new_col_index.size());

        // 单精度和双精度的不同
        if (matrix_struct->val_data_type == DOUBLE)
        {
            memcpy(matrix_struct->coo_value_cache, &new_double_var[0], sizeof(double) * new_double_var.size());
        }
        else
        {
            memcpy(matrix_struct->coo_value_cache, &new_float_var[0], sizeof(float) * new_float_var.size());
        }

        matrix_struct->is_blocked = true;
    }
    else
    {
        // 按照len分块
        int i;

        int cur_sub_block_index = 0;

        // 对某一个子块进一步分块，实际的最大最小索引和逻辑的最大最小索引，考虑到空行，并且考虑到CSR压缩处理不了头部和尾部的空行
        unsigned long begin_row_index = sub_block->min_dense_row_index;
        unsigned long end_row_index = sub_block->max_dense_row_index;
        unsigned long begin_col_index = sub_block->min_dense_col_index;
        unsigned long end_col_index = sub_block->max_dense_col_index;

        unsigned long begin_coo_index = sub_block->begin_coo_index;
        unsigned long end_coo_index = sub_block->end_coo_index;

        // 在coo中开始的偏移量
        unsigned long cur_coo_index = begin_coo_index;

        // 用一个数组存储新的coo数据
        vector<unsigned long> new_row_index;
        vector<unsigned long> new_col_index;
        vector<double> new_double_var;
        vector<float> new_float_var;

        vector<int> basic_coor(sub_block->block_coordinate);

        // 在表格中搜出这个元素、删除这个元素
        auto iter = std::remove(matrix_struct->block_coor_table.item_arr.begin(),
                                matrix_struct->block_coor_table.item_arr.end(), sub_block);
        matrix_struct->block_coor_table.item_arr.erase(iter, matrix_struct->block_coor_table.item_arr.end());

        // 将当前子块一点一点分列条带
        for (i = begin_col_index; i <= end_col_index; i = i + len)
        {
            dense_block_table_item_t *new_item = new dense_block_table_item_t();
            new_item->is_col_blocked = true;

            // 初始化范围
            new_item->min_dense_col_index = i;

            // 不超过矩阵上界的列索引范围，
            if ((i + len - 1) >= end_col_index)
            {
                new_item->max_dense_col_index = end_col_index;
            }
            else
            {
                new_item->max_dense_col_index = i + len - 1;
            }

            // 行索引范围就按照最大的来
            new_item->min_dense_row_index = begin_row_index;
            new_item->max_dense_row_index = end_row_index;

            if (matrix_struct->val_data_type == DOUBLE)
            {
                // 搜出对应列块的非零元
                vector<unsigned long> output_row_arr;
                vector<unsigned long> output_col_arr;
                vector<double> output_double_arr;

                // 在一定范围内找到对应列的非零元，
                find_all_col_double_element(matrix_struct, begin_coo_index, end_coo_index, new_item->min_dense_col_index,
                                            new_item->max_dense_col_index, &output_row_arr, &output_col_arr, &output_double_arr);

                // 如果不存在，就直接跳过
                if (output_row_arr.size() == 0)
                {
                    cout << "find empty col group:" << cur_sub_block_index << endl;
                    delete new_item;
                    continue;
                }

                assert(output_row_arr.size() == output_col_arr.size() && output_double_arr.size() == output_row_arr.size());
                // 起始位置是
                new_item->begin_coo_index = cur_coo_index;
                new_item->end_coo_index = new_item->begin_coo_index + output_row_arr.size() - 1;
                cur_coo_index = cur_coo_index + output_row_arr.size();

                // 坐标
                new_item->block_coordinate.insert(new_item->block_coordinate.end(), basic_coor.begin(), basic_coor.end());
                new_item->block_coordinate.push_back(cur_sub_block_index);
                cur_sub_block_index++;

                // 将找到的非零元拷贝到数组中
                new_row_index.insert(new_row_index.end(), output_row_arr.begin(), output_row_arr.end());
                new_col_index.insert(new_col_index.end(), output_col_arr.begin(), output_col_arr.end());
                new_double_var.insert(new_double_var.end(), output_double_arr.begin(), output_double_arr.end());

                // 没有压缩之前还没有指针
                new_item->compressed_block_ptr = NULL;

                // 子块的元数据加入到表中
                matrix_struct->block_coor_table.item_arr.push_back(new_item);
            }
            else
            {
                // 搜出对应列块的非零元
                vector<unsigned long> output_row_arr;
                vector<unsigned long> output_col_arr;
                vector<float> output_float_arr;

                // 在一定范围内找到对应列的非零元，
                find_all_col_float_element(matrix_struct, begin_coo_index, end_coo_index, new_item->min_dense_col_index,
                                           new_item->max_dense_col_index, &output_row_arr, &output_col_arr, &output_float_arr);

                // 如果不存在，就直接跳过
                if (output_row_arr.size() == 0)
                {
                    cout << "find empty col group:" << cur_sub_block_index << endl;
                    delete new_item;
                    continue;
                }

                assert(output_row_arr.size() == output_col_arr.size() && output_float_arr.size() == output_row_arr.size());
                // 起始位置是
                new_item->begin_coo_index = cur_coo_index;
                new_item->end_coo_index = new_item->begin_coo_index + output_row_arr.size() - 1;
                cur_coo_index = cur_coo_index + output_row_arr.size();

                // 坐标
                new_item->block_coordinate.insert(new_item->block_coordinate.end(), basic_coor.begin(), basic_coor.end());
                new_item->block_coordinate.push_back(cur_sub_block_index);
                cur_sub_block_index++;

                // 将找到的非零元拷贝到数组中
                new_row_index.insert(new_row_index.end(), output_row_arr.begin(), output_row_arr.end());
                new_col_index.insert(new_col_index.end(), output_col_arr.begin(), output_col_arr.end());
                new_float_var.insert(new_float_var.end(), output_float_arr.begin(), output_float_arr.end());

                // 没有压缩之前还没有指针
                new_item->compressed_block_ptr = NULL;

                // 子块的元数据加入到表中
                matrix_struct->block_coor_table.item_arr.push_back(new_item);
            }
        }
        // 新的coo数据覆盖老的
        assert(new_row_index.size() == (sub_block->end_coo_index - sub_block->begin_coo_index + 1) &&
               new_col_index.size() == (sub_block->end_coo_index - sub_block->begin_coo_index + 1));

        // 拷贝数据，注意拷贝的起始位置
        memcpy(&(((unsigned long *)(matrix_struct->coo_row_index_cache))[sub_block->begin_coo_index]), &new_row_index[0], sizeof(unsigned long) * new_row_index.size());
        memcpy(&(((unsigned long *)(matrix_struct->coo_col_index_cache))[sub_block->begin_coo_index]), &new_col_index[0], sizeof(unsigned long) * new_col_index.size());

        // 单精度和双精度的不同
        if (matrix_struct->val_data_type == DOUBLE)
        {
            memcpy(matrix_struct->coo_value_cache, &new_double_var[0], sizeof(double) * new_double_var.size());
        }
        else
        {
            memcpy(matrix_struct->coo_value_cache, &new_float_var[0], sizeof(float) * new_float_var.size());
        }
        delete sub_block;
    }
}

// 压缩，除了最终保留valarr之外，所有索引变成块内索引，当前还是COO的索引
void compress_dense_view(sparse_struct_t *matrix_struct)
{
    assert(matrix_struct != NULL);
    assert(matrix_struct->is_compressed == false);
    assert(matrix_struct->val_data_type == FLOAT || matrix_struct->val_data_type == DOUBLE);

    // 如果matrix还没有被分块过，那就整个数组成为一块
    if (matrix_struct->is_blocked == false && matrix_struct->block_coor_table.item_arr.size() == 0)
    {
        // 插入一个数据
        dense_block_table_item_t *item = new dense_block_table_item_t();

        item->block_coordinate.push_back(0);

        item->min_dense_row_index = 0;
        item->max_dense_row_index = matrix_struct->dense_row_number - 1;
        item->min_dense_col_index = 0;
        item->max_dense_col_index = matrix_struct->dense_col_number - 1;

        item->begin_coo_index = 0;
        item->end_coo_index = matrix_struct->nnz - 1;

        item->compressed_block_ptr = NULL;

        item->is_sorted = false;

        // 插入数据
        matrix_struct->block_coor_table.item_arr.push_back(item);
    }

    // 遍历所有的块为其产生一个压缩之后的结构
    cout << "scan all block" << endl;

    int i;
    for (i = 0; i < matrix_struct->block_coor_table.item_arr.size(); i++)
    {
        compressed_block_t *block_ptr = new compressed_block_t();
        block_ptr->share_row_with_other_block = matrix_struct->block_coor_table.item_arr[i]->is_col_blocked;

        // 一共两层索引，一个行索引，一个列索引。行索引和列索引是自己创建的
        index_of_compress_block_t *row_index = new index_of_compress_block_t();

        row_index->type_of_index = ROW_INDEX;
        row_index->level_of_this_index = OTHERS;
        row_index->index_compressed_type = COO;
        row_index->length = matrix_struct->block_coor_table.item_arr[i]->end_coo_index - matrix_struct->block_coor_table.item_arr[i]->begin_coo_index + 1;
        // 索引范围
        row_index->min_row_index = matrix_struct->block_coor_table.item_arr[i]->min_dense_row_index;
        row_index->max_row_index = matrix_struct->block_coor_table.item_arr[i]->max_dense_row_index;
        row_index->min_col_index = matrix_struct->block_coor_table.item_arr[i]->min_dense_col_index;
        row_index->max_col_index = matrix_struct->block_coor_table.item_arr[i]->max_dense_col_index;
        // 根据索引的范围确定块内索引所需要的数据类型

        row_index->index_data_type = find_most_suitable_data_type(row_index->max_row_index - row_index->min_row_index + 1);

        // 申请对应数据类型的索引
        row_index->index_arr = malloc_arr(row_index->length, row_index->index_data_type);

        // 数据拷贝，从coo index的起始和结束位置开始拷贝，并且根据地质的范围
        copy_unsigned_long_index_to_others(&(matrix_struct->coo_row_index_cache[matrix_struct->block_coor_table.item_arr[i]->begin_coo_index]),
                                           row_index->index_arr, row_index->index_data_type, row_index->length, row_index->min_row_index);

        // 处理一个块的列索引
        index_of_compress_block_t *col_index = new index_of_compress_block_t();
        col_index->type_of_index = COL_INDEX;
        col_index->level_of_this_index = OTHERS;
        col_index->index_compressed_type = COO;
        col_index->length = matrix_struct->block_coor_table.item_arr[i]->end_coo_index - matrix_struct->block_coor_table.item_arr[i]->begin_coo_index + 1;

        // 索引范围，主要是为了应对相对索引所带来的偏移量的记录
        col_index->min_row_index = matrix_struct->block_coor_table.item_arr[i]->min_dense_row_index;
        col_index->max_row_index = matrix_struct->block_coor_table.item_arr[i]->max_dense_row_index;
        col_index->min_col_index = matrix_struct->block_coor_table.item_arr[i]->min_dense_col_index;
        col_index->max_col_index = matrix_struct->block_coor_table.item_arr[i]->max_dense_col_index;

        col_index->index_data_type = find_most_suitable_data_type(col_index->max_col_index - col_index->min_col_index + 1);
        col_index->index_arr = malloc_arr(col_index->length, col_index->index_data_type);

        // 数据拷贝到col index中
        copy_unsigned_long_index_to_others(&(matrix_struct->coo_col_index_cache[matrix_struct->block_coor_table.item_arr[i]->begin_coo_index]),
                                           col_index->index_arr, col_index->index_data_type, col_index->length, col_index->min_col_index);

        // 在list中将内容写入对应索引
        block_ptr->read_index.push_back(row_index);
        block_ptr->read_index.push_back(col_index);

        // 与行索引和列索引不同，值数组是从稠密视图直接引用拷贝的
        // 包含的值数组的数量
        block_ptr->size = col_index->length;
        block_ptr->val_data_type = matrix_struct->val_data_type;

        // 根据值数组的数据类型来初始化当前块在值数组中的指针
        // if (block_ptr->val_data_type == DOUBLE)
        // {
        //     double *double_val_arr = (double *)(matrix_struct->coo_value_cache);
        //     block_ptr->val_arr = (void *)(&(double_val_arr[matrix_struct->block_coor_table.item_arr[i]->begin_coo_index]));
        // }
        // else if (block_ptr->val_data_type == FLOAT)
        // {
        //     float *float_val_arr = (float *)(matrix_struct->coo_value_cache);
        //     block_ptr->val_arr = (void *)(&(float_val_arr[matrix_struct->block_coor_table.item_arr[i]->begin_coo_index]));
        // }
        // else
        // {
        //     cout << "error" << endl;
        //     exit(-1);
        // }

        // 将压缩子图的值数组改成值拷贝，方便之后执行row_padding。
        if (block_ptr->val_data_type == DOUBLE)
        {
            block_ptr->val_arr = malloc_arr(block_ptr->size, block_ptr->val_data_type);
            // 执行拷贝
            memcpy(block_ptr->val_arr, matrix_struct->coo_value_cache, block_ptr->size * sizeof(double));
        }
        else if (block_ptr->val_data_type == FLOAT)
        {
            block_ptr->val_arr = malloc_arr(block_ptr->size, block_ptr->val_data_type);
            memcpy(block_ptr->val_arr, matrix_struct->coo_value_cache, block_ptr->size * sizeof(float));
        }
        else
        {
            cout << "error" << endl;
            assert(false);
        }

        // 将压缩视图存储到对应块中
        matrix_struct->block_coor_table.item_arr[i]->compressed_block_ptr = block_ptr;
    }

    matrix_struct->is_compressed = true;

    cout << "end compress" << endl;
}

data_type find_most_suitable_data_type(unsigned long max_index_number)
{

    // CHAR,
    // UNSIGNED_CHAR,
    // SHORT,
    // UNSIGNED_SHORT,
    // INT,
    // UNSIGNED_INT,
    // LONG,
    // UNSIGNED_LONG,
    // LONG_LONG,
    // UNSIGNED_LONG_LONG,
    // FLOAT,
    // DOUBLE

    // 主要是选择unsigned char、UNSIGNED_SHORT、UNSIGNED_INT、UNSIGNED_LONG
    if (max_index_number <= 255)
    {
        return UNSIGNED_CHAR;
    }
    else if (max_index_number <= 65535)
    {
        return UNSIGNED_SHORT;
    }
    else if (max_index_number <= 4294967295)
    {
        return UNSIGNED_INT;
    }

    return UNSIGNED_LONG;
}

data_type find_most_suitable_data_type(long max_number, long min_number)
{
    assert(max_number >= min_number);

    if (min_number >= 0)
    {
        // 返回无符号的数据类型
        return find_most_suitable_data_type(max_number);
    }
    else
    {
        if (max_number <= CHAR_MAX && min_number >= CHAR_MIN)
        {
            return CHAR;
        }

        if (max_number <= SHRT_MAX && min_number >= SHRT_MIN)
        {
            return SHORT;
        }

        if (max_number <= INT_MAX && min_number >= INT_MIN)
        {
            return INT;
        }

        return LONG;
    }

    assert(false);
    return LONG;
}

void sep_thread_level_row_csr(compressed_block_t *compressed_block)
{
    // 根据compressed_block
    assert(compressed_block != NULL);

    if (compressed_block->read_index.size() == 2)
    {
        cout << "have not blocked, error" << endl;
        assert(false);
    }
    else if (compressed_block->read_index.size() > 2)
    {
    }
    else
    {
        cout << "error, maybe is not compressed" << endl;
        assert(false);
    }
}

// 线程层次的行分块只能一行
void sep_thread_level_row_csr(compressed_block_t *compressed_block, vector<unsigned int> block_size_arr)
{
    cout << "not support multi row sep" << endl;
    assert(false);

    // 根据compressed_block
    assert(compressed_block != NULL);
    assert(block_size_arr.size() != 0);

    // 开始生成一个索引插入到头部和尾部之间
    // 先生成一个新的索引
    // 还没有被分块，那就处理处理第一个块
    if (compressed_block->read_index.size() == 2)
    {

        // 开没有被分块，一开始只有两个行和列索引
        cout << "not blocked only two index" << endl;

        // 检查所有划分的粒度合起来有没有足够的大小
        // 看看是不是列索引
        assert(compressed_block->read_index[1]->index_compressed_type == COO);
        assert(compressed_block->read_index[1]->type_of_index == COL_INDEX);
        unsigned long max_row_index = compressed_block->read_index[1]->max_row_index;
        unsigned long min_row_index = compressed_block->read_index[1]->min_row_index;

        // 行的数量
        unsigned long row_number = max_row_index - min_row_index + 1;

        unsigned long sum = 0;
        int i;
        for (i = 0; i < block_size_arr.size(); i++)
        {
            sum = sum + block_size_arr[i];
        }

        assert(sum == row_number);

        // 创造一个新的索引
        index_of_compress_block_t *new_index = new index_of_compress_block_t();

        // 元数据更新，这些元数据在合并之后要更新
        new_index->index_compressed_type = CSR;
        // 分块数量+1
        new_index->length = block_size_arr.size();
        // 根据非零元数量决定数据类型
        new_index->index_data_type = find_most_suitable_data_type(compressed_block->read_index[1]->length);

        // 根据我们
    }
    else if (compressed_block->read_index.size() > 2)
    {
        // 如果上一层有索引，那么就需要改变上一层索引
    }
    else
    {
        cout << "error, maybe is not compressed" << endl;
        assert(false);
    }
}

void init_op_manager(operator_manager_t *op_manager, sparse_struct_t *matrix)
{
    assert(op_manager != NULL);
    assert(matrix != NULL);

    // op_manager->is_block_level = false;
    // op_manager->is_warp_level = false;
    // op_manager->is_thread_level = false;

    op_manager->matrix = matrix;
}

operator_manager_t *init_op_manager(sparse_struct_t *matrix)
{
    operator_manager_t *return_op_manager = new operator_manager_t();
    init_op_manager(return_op_manager, matrix);
    return return_op_manager;
}

// 按照线程块粒度分块
void sep_tblock_level_row_csr(compressed_block_t *compressed_block, vector<unsigned int> block_size_arr)
{
    assert(compressed_block != NULL);
    assert(block_size_arr.size() != 0);
    assert(compressed_block->read_index.size() == 2);

    // 所包含的行的数量
    unsigned long row_num = compressed_block->read_index[1]->max_row_index - compressed_block->read_index[1]->min_row_index + 1;

    // 检查条带的大小，大小要正好等于这个子块所包含的行号数量
    // 并且找出块大小的最大值
    unsigned long i;
    unsigned long sum = 0;
    unsigned long max = 0;
    for (i = 0; i < block_size_arr.size(); i++)
    {
        // 行条带高度不可能是0
        assert(block_size_arr[i] != 0);
        sum = sum + block_size_arr[i];
        if (max < block_size_arr[i])
        {
            max = block_size_arr[i];
        }
    }

    assert(sum == row_num);

    // 申请一个新的索引
    index_of_compress_block_t *new_index = new index_of_compress_block_t();
    // 索引的元数据
    new_index->level_of_this_index = TBLOCK_LEVEL;
    // 按照CSR分块
    new_index->index_compressed_type = CSR;
    // 块数量
    new_index->block_num = block_size_arr.size();
    // 索引数组的长度
    new_index->length = new_index->block_num + 1;
    // 索引的归属
    new_index->type_of_index = BLOCK_INDEX;
    // 所有被分出的所有子块的索引范围
    new_index->max_row_index = compressed_block->read_index[1]->max_row_index;
    new_index->min_row_index = compressed_block->read_index[1]->min_row_index;
    new_index->max_col_index = compressed_block->read_index[1]->max_col_index;
    new_index->min_col_index = compressed_block->read_index[1]->min_col_index;

    // 更新块大小的数组
    // 选择合适的数据范围
    new_index->data_type_of_row_number_of_block_arr = find_most_suitable_data_type(max);
    // 申请对应大小的数组
    new_index->row_number_of_block_arr = malloc_arr(block_size_arr.size(), new_index->data_type_of_row_number_of_block_arr);
    // 将每个块的大小拷贝到自己
    copy_unsigned_int_arr_to_others(&block_size_arr[0], new_index->row_number_of_block_arr, new_index->data_type_of_row_number_of_block_arr, new_index->block_num);

    // 遍历块大小的矩阵
    // index_of_the_first_row_arr
    // coo_begin_index_arr

    // index_of_the_first_row_arr的数据类型和子块的行数量有关，所有子块的行号都是从0开始的
    // 都使用CSR的方式存储，每个数组的大小都是block_number + 1
    new_index->data_type_of_index_of_the_first_row_arr = find_most_suitable_data_type(sum + 1);
    new_index->index_of_the_first_row_arr = malloc_arr(new_index->length, new_index->data_type_of_index_of_the_first_row_arr);

    // coo_begin_index，线程块粒度的块对应的第一个val数组和col数组的偏移量
    // 数据类型的大小和这个压缩快的非零元数量相关
    new_index->data_type_of_coo_begin_index_arr = find_most_suitable_data_type(compressed_block->read_index[1]->length);
    new_index->coo_begin_index_arr = malloc_arr(new_index->length, new_index->data_type_of_coo_begin_index_arr);

    // 遍历块大小的矩阵，记录每一个矩阵的第一个块
    // 最近的起始行号
    unsigned long cur_first_row_index = 0;
    write_to_array_with_data_type(new_index->index_of_the_first_row_arr, new_index->data_type_of_index_of_the_first_row_arr, 0, cur_first_row_index);

    // 记录每一块的起始行号，最后一位是整个块中行的数量
    for (i = 0; i < new_index->block_num - 1; i++)
    {
        cur_first_row_index = cur_first_row_index + block_size_arr[i];
        write_to_array_with_data_type(new_index->index_of_the_first_row_arr, new_index->data_type_of_index_of_the_first_row_arr, i + 1, cur_first_row_index);
    }

    // 写每一块的coo的起始位置，用first_row找coo索引的起始位置
    // 遍历coo的row索引，然后每一个块第一行的起始位置，当然要处理空行
    // 如果是空条带，那就将对应new_index->row_number_of_block_arr设置为0，以待之后的整理
    unsigned long nnz_of_anc_block = compressed_block->read_index[0]->length;
    assert(compressed_block->read_index[0]->index_compressed_type == COO);
    assert(compressed_block->read_index[0]->type_of_index == ROW_INDEX);
    data_type coo_row_index_type = compressed_block->read_index[0]->index_data_type;
    assert(read_from_array_with_data_type(compressed_block->read_index[0]->index_arr,
                                          compressed_block->read_index[0]->index_data_type, 0) >= 0);
    // 遍历所有行索引，找出每一行的起始位置
    // 用一个索引遍历index_of_the_first_row_arr以及修改
    // row_number_of_block_arr
    // 记录空块的数量
    unsigned long empty_block_count = 0;
    // 记录上一个非零元所属的块的块号
    long block_index_of_last_block = -1;

    // 遍历coo所有的坐标，找出每个子块的coo起始位置，最后一位是整个大块中nnz数量
    for (i = 0; i < nnz_of_anc_block; i++)
    {
        unsigned long cur_nnz_row_index = read_from_array_with_data_type(compressed_block->read_index[0]->index_arr, compressed_block->read_index[0]->index_data_type, i);

        // 遍历block，查看一个行索引属于哪个块的区间
        long j;
        for (j = block_index_of_last_block + 1; j < new_index->block_num; j++)
        {
            unsigned long cur_block_first_row_index = read_from_array_with_data_type(new_index->index_of_the_first_row_arr, new_index->data_type_of_index_of_the_first_row_arr, j);
            unsigned long cur_block_row_number = read_from_array_with_data_type(new_index->row_number_of_block_arr, new_index->data_type_of_row_number_of_block_arr, j);

            // 查看是不是在当前遍历的块的范围内
            if ((cur_nnz_row_index >= cur_block_first_row_index) && (cur_nnz_row_index <= (cur_block_first_row_index + cur_block_row_number - 1)))
            {
                // cout << "搜到了" << j << endl;
                // 在范围内，并且因为行索引增序排列，搜到肯定是最小的
                write_to_array_with_data_type(new_index->coo_begin_index_arr, new_index->data_type_of_coo_begin_index_arr, j, i);

                // 和上一块之间的所有块都是空块，当一个块全是空行的时候，这个块会被登记下来，作为一个空块，最终不记录下来
                long k;
                // cout << block_index_of_last_block << "," << j << endl;
                for (k = block_index_of_last_block + 1; k <= j - 1; k++)
                {
                    // 唯一产生空块的位置
                    // cout << k << endl;
                    write_to_array_with_data_type(new_index->row_number_of_block_arr, new_index->data_type_of_row_number_of_block_arr, k, 0);
                    empty_block_count++;
                }

                // 最后记录一下上一个被记录的块索引
                block_index_of_last_block = j;
                // cout << "处理完了" << endl;
                break;
            }

            // 如果当前块的上界已经大于当前非零元的行号了，那就没有遍历下去的必要了，直接跳出
            if (cur_nnz_row_index < cur_block_first_row_index)
            {
                break;
            }
        }

        // // 如果遍历到的row_index和当前块的第一个行号相等，那就记录i
        // if (cur_nnz_row_index == cur_block_first_row_index)
        // {
        //     assert(index_of_block < new_index->block_num);
        //     // 在coo中的起始位置
        //     write_to_array_with_data_type(new_index->coo_begin_index_arr, new_index->data_type_of_coo_begin_index_arr, index_of_block, i);
        //     index_of_block++;
        // }
        // else if (cur_nnz_row_index > cur_block_first_row_index)
        // {
        //     // 这里说明存在空行，因为nnz_row_index一般是追着inner_block_row_index跑的
        //     // 不断遍历块，直到块的首行索引值不多余已经遍历到的行首元素为止
        //     while (cur_block_first_row_index < cur_nnz_row_index)
        //     {
        //         // 当前行条带是空条带，将row_number_of_block变为0
        //         write_to_array_with_data_type(new_index->row_number_of_block_arr, new_index->data_type_of_row_number_of_block_arr, index_of_block, 0);
        //         empty_block_count++;
        //         // 遍历下一个块
        //         index_of_block++;
        //         assert(index_of_block < new_index->block_num);
        //         cur_block_first_row_index = read_from_array_with_data_type(new_index->index_of_the_first_row_arr, new_index->data_type_of_index_of_the_first_row_arr, index_of_block);
        //     }
        //     // 如果二者相等，那就根据当前遍历的位置记录coo偏移
        //     if (cur_block_first_row_index == cur_nnz_row_index)
        //     {
        //         assert(index_of_block < new_index->block_num);
        //         // 在coo中的起始位置
        //         write_to_array_with_data_type(new_index->coo_begin_index_arr, new_index->data_type_of_coo_begin_index_arr, index_of_block, i);
        //         index_of_block++;
        //     }
        // }
    }

    // 如果block_index_of_last_block是小于new_index->block_num - 1的，那么剩下的都是空块
    unsigned long index_of_block = block_index_of_last_block + 1;
    while (index_of_block < new_index->block_num)
    {
        // 剩下的都是空行，一直补空行
        write_to_array_with_data_type(new_index->row_number_of_block_arr, new_index->data_type_of_row_number_of_block_arr, index_of_block, 0);
        empty_block_count++;
        index_of_block++;
    }

    if (empty_block_count > 0)
    {
        // 根据之前的记录去除空条带
        // 去除空条带，遍历所有条带的行数量，将行数量是0的块去除
        // 新的块数量
        unsigned long new_block_number = new_index->block_num - empty_block_count;

        // 按照新的块的数量申请新的index_of_the_first_row_arr、
        // row_number_of_block_arr、coo_begin_index_arr三个数组
        // 数据类型应该是不会变的，只需要变一下数组的大小即可
        // 创建三个新的数组
        
        void *new_index_of_the_first_row_arr = malloc_arr(new_block_number, new_index->data_type_of_index_of_the_first_row_arr);
        void *new_row_number_of_block_arr = malloc_arr(new_block_number, new_index->data_type_of_row_number_of_block_arr);
        void *new_coo_begin_index_arr = malloc_arr(new_block_number + 1, new_index->data_type_of_coo_begin_index_arr);

        // 这里做一个拷贝，如果block中行的数量是0，那就不将老的数组拷贝到新的数组
        unsigned long new_block_index = 0;
        for (i = 0; i < new_index->block_num; i++)
        {
            unsigned long row_num_of_cur_block = read_from_array_with_data_type(new_index->row_number_of_block_arr, new_index->data_type_of_row_number_of_block_arr, i);
            // 如果这里不等于0，那就放到新的数组中
            if (row_num_of_cur_block != 0)
            {
                // 读取三个数组的数据
                unsigned long old_cur_index_of_the_first_row = read_from_array_with_data_type(new_index->index_of_the_first_row_arr, new_index->data_type_of_index_of_the_first_row_arr, i);
                unsigned long old_cur_row_number_of_block = read_from_array_with_data_type(new_index->row_number_of_block_arr, new_index->data_type_of_row_number_of_block_arr, i);
                unsigned long old_cur_coo_begin_index = read_from_array_with_data_type(new_index->coo_begin_index_arr, new_index->data_type_of_coo_begin_index_arr, i);
                // 将三个数据一个个写到新的数组中
                write_to_array_with_data_type(new_index_of_the_first_row_arr, new_index->data_type_of_index_of_the_first_row_arr, new_block_index, old_cur_index_of_the_first_row);
                write_to_array_with_data_type(new_row_number_of_block_arr, new_index->data_type_of_row_number_of_block_arr, new_block_index, old_cur_row_number_of_block);
                write_to_array_with_data_type(new_coo_begin_index_arr, new_index->data_type_of_coo_begin_index_arr, new_block_index, old_cur_coo_begin_index);
                new_block_index++;
            }
        }

        // coo坐标最后一个是，这个块中非零元的数量
        write_to_array_with_data_type(new_coo_begin_index_arr, new_index->data_type_of_coo_begin_index_arr, new_block_number, nnz_of_anc_block);

        // print_arr_to_file_with_data_type(new_coo_begin_index_arr, new_index->data_type_of_coo_begin_index_arr, new_block_number + 1, "/home/duzhen/spmv_builder/data_source/test5-2.log");

        assert(new_block_index == new_block_number);
        // 析构三个数组
        delete_arr_with_data_type(new_index->index_of_the_first_row_arr, new_index->data_type_of_index_of_the_first_row_arr);
        new_index->index_of_the_first_row_arr = NULL;
        delete_arr_with_data_type(new_index->row_number_of_block_arr, new_index->data_type_of_row_number_of_block_arr);
        new_index->row_number_of_block_arr = NULL;
        delete_arr_with_data_type(new_index->coo_begin_index_arr, new_index->data_type_of_coo_begin_index_arr);
        new_index->coo_begin_index_arr = NULL;

        // 重新对三个数组赋值
        new_index->index_of_the_first_row_arr = new_index_of_the_first_row_arr;
        new_index->row_number_of_block_arr = new_row_number_of_block_arr;
        new_index->coo_begin_index_arr = new_coo_begin_index_arr;

        // 修改块的大小
        new_index->block_num = new_block_number;
        new_index->length = new_block_number + 1;
    }

    // 这里写coo
    write_to_array_with_data_type(new_index->coo_begin_index_arr, new_index->data_type_of_coo_begin_index_arr, new_index->block_num, nnz_of_anc_block);

    // 这个矩阵取决于下一个块会产生多少个中间结果，需要下一个快分块的时候才能决定./
    new_index->child_tmp_row_csr_index_arr = NULL;
    // 和下一个层次的CSR本体现在也是没法确定
    new_index->index_arr = NULL;

    // 将新的index放到
    compressed_block->read_index.push_back(new_index);
}

// 执行默认的BLB切分，将整个压缩块划分为一个BLB
void default_sep_tblock_level_row_csr(compressed_block_t* compressed_block_ptr)
{
    assert(compressed_block_ptr != NULL);
    assert(compressed_block_ptr->read_index.size() == 2);

    unsigned long sub_compressed_matrix_row_num = compressed_block_ptr->read_index[0]->max_row_index - compressed_block_ptr->read_index[0]->min_row_index + 1;
    
    vector<unsigned int> block_row_num;
    block_row_num.push_back(sub_compressed_matrix_row_num);

    sep_tblock_level_row_csr(compressed_block_ptr, block_row_num);
    
    assert(compressed_block_ptr->read_index.size() == 3);
}

void one_row_sep_tblock_level_row_csr(compressed_block_t* compressed_block_ptr)
{
    assert(compressed_block_ptr != NULL);
    assert(compressed_block_ptr->read_index.size() == 2);
    
    // 获取这个压缩子块的行号
    unsigned long row_num_of_sub_matrix = compressed_block_ptr->read_index[0]->max_row_index - compressed_block_ptr->read_index[0]->min_row_index + 1;
    // 行切分，一行一个块
    vector<unsigned int> block_row_num;

    for (unsigned long i = 0; i < row_num_of_sub_matrix; i++)
    {
        block_row_num.push_back(1);
    }

    sep_tblock_level_row_csr(compressed_block_ptr, block_row_num);

    assert(compressed_block_ptr->read_index.size() == 3);
}

// 按照线程块的粒度纵切分，并且可以在已经在行切分的基础上再纵切分一次，
// 纵切分是行对齐的，切分要在行交界处停止
// 如果要在已有的子块中再切，所以会改变已有索引，重点是在已有的索引中插入东西，这个时候两个数组都是有用的
// 如果非零元的数量不足以填满是没有关系的。
// 每个子块切分都有各自的策略，如果之前没有切分，那就只有一个切分策略
// 如果之前没有切分，那就生成一个新的索引
void sep_tblock_level_col_csr(compressed_block_t *compressed_block, vector<unsigned long> block_index_arr, vector<vector<unsigned int>> block_size_arr)
{
    // 两种情况，之前切分了，之前没有切分
    assert(compressed_block != NULL);
    // 两个数组的大小相等
    assert(block_index_arr.size() == block_size_arr.size() && block_index_arr.size() > 0);

    assert(compressed_block->read_index.size() >= 3);

    assert(compressed_block->read_index[2]->index_compressed_type == CSR && compressed_block->read_index[2]->type_of_index == BLOCK_INDEX &&
           compressed_block->read_index[2]->level_of_this_index == TBLOCK_LEVEL);

    assert(compressed_block->read_index[2]->block_num >= block_index_arr.size());

    index_of_compress_block_t *old_index = compressed_block->read_index[2];
    index_of_compress_block_t *global_row_index = compressed_block->read_index[0];
    index_of_compress_block_t *global_col_index = compressed_block->read_index[1];

    // 重新整理三个数组，index_of_the_first_row_arr，row_number_of_block_arr，coo_begin_index_arr
    // 先用最大数据类型的vector申请对应的边长数组，之后再换成定长的
    vector<unsigned long> new_index_of_the_first_row_vec;
    vector<unsigned long> new_row_number_of_block_vec;
    vector<unsigned long> new_coo_begin_index_vec;

    // 每个块的最大行数量
    unsigned long max_row_num_of_block = 1;
    // 遍历之前所有的块
    unsigned long block_index_need_to_sep = 0;
    unsigned long i;

    for (i = 0; i < old_index->block_num; i++)
    {
        // 查看当前块要不要被进一步分块，如果记录超了就不可能继续分块了
        if (block_index_need_to_sep < block_index_arr.size() && i == block_index_arr[block_index_need_to_sep])
        {
            // 有子块被列分块了
            compressed_block->share_row_with_other_block = true;

            vector<unsigned int> col_block_size_vec = block_size_arr[block_index_need_to_sep];
            assert(col_block_size_vec.size() > 0);
            assert(global_row_index->length > 0 && global_col_index->length > 0);
            // 这一块中的基本信息
            unsigned long first_row_index_of_this_block = read_from_array_with_data_type(old_index->index_of_the_first_row_arr, old_index->data_type_of_index_of_the_first_row_arr, i);
            unsigned long row_number_of_this_block = read_from_array_with_data_type(old_index->row_number_of_block_arr, old_index->data_type_of_row_number_of_block_arr, i);
            unsigned long coo_begin_index_of_this_block = read_from_array_with_data_type(old_index->coo_begin_index_arr, old_index->data_type_of_coo_begin_index_arr, i);
            unsigned long coo_begin_index_of_next_block = read_from_array_with_data_type(old_index->coo_begin_index_arr, old_index->data_type_of_coo_begin_index_arr, i + 1);
            // 遍历这个块的所有COO列索引和行索引
            // 设计一个变量负责检测换行
            unsigned long last_row_index = read_from_array_with_data_type(global_row_index->index_arr, global_row_index->index_data_type, coo_begin_index_of_this_block);

            unsigned long j;
            // 一行中经过的累计nz的做引
            unsigned long row_nz_index = 0;
            // 记录不同块的起始位置，也就是上一次分块的位置
            unsigned long col_block_begin_coo = coo_begin_index_of_this_block;
            // 不同列分块累计nz数量
            unsigned long acc_col_block_nz = 0;
            // 当前所在的列快
            unsigned long cur_col_block_index = 0;
            // 被遍历到的列快内的非零元数量
            unsigned long inner_cur_col_block_acc_number = 0;
            for (j = coo_begin_index_of_this_block; j < coo_begin_index_of_next_block; j++)
            {
                // 获取当前的行号和列号
                assert(j < global_row_index->length);
                assert(j < global_col_index->length);
                unsigned long cur_row_index = read_from_array_with_data_type(global_row_index->index_arr, global_row_index->index_data_type, j);
                unsigned long cur_col_index = read_from_array_with_data_type(global_col_index->index_arr, global_col_index->index_data_type, j);

                // 扫描到行末的时候分块，如果行末和列块末重合了，就没有必要再加入
                if (last_row_index < cur_row_index)
                {
                    // 普通的换行和带分块的换行
                    if (inner_cur_col_block_acc_number > 0)
                    {
                        // 带分块的换行
                        // 换行强制触发加入新块
                        new_index_of_the_first_row_vec.push_back(last_row_index);
                        new_row_number_of_block_vec.push_back(1);
                        new_coo_begin_index_vec.push_back(col_block_begin_coo);

                        // 当前位置是下一个块的起点
                        col_block_begin_coo = j;
                        // 这里换行
                        row_nz_index = 0;
                        acc_col_block_nz = 0;
                        cur_col_block_index = 0;
                        inner_cur_col_block_acc_number = 0;
                    }
                    else
                    {
                        // 在列分块边界的时候已经分过，然后不带分块的换行
                        // 这里换行
                        row_nz_index = 0;
                        acc_col_block_nz = 0;
                        cur_col_block_index = 0;
                        inner_cur_col_block_acc_number = 0;
                    }
                }

                // 扫到一列中最后一个位置和行末的时候分块
                // 列块的最后一个位置
                // assert(cur_col_block_index < col_block_size_vec.size());

                if (cur_col_block_index >= col_block_size_vec.size())
                {
                    cout << "error:col block is not enough, acc block index:" << i << ", coo index:" << j << ", cur col block index:" << cur_col_block_index << endl;
                    assert(false);
                }

                if (row_nz_index == acc_col_block_nz + col_block_size_vec[cur_col_block_index] - 1)
                {

                    if (cur_col_block_index >= col_block_size_vec.size())
                    {
                        // 分块的数量没法满足行非零元数量的要求
                        cout << "error:col block is not enough, acc block index:" << i << ", coo index:" << j << ", cur col block index:" << cur_col_block_index << endl;
                        assert(false);
                    }

                    // 加入一个新的块，块的起始位置是上一个块结束的位置，也就是acc_col_block_nz
                    assert(cur_row_index >= first_row_index_of_this_block && cur_row_index < (first_row_index_of_this_block + row_number_of_this_block));
                    // if(cur_row_index == 0 || cur_row_index == 1){
                    //     cout << cur_row_index << "," << 1 << "," << col_block_begin_coo << endl;
                    // }

                    new_index_of_the_first_row_vec.push_back(cur_row_index);
                    new_row_number_of_block_vec.push_back(1);
                    new_coo_begin_index_vec.push_back(col_block_begin_coo);

                    // 下一个块的coo起始位置，
                    col_block_begin_coo = j + 1;

                    // 当前行一个列块的起始位置
                    acc_col_block_nz = acc_col_block_nz + col_block_size_vec[cur_col_block_index];
                    cur_col_block_index++;
                    inner_cur_col_block_acc_number = 0;
                }
                else
                {
                    // 如果没到块末那就自增
                    inner_cur_col_block_acc_number++;
                }

                row_nz_index++;
                last_row_index = cur_row_index;
            }

            // 父块结束的位置也要强制分块
            // 查看父块结束的位置有没有分块
            assert(col_block_begin_coo <= coo_begin_index_of_next_block);
            // 这里代表在父块的边界处没有精力过分块，行分块又不cover，所以在父块边界上要做一个分块
            if (col_block_begin_coo != coo_begin_index_of_next_block)
            {
                // 父块结束的时候没有强制分子块，这里要补一个分块
                // 块的行索引是父块的最后一行
                // cout << "block edge block" << col_block_begin_coo << "," << coo_begin_index_of_next_block << endl;
                new_index_of_the_first_row_vec.push_back(first_row_index_of_this_block + row_number_of_this_block - 1);
                new_row_number_of_block_vec.push_back(1);
                // 列块的起始coo索引
                new_coo_begin_index_vec.push_back(col_block_begin_coo);
            }

            // 需要进一步分块的块索引
            block_index_need_to_sep++;
        }
        else
        {
            // 如果没有进一步分块就直接拷贝就好了
            new_index_of_the_first_row_vec.push_back(read_from_array_with_data_type(old_index->index_of_the_first_row_arr,
                                                                                    old_index->data_type_of_index_of_the_first_row_arr, i));

            // 当前块的行数量
            unsigned long cur_row_num_of_block = read_from_array_with_data_type(old_index->row_number_of_block_arr,
                                                                                old_index->data_type_of_row_number_of_block_arr, i);

            if (max_row_num_of_block < cur_row_num_of_block)
            {
                max_row_num_of_block = cur_row_num_of_block;
            }

            new_row_number_of_block_vec.push_back(cur_row_num_of_block);
            new_coo_begin_index_vec.push_back(read_from_array_with_data_type(old_index->coo_begin_index_arr, old_index->data_type_of_coo_begin_index_arr, i));
        }
    }

    // 三个数组的大小现在是相等的
    assert(new_index_of_the_first_row_vec.size() == new_row_number_of_block_vec.size() && new_row_number_of_block_vec.size() == new_coo_begin_index_vec.size());

    // 为coo数组最后加一个父块的nnz数量
    unsigned long end_of_coo_begin_arr = read_from_array_with_data_type(old_index->coo_begin_index_arr, old_index->data_type_of_coo_begin_index_arr, old_index->length - 1);
    assert(end_of_coo_begin_arr == global_row_index->length);
    new_coo_begin_index_vec.push_back(end_of_coo_begin_arr);

    // 修改old_index
    unsigned long new_block_num = new_index_of_the_first_row_vec.size();
    // 三个新的数组，其中每个块的行数量的数据类型需要重新定
    void *new_index_of_the_first_row_arr = malloc_arr(new_block_num, old_index->data_type_of_index_of_the_first_row_arr);
    // 新的row_num数据类型
    data_type data_type_of_new_row_num_arr = find_most_suitable_data_type(max_row_num_of_block);
    void *new_row_number_of_block_arr = malloc_arr(new_block_num, data_type_of_new_row_num_arr);
    // 新的coo begin，数组大一号
    void *new_coo_begin_index_arr = malloc_arr(new_block_num + 1, old_index->data_type_of_coo_begin_index_arr);

    // 拷贝三个数组
    copy_unsigned_long_arr_to_others(&(new_index_of_the_first_row_vec[0]), new_index_of_the_first_row_arr, old_index->data_type_of_index_of_the_first_row_arr, new_block_num);
    copy_unsigned_long_arr_to_others(&(new_row_number_of_block_vec[0]), new_row_number_of_block_arr, data_type_of_new_row_num_arr, new_block_num);
    assert(new_coo_begin_index_vec.size() == new_block_num + 1);
    copy_unsigned_long_arr_to_others(&(new_coo_begin_index_vec[0]), new_coo_begin_index_arr, old_index->data_type_of_coo_begin_index_arr, new_block_num + 1);

    // 修改原索引
    delete_arr_with_data_type(old_index->index_of_the_first_row_arr, old_index->data_type_of_index_of_the_first_row_arr);
    old_index->index_of_the_first_row_arr = NULL;
    delete_arr_with_data_type(old_index->row_number_of_block_arr, old_index->data_type_of_row_number_of_block_arr);
    old_index->row_number_of_block_arr = NULL;
    delete_arr_with_data_type(old_index->coo_begin_index_arr, old_index->data_type_of_coo_begin_index_arr);
    old_index->coo_begin_index_arr = NULL;

    old_index->index_of_the_first_row_arr = new_index_of_the_first_row_arr;
    old_index->row_number_of_block_arr = new_row_number_of_block_arr;
    old_index->coo_begin_index_arr = new_coo_begin_index_arr;

    // 修改数组大小
    old_index->block_num = new_block_num;
    old_index->length = new_block_num + 1;

    // 修改数据类型
    old_index->data_type_of_row_number_of_block_arr = data_type_of_new_row_num_arr;
}

// 从头开始纵切分
void sep_tblock_level_col_csr(compressed_block_t *compressed_block, vector<unsigned int> block_size_arr)
{
    // 对完整的矩阵纵切分，然后生成新的索引。
    assert(compressed_block->read_index.size() == 2);

    // 现有的两个索引都是coo的
    assert(compressed_block->read_index[0]->type_of_index == ROW_INDEX && compressed_block->read_index[1]->type_of_index == COL_INDEX);

    assert(compressed_block->read_index[0]->index_compressed_type == COO && compressed_block->read_index[1]->index_compressed_type == COO);

    assert(block_size_arr.size() > 0);

    // 行与列的密集视图
    index_of_compress_block_t *global_row_index = compressed_block->read_index[0];
    index_of_compress_block_t *global_col_index = compressed_block->read_index[1];

    // 重新整理三个数组，index_of_the_first_row_arr，row_number_of_block_arr，coo_begin_index_arr
    // 先用最大数据类型的vector申请对应的边长数组，之后再换成定长的
    vector<unsigned long> new_index_of_the_first_row_vec;
    vector<unsigned long> new_row_number_of_block_vec;
    vector<unsigned long> new_coo_begin_index_vec;

    assert(global_row_index->length == global_col_index->length && global_row_index->length > 0 && global_row_index->block_num == 0 && global_row_index->block_num == global_col_index->block_num);

    compressed_block->share_row_with_other_block = true;

    // 遍历所有的非零元
    unsigned long i;
    // 纵分块的索引
    unsigned long cur_col_block_index = 0;
    // 在一个纵分块内部遍历到的非零元数量
    unsigned long inner_cur_col_block_acc_number = 0;
    // 一行中经过的累计nz的做引
    unsigned long row_nz_index = 0;
    // 记录不同块的起始位置，也就是上一次分块的位置
    unsigned long col_block_begin_coo = 0;
    // 已经经过的列分块累计nz数量
    unsigned long acc_col_block_nz = 0;
    // 上一行
    unsigned long last_row_index = 0;
    for (i = 0; i < global_row_index->length; i++)
    {
        // 遍历所有的非零元
        unsigned long cur_row_index = read_from_array_with_data_type(global_row_index->index_arr, global_row_index->index_data_type, i);
        unsigned long cur_col_index = read_from_array_with_data_type(global_col_index->index_arr, global_col_index->index_data_type, i);

        assert(last_row_index <= cur_row_index);
        // 换行了
        if (last_row_index < cur_row_index)
        {
            // 带分块的换行
            if (inner_cur_col_block_acc_number > 0)
            {

                // 执行分块
                new_index_of_the_first_row_vec.push_back(last_row_index);
                new_row_number_of_block_vec.push_back(1);
                new_coo_begin_index_vec.push_back(col_block_begin_coo);
                col_block_begin_coo = i;

                // 几个关键的索引清零
                row_nz_index = 0;
                acc_col_block_nz = 0;
                cur_col_block_index = 0;
                inner_cur_col_block_acc_number = 0;
            }
            else
            {
                // 在列分块边界的时候已经分过，然后不带分块的换行
                // 这里换行
                row_nz_index = 0;
                acc_col_block_nz = 0;
                cur_col_block_index = 0;
                inner_cur_col_block_acc_number = 0;
            }
        }

        if (cur_col_block_index >= block_size_arr.size())
        {
            cout << "error:col block is not enough, coo index:" << i << ", cur col block index:" << cur_col_block_index << endl;
            assert(false);
        }

        if (row_nz_index == acc_col_block_nz + block_size_arr[cur_col_block_index] - 1)
        {
            if (cur_col_block_index >= block_size_arr.size())
            {
                // 分块的数量没法满足行非零元数量的要求
                cout << "error:col block is not enough, coo index:" << i << ", cur col block index:" << cur_col_block_index << endl;
                assert(false);
            }

            // 当前非零元的行号
            assert(cur_row_index >= global_row_index->min_row_index && cur_row_index <= global_row_index->max_row_index);
            assert(cur_col_index >= global_col_index->min_col_index && cur_col_index <= global_col_index->max_col_index);

            // 到块末了，增加一个块
            new_index_of_the_first_row_vec.push_back(cur_row_index);
            new_row_number_of_block_vec.push_back(1);
            new_coo_begin_index_vec.push_back(col_block_begin_coo);

            col_block_begin_coo = i + 1;

            // 已经过的块的非零元数量、已经经过的块的数量、在当前块内部经过的非零元数量
            acc_col_block_nz = acc_col_block_nz + block_size_arr[cur_col_block_index];
            cur_col_block_index++;
            inner_cur_col_block_acc_number = 0;
        }
        else
        {
            // 没到块末就自增
            inner_cur_col_block_acc_number++;
        }

        row_nz_index++;
        last_row_index = cur_row_index;
    }

    // 父块结束时的强制分块
    assert(col_block_begin_coo <= global_row_index->length);
    if (col_block_begin_coo != global_row_index->length)
    {
        // 父块末尾的分块
        new_index_of_the_first_row_vec.push_back(global_row_index->max_row_index);
        new_row_number_of_block_vec.push_back(1);
        new_coo_begin_index_vec.push_back(col_block_begin_coo);
    }

    assert(new_index_of_the_first_row_vec.size() > 0);

    assert(new_index_of_the_first_row_vec.size() == new_row_number_of_block_vec.size() && new_row_number_of_block_vec.size() == new_coo_begin_index_vec.size());

    // coo是一个block_num + 1大小的数组，最后附带父块nnz
    new_coo_begin_index_vec.push_back(global_row_index->length);

    // 加入一个新的索引
    index_of_compress_block_t *new_index = new index_of_compress_block_t();

    // 一些元数据
    new_index->level_of_this_index = TBLOCK_LEVEL;
    new_index->index_compressed_type = CSR;
    new_index->block_num = new_index_of_the_first_row_vec.size();
    // 针对更低一层的CSR索引先不存在，数据类型也不存在
    new_index->index_arr = NULL;
    new_index->length = new_coo_begin_index_vec.size();

    new_index->type_of_index = BLOCK_INDEX;
    new_index->max_col_index = global_col_index->max_col_index;
    new_index->min_col_index = global_col_index->min_col_index;
    new_index->max_row_index = global_row_index->max_row_index;
    new_index->min_row_index = global_row_index->min_row_index;

    // 找到三个数组的数据类型
    new_index->data_type_of_index_of_the_first_row_arr = find_most_suitable_data_type(new_index_of_the_first_row_vec[new_index_of_the_first_row_vec.size() - 1] + 1);
    new_index->data_type_of_row_number_of_block_arr = find_most_suitable_data_type(new_row_number_of_block_vec[0]);
    new_index->data_type_of_coo_begin_index_arr = find_most_suitable_data_type(new_coo_begin_index_vec[new_coo_begin_index_vec.size() - 1]);

    // 申请三个数组
    new_index->index_of_the_first_row_arr = malloc_arr(new_index_of_the_first_row_vec.size(), new_index->data_type_of_index_of_the_first_row_arr);
    new_index->row_number_of_block_arr = malloc_arr(new_row_number_of_block_vec.size(), new_index->data_type_of_row_number_of_block_arr);
    new_index->coo_begin_index_arr = malloc_arr(new_coo_begin_index_vec.size(), new_index->data_type_of_coo_begin_index_arr);

    // 将数据拷贝进来
    copy_unsigned_long_arr_to_others(&(new_index_of_the_first_row_vec[0]), new_index->index_of_the_first_row_arr, new_index->data_type_of_index_of_the_first_row_arr, new_index->block_num);
    copy_unsigned_long_arr_to_others(&(new_row_number_of_block_vec[0]), new_index->row_number_of_block_arr, new_index->data_type_of_row_number_of_block_arr, new_index->block_num);
    copy_unsigned_long_arr_to_others(&(new_coo_begin_index_vec[0]), new_index->coo_begin_index_arr, new_index->data_type_of_coo_begin_index_arr, new_index->length);

    // 剩下没有用的数组
    new_index->child_tmp_row_csr_index_arr = NULL;
    compressed_block->read_index.push_back(new_index);
}

// BLB基础上执行分块，WLB和BLB的分块都会省略掉没有非零元的空块
void sep_warp_level_row_csr(compressed_block_t *compressed_block, vector<unsigned long> block_index_arr, vector<vector<unsigned int>> row_block_size_arr)
{
    assert(compressed_block != NULL && compressed_block->read_index.size() > 2);
    assert(compressed_block->read_index[2]->index_compressed_type == CSR && compressed_block->read_index[2]->type_of_index == BLOCK_INDEX &&
           compressed_block->read_index[2]->level_of_this_index == TBLOCK_LEVEL);

    assert(compressed_block->read_index[2]->block_num >= block_index_arr.size());
    assert(block_index_arr.size() == row_block_size_arr.size() && row_block_size_arr.size() >= 0);

    // 记录每个数组的最大值，用来规定数据类型
    unsigned long max_first_row_index = 0;
    unsigned long max_row_num = 0;
    unsigned long max_coo_begin_index = 0;
    unsigned long max_new_child_tmp_row_csr_index = 0;
    unsigned long max_coo_block_size = 0;

    // 父索引
    index_of_compress_block_t *tblock_index = compressed_block->read_index[2];
    // 全局的行索引
    index_of_compress_block_t *global_row_index = compressed_block->read_index[0];

    assert(tblock_index->level_of_this_index == TBLOCK_LEVEL && tblock_index->index_compressed_type == CSR);
    // 这个时候父索引的很多部分还没有建立，这个函数的目标就是建立这些
    assert(tblock_index->index_arr == NULL && tblock_index->child_tmp_row_csr_index_arr == NULL);
    assert(global_row_index->index_compressed_type == COO);
    assert(global_row_index->type_of_index == ROW_INDEX);

    // 重新整理三个数组
    vector<unsigned long> new_first_row_index_in_sub_block_vec;
    vector<unsigned long> new_coo_begin_index_in_sub_block_vec;
    vector<unsigned long> new_block_row_num_in_sub_block_vec;
    vector<unsigned long> new_coo_block_size_vec;

    // 每个线程块粒度的分块到warp块的CSR索引
    vector<unsigned long> new_tblock_index_vec;
    // 从0开始计算
    new_tblock_index_vec.push_back(0);

    // 每个线程块进行中间结果的归约，这个数组的取值范围和每个warp产生的中间结果的数量的综合有关
    // 这个数组的长度等于一个线程块内的理论warp块的理论(行数量+1)只和
    vector<unsigned long> new_child_tmp_row_csr_index_vec;
    vector<unsigned long> new_block_begin_index_in_tmp_row_csr_index_vec;

    // 要被切分的块的元数据的索引
    unsigned long index_of_block_index_arr = 0;

    unsigned long i;
    // 遍历所有的线程块粒度的分块
    for (i = 0; i < tblock_index->block_num; i++)
    {
        // 当前行中间结果的数量，是归约数组的增量
        unsigned long warp_tmp_result_of_last_row = 1;

        // 查看这个块的是不是需要继续分块
        if (index_of_block_index_arr < block_index_arr.size() && i == block_index_arr[index_of_block_index_arr])
        {
            // 进一步行分块的块大小
            vector<unsigned int> row_block_size_of_this_block_vec = row_block_size_arr[index_of_block_index_arr];
            // 分块至少分两块
            assert(row_block_size_of_this_block_vec.size() > 1);

            // 分块的大小只和应该只能等于行条带的大小，这里要做一个检查
            unsigned long j;
            unsigned long row_block_size_sum = 0;
            for (j = 0; j < row_block_size_of_this_block_vec.size(); j++)
            {
                row_block_size_sum = row_block_size_sum + row_block_size_of_this_block_vec[j];
            }

            // cout << row_block_size_sum << "," << (tblock_index->max_row_index - tblock_index->min_row_index + 1) << endl;
            // 父块的行数量和子块行数量只和相等
            assert(row_block_size_sum == read_from_array_with_data_type(tblock_index->row_number_of_block_arr,
                                                                        tblock_index->data_type_of_row_number_of_block_arr, i));

            // 首先得到在没有空块的前提下，每个块的首行行号
            vector<unsigned long> ideal_block_first_row_index_vec;
            ideal_block_first_row_index_vec.push_back(0);

            for (j = 0; j < row_block_size_of_this_block_vec.size(); j++)
            {
                ideal_block_first_row_index_vec.push_back(ideal_block_first_row_index_vec[ideal_block_first_row_index_vec.size() - 1] + row_block_size_of_this_block_vec[j]);
            }

            assert(ideal_block_first_row_index_vec.size() == row_block_size_of_this_block_vec.size() + 1);

            // 当前父块的第一行行号
            unsigned long first_global_row_index_of_this_block = read_from_array_with_data_type(tblock_index->index_of_the_first_row_arr, tblock_index->data_type_of_index_of_the_first_row_arr, i);

            // 记录上一个非零元所属于相对块号
            unsigned long last_relat_block_index = 0;

            // 当前父块的起始位置和结束位置
            unsigned long coo_begin_index_of_acc_block = read_from_array_with_data_type(tblock_index->coo_begin_index_arr, tblock_index->data_type_of_coo_begin_index_arr, i);
            unsigned long coo_end_index_of_acc_block = read_from_array_with_data_type(tblock_index->coo_begin_index_arr, tblock_index->data_type_of_coo_begin_index_arr, i + 1) - 1;

            // cout << coo_begin_index_of_acc_block << "," << coo_end_index_of_acc_block << endl;
            // // 用一个数组记录所有的非空块
            // vector<unsigned long> no_empty_block_record;
            // 用一个变量存储这个父块中warp子块的数量
            unsigned long num_of_warp_sub_block = 0;

            for (j = coo_begin_index_of_acc_block; j <= coo_end_index_of_acc_block; j++)
            {
                assert(j < global_row_index->length);
                // 获取对应的非零元的行号
                unsigned long cur_global_row_index = read_from_array_with_data_type(global_row_index->index_arr, global_row_index->index_data_type, j);
                unsigned long cur_relat_row_index = cur_global_row_index - first_global_row_index_of_this_block;
                // cout << cur_global_row_index << "," << first_global_row_index_of_this_block << endl;
                // 搜索当前的行号所属于的块，遍历所有的行分块
                unsigned long k;
                // 用bool值判断有没有找到对应的非零元属于的warp块
                bool is_found = false;
                for (k = last_relat_block_index; k < row_block_size_of_this_block_vec.size(); k++)
                {
                    // 能不能搜得到
                    if (cur_relat_row_index >= ideal_block_first_row_index_vec[k] && cur_relat_row_index < ideal_block_first_row_index_vec[k + 1])
                    {
                        // 这里代表搜到了
                        // 看看是不是第一个非零元
                        if (j == coo_begin_index_of_acc_block)
                        {
                            num_of_warp_sub_block++;
                            // no_empty_block_record.push_back(k);
                            if (max_first_row_index < ideal_block_first_row_index_vec[k])
                            {
                                max_first_row_index = ideal_block_first_row_index_vec[k];
                            }
                            // 如果是第一个非零元，直接写元数据
                            new_first_row_index_in_sub_block_vec.push_back(ideal_block_first_row_index_vec[k]);

                            if (j - coo_begin_index_of_acc_block > max_coo_begin_index)
                            {
                                max_coo_begin_index = j - coo_begin_index_of_acc_block;
                            }

                            new_coo_begin_index_in_sub_block_vec.push_back(j - coo_begin_index_of_acc_block);

                            // 记录这一子块一共多少行
                            if (max_row_num < row_block_size_of_this_block_vec[k])
                            {
                                max_row_num = row_block_size_of_this_block_vec[k];
                            }

                            new_block_row_num_in_sub_block_vec.push_back(row_block_size_of_this_block_vec[k]);

                            last_relat_block_index = k;
                        }
                        else
                        {
                            // 不是线程块粒度子块的第一个非零元，需要比较才能决定是不是有一个新的块
                            assert(last_relat_block_index <= k);
                            // 如果不是就比较一下再写，说明当前非零元来自于一个新的warp块
                            // 这个新的块的几个元数据的记录会自动跳过完全空的WLB。无论是BLB还是WLB的行分块，都会将空块忽视掉
                            if (last_relat_block_index < k)
                            {
                                num_of_warp_sub_block++;
                                // no_empty_block_record.push_back(k);
                                // 写三个元数据
                                if (max_first_row_index < ideal_block_first_row_index_vec[k])
                                {
                                    max_first_row_index = ideal_block_first_row_index_vec[k];
                                }

                                new_first_row_index_in_sub_block_vec.push_back(ideal_block_first_row_index_vec[k]);

                                if (max_coo_begin_index < j - coo_begin_index_of_acc_block)
                                {
                                    max_coo_begin_index = j - coo_begin_index_of_acc_block;
                                }

                                new_coo_begin_index_in_sub_block_vec.push_back(j - coo_begin_index_of_acc_block);

                                // 记录这一子块一共多少行
                                if (max_row_num < row_block_size_of_this_block_vec[k])
                                {
                                    max_row_num = row_block_size_of_this_block_vec[k];
                                }

                                new_block_row_num_in_sub_block_vec.push_back(row_block_size_of_this_block_vec[k]);
                            }
                            last_relat_block_index = k;
                        }

                        // 如果搜到了，就不用再搜了
                        is_found = true;
                        break;
                    }
                }

                // 如果还是没找到说明出错了
                if (is_found == false)
                {
                    cout << "nz is not belong to any warp blocks" << i << "," << j << "," << cur_relat_row_index << endl;
                    assert(false);
                }
            }
            // 找下一个块
            index_of_block_index_arr++;

            // 记录这个父块中在自索引中的偏移量
            new_tblock_index_vec.push_back(new_tblock_index_vec[new_tblock_index_vec.size() - 1] + num_of_warp_sub_block);
        }
        else
        {
            // 这一块是不用分的，直接放到新的索引中
            // 所有的索引都是相对子块，每个子块从0开始计算
            new_first_row_index_in_sub_block_vec.push_back(0);

            unsigned long tblock_row_num = read_from_array_with_data_type(tblock_index->row_number_of_block_arr,
                                                                          tblock_index->data_type_of_row_number_of_block_arr, i);
            // 和父索引一致的行数量
            if (max_row_num < tblock_row_num)
            {
                max_row_num = tblock_row_num;
            }

            new_block_row_num_in_sub_block_vec.push_back(tblock_row_num);
            new_coo_begin_index_in_sub_block_vec.push_back(0);

            // 记录每个线程块的块索引偏移，不考虑大量空行导致的空块，只需要考虑实际块即可
            new_tblock_index_vec.push_back(new_tblock_index_vec[new_tblock_index_vec.size() - 1] + 1);
        }

        // 为当前block添加归约元数据
        // warp之间没有公共部分，按照块内索引一行一个中间结果，每个warp根据在处理的非零元的相对行号直接将数据归约到共享内存的特定位置
        // 遍历block中的所有行，一行一个结果，并且一行不可能有两个结果
        // 因为归约信息是CSR格式的，所以每个块的归约信息大小为行数+1个
        vector<unsigned long> inner_block_tmp_row_csr_vec;

        // 父块的行数量
        unsigned long tblock_row_num = read_from_array_with_data_type(tblock_index->row_number_of_block_arr,
                                                                      tblock_index->data_type_of_row_number_of_block_arr, i);

        unsigned long j;
        for (j = 0; j < tblock_row_num; j++)
        {
            // inner_block_tmp_row_csr_vec.push_back(j);
        }

        // inner_block_tmp_row_csr_vec.push_back(tblock_row_num);

        // 拷贝到全局的数组中
        // new_block_begin_index_in_tmp_row_csr_index_vec.push_back(new_child_tmp_row_csr_index_vec.size());

        // 将inner_block_tmp_row_csr_vec拷贝到new_child_tmp_row_csr_index_vec
        for (j = 0; j < inner_block_tmp_row_csr_vec.size(); j++)
        {
            if (max_new_child_tmp_row_csr_index < inner_block_tmp_row_csr_vec[j])
            {
                max_new_child_tmp_row_csr_index = inner_block_tmp_row_csr_vec[j];
            }
            // new_child_tmp_row_csr_index_vec.push_back(inner_block_tmp_row_csr_vec[j]);
        }
    }

    assert(new_tblock_index_vec.size() == tblock_index->length);
    // 遍历所有的父块，推算出所有子块的大小
    for (i = 0; i < tblock_index->block_num; i++)
    {
        // 用一个变量存储累计的非零元数量
        unsigned long acele_nzz = 0;
        // 遍历父块中所有的warp块，先得到开始和结束的索引
        unsigned long j;
        for (j = new_tblock_index_vec[i]; j < new_tblock_index_vec[i + 1] - 1; j++)
        {
            // if(i == tblock_index->block_num - 1){
            //     //
            // }

            unsigned long warp_block_nnz = new_coo_begin_index_in_sub_block_vec[j + 1] - new_coo_begin_index_in_sub_block_vec[j];

            if (max_coo_block_size < warp_block_nnz)
            {
                max_coo_block_size = warp_block_nnz;
            }

            // j代表的是warp块号
            new_coo_block_size_vec.push_back(warp_block_nnz);
            acele_nzz = acele_nzz + warp_block_nnz;
        }

        // 最后一个warp块的数量用整个父块的非零元数量来减
        // 找到父块的非零元数量
        // 当前父块的起始位置和结束位置
        unsigned long coo_begin_index_of_acc_block = read_from_array_with_data_type(tblock_index->coo_begin_index_arr, tblock_index->data_type_of_coo_begin_index_arr, i);

        assert(tblock_index->coo_block_size_arr == NULL);

        unsigned long coo_end_index_of_acc_block = read_from_array_with_data_type(tblock_index->coo_begin_index_arr, tblock_index->data_type_of_coo_begin_index_arr, i + 1) - 1;

        if (i == tblock_index->block_num - 1)
        {
            cout << "coo_begin_index_of_acc_block:" << coo_begin_index_of_acc_block << ",coo_end_index_of_acc_block:" << coo_end_index_of_acc_block << endl;
        }

        unsigned long tblock_nnz = coo_end_index_of_acc_block - coo_begin_index_of_acc_block + 1;
        unsigned long warp_block_nnz = tblock_nnz - acele_nzz;

        // cout << tblock_nnz << "," << warp_block_nnz << "," << acele_nzz << endl;

        if (max_coo_block_size < warp_block_nnz)
        {
            max_coo_block_size = warp_block_nnz;
        }

        new_coo_block_size_vec.push_back(warp_block_nnz);
    }

    // 几个数组的大小是相等的
    assert(new_coo_block_size_vec.size() == new_first_row_index_in_sub_block_vec.size());
    assert(new_first_row_index_in_sub_block_vec.size() == new_coo_begin_index_in_sub_block_vec.size());
    assert(new_coo_begin_index_in_sub_block_vec.size() == new_block_row_num_in_sub_block_vec.size());
    assert(new_tblock_index_vec.size() == tblock_index->length);

    // 计算父索引中所有块的行的数量
    unsigned long tblock_index_row_num_sum = 0;

    // 遍历所有父索引
    // for (i = 0; i < tblock_index->block_num; i++)
    // {
    //     tblock_index_row_num_sum = tblock_index_row_num_sum + read_from_array_with_data_type(tblock_index->row_number_of_block_arr, tblock_index->data_type_of_row_number_of_block_arr, i);
    // }

    // 检查一下new_child_tmp_row_csr_index_vec和new_block_begin_index_in_tmp_row_csr_index_vec大小
    // assert(new_block_begin_index_in_tmp_row_csr_index_vec.size() == tblock_index->block_num);
    // new_child_tmp_row_csr_index_vec的大小是行的数量+块的数量
    // assert(new_child_tmp_row_csr_index_vec.size() == tblock_index->block_num + tblock_index_row_num_sum);

    assert(new_block_begin_index_in_tmp_row_csr_index_vec.size() == 0 && new_child_tmp_row_csr_index_vec.size() == 0);

    // // 将new_block_begin_index_in_tmp_row_csr_index_vec和new_child_tmp_row_csr_index_vec写到父索引中
    // tblock_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block = find_most_suitable_data_type(new_block_begin_index_in_tmp_row_csr_index_vec[new_block_begin_index_in_tmp_row_csr_index_vec.size() - 1]);
    // tblock_index->data_type_of_child_tmp_row_csr_index = find_most_suitable_data_type(max_new_child_tmp_row_csr_index);
    tblock_index->index_data_type = find_most_suitable_data_type(new_tblock_index_vec[new_tblock_index_vec.size() - 1]);

    // // 创建数组
    // tblock_index->child_tmp_row_csr_index_arr = malloc_arr(new_child_tmp_row_csr_index_vec.size(), tblock_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block);
    // tblock_index->begin_index_in_tmp_row_csr_arr_of_block = malloc_arr(new_block_begin_index_in_tmp_row_csr_index_vec.size(), tblock_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block);
    tblock_index->index_arr = malloc_arr(new_tblock_index_vec.size(), tblock_index->index_data_type);

    // // 拷贝数组
    // copy_unsigned_long_arr_to_others(&(new_child_tmp_row_csr_index_vec[0]), tblock_index->child_tmp_row_csr_index_arr, tblock_index->data_type_of_child_tmp_row_csr_index, new_child_tmp_row_csr_index_vec.size());
    // copy_unsigned_long_arr_to_others(&(new_block_begin_index_in_tmp_row_csr_index_vec[0]), tblock_index->begin_index_in_tmp_row_csr_arr_of_block, tblock_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block, new_block_begin_index_in_tmp_row_csr_index_vec.size());
    copy_unsigned_long_arr_to_others(&(new_tblock_index_vec[0]), tblock_index->index_arr, tblock_index->index_data_type, new_tblock_index_vec.size());

    // tblock_index->size_of_child_tmp_row_csr_index = new_child_tmp_row_csr_index_vec.size();

    // 以上修改完父块

    // 遍历完所有的块，写索引
    index_of_compress_block_t *warp_level_index = new index_of_compress_block_t();

    warp_level_index->level_of_this_index = WRAP_LEVEL;
    warp_level_index->index_compressed_type = CSR;

    warp_level_index->block_num = new_first_row_index_in_sub_block_vec.size();
    // 具体的索引还是置空
    warp_level_index->index_arr = NULL;
    warp_level_index->length = warp_level_index->block_num + 1;
    warp_level_index->type_of_index = BLOCK_INDEX;

    warp_level_index->data_type_of_index_of_the_first_row_arr = find_most_suitable_data_type(max_first_row_index + 1);
    warp_level_index->index_of_the_first_row_arr = malloc_arr(warp_level_index->block_num, warp_level_index->data_type_of_index_of_the_first_row_arr);
    copy_unsigned_long_arr_to_others(&(new_first_row_index_in_sub_block_vec[0]), warp_level_index->index_of_the_first_row_arr,
                                     warp_level_index->data_type_of_index_of_the_first_row_arr, warp_level_index->block_num);

    warp_level_index->data_type_of_row_number_of_block_arr = find_most_suitable_data_type(max_coo_block_size);
    warp_level_index->row_number_of_block_arr = malloc_arr(warp_level_index->block_num, warp_level_index->data_type_of_row_number_of_block_arr);
    copy_unsigned_long_arr_to_others(&(new_block_row_num_in_sub_block_vec[0]), warp_level_index->row_number_of_block_arr, warp_level_index->data_type_of_row_number_of_block_arr, warp_level_index->block_num);

    warp_level_index->max_row_index = tblock_index->max_row_index;
    warp_level_index->min_row_index = tblock_index->min_row_index;
    warp_level_index->max_col_index = tblock_index->max_col_index;
    warp_level_index->min_col_index = tblock_index->min_col_index;

    warp_level_index->data_type_of_coo_begin_index_arr = find_most_suitable_data_type(max_coo_begin_index);
    warp_level_index->coo_begin_index_arr = malloc_arr(warp_level_index->block_num, warp_level_index->data_type_of_coo_begin_index_arr);
    copy_unsigned_long_arr_to_others(&(new_coo_begin_index_in_sub_block_vec[0]), warp_level_index->coo_begin_index_arr, warp_level_index->data_type_of_coo_begin_index_arr, warp_level_index->block_num);

    // 打印coo size
    // print_arr_to_file_with_data_type(&(new_coo_block_size_vec[0]), UNSIGNED_LONG, new_coo_block_size_vec.size(), "/home/duzhen/spmv_builder/data_source/test5-1.log");
    warp_level_index->data_type_of_coo_block_size_arr = find_most_suitable_data_type(max_coo_block_size);
    warp_level_index->coo_block_size_arr = malloc_arr(warp_level_index->block_num, warp_level_index->data_type_of_coo_block_size_arr);
    copy_unsigned_long_arr_to_others(&(new_coo_block_size_vec[0]), warp_level_index->coo_block_size_arr, warp_level_index->data_type_of_coo_block_size_arr, warp_level_index->block_num);

    // 剩下的几个矩阵由下一层决定

    // 将warp数组放到块矩阵中
    compressed_block->read_index.push_back(warp_level_index);
}

void default_sep_warp_level_row_csr(compressed_block_t *compressed_block_ptr)
{
    assert(compressed_block_ptr != NULL);
    assert(compressed_block_ptr->read_index.size() == 3);

    // 执行默认的WLB分块
    // 如果没有warp层次的切分，需要补一个默认的warp层次的切分，将WLB和BLB合为一体
    vector<vector<unsigned int>> arr_of_row_block_size_arr;
    vector<unsigned long> sep_block_id_arr;

    sep_warp_level_row_csr(compressed_block_ptr, sep_block_id_arr, arr_of_row_block_size_arr);

    assert(compressed_block_ptr->read_index.size() == 4);
}

void one_row_sep_warp_level_row_csr(compressed_block_t *compressed_block_ptr)
{
    assert(compressed_block_ptr != NULL);
    assert(compressed_block_ptr->read_index.size() == 3);

    vector<unsigned long> sep_block_id;
    vector<vector<unsigned int>> spec_WLB_row_num_of_a_BLB;

    // 一行一个warp
    // 遍历所有的BLB
    assert(compressed_block_ptr->read_index[2]->block_num > 0);
    index_of_compress_block_t* BLB_index = compressed_block_ptr->read_index[2];
    assert(BLB_index->row_number_of_block_arr != NULL);
    for (unsigned long BLB_id = 0; BLB_id < compressed_block_ptr->read_index[2]->block_num; BLB_id++)
    {
        // 获取当前BLB的行数量
        unsigned long cur_BLB_row_num = read_from_array_with_data_type(BLB_index->row_number_of_block_arr, BLB_index->data_type_of_row_number_of_block_arr, BLB_id);
        // 如果行数量大于1，那就需要进一步分块
        if (cur_BLB_row_num >= 1)
        {
            // 进一步分块
            sep_block_id.push_back(BLB_id);
            // 初始化一个进一步分块的WLB行号
            vector<unsigned int> WLB_row_num;

            for (unsigned long i = 0; i < cur_BLB_row_num; i++)
            {
                WLB_row_num.push_back(1);
            }

            spec_WLB_row_num_of_a_BLB.push_back(WLB_row_num);
        }
    }

    assert(sep_block_id.size() > 0 && spec_WLB_row_num_of_a_BLB.size() > 0);
    // 这里执行一个WLB级别的行分块，一行一个WLB
    sep_warp_level_row_csr(compressed_block_ptr, sep_block_id, spec_WLB_row_num_of_a_BLB);
    assert(compressed_block_ptr->read_index.size() == 4);
}

void sep_warp_level_col_csr(compressed_block_t *compressed_block, vector<unsigned long> block_index_arr, vector<vector<unsigned int>> col_block_size_arr)
{
    assert(compressed_block != NULL && block_index_arr.size() != 0);
    assert(compressed_block->read_index.size() == 4);
    assert(compressed_block->read_index[2]->level_of_this_index == TBLOCK_LEVEL && compressed_block->read_index[3]->level_of_this_index == WRAP_LEVEL);

    assert(compressed_block->read_index[2]->index_compressed_type == CSR && compressed_block->read_index[3]->index_compressed_type == CSR);

    index_of_compress_block_t *old_tblock_index = compressed_block->read_index[2];
    index_of_compress_block_t *old_warp_index = compressed_block->read_index[3];
    index_of_compress_block_t *global_row_index = compressed_block->read_index[0];

    assert(global_row_index->index_compressed_type == COO && global_row_index->type_of_index == ROW_INDEX);

    compressed_block->share_row_with_other_warp = true;
    // 先遍历所有的线程块分块，然后遍历所有warp分块，纵分块会同时改变两层索引，对于父索引改变的是child_tmp_row_csr_index_arr，begin_index_in_tmp_row_csr_arr_of_block，以及index_arr
    // 对于子索引来说，index_of_the_first_row_arr、row_number_of_block_arr、coo_begin_index_arr、coo_block_size_arr、tmp_result_write_index_arr五个数组
    // 分别申请这8个数组
    // 父索引的
    unsigned long max_tblock_child_tmp_row_csr_index = 0;
    // 因为分块是对齐的，所以这个数组的大小为，行的数量+block块的数量。但是在总分块中，不再给空行和空块预留中间结果的存储空间，
    // 因为额外引入了new_warp_tmp_result_write_index_vec来存储每个warp中间结果，所以不再需要一行一行一一对应了
    vector<unsigned long> new_tblock_child_tmp_row_csr_index_vec;
    // 和block_num大小一致
    vector<unsigned long> new_tblock_begin_index_in_tmp_csr_vec;
    vector<unsigned long> new_tblock_index_arr;
    new_tblock_index_arr.push_back(0);

    // 子索引的相关数组
    unsigned long max_warp_first_row_index = 0;
    vector<unsigned long> new_warp_first_row_index_vec;
    unsigned long max_warp_row_num = 0;
    vector<unsigned long> new_warp_row_num_vec;
    unsigned long max_warp_coo_begin_index = 0;
    vector<unsigned long> new_warp_coo_begin_index_vec;
    unsigned long max_coo_block_size = 0;
    vector<unsigned long> new_warp_coo_block_size_vec;
    // 总切块导致多个warp负责一行，中间结果的存储位置不再和行号对应，所以需要加一个数组
    // 这个数组本质上是块内每个warp的块内行号的累加，大小是warp的数量
    unsigned long max_tmp_result_write_index = 0;
    vector<unsigned long> new_warp_tmp_result_write_index_vec;

    // 这里纵分块，遍历所有的子块，在子块的基础上进一步纵分块
    unsigned long old_global_cur_warp_index = 0;

    // block_index_arr的索引
    unsigned long cur_index_in_block_index_arr = 0;

    // 遍历所有的block
    unsigned long global_cur_tblock_index;
    for (global_cur_tblock_index = 0; global_cur_tblock_index < old_tblock_index->block_num; global_cur_tblock_index++)
    {
        // 记录一个块内部的规约信息
        vector<unsigned long> inner_tmp_csr_index;
        // 最后一行的中间结果数量
        unsigned long result_number_of_last_row = 1;

        // 用一个变量存储当前tblock中warp块的数量
        unsigned long inner_block_warp_block_num = 0;

        // 用一个bool判断新的warp块是不是tblock中的第一个块
        bool is_first_warp_block = true;

        // 遍历所有的子块
        unsigned long warp_begin = read_from_array_with_data_type(old_tblock_index->index_arr, old_tblock_index->index_data_type, global_cur_tblock_index);
        unsigned long warp_end = read_from_array_with_data_type(old_tblock_index->index_arr, old_tblock_index->index_data_type, global_cur_tblock_index + 1) - 1;

        // 块的coo起始位置
        unsigned long tblock_coo_begin = read_from_array_with_data_type(old_tblock_index->coo_begin_index_arr, old_tblock_index->data_type_of_coo_begin_index_arr, global_cur_tblock_index);
        unsigned long tblock_coo_end = read_from_array_with_data_type(old_tblock_index->coo_begin_index_arr, old_tblock_index->data_type_of_coo_begin_index_arr, global_cur_tblock_index + 1) - 1;

        // tblock的首行行号
        unsigned long t_block_first_row_index = read_from_array_with_data_type(old_tblock_index->index_of_the_first_row_arr, old_tblock_index->data_type_of_index_of_the_first_row_arr, global_cur_tblock_index);
        unsigned long t_block_row_num = read_from_array_with_data_type(old_tblock_index->row_number_of_block_arr, old_tblock_index->data_type_of_row_number_of_block_arr, global_cur_tblock_index);

        // 遍历一个tblock中所有warp块
        for (old_global_cur_warp_index = warp_begin; old_global_cur_warp_index <= warp_end; old_global_cur_warp_index++)
        {
            // 查看这个块是不是需要进一步列分块的
            if (cur_index_in_block_index_arr < block_index_arr.size() && old_global_cur_warp_index == block_index_arr[cur_index_in_block_index_arr])
            {
                // cout << "find warp block need to be sep" << endl;
                // 当前子块进一步分块的大小
                vector<unsigned int> further_col_block_size = col_block_size_arr[cur_index_in_block_index_arr];

                // 这里说明要进一步列分块，遍历所有分非零元。获取局部块索引
                unsigned long local_warp_coo_begin = read_from_array_with_data_type(old_warp_index->coo_begin_index_arr, old_warp_index->data_type_of_coo_begin_index_arr, old_global_cur_warp_index);
                unsigned long local_warp_coo_end = local_warp_coo_begin + read_from_array_with_data_type(old_warp_index->coo_block_size_arr, old_warp_index->data_type_of_coo_block_size_arr, old_global_cur_warp_index) - 1;

                // 全局的索引
                unsigned long global_warp_coo_begin = local_warp_coo_begin + tblock_coo_begin;
                unsigned long global_warp_coo_end = tblock_coo_begin + local_warp_coo_end;
                assert(global_warp_coo_end <= tblock_coo_end);

                // 用一个变量存储warp块内非零元数量
                unsigned long nnz_in_cur_block = 0;

                // 在一行中遍历的非零元数量
                // 用一个变量存储当前遍历的非零元在一行中列块的编号
                unsigned long col_block_index_in_row = 0;

                // 用一个变量存储上一个非零元的行号
                unsigned long local_row_index_of_last_nz;

                // 遍历对应位的行号
                unsigned long global_coo_index;
                // cout << "global_warp_coo_begin:" << global_warp_coo_begin << ",global_warp_coo_end:" << global_warp_coo_end << endl;
                for (global_coo_index = global_warp_coo_begin; global_coo_index <= global_warp_coo_end; global_coo_index++)
                {
                    // 获取行号
                    unsigned long global_cur_coo_row_index = read_from_array_with_data_type(global_row_index->index_arr, global_row_index->index_data_type, global_coo_index);
                    // 获取相对行号
                    unsigned long local_cur_coo_row_index = global_cur_coo_row_index - t_block_first_row_index;

                    // cout << "global_cur_coo_row_index:" << global_cur_coo_row_index << ",local_cur_coo_row_index:" << local_cur_coo_row_index << endl;

                    // 当前块的非零元数量，用来计算一个块的coo_size，在每次存储coo size的时候都要归零
                    nnz_in_cur_block++;

                    // 如果没有下一个非零元或者下一行和自己不一样那就意味着到了行的边界和块的边界，要加入一个新的warp块
                    if (global_coo_index == global_warp_coo_end || global_cur_coo_row_index != read_from_array_with_data_type(global_row_index->index_arr, global_row_index->index_data_type, global_coo_index + 1))
                    {
                        // cout << "row edge!" << endl;
                        inner_block_warp_block_num++;
                        // 这里代表需要在此处分块，修改
                        // index_of_the_first_row_arr、row_number_of_block_arr、coo_begin_index_arr、coo_block_size_arr、tmp_result_write_index_arr五个数组
                        // 第一个warp的首行单独处理
                        if (is_first_warp_block == true)
                        {
                            // cout << "first warp block in tblock" << endl;
                            // 第一行的相对行号可以直接读出来，在进一步的分块中，空行会被进一步排除，但是之前还是有空行，需要在更新中间结果行偏移的时候考虑
                            // tblock块内的第一个块的块号
                            if (max_warp_first_row_index < local_cur_coo_row_index)
                            {
                                max_warp_first_row_index = local_cur_coo_row_index;
                            }

                            new_warp_first_row_index_vec.push_back(local_cur_coo_row_index);
                            // coo的相对起始位置也从一上来开始计算
                            new_warp_coo_begin_index_vec.push_back(0);

                            new_warp_tmp_result_write_index_vec.push_back(0);

                            // 行的数量肯定是1
                            new_warp_row_num_vec.push_back(1);

                            // 行的数量
                            if (max_warp_row_num < 1)
                            {
                                max_warp_row_num = 1;
                            }

                            assert(nnz_in_cur_block != 0);
                            new_warp_coo_block_size_vec.push_back(nnz_in_cur_block);

                            if (max_coo_block_size < nnz_in_cur_block)
                            {
                                max_coo_block_size = nnz_in_cur_block;
                            }

                            // 更新父块要用的inner_tmp_csr_index
                            // 这个标记了所有中间结果的换行位置，需要检查空行问题
                            // 从0到local_cur_coo_row_index全是0
                            unsigned long i;
                            for (i = 0; i <= local_cur_coo_row_index; i++)
                            {
                                // 一直到当前位置全是0
                                inner_tmp_csr_index.push_back(0);
                            }

                            is_first_warp_block = false;
                        }
                        else
                        {
                            // 用上一个warp块的first row和row size来处理，first row永远都是当前行
                            unsigned long new_warp_first_row_index = local_cur_coo_row_index;
                            // unsigned long new_warp_first_row_index = new_warp_first_row_index_vec[new_warp_first_row_index_vec.size() - 1] + new_warp_row_num_vec[new_warp_row_num_vec.size() - 1];
                            unsigned long new_warp_coo_begin_index = new_warp_coo_begin_index_vec[new_warp_coo_begin_index_vec.size() - 1] + new_warp_coo_block_size_vec[new_warp_coo_block_size_vec.size() - 1];
                            if (max_warp_first_row_index < new_warp_first_row_index)
                            {
                                max_warp_first_row_index = new_warp_first_row_index;
                            }
                            if (max_warp_coo_begin_index < new_warp_coo_begin_index)
                            {
                                max_warp_coo_begin_index = new_warp_coo_begin_index;
                            }

                            new_warp_first_row_index_vec.push_back(new_warp_first_row_index);
                            new_warp_coo_begin_index_vec.push_back(new_warp_coo_begin_index);

                            unsigned long new_warp_tmp_result_write_index = new_warp_tmp_result_write_index_vec[new_warp_tmp_result_write_index_vec.size() - 1] + 1;

                            if (max_tmp_result_write_index > new_warp_tmp_result_write_index)
                            {
                                max_tmp_result_write_index = new_warp_tmp_result_write_index;
                            }

                            // 每个结果是当前行数量，在列分块中
                            new_warp_tmp_result_write_index_vec.push_back(new_warp_tmp_result_write_index);

                            // 行的数量肯定是1
                            new_warp_row_num_vec.push_back(1);

                            // 行的数量
                            if (max_warp_row_num < 1)
                            {
                                max_warp_row_num = 1;
                            }

                            assert(nnz_in_cur_block != 0);
                            new_warp_coo_block_size_vec.push_back(nnz_in_cur_block);

                            if (max_coo_block_size < nnz_in_cur_block)
                            {
                                max_coo_block_size = nnz_in_cur_block;
                            }

                            // inner_tmp_csr_index的更新，首先找到上一个块的首地址
                            assert(inner_tmp_csr_index.size() > 0);
                            // 上一个块的行号和当前块之间的关系导致的不同处理
                            unsigned long last_block_row_index = inner_tmp_csr_index.size() - 1;

                            if (local_cur_coo_row_index == last_block_row_index)
                            {
                                result_number_of_last_row++;
                            }
                            else if (local_cur_coo_row_index < last_block_row_index)
                            {
                                assert(false);
                            }
                            else
                            {
                                // local_cur_coo_row_index > last_block_row_index
                                // 从last_block_row_index到local_cur_coo_row_index中，第一行之前的结果数量加一下，剩下的都是0
                                unsigned long i;
                                for (i = last_block_row_index + 1; i <= local_cur_coo_row_index; i++)
                                {
                                    // 这之间的都是空行
                                    if (i == last_block_row_index + 1)
                                    {
                                        inner_tmp_csr_index.push_back(result_number_of_last_row + inner_tmp_csr_index[inner_tmp_csr_index.size() - 1]);
                                        result_number_of_last_row = 0;
                                        continue;
                                    }

                                    inner_tmp_csr_index.push_back(result_number_of_last_row + inner_tmp_csr_index[inner_tmp_csr_index.size() - 1]);
                                }

                                result_number_of_last_row = 1;
                            }
                        }

                        // 块非零元数量
                        nnz_in_cur_block = 0;
                        // 当前非零元所在列块的索引
                        col_block_index_in_row = 0;

                        // 这里不清零，因为下一行的在结果中的起始位置依赖于这个。
                        // 这个变量在每次给inner_tmp_csr_index赋值之后就要回归到1了，其他时候都不应该进行重置
                        // result_number_of_last_row = 1;

                        // 直接跳过，没必要列分块了
                        continue;
                    }

                    // col_block_index_in_row不能超过further_col_block_size的范围，如果超过了就是分的块数不够
                    if (col_block_index_in_row >= further_col_block_size.size())
                    {
                        cout << "col block num is not enough" << endl;
                        assert(false);
                    }

                    // 当前非零元在行中的索引号等于列块中的最后一个非零元的行中索引，那就划分一个新的块
                    if (nnz_in_cur_block == further_col_block_size[col_block_index_in_row])
                    {
                        inner_block_warp_block_num++;

                        // 看看是不是block中的第一个块，用不同的处理方式
                        if (is_first_warp_block == true)
                        {
                            // cout << "first warp block in tblock" << endl;
                            // 第一个块
                            // 第一行的相对行号可以直接读出来，在进一步的分块中，空行会被进一步排除，但是之前还是有空行，需要在更新中间结果行偏移的时候考虑
                            // tblock块内的第一个块的块号
                            if (max_warp_first_row_index < local_cur_coo_row_index)
                            {
                                max_warp_first_row_index = local_cur_coo_row_index;
                            }

                            new_warp_first_row_index_vec.push_back(local_cur_coo_row_index);
                            // coo的相对起始位置也从一上来开始计算
                            new_warp_coo_begin_index_vec.push_back(0);

                            new_warp_tmp_result_write_index_vec.push_back(0);

                            // 行的数量肯定是1
                            new_warp_row_num_vec.push_back(1);

                            // 行的数量
                            if (max_warp_row_num < 1)
                            {
                                max_warp_row_num = 1;
                            }

                            assert(nnz_in_cur_block != 0);
                            new_warp_coo_block_size_vec.push_back(nnz_in_cur_block);

                            if (max_coo_block_size < nnz_in_cur_block)
                            {
                                max_coo_block_size = nnz_in_cur_block;
                            }

                            // 更新父块要用的inner_tmp_csr_index
                            // 这个标记了所有中间结果的换行位置，需要检查空行问题
                            // 从0到local_cur_coo_row_index全是0
                            unsigned long i;
                            for (i = 0; i <= local_cur_coo_row_index; i++)
                            {
                                // 一直到当前位置全是0
                                inner_tmp_csr_index.push_back(0);
                            }

                            // unsigned long vector_index;
                            // for(vector_index = 0; vector_index < inner_tmp_csr_index.size(); vector_index++){
                            //     cout << inner_tmp_csr_index[vector_index] << ",";
                            // }
                            // cout << endl;

                            is_first_warp_block = false;
                        }
                        else
                        {
                            // 不是第一行
                            // 用上一个warp块的first row和row size来处理，first row永远都是当前行
                            unsigned long new_warp_first_row_index = local_cur_coo_row_index;
                            // unsigned long new_warp_first_row_index = new_warp_first_row_index_vec[new_warp_first_row_index_vec.size() - 1] + new_warp_row_num_vec[new_warp_row_num_vec.size() - 1];
                            unsigned long new_warp_coo_begin_index = new_warp_coo_begin_index_vec[new_warp_coo_begin_index_vec.size() - 1] + new_warp_coo_block_size_vec[new_warp_coo_block_size_vec.size() - 1];
                            if (max_warp_first_row_index < new_warp_first_row_index)
                            {
                                max_warp_first_row_index = new_warp_first_row_index;
                            }
                            if (max_warp_coo_begin_index < new_warp_coo_begin_index)
                            {
                                max_warp_coo_begin_index = new_warp_coo_begin_index;
                            }
                            new_warp_first_row_index_vec.push_back(new_warp_first_row_index);
                            new_warp_coo_begin_index_vec.push_back(new_warp_coo_begin_index);

                            unsigned long new_warp_tmp_result_write_index = new_warp_tmp_result_write_index_vec[new_warp_tmp_result_write_index_vec.size() - 1] + 1;

                            if (max_tmp_result_write_index > new_warp_tmp_result_write_index)
                            {
                                max_tmp_result_write_index = new_warp_tmp_result_write_index;
                            }

                            // 每个结果是当前行数量，在列分块中
                            new_warp_tmp_result_write_index_vec.push_back(new_warp_tmp_result_write_index);

                            // 行的数量肯定是1
                            new_warp_row_num_vec.push_back(1);

                            assert(nnz_in_cur_block != 0);
                            new_warp_coo_block_size_vec.push_back(nnz_in_cur_block);

                            // inner_tmp_csr_index的更新，首先找到上一个块的首地址
                            assert(inner_tmp_csr_index.size() > 0);
                            // 上一个块的行号和当前块之间的关系导致的不同处理
                            unsigned long last_block_row_index = inner_tmp_csr_index.size() - 1;

                            if (local_cur_coo_row_index == last_block_row_index)
                            {

                                result_number_of_last_row++;
                            }
                            else if (local_cur_coo_row_index < last_block_row_index)
                            {
                                // unsigned long vector_index;
                                // for(vector_index = 0; vector_index < inner_tmp_csr_index.size(); vector_index++){
                                //     cout << inner_tmp_csr_index[vector_index] << ",";
                                // }
                                // cout << endl;
                                // cout << "local_cur_coo_row_index:" << local_cur_coo_row_index << ",last_block_row_index:" << last_block_row_index << endl;
                                assert(false);
                            }
                            else
                            {
                                // local_cur_coo_row_index > last_block_row_index
                                // 从last_block_row_index到local_cur_coo_row_index中，第一行之前的结果数量加一下，剩下的都是0
                                unsigned long i;
                                for (i = last_block_row_index + 1; i <= local_cur_coo_row_index; i++)
                                {
                                    // 这之间的都是空行
                                    if (i == last_block_row_index + 1)
                                    {
                                        // cout << "result_number_of_last_row:" << result_number_of_last_row << endl;
                                        inner_tmp_csr_index.push_back(result_number_of_last_row + inner_tmp_csr_index[inner_tmp_csr_index.size() - 1]);
                                        result_number_of_last_row = 0;
                                        continue;
                                    }

                                    inner_tmp_csr_index.push_back(result_number_of_last_row + inner_tmp_csr_index[inner_tmp_csr_index.size() - 1]);
                                }

                                // unsigned long vector_index;
                                // for(vector_index = 0; vector_index < inner_tmp_csr_index.size(); vector_index++){
                                //     cout << inner_tmp_csr_index[vector_index] << ",";
                                // }
                                // cout << endl;

                                result_number_of_last_row = 1;
                            }
                        }

                        nnz_in_cur_block = 0;
                        col_block_index_in_row++;
                    }
                }

                cur_index_in_block_index_arr++;
            }
            else
            {
                inner_block_warp_block_num++;
                // 不用进一步分块，进一步拷贝进来
                unsigned long old_warp_first_row_index = read_from_array_with_data_type(old_warp_index->index_of_the_first_row_arr, old_warp_index->data_type_of_index_of_the_first_row_arr, old_global_cur_warp_index);
                unsigned long old_warp_row_size = read_from_array_with_data_type(old_warp_index->row_number_of_block_arr, old_warp_index->data_type_of_row_number_of_block_arr, old_global_cur_warp_index);
                unsigned long old_warp_coo_begin_index = read_from_array_with_data_type(old_warp_index->coo_begin_index_arr, old_warp_index->data_type_of_coo_begin_index_arr, old_global_cur_warp_index);
                unsigned long old_warp_coo_block_size = read_from_array_with_data_type(old_warp_index->coo_block_size_arr, old_warp_index->data_type_of_coo_block_size_arr, old_global_cur_warp_index);

                // cout << "old_warp_first_row_index:" << old_warp_first_row_index << ",old_warp_row_size:" << old_warp_row_size << ",old_warp_coo_begin_index:" << old_warp_coo_begin_index << ",old_warp_coo_block_size:" << old_warp_coo_block_size << endl;

                // 记录最大值，更新5个数组
                if (max_warp_first_row_index < old_warp_first_row_index)
                {
                    max_warp_first_row_index = old_warp_first_row_index;
                }

                if (max_warp_row_num < old_warp_row_size)
                {
                    max_warp_row_num = old_warp_row_size;
                }

                if (max_warp_coo_begin_index < old_warp_coo_begin_index)
                {
                    max_warp_coo_begin_index = old_warp_coo_begin_index;
                }

                if (max_coo_block_size < old_warp_coo_block_size)
                {
                    max_coo_block_size = old_warp_coo_block_size;
                }

                // 将数据将数据拷贝到新的warp索引元数据的末尾
                new_warp_first_row_index_vec.push_back(old_warp_first_row_index);
                new_warp_row_num_vec.push_back(old_warp_row_size);
                new_warp_coo_begin_index_vec.push_back(old_warp_coo_begin_index);
                new_warp_coo_block_size_vec.push_back(old_warp_coo_block_size);

                // 根据当前块的已累计行数量来决定每个中间结果在块内共享内存中的存储位置
                if (is_first_warp_block == true)
                {
                    // cout << "first warp of tblock" << endl;
                    // 当前tblock的第一个warp
                    new_warp_tmp_result_write_index_vec.push_back(0);
                    is_first_warp_block = false;

                    // 更新inner_tmp_csr_index，存储的是最终结果的行边界
                    // 这里是整个block的第一个块，但是第一行可能是和
                    unsigned long i;
                    for (i = 0; i <= old_warp_first_row_index; i++)
                    {
                        // cout << "不应该进到这里" << endl;
                        // 一直到当前位置全是0
                        inner_tmp_csr_index.push_back(0);
                    }

                    // 剩下的几行让中间结果的索引依次增加1，一位一个中间结果就正好是一行
                    for (i = old_warp_first_row_index + 1; i < old_warp_first_row_index + old_warp_row_size; i++)
                    {
                        inner_tmp_csr_index.push_back(inner_tmp_csr_index[inner_tmp_csr_index.size() - 1] + 1);
                    }

                    unsigned long last_block_row_index = inner_tmp_csr_index.size() - 1;
                    // cout << "last_block_row_index:" << last_block_row_index << ",old_warp_first_row_index:" << old_warp_first_row_index << ",inner_block_warp_block_num:" << inner_block_warp_block_num<< ",global_cur_tblock_index:" << global_cur_tblock_index << endl;
                }
                else
                {
                    // cout << "not first warp of tblock, inner_tmp_csr_index.size():" << inner_tmp_csr_index.size() << endl;

                    // unsigned long inner_tmp_csr_index_index;
                    // for(){

                    // }

                    assert(inner_block_warp_block_num >= 2);
                    assert(new_warp_tmp_result_write_index_vec.size() + 1 == new_warp_row_num_vec.size());
                    // 不是第一个warp块，用上一个warp的row num来加得到这个warp的中间结果偏移量
                    unsigned long new_warp_tmp_result_write_index = new_warp_tmp_result_write_index_vec[new_warp_tmp_result_write_index_vec.size() - 1] + new_warp_row_num_vec[new_warp_row_num_vec.size() - 2];
                    if (max_tmp_result_write_index < new_warp_tmp_result_write_index)
                    {
                        max_tmp_result_write_index = new_warp_tmp_result_write_index;
                    }
                    new_warp_tmp_result_write_index_vec.push_back(new_warp_tmp_result_write_index);

                    // 还是要处理空块问题，获取上一个块的行号，通常来讲，在之前是肯定有一个块的，这个块肯定是某一行的第一个块，所以一定会有对应的中间结果偏移
                    unsigned long last_block_row_index = inner_tmp_csr_index.size() - 1;

                    if (last_block_row_index > old_warp_first_row_index)
                    {
                        cout << "last_block_row_index:" << last_block_row_index << ",old_warp_first_row_index:" << old_warp_first_row_index << ",inner_block_warp_block_num:" << inner_block_warp_block_num << ",global_cur_tblock_index:" << global_cur_tblock_index << endl;
                    }

                    assert(last_block_row_index <= old_warp_first_row_index);

                    if (last_block_row_index == old_warp_first_row_index)
                    {
                        // 不用考虑空行，这里考虑的是一行中有多个块的情况，但是这种情况在只经过行分块的块中是不存在的
                        cout << "more than 1 block in a row before col blocking" << endl;
                        assert(false);
                    }
                    else
                    {
                        // 这里在考虑空行的情况下，需要对空行到当前行之间的每一行赋值
                        unsigned long i;
                        for (i = last_block_row_index + 1; i <= old_warp_first_row_index; i++)
                        {
                            if (i == last_block_row_index)
                            {
                                inner_tmp_csr_index.push_back(inner_tmp_csr_index[inner_tmp_csr_index.size() - 1] + result_number_of_last_row);
                                result_number_of_last_row = 1;
                            }
                            else
                            {
                                // 剩下的都是空行
                                inner_tmp_csr_index.push_back(inner_tmp_csr_index[inner_tmp_csr_index.size() - 1]);
                            }
                        }

                        // 这个warp块剩下的部分一行一个结果
                        for (i = old_warp_first_row_index + 1; i < old_warp_first_row_index + old_warp_row_size; i++)
                        {
                            inner_tmp_csr_index.push_back(inner_tmp_csr_index[inner_tmp_csr_index.size() - 1] + result_number_of_last_row);
                            result_number_of_last_row = 1;
                        }
                    }
                }

                // 遍历这一warp块的所有行，这里肯定只有行分块，所以一个个加到inner_tmp_csr_index之后
                // 给父块使用
                // unsigned long row_index_in_warp;
                // for (row_index_in_warp = 0; row_index_in_warp < old_warp_row_size; row_index_in_warp++)
                // {
                //     inner_tmp_csr_index.push_back(inner_tmp_csr_index[inner_tmp_csr_index.size() - 1] + result_number_of_last_row);
                //     result_number_of_last_row = 1;
                // }

                is_first_warp_block = false;
            }
        }

        // 这里要对inner_tmp_csr_index封底，因为其为csr结构，需要将最后一个块的结束位置显示出来
        assert(inner_tmp_csr_index.size() > 0);
        inner_tmp_csr_index.push_back(inner_tmp_csr_index[inner_tmp_csr_index.size() - 1] + result_number_of_last_row);

        // 这里处理block的所有信息
        // child_tmp_row_csr_index_arr，begin_index_in_tmp_row_csr_arr_of_block，以及index_arr
        // 用inner_tmp_csr_index更新child_tmp_row_csr_index_arr，用child_tmp_row_csr_index_arr更新begin_index_in_tmp_row_csr_arr_of_block
        // 用tblock中的warp数量更新index_arr
        assert(inner_tmp_csr_index.size() == t_block_row_num + 1);

        // 更新块中间结果的起始位置
        // new_tblock_begin_index_in_tmp_csr_vec.push_back(new_tblock_child_tmp_row_csr_index_vec.size());

        unsigned long i;
        for (i = 0; i < inner_tmp_csr_index.size(); i++)
        {
            if (max_tblock_child_tmp_row_csr_index < inner_tmp_csr_index[i])
            {
                max_tblock_child_tmp_row_csr_index = inner_tmp_csr_index[i];
            }
            // new_tblock_child_tmp_row_csr_index_vec.push_back(inner_tmp_csr_index[i]);
        }

        // 这里处理index_arr
        new_tblock_index_arr.push_back(new_tblock_index_arr[new_tblock_index_arr.size() - 1] + inner_block_warp_block_num);
    }

    // 全部处理完，检查一下各个数组的大小是不是符合要求
    // 中间结果的偏移量数量很容易，和当前块行的数量相同
    // assert(new_tblock_begin_index_in_tmp_csr_vec.size() == old_tblock_index->block_num);
    assert(new_tblock_index_arr.size() == old_tblock_index->length);
    // new_tblock_child_tmp_row_csr_index_vec的大小应该是不会变化的，中间结果的数量虽然增多，但是每一个块中行的数量是一样的，所以归约结果的数量也是一致的，new_tblock_child_tmp_row_csr_index_vec的大小和每个父块结果的数量是一致的
    // assert(new_tblock_child_tmp_row_csr_index_vec.size() == old_tblock_index->size_of_child_tmp_row_csr_index);

    assert(new_tblock_child_tmp_row_csr_index_vec.size() == 0 && new_tblock_begin_index_in_tmp_csr_vec.size() == 0);

    // // 子索引的相关数组
    // unsigned long max_warp_first_row_index = 0;
    // vector<unsigned long> new_warp_first_row_index_vec;
    // unsigned long max_warp_row_num = 0;
    // vector<unsigned long> new_warp_row_num_vec;
    // unsigned long max_warp_coo_begin_index = 0;
    // vector<unsigned long> new_warp_coo_begin_index_vec;
    // unsigned long max_coo_block_size = 0;
    // vector<unsigned long> new_warp_coo_block_size_vec;
    // // 总切块导致多个warp负责一行，中间结果的存储位置不再和行号对应，所以需要加一个数组
    // // 这个数组本质上是块内每个warp的块内行号的累加，大小是warp的数量
    // unsigned long max_tmp_result_write_index = 0;
    // vector<unsigned long> new_warp_tmp_result_write_index_vec;
    // warp中不同块的大小
    assert(new_warp_first_row_index_vec.size() == new_warp_row_num_vec.size() && new_warp_coo_begin_index_vec.size() == new_warp_row_num_vec.size());
    assert(new_warp_coo_block_size_vec.size() == new_warp_first_row_index_vec.size());

    if (!(max_warp_first_row_index > 0 && max_warp_row_num > 0 && max_warp_coo_begin_index > 0 && max_coo_block_size > 0))
    {
        cout << "max_warp_first_row_index:" << max_warp_first_row_index << endl;
        cout << "max_warp_row_num:" << max_warp_row_num << endl;
        cout << "max_warp_coo_begin_index" << max_warp_coo_begin_index << endl;
        cout << "max_coo_block_size" << max_coo_block_size << endl;
    }

    assert(max_warp_first_row_index > 0 && max_warp_row_num > 0 && max_warp_coo_begin_index > 0 && max_coo_block_size > 0);

    // 首先是index_arr的重构，增序数组
    // 先删除已有的数组
    delete_arr_with_data_type(old_tblock_index->index_arr, old_tblock_index->index_data_type);
    old_tblock_index->index_arr = NULL;
    // 构造新的数组
    old_tblock_index->index_data_type = find_most_suitable_data_type(new_tblock_index_arr[new_tblock_index_arr.size() - 1]);
    old_tblock_index->index_arr = malloc_arr(old_tblock_index->length, old_tblock_index->index_data_type);
    // 复制新的数组
    copy_unsigned_long_arr_to_others(&(new_tblock_index_arr[0]), old_tblock_index->index_arr, old_tblock_index->index_data_type, old_tblock_index->length);

    // // child_tmp_row_csr_index_arr
    // delete_arr_with_data_type(old_tblock_index->child_tmp_row_csr_index_arr, old_tblock_index->data_type_of_child_tmp_row_csr_index);
    // old_tblock_index->child_tmp_row_csr_index_arr = NULL;
    // // 构造新的数组
    // old_tblock_index->data_type_of_child_tmp_row_csr_index = find_most_suitable_data_type(max_tblock_child_tmp_row_csr_index);
    // // print_data_type(old_tblock_index->data_type_of_child_tmp_row_csr_index);
    // // cout << "max_tblock_child_tmp_row_csr_index:" << max_tblock_child_tmp_row_csr_index <<endl;
    // old_tblock_index->child_tmp_row_csr_index_arr = malloc_arr(new_tblock_child_tmp_row_csr_index_vec.size(), old_tblock_index->data_type_of_child_tmp_row_csr_index);
    // copy_unsigned_long_arr_to_others(&(new_tblock_child_tmp_row_csr_index_vec[0]), old_tblock_index->child_tmp_row_csr_index_arr, old_tblock_index->data_type_of_child_tmp_row_csr_index,
    //                                  new_tblock_child_tmp_row_csr_index_vec.size());

    // // begin_index_in_tmp_row_csr_arr_of_block
    // delete_arr_with_data_type(old_tblock_index->begin_index_in_tmp_row_csr_arr_of_block, old_tblock_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block);
    // old_tblock_index->begin_index_in_tmp_row_csr_arr_of_block = NULL;
    // old_tblock_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block = find_most_suitable_data_type(new_tblock_begin_index_in_tmp_csr_vec[new_tblock_begin_index_in_tmp_csr_vec.size() - 1]);
    // old_tblock_index->begin_index_in_tmp_row_csr_arr_of_block = malloc_arr(new_tblock_begin_index_in_tmp_csr_vec.size(), old_tblock_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block);
    // copy_unsigned_long_arr_to_others(&(new_tblock_begin_index_in_tmp_csr_vec[0]), old_tblock_index->begin_index_in_tmp_row_csr_arr_of_block, old_tblock_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block, new_tblock_begin_index_in_tmp_csr_vec.size());

    // 修改warp索引中的内容，主要是5个数组：index_of_the_first_row_arr、row_number_of_block_arr、coo_begin_index_arr、coo_block_size_arr、tmp_result_write_index_arr五个数组。其中最后一个数组一开始应该是空的
    assert(old_warp_index->tmp_result_write_index_arr == NULL);

    // warp索引的大小发生了显而易见的变化，所以一切都要重来
    old_warp_index->block_num = new_warp_first_row_index_vec.size();
    old_warp_index->length = new_warp_first_row_index_vec.size() + 1;

    delete_arr_with_data_type(old_warp_index->index_of_the_first_row_arr, old_warp_index->data_type_of_index_of_the_first_row_arr);
    old_warp_index->index_of_the_first_row_arr = NULL;
    old_warp_index->data_type_of_index_of_the_first_row_arr = find_most_suitable_data_type(max_warp_first_row_index + 1);
    old_warp_index->index_of_the_first_row_arr = malloc_arr(new_warp_first_row_index_vec.size(), old_warp_index->data_type_of_index_of_the_first_row_arr);
    copy_unsigned_long_arr_to_others(&(new_warp_first_row_index_vec[0]), old_warp_index->index_of_the_first_row_arr, old_warp_index->data_type_of_index_of_the_first_row_arr, new_warp_first_row_index_vec.size());

    delete_arr_with_data_type(old_warp_index->row_number_of_block_arr, old_warp_index->data_type_of_row_number_of_block_arr);
    old_warp_index->row_number_of_block_arr = NULL;
    old_warp_index->data_type_of_row_number_of_block_arr = find_most_suitable_data_type(max_warp_row_num);
    old_warp_index->row_number_of_block_arr = malloc_arr(new_warp_row_num_vec.size(), old_warp_index->data_type_of_row_number_of_block_arr);
    copy_unsigned_long_arr_to_others(&(new_warp_row_num_vec[0]), old_warp_index->row_number_of_block_arr, old_warp_index->data_type_of_row_number_of_block_arr, new_warp_row_num_vec.size());

    delete_arr_with_data_type(old_warp_index->coo_begin_index_arr, old_warp_index->data_type_of_coo_begin_index_arr);
    old_warp_index->coo_begin_index_arr = NULL;
    old_warp_index->data_type_of_coo_begin_index_arr = find_most_suitable_data_type(max_warp_coo_begin_index);
    old_warp_index->coo_begin_index_arr = malloc_arr(new_warp_coo_begin_index_vec.size(), old_warp_index->data_type_of_coo_begin_index_arr);
    copy_unsigned_long_arr_to_others(&(new_warp_coo_begin_index_vec[0]), old_warp_index->coo_begin_index_arr, old_warp_index->data_type_of_coo_begin_index_arr, new_warp_coo_begin_index_vec.size());

    delete_arr_with_data_type(old_warp_index->coo_block_size_arr, old_warp_index->data_type_of_coo_block_size_arr);
    old_warp_index->coo_block_size_arr = NULL;
    old_warp_index->data_type_of_coo_block_size_arr = find_most_suitable_data_type(max_coo_block_size);
    old_warp_index->coo_block_size_arr = malloc_arr(new_warp_coo_block_size_vec.size(), old_warp_index->data_type_of_coo_block_size_arr);
    copy_unsigned_long_arr_to_others(&(new_warp_coo_block_size_vec[0]), old_warp_index->coo_block_size_arr, old_warp_index->data_type_of_coo_block_size_arr, new_warp_coo_block_size_vec.size());

    // // tmp_result_write_index_arr是空的，直接申请数组
    // old_warp_index->data_type_of_tmp_result_write_index_arr = find_most_suitable_data_type(max_tmp_result_write_index);
    // old_warp_index->tmp_result_write_index_arr = malloc_arr(new_warp_tmp_result_write_index_vec.size(), old_warp_index->data_type_of_tmp_result_write_index_arr);
    // copy_unsigned_long_arr_to_others(&(new_warp_tmp_result_write_index_vec[0]), old_warp_index->tmp_result_write_index_arr, old_warp_index->data_type_of_tmp_result_write_index_arr, new_warp_tmp_result_write_index_vec.size());
}

// 所有的for循环的遍历范围都是包含下界，
void sep_thread_level_col_ell_with_padding(compressed_block_t *compressed_block, vector<unsigned long> block_index_arr, vector<unsigned long> thread_col_block_size)
{
    assert(compressed_block != NULL && block_index_arr.size() == thread_col_block_size.size());

    // 一系列新的数组，对于block级别的索引来说，需要修改的是coo_begin_index_arr
    vector<unsigned long> new_tblock_coo_begin_vec;
    // 是CSR格式的，最后的大小是block_num + 1
    new_tblock_coo_begin_vec.push_back(0);

    // 对于warp索引，coo_begin_index_arr，arr_index（用来标记每个warp内部的线程对应的块数量），coo_block_size_arr，child_tmp_row_csr_index_arr，begin_index_in_tmp_row_csr_arr_of_block
    // coo begin有一个最大值
    unsigned long max_warp_coo_begin_index = 0;
    vector<unsigned long> new_warp_coo_begin_index_vec;
    vector<unsigned long> new_warp_index_vec;
    // 从0开始算
    new_warp_index_vec.push_back(0);
    // 反映新的warp块的非零元数量
    unsigned long max_warp_coo_block_size = 0;
    vector<unsigned long> new_warp_coo_block_size_vec;
    // 子块中间结果的行边界
    unsigned long max_warp_child_tmp_row_csr_index = 0;
    vector<unsigned long> new_warp_child_tmp_row_csr_index_vec;
    vector<unsigned long> new_begin_index_in_tmp_row_csr_arr_of_block_vec;

    // 同一个block不同warp中间结果的位置和以前的一样，不用修改，warp中间结果写的位置只和一个warp中行的数量有关
    // vector<unsigned long> new_warp_tmp_result_begin_index_vec;

    // 对于新的数组来说，只有coo_block_size_arr是必要的，并且因为是ELL的方式，相同父块的每一个子块的大小是一样的，所以coo_block_size_arr的长度和父索引长度一致即可。当然对于最细粒度的块的行号的记录是必须的
    // 并且长度就是线程粒度的块的数量，剩下的元数据就不再需要了
    unsigned long max_thread_coo_block_size = 0;
    vector<unsigned long> thread_coo_block_size_vec;
    unsigned long max_thread_first_row_index = 0;
    vector<unsigned long> thread_first_row_index_vec;

    // 还需要两个新的col索引，数据类型的已有的保持一致
    vector<unsigned long> padding_col_index_vec;
    vector<unsigned long> staggered_padding_col_index_vec;

    // 还需要两个新的val索引，数据类型和未重排的列数组保持一致
    vector<double> padding_val_vec;
    vector<double> staggered_padding_val_vec;

    // 已有的一些索引
    assert(compressed_block->read_index.size() == 4);
    index_of_compress_block_t *old_global_row_index = compressed_block->read_index[0];
    index_of_compress_block_t *old_global_col_index = compressed_block->read_index[1];
    index_of_compress_block_t *old_tblock_index = compressed_block->read_index[2];
    index_of_compress_block_t *old_warp_index = compressed_block->read_index[3];

    assert(old_global_row_index != NULL && old_global_col_index != NULL && old_tblock_index != NULL && old_warp_index != NULL);
    assert(old_global_row_index->index_compressed_type == COO && old_global_col_index->index_compressed_type == COO && old_tblock_index->index_compressed_type == CSR && old_tblock_index->index_compressed_type == CSR);

    // 累计的总warp号
    unsigned long global_warp_block_index = 0;
    // 记录需要进一步分块的warp块的索引
    unsigned long cur_index_of_block_index_arr = 0;

    // 遍历所有的warp块，从block块开始遍历
    unsigned long global_tblock_index;
    for (global_tblock_index = 0; global_tblock_index < old_tblock_index->block_num; global_tblock_index++)
    {
        // cout << "begin scan all block:" << global_tblock_index << endl;
        // 当前tblock的warp范围，不包含下界
        assert(old_tblock_index->index_arr != NULL && global_tblock_index < old_tblock_index->block_num);
        unsigned long warp_begin_index = read_from_array_with_data_type(old_tblock_index->index_arr, old_tblock_index->index_data_type, global_tblock_index);
        unsigned long warp_end_index = read_from_array_with_data_type(old_tblock_index->index_arr, old_tblock_index->index_data_type, global_tblock_index + 1) - 1;

        // 当前tblock在coo索引中的起始位置，这里是包含上界和下界的
        assert(old_tblock_index->coo_begin_index_arr != NULL && global_tblock_index < old_tblock_index->block_num);
        unsigned long tblock_coo_begin_index = read_from_array_with_data_type(old_tblock_index->coo_begin_index_arr, old_tblock_index->data_type_of_coo_begin_index_arr, global_tblock_index);
        unsigned long tblock_coo_end_index = read_from_array_with_data_type(old_tblock_index->coo_begin_index_arr, old_tblock_index->data_type_of_coo_begin_index_arr, global_tblock_index + 1) - 1;

        // 用一个变量存储一个block所有非零元数量，用block内部每个warp的非零元数量相加
        unsigned long nnz_of_cur_tblock = 0;

        // 当前tblock的第一个行，行号
        unsigned long tblock_first_row_index = read_from_array_with_data_type(old_tblock_index->index_of_the_first_row_arr, old_tblock_index->data_type_of_index_of_the_first_row_arr, global_tblock_index);

        unsigned long local_warp_index;

        for (local_warp_index = warp_begin_index; local_warp_index <= warp_end_index; local_warp_index++)
        {
            // cout << "begin scan all warp:" << local_warp_index << ",warp_end_index:" << warp_end_index << endl;

            assert(local_warp_index == global_warp_block_index);
            // 先进行分块，然后将分块数量补到32的整数倍，将多出来的分块均匀分配到warp中的多个行
            assert(old_warp_index->row_number_of_block_arr != NULL && global_warp_block_index < old_warp_index->block_num);
            unsigned long warp_row_num = read_from_array_with_data_type(old_warp_index->row_number_of_block_arr, old_warp_index->data_type_of_row_number_of_block_arr, global_warp_block_index);
            // 块内临时缓冲，分别存储warp内部的新的padding列索引、新的交错列索引、新的padding值数组，新的交错值数组
            // 先用一个二维数组来存储每一行的数据
            vector<vector<unsigned long>> dim_2_inner_warp_padding_col_index_vec(warp_row_num);
            vector<vector<double>> dim_2_inner_warp_padding_val_vec(warp_row_num);

            // 每一行的数据拉平并整理
            vector<unsigned long> inner_warp_padding_col_index_vec;
            vector<unsigned long> inner_warp_padding_stagger_col_index_vec;
            vector<double> inner_warp_padding_val_vec;
            vector<double> inner_warp_padding_stagger_val_vec;

            // 用来存储线程中间结果的行边界
            vector<unsigned long> inner_warp_tmp_result_csr_vec;

            // 局部的coo索引
            // 这里说明要进一步列分块，遍历所有分非零元。获取局部块索引
            unsigned long local_warp_coo_begin = read_from_array_with_data_type(old_warp_index->coo_begin_index_arr, old_warp_index->data_type_of_coo_begin_index_arr, global_warp_block_index);

            unsigned long local_warp_coo_end = local_warp_coo_begin + read_from_array_with_data_type(old_warp_index->coo_block_size_arr, old_warp_index->data_type_of_coo_block_size_arr, global_warp_block_index) - 1;

            // if(local_warp_index == warp_end_index){
            //     cout << "local_warp_coo_size:" << read_from_array_with_data_type(old_warp_index->coo_block_size_arr, old_warp_index->data_type_of_coo_block_size_arr, global_warp_block_index) << ",local_warp_coo_end:" << local_warp_coo_end << ",tblock_coo_begin_index:" << tblock_coo_begin_index << endl;
            // }

            // warp全局的非零元范围
            unsigned long global_warp_coo_begin = local_warp_coo_begin + tblock_coo_begin_index;
            unsigned long global_warp_coo_end = local_warp_coo_end + tblock_coo_begin_index;

            // if(local_warp_index == warp_end_index){
            //     cout << "local_warp_coo_begin:" << local_warp_coo_begin << ",local_warp_coo_end:" << local_warp_coo_end << ",tblock_coo_begin_index:" << tblock_coo_begin_index << endl;
            // }

            unsigned long local_warp_first_row_index = read_from_array_with_data_type(old_warp_index->index_of_the_first_row_arr, old_warp_index->data_type_of_index_of_the_first_row_arr, global_warp_block_index);

            unsigned long global_warp_first_row_index = local_warp_first_row_index + tblock_first_row_index;
            // cout << "global_warp_first_row_index:" << global_warp_first_row_index << endl;

            // 判断是不是需要进一步列分块的
            // cout << cur_index_of_block_index_arr << "," << global_warp_block_index << endl;
            if (cur_index_of_block_index_arr < block_index_arr.size() && global_warp_block_index == block_index_arr[cur_index_of_block_index_arr])
            {
                // cout << "need col sep" << endl;

                compressed_block->share_row_with_other_thread = true;
                // 当前分块的长度
                unsigned long thread_block_size = thread_col_block_size[cur_index_of_block_index_arr];

                // 查看warp对应的coo索引中每一行按照thread分块之后块的数量，看看要补多少个块，最终让每一行有多少个块
                vector<unsigned long> thread_level_block_num_of_each_row = find_thread_block_num_of_each_line_after_padding_in_thread_and_warp_level(old_global_row_index,
                                                                                                                                                     global_warp_first_row_index, warp_row_num, global_warp_coo_begin, global_warp_coo_end, thread_block_size);

                // 用来判断当前线程对应的块是不是warp块中的第一个块
                bool is_first_thread_level_block_in_warp = true;

                // 空行要补的列号，和这个warp其他列的某一个列号相同
                // unsigned long col_index_need_to_be_padding = 0;

                // 遍历每一个非零元，进行padding，存到inner_warp_padding_col_index_vec和inner_warp_padding_var_vec中
                // thread_level_block_num_of_each_row可以推测出每一行的非零元数量
                unsigned long global_coo_index;
                for (global_coo_index = global_warp_coo_begin; global_coo_index <= global_warp_coo_end; global_coo_index++)
                {
                    // cout << "begin scan nz:" << global_coo_index << endl;
                    // 获取当前的全局行号列号和值
                    unsigned long global_cur_row_index = read_from_array_with_data_type(old_global_row_index->index_arr, old_global_row_index->index_data_type, global_coo_index);
                    unsigned long global_cur_col_index = read_from_array_with_data_type(old_global_col_index->index_arr, old_global_col_index->index_data_type, global_coo_index);

                    double global_val = read_double_from_array_with_data_type(compressed_block->val_arr, compressed_block->val_data_type, global_coo_index);
                    // 将全局行号换成局部行号，并且是warp内部的局部行号
                    unsigned long local_cur_row_index = global_cur_row_index - global_warp_first_row_index;

                    // 对应位置的拷贝
                    assert(local_cur_row_index >= 0 && local_cur_row_index < dim_2_inner_warp_padding_col_index_vec.size());
                    dim_2_inner_warp_padding_col_index_vec[local_cur_row_index].push_back(global_cur_col_index);
                    dim_2_inner_warp_padding_val_vec[local_cur_row_index].push_back(global_val);

                    // 检查是不是行末，在行末和块末的时候做几件事情：padding（影响列坐标和值数组）
                    if (global_coo_index == global_warp_coo_end || global_cur_row_index < read_from_array_with_data_type(old_global_row_index->index_arr, old_global_row_index->index_data_type, global_coo_index + 1))
                    {
                        // 作为warp中的第一个块，需要考虑的不那么多
                        // 首先集中进行padding工作，查看当前行实际需要的非零元数量
                        unsigned long target_nnz_of_this_row = thread_level_block_num_of_each_row[local_cur_row_index] * thread_block_size;
                        assert(dim_2_inner_warp_padding_col_index_vec[local_cur_row_index].size() == dim_2_inner_warp_padding_val_vec[local_cur_row_index].size());
                        assert(dim_2_inner_warp_padding_col_index_vec[local_cur_row_index].size() <= target_nnz_of_this_row);

                        unsigned long nz_need_to_be_padded = target_nnz_of_this_row - dim_2_inner_warp_padding_val_vec[local_cur_row_index].size();

                        unsigned long i;
                        for (i = 0; i < nz_need_to_be_padded; i++)
                        {
                            assert(dim_2_inner_warp_padding_val_vec[local_cur_row_index].size() > 0);
                            // 套用本行最后一列的列索引
                            // cout << dim_2_inner_warp_padding_col_index_vec[local_cur_row_index][dim_2_inner_warp_padding_col_index_vec[local_cur_row_index].size() - 1] << endl;
                            dim_2_inner_warp_padding_col_index_vec[local_cur_row_index].push_back(dim_2_inner_warp_padding_col_index_vec[local_cur_row_index][dim_2_inner_warp_padding_col_index_vec[local_cur_row_index].size() - 1]);
                            dim_2_inner_warp_padding_val_vec[local_cur_row_index].push_back(0);
                        }
                    }
                }

                // 对空行执行padding，对inner_warp_tmp_result_csr_vec执行对应的
                assert(thread_level_block_num_of_each_row.size() == warp_row_num);
                // 遍历每一行，对空行执行padding，对每一行的每一个线程粒度的块填充元数据
                unsigned long i;
                for (i = 0; i < warp_row_num; i++)
                {
                    // 当前行应该的块数量
                    unsigned long target_thread_level_block_num = thread_level_block_num_of_each_row[i];
                    // 当前行应该的非零元数量
                    unsigned long target_nnz_in_row = target_thread_level_block_num * thread_block_size;
                    // 当前行实际的非零元数量
                    unsigned long real_nnz_in_row = dim_2_inner_warp_padding_col_index_vec[i].size();

                    // 行实际非零元的数量要么是0，要么已经满了
                    assert(real_nnz_in_row == 0 || real_nnz_in_row == target_nnz_in_row);

                    // 将每一行补齐，但是补的东西最好和其他行补的东西相近，尽可能利用cache
                    unsigned long j;
                    for (j = real_nnz_in_row; j < target_nnz_in_row; j++)
                    {
                        dim_2_inner_warp_padding_col_index_vec[i].push_back(old_warp_index->min_col_index);
                        dim_2_inner_warp_padding_val_vec[i].push_back(0);
                    }

                    // 记录每一个线程块的第一个行行号，但是warp内部的相对行号
                    for (j = 0; j < target_thread_level_block_num; j++)
                    {
                        if (max_thread_first_row_index < i)
                        {
                            max_thread_first_row_index = i;
                        }

                        // if(thread_first_row_index_vec.size() > 0 && i == 0){
                        //     thread_first_row_index_vec.push_back(i);
                        // }

                        thread_first_row_index_vec.push_back(i);
                    }
                }

                // 更新thread的coo数量
                if (max_thread_coo_block_size < thread_block_size)
                {
                    max_thread_coo_block_size = thread_block_size;
                }
                thread_coo_block_size_vec.push_back(thread_block_size);

                // inner_warp_tmp_result_csr_vec修改
                // 记录每一行的交界位置，有数组已经记录了每行结果的数量
                inner_warp_tmp_result_csr_vec.push_back(0);

                for (i = 0; i < thread_level_block_num_of_each_row.size(); i++)
                {
                    // 每个线程的块产生一个结果
                    inner_warp_tmp_result_csr_vec.push_back(inner_warp_tmp_result_csr_vec[inner_warp_tmp_result_csr_vec.size() - 1] + thread_level_block_num_of_each_row[i]);
                }

                assert(inner_warp_tmp_result_csr_vec.size() == warp_row_num + 1);

                cur_index_of_block_index_arr++;
            }
            else
            {
                // cout << "one row one block" << endl;
                // 对于不需要进一步列分块的，直接按行分块，一行一个线程对应的块，所以一开始要确定thread size的大小
                unsigned long thread_block_size = find_max_row_nnz_in_coo_sub_block(old_global_row_index, global_warp_first_row_index, warp_row_num, global_warp_coo_begin, global_warp_coo_end);

                // cout << "thread_block_size:" << thread_block_size << endl;

                // if(local_warp_index == warp_end_index){
                //     cout << "finish get thread_block_size, global_warp_coo_begin:" << global_warp_coo_begin << ",global_warp_coo_end:" << global_warp_coo_end << endl;
                // }
                // 查看warp对应的coo索引中每一行按照thread分块之后块的数量，看看要补多少个块，最终让每一行有多少个块，
                vector<unsigned long> thread_level_block_num_of_each_row = find_thread_block_num_of_each_line_after_padding_in_thread_and_warp_level(old_global_row_index,
                                                                                                                                                     global_warp_first_row_index, warp_row_num, global_warp_coo_begin, global_warp_coo_end, thread_block_size);

                // 遍历每一行的线程数量，如果有一行超过两个线程，说明行被线程共享了
                for (int thread_level_block_num_of_each_row_index = 0; thread_level_block_num_of_each_row_index < thread_level_block_num_of_each_row.size(); thread_level_block_num_of_each_row_index++)
                {
                    assert(thread_level_block_num_of_each_row[thread_level_block_num_of_each_row_index] > 0);
                    if (thread_level_block_num_of_each_row[thread_level_block_num_of_each_row_index] >= 2)
                    {
                        // 存在行共享
                        compressed_block->share_row_with_other_thread = true;
                        break;
                    }
                }
                // if(local_warp_index == warp_end_index){
                //     cout << "finish sort" << endl;
                // }
                // 不见得一行只有一个块，剩下的一样处理
                // 用来判断当前线程对应的块是不是warp块中的第一个块
                bool is_first_thread_level_block_in_warp = true;

                // 遍历每一个非零元，进行padding，存到inner_warp_padding_col_index_vec和inner_warp_padding_var_vec中
                // thread_level_block_num_of_each_row可以推测出每一行的非零元数量
                unsigned long global_coo_index;
                for (global_coo_index = global_warp_coo_begin; global_coo_index <= global_warp_coo_end; global_coo_index++)
                {
                    // cout << "begin scan nz:" << global_coo_index << endl;
                    // 获取当前的全局行号列号和值
                    unsigned long global_cur_row_index = read_from_array_with_data_type(old_global_row_index->index_arr, old_global_row_index->index_data_type, global_coo_index);
                    unsigned long global_cur_col_index = read_from_array_with_data_type(old_global_col_index->index_arr, old_global_col_index->index_data_type, global_coo_index);

                    double global_val = read_double_from_array_with_data_type(compressed_block->val_arr, compressed_block->val_data_type, global_coo_index);
                    // 将全局行号换成局部行号
                    unsigned long local_cur_row_index = global_cur_row_index - global_warp_first_row_index;

                    // 对应位置的拷贝
                    // cout << "local_cur_row_index:" << local_cur_row_index << ",dim_2_inner_warp_padding_col_index_vec.size():" << dim_2_inner_warp_padding_col_index_vec.size() << endl;
                    assert(local_cur_row_index >= 0 && local_cur_row_index < dim_2_inner_warp_padding_col_index_vec.size());
                    dim_2_inner_warp_padding_col_index_vec[local_cur_row_index].push_back(global_cur_col_index);
                    dim_2_inner_warp_padding_val_vec[local_cur_row_index].push_back(global_val);

                    // 检查是不是行末，在行末和块末的时候做几件事情：padding（影响列坐标和值数组）
                    if (global_coo_index == global_warp_coo_end || global_cur_row_index < read_from_array_with_data_type(old_global_row_index->index_arr, old_global_row_index->index_data_type, global_coo_index + 1))
                    {
                        // cout << "end of row or block" << endl;
                        // 作为warp中的第一个块，需要考虑的不那么多
                        // 首先集中进行padding工作，查看当前行实际需要的非零元数量
                        unsigned long target_nnz_of_this_row = thread_level_block_num_of_each_row[local_cur_row_index] * thread_block_size;
                        assert(dim_2_inner_warp_padding_col_index_vec[local_cur_row_index].size() == dim_2_inner_warp_padding_val_vec[local_cur_row_index].size());
                        assert(dim_2_inner_warp_padding_col_index_vec[local_cur_row_index].size() <= target_nnz_of_this_row);

                        unsigned long nz_need_to_be_padded = target_nnz_of_this_row - dim_2_inner_warp_padding_val_vec[local_cur_row_index].size();

                        unsigned long i;
                        for (i = 0; i < nz_need_to_be_padded; i++)
                        {
                            assert(dim_2_inner_warp_padding_val_vec[local_cur_row_index].size() > 0);
                            // 套用本行最后一列的列索引
                            // cout << dim_2_inner_warp_padding_val_vec[local_cur_row_index][dim_2_inner_warp_padding_val_vec[local_cur_row_index].size() - 1] << endl;
                            dim_2_inner_warp_padding_col_index_vec[local_cur_row_index].push_back(dim_2_inner_warp_padding_col_index_vec[local_cur_row_index][dim_2_inner_warp_padding_col_index_vec[local_cur_row_index].size() - 1]);
                            dim_2_inner_warp_padding_val_vec[local_cur_row_index].push_back(0);
                        }
                    }
                }

                // 对空行执行padding，对inner_warp_tmp_result_csr_vec执行对应的
                assert(thread_level_block_num_of_each_row.size() == warp_row_num);
                // 遍历每一行，对空行执行padding，对每一行的每一个线程粒度的块填充元数据
                unsigned long i;
                for (i = 0; i < warp_row_num; i++)
                {
                    // 当前行应该的块数量
                    unsigned long target_thread_level_block_num = thread_level_block_num_of_each_row[i];
                    // 当前行应该的非零元数量
                    unsigned long target_nnz_in_row = target_thread_level_block_num * thread_block_size;
                    // 当前行实际的非零元数量
                    unsigned long real_nnz_in_row = dim_2_inner_warp_padding_col_index_vec[i].size();

                    // 行实际非零元的数量要么是0，要么已经满了，
                    // cout << "real_nnz_in_row:" << real_nnz_in_row << ",target_nnz_in_row:" << target_nnz_in_row <<endl;
                    assert(real_nnz_in_row == 0 || real_nnz_in_row == target_nnz_in_row);

                    // 将每一行补齐，但是补的东西最好和其他行补的东西相近，尽可能利用cache
                    unsigned long j;
                    for (j = real_nnz_in_row; j < target_nnz_in_row; j++)
                    {
                        // padding的内容要符合逻辑，padding当前父块的最小列号
                        dim_2_inner_warp_padding_col_index_vec[i].push_back(old_warp_index->min_col_index);
                        dim_2_inner_warp_padding_val_vec[i].push_back(0);
                    }

                    // 记录每一个线程块的第一个行行号，但是warp内部的相对行号
                    for (j = 0; j < target_thread_level_block_num; j++)
                    {
                        if (max_thread_first_row_index < i)
                        {
                            max_thread_first_row_index = i;
                        }

                        thread_first_row_index_vec.push_back(i);
                    }
                }

                // 更新thread的coo数量
                if (max_thread_coo_block_size < thread_block_size)
                {
                    max_thread_coo_block_size = thread_block_size;
                }
                thread_coo_block_size_vec.push_back(thread_block_size);

                // inner_warp_tmp_result_csr_vec修改
                // 记录每一行的交界位置，有数组已经记录了每行结果的数量
                inner_warp_tmp_result_csr_vec.push_back(0);

                for (i = 0; i < thread_level_block_num_of_each_row.size(); i++)
                {
                    // 每个线程的块产生一个结果
                    inner_warp_tmp_result_csr_vec.push_back(inner_warp_tmp_result_csr_vec[inner_warp_tmp_result_csr_vec.size() - 1] + thread_level_block_num_of_each_row[i]);
                }

                // if(warp_row_num == 46){
                //     cout << "here, thread_level_block_num_of_each_row.size()" << thread_level_block_num_of_each_row.size() << endl;
                //     exit(-1);
                // }

                assert(inner_warp_tmp_result_csr_vec.size() == warp_row_num + 1);
            }

            // 修改warp的一些元数据
            // 将分行的数据拉直
            unsigned long i;
            assert(inner_warp_padding_col_index_vec.size() == 0);
            for (i = 0; i < dim_2_inner_warp_padding_col_index_vec.size(); i++)
            {
                unsigned long j;
                assert(dim_2_inner_warp_padding_col_index_vec[i].size() % thread_coo_block_size_vec[thread_coo_block_size_vec.size() - 1] == 0);
                for (j = 0; j < dim_2_inner_warp_padding_col_index_vec[i].size(); j++)
                {
                    inner_warp_padding_col_index_vec.push_back(dim_2_inner_warp_padding_col_index_vec[i][j]);
                    inner_warp_padding_val_vec.push_back(dim_2_inner_warp_padding_val_vec[i][j]);
                }
            }

            assert(inner_warp_padding_col_index_vec.size() > 0);

            // 累加一个warp内的非零元
            nnz_of_cur_tblock = nnz_of_cur_tblock + inner_warp_padding_col_index_vec.size();

            // 两个新的数组的大小一样，并且是32的倍数
            assert(inner_warp_padding_col_index_vec.size() == inner_warp_padding_val_vec.size());
            // thread_coo_block_size_vec的最后一位是当前warp的所有
            assert(inner_warp_padding_col_index_vec.size() % 32 == 0 && inner_warp_padding_col_index_vec.size() % (32 * thread_coo_block_size_vec[thread_coo_block_size_vec.size() - 1]) == 0);

            // warp在coo的起始位置，对于一个块中的第一个warp来说，coo的相对其实位置是0
            if (local_warp_index == warp_begin_index)
            {
                new_warp_coo_begin_index_vec.push_back(0);
            }
            else
            {
                unsigned long new_warp_coo_begin_index = new_warp_coo_begin_index_vec[new_warp_coo_begin_index_vec.size() - 1] + new_warp_coo_block_size_vec[new_warp_coo_block_size_vec.size() - 1];
                if (max_warp_coo_begin_index < new_warp_coo_begin_index)
                {
                    max_warp_coo_begin_index = new_warp_coo_begin_index;
                }
                new_warp_coo_begin_index_vec.push_back(new_warp_coo_begin_index);
            }

            // 用新的局部非零元数组初始化coo size
            if (max_warp_coo_block_size < inner_warp_padding_col_index_vec.size())
            {
                max_warp_coo_block_size = inner_warp_padding_col_index_vec.size();
            }
            new_warp_coo_block_size_vec.push_back(inner_warp_padding_col_index_vec.size());

            // 用现成粒度的块的数量更新warp层次的index_arr，index_arr是全局的偏移量
            assert(inner_warp_padding_col_index_vec.size() % thread_coo_block_size_vec[thread_coo_block_size_vec.size() - 1] == 0);
            assert((inner_warp_padding_col_index_vec.size() / (thread_coo_block_size_vec[thread_coo_block_size_vec.size() - 1])) % 32 == 0);
            // if(new_warp_index_vec[new_warp_index_vec.size() - 1] + (inner_warp_padding_col_index_vec.size() / (thread_coo_block_size_vec[thread_coo_block_size_vec.size() - 1])) == 16777472)
            // {
            //     // 相关变量全部打印
            //     cout << "new_warp_index_vec.size():" << new_warp_index_vec.size() << endl;
            //     cout << "new_warp_index_vec[new_warp_index_vec.size() - 1]:" << new_warp_index_vec[new_warp_index_vec.size() - 1] << endl;
            //     cout << "(inner_warp_padding_col_index_vec.size() / (thread_coo_block_size_vec[thread_coo_block_size_vec.size() - 1]):" << inner_warp_padding_col_index_vec.size() / (thread_coo_block_size_vec[thread_coo_block_size_vec.size() - 1]) << endl;
            //     // cout <<
            //     assert(false);
            // }
            new_warp_index_vec.push_back(new_warp_index_vec[new_warp_index_vec.size() - 1] + (inner_warp_padding_col_index_vec.size() / (thread_coo_block_size_vec[thread_coo_block_size_vec.size() - 1])));

            // tmp_result_write_index_arr本质上是warp在归约之后放自己每一行中间结果的头部位置
            // new_begin_index_in_tmp_row_csr_arr_of_block_vec是每个warp查看自己归约索引
            // new_warp_child_tmp_row_csr_index_vec的拷贝，中间结果行边界数组的偏移量
            // new_begin_index_in_tmp_row_csr_arr_of_block_vec.push_back(new_warp_child_tmp_row_csr_index_vec.size());

            // 拷贝归约索引的偏移量
            // 最终new_warp_child_tmp_row_csr_index_vec数组的大小是warp中行的数量只和加上warp块的数量
            for (i = 0; i < inner_warp_tmp_result_csr_vec.size(); i++)
            {
                // inner_warp_tmp_result_csr_vec是一个warp内部的规约索引
                // new_warp_child_tmp_row_csr_index_vec.push_back(inner_warp_tmp_result_csr_vec[i]);
            }

            // 将padding之后的东西在进一步交错存储，交错存储，交错存储以32个块为一组，分别交错，也就是说，32n号块的第一个非零元所处的位置是没有变化的，可以做一个检查
            // 按照组的方式分有一个好处，保证每个warp的连续读取显存上的内容
            unsigned long stagger_group_index;
            // 每32个线程对应的块一个组，除一下得到组的数量
            unsigned long stagger_group_num = inner_warp_padding_col_index_vec.size() / (thread_coo_block_size_vec[thread_coo_block_size_vec.size() - 1] * 32);
            assert(inner_warp_padding_col_index_vec.size() % (thread_coo_block_size_vec[thread_coo_block_size_vec.size() - 1] * 32) == 0);
            for (stagger_group_index = 0; stagger_group_index < stagger_group_num; stagger_group_index++)
            {
                // 一个组内部的非零元穿插，每个组有thread_coo_block_size_vec[thread_coo_block_size_vec.size() - 1] * 32和元素
                unsigned long stagger_warp_group_index;
                for (stagger_warp_group_index = 0; stagger_warp_group_index < thread_coo_block_size_vec[thread_coo_block_size_vec.size() - 1]; stagger_warp_group_index++)
                {
                    unsigned long nz_index_inner_warp_group;
                    for (nz_index_inner_warp_group = 0; nz_index_inner_warp_group < 32; nz_index_inner_warp_group++)
                    {
                        // 根据之前的索引从仅padding的数组中取数据
                        // 需要取的数据
                        unsigned long cor_index = thread_coo_block_size_vec[thread_coo_block_size_vec.size() - 1] * nz_index_inner_warp_group + stagger_warp_group_index + stagger_group_index * 32 * thread_coo_block_size_vec[thread_coo_block_size_vec.size() - 1];
                        inner_warp_padding_stagger_col_index_vec.push_back(inner_warp_padding_col_index_vec[cor_index]);
                        inner_warp_padding_stagger_val_vec.push_back(inner_warp_padding_val_vec[cor_index]);
                    }
                }
            }

            assert(inner_warp_padding_stagger_col_index_vec.size() == inner_warp_padding_col_index_vec.size() && inner_warp_padding_stagger_val_vec.size() == inner_warp_padding_val_vec.size());

            // 每个交错组的一个非零元是一致的
            for (stagger_group_index = 0; stagger_group_index < stagger_group_num; stagger_group_index++)
            {
                // 计算每一个stagger_group的第一个非零元的索引
                unsigned long first_ele_index_of_stagger_group = stagger_group_index * 32 * thread_coo_block_size_vec[thread_coo_block_size_vec.size() - 1];
                assert(inner_warp_padding_stagger_col_index_vec[first_ele_index_of_stagger_group] == inner_warp_padding_col_index_vec[first_ele_index_of_stagger_group]);
                assert(inner_warp_padding_stagger_val_vec[first_ele_index_of_stagger_group] == inner_warp_padding_val_vec[first_ele_index_of_stagger_group]);
            }

            // 将col和val的局部缓存拷贝给全局，一共四个数组要考虑
            for (i = 0; i < inner_warp_padding_stagger_col_index_vec.size(); i++)
            {
                padding_col_index_vec.push_back(inner_warp_padding_col_index_vec[i]);
                padding_val_vec.push_back(inner_warp_padding_val_vec[i]);
                staggered_padding_col_index_vec.push_back(inner_warp_padding_stagger_col_index_vec[i]);
                staggered_padding_val_vec.push_back(inner_warp_padding_stagger_val_vec[i]);
            }

            global_warp_block_index++;
        }

        // 修改new_tblock_coo_begin_vec，第一个块是0，之后每次新的列索引和值数组的大小，就是coo
        new_tblock_coo_begin_vec.push_back(new_tblock_coo_begin_vec[new_tblock_coo_begin_vec.size() - 1] + nnz_of_cur_tblock);
    }

    // cout << 1234 << endl;

    // 检查几个数组的大小
    assert(new_tblock_coo_begin_vec.size() == old_tblock_index->length && new_warp_index_vec.size() == old_warp_index->length);
    assert(padding_col_index_vec.size() == padding_val_vec.size() && padding_val_vec.size() == staggered_padding_col_index_vec.size() && staggered_padding_col_index_vec.size() == staggered_padding_val_vec.size());
    assert(thread_coo_block_size_vec.size() == old_warp_index->block_num && new_warp_coo_block_size_vec.size() == old_warp_index->block_num);

    // 计算每个warp的行数量只和和warp块的数量只和
    unsigned long row_num_sum_of_each_row = 0;
    unsigned long i;

    // cout << "old_warp_index->block_num:" << old_warp_index->block_num << endl;
    // for (i = 0; i < old_warp_index->block_num; i++)
    // {
    //     // cout << i << endl;
    //     row_num_sum_of_each_row = row_num_sum_of_each_row + read_from_array_with_data_type(old_warp_index->row_number_of_block_arr, old_warp_index->data_type_of_row_number_of_block_arr, i);
    // }
    // assert(new_warp_child_tmp_row_csr_index_vec.size() == row_num_sum_of_each_row + old_warp_index->block_num);
    assert(new_begin_index_in_tmp_row_csr_arr_of_block_vec.size() == 0 && new_warp_child_tmp_row_csr_index_vec.size() == 0);

    // 修改线程块层次的元数据，修改coo begin
    assert(old_tblock_index->coo_begin_index_arr != NULL && old_tblock_index->length == new_tblock_coo_begin_vec.size());
    data_type new_data_type_of_tblock_coo_begin_index_arr = find_most_suitable_data_type(new_tblock_coo_begin_vec[new_tblock_coo_begin_vec.size() - 1]);
    if (new_data_type_of_tblock_coo_begin_index_arr != old_tblock_index->data_type_of_coo_begin_index_arr)
    {
        // 删除原来的数组，创建新数组
        delete_arr_with_data_type(old_tblock_index->coo_begin_index_arr, old_tblock_index->data_type_of_coo_begin_index_arr);
        old_tblock_index->coo_begin_index_arr = NULL;
        old_tblock_index->data_type_of_coo_begin_index_arr = new_data_type_of_tblock_coo_begin_index_arr;
        old_tblock_index->coo_begin_index_arr = malloc_arr(old_tblock_index->length, old_tblock_index->data_type_of_coo_begin_index_arr);
    }

    copy_unsigned_long_arr_to_others(&(new_tblock_coo_begin_vec[0]), old_tblock_index->coo_begin_index_arr, old_tblock_index->data_type_of_coo_begin_index_arr, old_tblock_index->length);

    // 修改warp层次的几个索引，new_warp_coo_begin_index_vec，new_warp_index_vec，new_warp_tmp_result_begin_index_vec，new_warp_child_tmp_row_csr_index_vec，new_warp_coo_block_size_vec
    assert(old_warp_index->coo_begin_index_arr != NULL && old_warp_index->block_num == new_warp_coo_begin_index_vec.size());
    // 根据数据类型的变化， 申请新的数组
    data_type new_data_type_of_coo_begin_index_arr = find_most_suitable_data_type(max_warp_coo_begin_index);
    if (new_data_type_of_coo_begin_index_arr != old_warp_index->data_type_of_coo_begin_index_arr)
    {
        // 删除原来的数组，创建新数组
        delete_arr_with_data_type(old_warp_index->coo_begin_index_arr, old_warp_index->data_type_of_coo_begin_index_arr);
        old_warp_index->coo_begin_index_arr = NULL;
        old_warp_index->data_type_of_coo_begin_index_arr = new_data_type_of_coo_begin_index_arr;
        old_warp_index->coo_begin_index_arr = malloc_arr(old_warp_index->block_num, old_warp_index->data_type_of_coo_begin_index_arr);
    }
    // old_warp_index->data_type_of_coo_begin_index_arr = find_most_suitable_data_type(max_warp_coo_begin_index);
    copy_unsigned_long_arr_to_others(&(new_warp_coo_begin_index_vec[0]), old_warp_index->coo_begin_index_arr, old_warp_index->data_type_of_coo_begin_index_arr, old_warp_index->block_num);

    // unsigned int* new_ptr = (unsigned int*)malloc(old_warp_index->length * sizeof(unsigned int));

    // new_ptr[0] = 123;

    // new_warp_index_vec
    assert(old_warp_index->index_arr == NULL);

    // print_arr_to_file_with_data_type(&(new_warp_index_vec[0]), UNSIGNED_LONG, new_warp_index_vec.size(), "/home/duzhen/spmv_builder/data_source/test0-7.log");

    // cout << new_warp_index_vec[0] << endl;

    old_warp_index->index_data_type = find_most_suitable_data_type(new_warp_index_vec[new_warp_index_vec.size() - 1]);
    old_warp_index->index_arr = malloc_arr(old_warp_index->length, old_warp_index->index_data_type);
    copy_unsigned_long_arr_to_others(&(new_warp_index_vec[0]), old_warp_index->index_arr, old_warp_index->index_data_type, old_warp_index->length);

    // cout << read_from_array_with_data_type(old_warp_index->index_arr, old_warp_index->index_data_type, 0) << endl;

    // cout << old_warp_index->length << endl;
    // cout << old_warp_index->block_num << endl;

    // print_arr_to_file_with_data_type(old_warp_index->index_arr, old_warp_index->index_data_type, old_warp_index->length, "/home/duzhen/spmv_builder/data_source/test0-6.log");

    // new_begin_index_in_tmp_row_csr_arr_of_block_vec
    // assert(old_warp_index->begin_index_in_tmp_row_csr_arr_of_block == NULL);
    // old_warp_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block = find_most_suitable_data_type(new_begin_index_in_tmp_row_csr_arr_of_block_vec[new_begin_index_in_tmp_row_csr_arr_of_block_vec.size() - 1]);
    // old_warp_index->begin_index_in_tmp_row_csr_arr_of_block = malloc_arr(old_warp_index->block_num, old_warp_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block);
    // copy_unsigned_long_arr_to_others(&(new_begin_index_in_tmp_row_csr_arr_of_block_vec[0]), old_warp_index->begin_index_in_tmp_row_csr_arr_of_block, old_warp_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block, old_warp_index->block_num);

    // cout << read_from_array_with_data_type(old_warp_index->index_arr, old_warp_index->index_data_type, 0) << endl;

    // new_warp_coo_block_size_vec
    assert(old_warp_index->coo_block_size_arr != NULL && new_warp_coo_block_size_vec.size() == old_warp_index->block_num);
    // 可能数据类型会发生变化，所以要重新申请数组
    data_type new_data_type = find_most_suitable_data_type(max_warp_coo_block_size);
    if (new_data_type != old_warp_index->data_type_of_coo_block_size_arr)
    {
        // 删除原来的数组，申请新的数组
        delete_arr_with_data_type(old_warp_index->coo_block_size_arr, old_warp_index->data_type_of_coo_block_size_arr);
        old_warp_index->coo_block_size_arr = NULL;
        old_warp_index->coo_block_size_arr = malloc_arr(old_warp_index->block_num, new_data_type);
        old_warp_index->data_type_of_coo_block_size_arr = new_data_type;
    }

    copy_unsigned_long_arr_to_others(&(new_warp_coo_block_size_vec[0]), old_warp_index->coo_block_size_arr, old_warp_index->data_type_of_coo_block_size_arr, old_warp_index->block_num);

    // cout << read_from_array_with_data_type(old_warp_index->index_arr, old_warp_index->index_data_type, 0) << endl;

    // new_warp_child_tmp_row_csr_index_vec
    // assert(old_warp_index->child_tmp_row_csr_index_arr == NULL);
    // old_warp_index->size_of_child_tmp_row_csr_index = new_warp_child_tmp_row_csr_index_vec.size();
    // old_warp_index->data_type_of_child_tmp_row_csr_index = find_most_suitable_data_type(max_warp_child_tmp_row_csr_index);
    // old_warp_index->child_tmp_row_csr_index_arr = malloc_arr(old_warp_index->size_of_child_tmp_row_csr_index, old_warp_index->data_type_of_child_tmp_row_csr_index);
    // // 这里没拷贝？？？？？我TM吐了
    // copy_unsigned_long_arr_to_others(&(new_warp_child_tmp_row_csr_index_vec[0]), old_warp_index->child_tmp_row_csr_index_arr, old_warp_index->data_type_of_child_tmp_row_csr_index, old_warp_index->size_of_child_tmp_row_csr_index);

    // print_arr_to_file_with_data_type

    // 修改thread粒度的块的几个索引
    index_of_compress_block_t *thread_level_block_index = new index_of_compress_block_t();
    thread_level_block_index->level_of_this_index = THREAD_LEVEL;
    thread_level_block_index->index_compressed_type = ELL;
    thread_level_block_index->block_num = thread_first_row_index_vec.size();
    thread_level_block_index->type_of_index = BLOCK_INDEX;

    // thread块的行首地址
    // cout << "max_thread_first_row_index:" << max_thread_first_row_index << endl;
    thread_level_block_index->data_type_of_index_of_the_first_row_arr = find_most_suitable_data_type(max_thread_first_row_index + 1);
    thread_level_block_index->index_of_the_first_row_arr = malloc_arr(thread_first_row_index_vec.size(), thread_level_block_index->data_type_of_index_of_the_first_row_arr);
    copy_unsigned_long_arr_to_others(&(thread_first_row_index_vec[0]), thread_level_block_index->index_of_the_first_row_arr, thread_level_block_index->data_type_of_index_of_the_first_row_arr, thread_level_block_index->block_num);

    // thread写中间结果的位置，只要根据一个warp对应的块内部的thread索引号，就可以了
    thread_level_block_index->max_row_index = old_warp_index->max_row_index;
    thread_level_block_index->min_row_index = old_warp_index->min_row_index;
    thread_level_block_index->max_col_index = old_warp_index->max_col_index;
    thread_level_block_index->min_col_index = old_warp_index->min_col_index;

    // 用一个和warp数量相同的数组来存储块的大小，这里每个warp中thread的大小相同
    thread_level_block_index->data_type_of_coo_block_size_arr = find_most_suitable_data_type(max_thread_coo_block_size);
    thread_level_block_index->coo_block_size_arr = malloc_arr(old_warp_index->block_num, thread_level_block_index->data_type_of_coo_block_size_arr);
    copy_unsigned_long_arr_to_others(&(thread_coo_block_size_vec[0]), thread_level_block_index->coo_block_size_arr, thread_level_block_index->data_type_of_coo_block_size_arr, old_warp_index->block_num);

    // 将新的索引记录在数组中
    compressed_block->read_index.push_back(thread_level_block_index);

    // 还有新的col索引，各两个
    index_of_compress_block_t *new_padding_col_arr = new index_of_compress_block_t();
    new_padding_col_arr->level_of_this_index = OTHERS;
    new_padding_col_arr->index_compressed_type = COO;
    new_padding_col_arr->length = padding_col_index_vec.size();
    new_padding_col_arr->index_data_type = old_global_col_index->index_data_type;
    new_padding_col_arr->type_of_index = COL_INDEX;

    new_padding_col_arr->max_row_index = old_global_col_index->max_row_index;
    new_padding_col_arr->min_row_index = old_global_col_index->min_row_index;
    new_padding_col_arr->max_col_index = old_global_col_index->max_col_index;
    new_padding_col_arr->min_col_index = old_global_col_index->min_col_index;

    new_padding_col_arr->index_arr = malloc_arr(new_padding_col_arr->length, new_padding_col_arr->index_data_type);
    copy_unsigned_long_arr_to_others(&(padding_col_index_vec[0]), new_padding_col_arr->index_arr, new_padding_col_arr->index_data_type, new_padding_col_arr->length);

    // 将索引记录
    compressed_block->read_index.push_back(new_padding_col_arr);

    index_of_compress_block_t *new_padding_stagger_col_arr = new index_of_compress_block_t();
    new_padding_stagger_col_arr->level_of_this_index = OTHERS;
    new_padding_stagger_col_arr->index_compressed_type = COO;
    new_padding_stagger_col_arr->length = staggered_padding_col_index_vec.size();
    new_padding_stagger_col_arr->index_data_type = old_global_col_index->index_data_type;
    new_padding_stagger_col_arr->type_of_index = COL_INDEX;

    new_padding_stagger_col_arr->max_row_index = old_global_col_index->max_row_index;
    new_padding_stagger_col_arr->min_row_index = old_global_col_index->min_row_index;
    new_padding_stagger_col_arr->max_col_index = old_global_col_index->max_col_index;
    new_padding_stagger_col_arr->min_col_index = old_global_col_index->min_col_index;

    new_padding_stagger_col_arr->index_arr = malloc_arr(staggered_padding_col_index_vec.size(), new_padding_stagger_col_arr->index_data_type);
    copy_unsigned_long_arr_to_others(&(staggered_padding_col_index_vec[0]), new_padding_stagger_col_arr->index_arr, new_padding_stagger_col_arr->index_data_type, new_padding_stagger_col_arr->length);

    compressed_block->read_index.push_back(new_padding_stagger_col_arr);

    // 将padding的val处理进来
    assert(compressed_block->padding_val_arr == NULL);
    compressed_block->padding_arr_size = padding_val_vec.size();
    compressed_block->padding_val_arr = malloc_arr(compressed_block->padding_arr_size, compressed_block->val_data_type);
    copy_double_arr_to_others(&(padding_val_vec[0]), compressed_block->padding_val_arr, compressed_block->val_data_type, compressed_block->padding_arr_size);

    // 将stagger的val存起来
    assert(compressed_block->staggered_padding_val_arr == NULL);
    compressed_block->staggered_padding_val_arr_size = staggered_padding_val_vec.size();
    compressed_block->staggered_padding_val_arr = malloc_arr(compressed_block->staggered_padding_val_arr_size, compressed_block->val_data_type);
    copy_double_arr_to_others(&(staggered_padding_val_vec[0]), compressed_block->staggered_padding_val_arr, compressed_block->val_data_type, compressed_block->staggered_padding_val_arr_size);

    assert(compressed_block->read_index.size() == 7);

    assert(compressed_block->read_index[6]->type_of_index == COL_INDEX);
}

// 执行按照非零元的分块，在TLB级别的分块之前，需要保证BLB和WLB的分块都被放弃了。用来支持CSR5
// 不做padding，和交错存储。所以尾部会空出来
// 产生的元数据感觉没啥用，其实应该把所有的元数据准备全部放到模板那一层才是正道
void sep_thread_level_acc_to_nnz(compressed_block_t *compressed_block, unsigned long thread_level_block_size)
{
    assert(compressed_block != NULL && thread_level_block_size > 0);

    // 保证之前已经有对应的索引
    assert(compressed_block->read_index.size() == 4);
    index_of_compress_block_t *old_global_row_index = compressed_block->read_index[0];
    index_of_compress_block_t *old_global_col_index = compressed_block->read_index[1];
    index_of_compress_block_t *old_tblock_index = compressed_block->read_index[2];
    index_of_compress_block_t *old_warp_index = compressed_block->read_index[3];

    assert(old_global_row_index != NULL && old_global_col_index != NULL && old_tblock_index != NULL && old_warp_index != NULL);
    assert(old_global_row_index->index_compressed_type == COO && old_global_col_index->index_compressed_type == COO && old_tblock_index->index_compressed_type == CSR && old_tblock_index->index_compressed_type == CSR);

    assert(old_global_col_index->length > 0);
    // 这里查看前面两层是不是放弃分块了
    if (old_global_col_index->length != 1)
    {
        assert(old_tblock_index->block_num == 1);
        assert(old_warp_index->block_num == 1);
    }

    // 执行分块，这类工作先不做padding和交错存储，但是为了保证逻辑的统一性，需要给所有的padding和交错存储产生的数据赋一个没有padding和交错存储之前的指针。
    // 具体要怎么处理分块之后的数据，交给模板。

    // 最大的TLB行数量，用来确定数据类型
    unsigned long max_row_num_in_TLB = 0;

    // 遍历所有的行号，获取首行的索引。thread_level_block_index需要的是首行索引和线程粒度块的大小，以及每个线程所占的行的数量。
    vector<unsigned long> thread_level_first_row_index_vec;
    vector<unsigned long> thread_level_block_size_vec;
    vector<unsigned long> thread_level_row_num_vec;

    thread_level_block_size_vec.push_back(thread_level_block_size);

    // 遍历所有的非零元，找出所有线程的首行和线程所包含的行号
    assert(compressed_block->size == old_global_row_index->length && old_global_row_index->length == old_global_col_index->length);
    
    unsigned long nnz = compressed_block->size;

    // 找出TLB数量，找出首行行号和TLB所占用的行数量
    unsigned long TLB_num;

    if (nnz % thread_level_block_size == 0)
    {
        TLB_num = nnz / thread_level_block_size;
    }
    else
    {
        TLB_num = nnz / thread_level_block_size + 1;
    }

    // 遍历所有的TLB，找出其对应的首行索引和行数量
    for (unsigned long TLB_id = 0; TLB_id < TLB_num; TLB_id++)
    {
        // TLB的第一个非零元索引
        unsigned long TLB_first_nz_index = TLB_id * thread_level_block_size;
        assert(TLB_first_nz_index < nnz);

        // 最后一个非零元索引
        unsigned long TLB_last_nz_index = TLB_first_nz_index + thread_level_block_size - 1;

        if (TLB_last_nz_index >= nnz)
        {
            TLB_last_nz_index = nnz - 1;
        }

        assert(TLB_last_nz_index >= TLB_first_nz_index);

        // 找到第一个非零元行索引
        unsigned long row_index_of_first_TLB_nz = read_from_array_with_data_type(old_global_row_index->index_arr, old_global_row_index->index_data_type, TLB_first_nz_index);
        unsigned long row_index_of_last_TLB_nz = read_from_array_with_data_type(old_global_row_index->index_arr, old_global_row_index->index_data_type, TLB_last_nz_index);

        assert(row_index_of_first_TLB_nz <= row_index_of_last_TLB_nz);

        // 行的数量
        unsigned long TLB_row_num = row_index_of_last_TLB_nz - row_index_of_first_TLB_nz + 1;

        // 将两个元数据记录到数组中
        thread_level_first_row_index_vec.push_back(row_index_of_first_TLB_nz);

        if (max_row_num_in_TLB < TLB_row_num)
        {
            max_row_num_in_TLB = TLB_row_num;
        }

        thread_level_row_num_vec.push_back(TLB_row_num);
    }

    // 修改thread粒度的块的几个索引
    index_of_compress_block_t *thread_level_block_index = new index_of_compress_block_t();
    thread_level_block_index->level_of_this_index = THREAD_LEVEL;
    thread_level_block_index->index_compressed_type = NO_INDEX;
    thread_level_block_index->block_num = thread_level_first_row_index_vec.size();
    thread_level_block_index->type_of_index = BLOCK_INDEX;
    
    // 首行索引
    // 数据类型和行索引用同一个
    thread_level_block_index->data_type_of_index_of_the_first_row_arr = old_global_row_index->index_data_type;
    thread_level_block_index->index_of_the_first_row_arr = malloc_arr(thread_level_first_row_index_vec.size(), thread_level_block_index->data_type_of_index_of_the_first_row_arr);
    copy_unsigned_long_arr_to_others(&(thread_level_first_row_index_vec[0]), thread_level_block_index->index_of_the_first_row_arr, thread_level_block_index->data_type_of_index_of_the_first_row_arr, thread_level_block_index->block_num);

    thread_level_block_index->max_row_index = old_warp_index->max_row_index;
    thread_level_block_index->min_row_index = old_warp_index->min_row_index;
    thread_level_block_index->max_col_index = old_warp_index->max_col_index;
    thread_level_block_index->min_col_index = old_warp_index->min_col_index;

    // 用一个和warp数量相同的数组来存储块的大小，因为在block和warp阶段默认放弃了分块，所以TLB块的大小只有一位
    thread_level_block_index->data_type_of_coo_block_size_arr = find_most_suitable_data_type(thread_level_block_size);
    thread_level_block_index->coo_block_size_arr = malloc_arr(old_warp_index->block_num, thread_level_block_index->data_type_of_coo_block_size_arr);
    copy_unsigned_long_arr_to_others(&(thread_level_block_size_vec[0]), thread_level_block_index->coo_block_size_arr, thread_level_block_index->data_type_of_coo_block_size_arr, old_warp_index->block_num);

    // 存储每个线程行的数量
    thread_level_block_index->data_type_of_row_number_of_block_arr = find_most_suitable_data_type(max_row_num_in_TLB);
    thread_level_block_index->row_number_of_block_arr = malloc_arr(thread_level_row_num_vec.size(), thread_level_block_index->data_type_of_row_number_of_block_arr);
    copy_unsigned_long_arr_to_others(&(thread_level_row_num_vec[0]), thread_level_block_index->row_number_of_block_arr, thread_level_block_index->data_type_of_row_number_of_block_arr, thread_level_block_index->block_num);
    
    // 将新的索引记录在数组中
    compressed_block->read_index.push_back(thread_level_block_index);

    // 不做任何padding和交错存储，直接在对应位置放原始数据的指针
    compressed_block->read_index.push_back(old_global_col_index);
    compressed_block->read_index.push_back(old_global_col_index);

    // 值也做相同的处理
    assert(compressed_block->padding_val_arr == NULL);
    compressed_block->padding_arr_size = compressed_block->size;
    compressed_block->padding_val_arr = compressed_block->val_arr;

    assert(compressed_block->staggered_padding_val_arr == NULL);
    compressed_block->staggered_padding_val_arr_size = compressed_block->size;
    compressed_block->staggered_padding_val_arr = compressed_block->val_arr;

    // 最终的要求
    assert(compressed_block->padding_val_arr != NULL);
    assert(compressed_block->staggered_padding_val_arr != NULL);
    assert(compressed_block->read_index[6]->type_of_index == COL_INDEX);
    assert(compressed_block->read_index.size() == 7);
}

// 根据一段实际的非零元，找到每一行实际的thread块数量，在thread层次和warp层次同时padding，保证每个thread非零元数量相等的同时，一个warp的thread分块的数量是32的倍数，最激进的padding策略
// 参数是全局的行索引，以及实际非零元的上界和下界（包含上界和下界的值），返回每一行的thread块数量
vector<unsigned long> find_thread_block_num_of_each_line_after_padding_in_thread_and_warp_level(index_of_compress_block_t *old_global_row_index, unsigned long global_warp_first_row, unsigned long warp_row_num, unsigned long global_warp_coo_start, unsigned long global_warp_coo_end, unsigned long thread_level_block_size)
{
    assert(old_global_row_index != NULL && old_global_row_index->index_compressed_type == COO && old_global_row_index->type_of_index == ROW_INDEX);

    // 首先得到每一行的长度
    vector<unsigned long> nnz_of_each_row;
    vector<unsigned long> thread_level_block_num_of_each_row;

    // 按照每一行的长度赋值为0
    unsigned long nnz_of_each_row_index;
    for (nnz_of_each_row_index = 0; nnz_of_each_row_index < warp_row_num; nnz_of_each_row_index++)
    {
        nnz_of_each_row.push_back(0);
    }

    unsigned long global_coo_index;
    // 对于coo的遍历是不包含下界的
    for (global_coo_index = global_warp_coo_start; global_coo_index <= global_warp_coo_end; global_coo_index++)
    {
        // 获取当前行的行号
        assert(global_coo_index < old_global_row_index->length);
        unsigned long global_row_index = read_from_array_with_data_type(old_global_row_index->index_arr, old_global_row_index->index_data_type, global_coo_index);
        assert(global_row_index >= global_warp_first_row);
        // 换算成warp内部的行号
        unsigned long local_row_index = global_row_index - global_warp_first_row;
        assert(local_row_index < warp_row_num);

        // 对应行元素+1
        nnz_of_each_row[local_row_index] = nnz_of_each_row[local_row_index] + 1;
    }

    // cout << "nnz_of_each_row[";

    // unsigned long x;
    // for(x = 0; x < nnz_of_each_row.size(); x++){
    //     cout << nnz_of_each_row[x] << ",";
    // }

    // cout << "]" << endl;

    // cout << "thread_level_block_size:" << thread_level_block_size << endl;
    // 计算每一行线程粒度的块的数量
    for (nnz_of_each_row_index = 0; nnz_of_each_row_index < warp_row_num; nnz_of_each_row_index++)
    {
        // 用每一行的行数量除块数量，然后用模判断能不能整除，从而决定每一行块的数量
        thread_level_block_num_of_each_row.push_back((unsigned long)(nnz_of_each_row[nnz_of_each_row_index] / thread_level_block_size));
        // 如果，不能整数，那就额外再加一
        // 所以TLB的大小可以远大于一行，从而保证一整行都是一个
        if (nnz_of_each_row[nnz_of_each_row_index] % thread_level_block_size != 0)
        {
            thread_level_block_num_of_each_row[nnz_of_each_row_index] = thread_level_block_num_of_each_row[nnz_of_each_row_index] + 1;
        }
    }

    // 查看当前块的数量和32的倍数是多少
    unsigned long thread_level_block_num_sum = 0;
    unsigned long thread_level_block_num_of_each_row_index;
    for (thread_level_block_num_of_each_row_index = 0; thread_level_block_num_of_each_row_index < thread_level_block_num_of_each_row.size(); thread_level_block_num_of_each_row_index++)
    {
        thread_level_block_num_sum = thread_level_block_num_sum + thread_level_block_num_of_each_row[thread_level_block_num_of_each_row_index];
    }

    unsigned long target_thread_level_block_num;
    // 应该的总块数
    if (thread_level_block_num_sum % 32 == 0)
    {
        target_thread_level_block_num = thread_level_block_num_sum;
    }
    else
    {
        //这里代表块数没有办法被32整除，要按照32的向上取整
        target_thread_level_block_num = ((unsigned long)(thread_level_block_num_sum / 32) + 1) * 32;
    }

    // 实际上需要增加的块数量
    unsigned long block_num_need_to_be_added = target_thread_level_block_num - thread_level_block_num_sum;

    // cout << "target_thread_level_block_num:" << target_thread_level_block_num << ",thread_level_block_num_sum:" << thread_level_block_num_sum << ",block_num_need_to_be_added:" << block_num_need_to_be_added << endl;

    assert(target_thread_level_block_num % 32 == 0);

    assert(thread_level_block_num_of_each_row.size() == warp_row_num);
    // 将块分配给不同的行
    unsigned long row_index_to_add_new_block = 0;
    // 遍历所有需要增加的块
    unsigned long i;
    for (i = 0; i < block_num_need_to_be_added; i++)
    {
        thread_level_block_num_of_each_row[row_index_to_add_new_block]++;
        row_index_to_add_new_block = (row_index_to_add_new_block + 1) % warp_row_num;
    }

    // 检查一下，重新计算一下所有行的块数量
    thread_level_block_num_sum = 0;
    for (thread_level_block_num_of_each_row_index = 0; thread_level_block_num_of_each_row_index < thread_level_block_num_of_each_row.size(); thread_level_block_num_of_each_row_index++)
    {
        thread_level_block_num_sum = thread_level_block_num_sum + thread_level_block_num_of_each_row[thread_level_block_num_of_each_row_index];
    }

    // cout << "thread_level_block_num_sum:" << thread_level_block_num_sum << endl;
    assert(thread_level_block_num_sum % 32 == 0);

    return thread_level_block_num_of_each_row;
}

// 找到一段coo格式的子矩阵中最大行非零元数量，传入的coo范围不包括下界
unsigned long find_max_row_nnz_in_coo_sub_block(index_of_compress_block_t *old_global_row_index, unsigned long global_warp_first_row, unsigned long warp_row_num, unsigned long global_warp_coo_start, unsigned long global_warp_coo_end)
{
    assert(old_global_row_index != NULL && old_global_row_index->index_compressed_type == COO && old_global_row_index->type_of_index == ROW_INDEX);

    // 首先得到每一行的长度
    vector<unsigned long> nnz_of_each_row;

    // 按照每一行的长度赋值为0
    unsigned long nnz_of_each_row_index;
    for (nnz_of_each_row_index = 0; nnz_of_each_row_index < warp_row_num; nnz_of_each_row_index++)
    {
        nnz_of_each_row.push_back(0);
    }

    unsigned long global_coo_index;
    // cout << "global_warp_coo_end:" << global_warp_coo_end << ",global_warp_coo_start:" << global_warp_coo_start << endl;
    for (global_coo_index = global_warp_coo_start; global_coo_index <= global_warp_coo_end; global_coo_index++)
    {
        // 获取当前行的行号
        assert(global_coo_index < old_global_row_index->length);
        unsigned long global_row_index = read_from_array_with_data_type(old_global_row_index->index_arr, old_global_row_index->index_data_type, global_coo_index);
        // cout << global_row_index << "," << global_warp_first_row << endl;
        assert(global_row_index >= global_warp_first_row);
        // 换算成warp内部的行号
        unsigned long local_row_index = global_row_index - global_warp_first_row;
        // cout << global_coo_index << "," << global_row_index << "," << global_warp_first_row << "," << warp_row_num << "," << local_row_index << endl;
        assert(local_row_index < warp_row_num);

        // 对应行元素+1
        nnz_of_each_row[local_row_index] = nnz_of_each_row[local_row_index] + 1;
    }

    unsigned long max_row_num = 0;
    // 遍历nnz_of_each_row数组
    for (nnz_of_each_row_index = 0; nnz_of_each_row_index < warp_row_num; nnz_of_each_row_index++)
    {
        if (nnz_of_each_row[nnz_of_each_row_index] > max_row_num)
        {
            max_row_num = nnz_of_each_row[nnz_of_each_row_index];
        }
    }

    // cout << "max_row_num:" << max_row_num << endl;

    // 返回最大的非零元数量
    return max_row_num;
}

// 对压缩视图的子块中的某一些线程块对应的子块
void compressed_block_sort(compressed_block_t *compressed_block, vector<unsigned long> sort_block_index_arr)
{
    assert(compressed_block != NULL);
    assert(compressed_block->read_index.size() == 3);
    assert(compressed_block->read_index[2]->level_of_this_index == TBLOCK_LEVEL);
    assert(compressed_block->read_index[2]->is_sort_arr == NULL);
    assert(compressed_block->read_index[2]->index_compressed_type == CSR);
    assert(sort_block_index_arr.size() > 0);
    // 之前不能被分块
    assert(compressed_block->is_sorted == false && compressed_block->y_write_index.size() == 0);

    index_of_compress_block_t *old_tblock_index = compressed_block->read_index[2];
    index_of_compress_block_t *old_global_row_index = compressed_block->read_index[0];
    index_of_compress_block_t *old_global_col_index = compressed_block->read_index[1];

    assert(old_tblock_index != NULL);
    assert(old_global_row_index->length == old_global_col_index->length);

    // 用一个bool类型的数组决定是不是要分块
    vector<bool> new_is_sorted(old_tblock_index->block_num);

    // 这个数组的大小一开始每一位的内容与其索引是一样的，计算这个子块中的行数量，空行也加入考虑
    unsigned long row_num_of_this_compressed_block = old_tblock_index->max_row_index - old_tblock_index->min_row_index + 1;

    assert(old_tblock_index->block_num >= sort_block_index_arr.size());

    // y_write_index中会增加一个行索引，用来做输出索引，这个行索引本质上是逻辑行号和实际行号的对应，这个行索引本质上也有对应的
    vector<unsigned long> new_y_write_index_vec(row_num_of_this_compressed_block);

    // new_y_write_index为一个压缩块专属，行索引从0开始计算。
    unsigned long i;
    for (i = 0; i < new_y_write_index_vec.size(); i++)
    {
        new_y_write_index_vec[i] = i;
    }

    for (i = 0; i < new_is_sorted.size(); i++)
    {
        new_is_sorted[i] = false;
    }

    unsigned long sort_block_index_arr_index;
    for (sort_block_index_arr_index = 0; sort_block_index_arr_index < sort_block_index_arr.size(); sort_block_index_arr_index++)
    {
        // print_arr_to_file_with_data_type(old_global_row_index->index_arr, old_global_row_index->index_data_type, old_global_row_index->length, "/home/duzhen/spmv_builder/data_source/test0-2.log");
        unsigned long block_index_need_to_be_sorted = sort_block_index_arr[sort_block_index_arr_index];
        assert(block_index_need_to_be_sorted < old_tblock_index->block_num);
        // 这个范围包含上界和下界
        assert(old_tblock_index->coo_begin_index_arr != NULL && old_tblock_index->row_number_of_block_arr != NULL);
        unsigned long block_coo_begin = read_from_array_with_data_type(old_tblock_index->coo_begin_index_arr, old_tblock_index->data_type_of_coo_begin_index_arr, block_index_need_to_be_sorted);
        unsigned long block_coo_end = read_from_array_with_data_type(old_tblock_index->coo_begin_index_arr, old_tblock_index->data_type_of_coo_begin_index_arr, block_index_need_to_be_sorted + 1) - 1;
        // cout << "block_index_need_to_be_sorted:" << block_index_need_to_be_sorted << ",block_coo_begin:" << block_coo_begin << ",block_coo_end:" << block_coo_end << endl;
        // 子块行的数量
        unsigned long block_row_num = read_from_array_with_data_type(old_tblock_index->row_number_of_block_arr, old_tblock_index->data_type_of_row_number_of_block_arr, block_index_need_to_be_sorted);

        // 子块的第一个行
        unsigned long block_first_row_index = read_from_array_with_data_type(old_tblock_index->index_of_the_first_row_arr, old_tblock_index->data_type_of_index_of_the_first_row_arr, block_index_need_to_be_sorted);

        unsigned long coo_num_of_this_block = block_coo_end - block_coo_begin + 1;

        assert(coo_num_of_this_block > 0);
        // 用三个数组分别tblock排序之后的row、col和val的一部分，用来替换最终数组的一部分
        // 使用的是二维数组，一行一个小数组
        vector<vector<unsigned long>> dim_2_inner_block_col_index_vec(block_row_num);
        vector<vector<double>> dim_2_inner_block_val_index_vec(block_row_num);

        assert(new_is_sorted[block_index_need_to_be_sorted] == false);

        new_is_sorted[block_index_need_to_be_sorted] = true;

        // 获取当前block每一个row非零元的长度
        // cout << "block_index_need_to_be_sorted:" << block_index_need_to_be_sorted << ",block_coo_begin:" << block_coo_begin << ",block_coo_end:" << block_coo_end << endl;
        vector<unsigned long> nnz_in_each_row_of_block = get_nnz_of_each_row_in_spec_range(old_global_row_index, block_first_row_index, block_first_row_index + block_row_num - 1, block_coo_begin, block_coo_end);
        // 块内行排序，先是新行号和老的行号的映射，用来写y_write_index，然后是老行号到新行号的转变，用来交换索引和值的位置
        // cout << "2" << endl;
        vector<vector<unsigned long>> two_direct_index_map = index_map_after_sorting(nnz_in_each_row_of_block);

        // print_arr_to_file_with_data_type(&(two_direct_index_map[0][0]), UNSIGNED_LONG, two_direct_index_map[0].size(), "/home/duzhen/spmv_builder/data_source/test0-1.log");

        // 将数据放到dim_2_inner_block_col_index_vec和dim_2_inner_block_val_index_vec中
        // 遍历子块中所有非零元
        unsigned long nz_index_inner_block;
        for (nz_index_inner_block = 0; nz_index_inner_block < coo_num_of_this_block; nz_index_inner_block++)
        {
            // cout << "nz_index_inner_block:" << nz_index_inner_block << endl;
            // 全局的coo索引
            unsigned long global_nz_index = nz_index_inner_block + block_coo_begin;
            assert(global_nz_index < old_global_row_index->length && global_nz_index < compressed_block->size);
            // 获取全局行号
            unsigned long global_row_index = read_from_array_with_data_type(old_global_row_index->index_arr, old_global_row_index->index_data_type, global_nz_index);
            // 全局列号
            unsigned long global_col_index = read_from_array_with_data_type(old_global_col_index->index_arr, old_global_col_index->index_data_type, global_nz_index);
            // 全局值
            double global_val = read_double_from_array_with_data_type(compressed_block->val_arr, compressed_block->val_data_type, global_nz_index);
            // 获取局部行号
            unsigned long local_row_index = global_row_index - block_first_row_index;
            assert(local_row_index < two_direct_index_map[1].size());
            // 获取局部行号的新行号，需要老行号与新行号之间的转换
            unsigned long new_row_local_index_cor_to_old = two_direct_index_map[1][local_row_index];
            // 在二维数组中的新位置存储列号和值
            dim_2_inner_block_col_index_vec[new_row_local_index_cor_to_old].push_back(global_col_index);
            dim_2_inner_block_val_index_vec[new_row_local_index_cor_to_old].push_back(global_val);
        }

        // cout << "1" << endl;

        // 检查新的二维数组中每一行的大小是不是满足要求，一看是不是增序，二看每一行的非零元数量是不是满足要求
        unsigned long new_local_row_index;
        for (new_local_row_index = 0; new_local_row_index < block_row_num; new_local_row_index++)
        {
            // 对应的旧索引
            unsigned long old_local_row_index = two_direct_index_map[0][new_local_row_index];
            // 旧索引对应的行非零元数量和转换之后对应行非零元数量进行比较
            assert(nnz_in_each_row_of_block[old_local_row_index] == dim_2_inner_block_val_index_vec[new_local_row_index].size() && dim_2_inner_block_col_index_vec[new_local_row_index].size() == dim_2_inner_block_val_index_vec[new_local_row_index].size());

            // 查看是不是从大到小排序
            if (new_local_row_index < block_row_num - 1)
            {
                assert(dim_2_inner_block_val_index_vec[new_local_row_index].size() >= dim_2_inner_block_val_index_vec[new_local_row_index + 1].size());
            }
        }

        // 覆盖row col val三个数组的对应位置
        void *coo_row_index = old_global_row_index->index_arr;
        void *coo_col_index = old_global_col_index->index_arr;
        void *coo_val = compressed_block->val_arr;

        // 将数据覆盖到对应位置
        // row没有办法直接拷贝，要一个一个写，这里是要写的位置
        unsigned long global_coo_index_in_block = block_coo_begin;

        // cout << "global_coo_index_in_block:" << global_coo_index_in_block << ",block_first_row_index:" << block_first_row_index << endl;

        for (new_local_row_index = 0; new_local_row_index < block_row_num; new_local_row_index++)
        {

            unsigned long new_global_row_index = new_local_row_index + block_first_row_index;

            // 排序后这一行的非零元数量，
            unsigned long index_in_this_row;
            for (index_in_this_row = 0; index_in_this_row < dim_2_inner_block_col_index_vec[new_local_row_index].size(); index_in_this_row++)
            {
                // 查看写的位置是不是出了范围
                assert(global_coo_index_in_block >= block_coo_begin && global_coo_index_in_block <= block_coo_end);
                // 每一行
                // cout << "global_coo_index_in_block:" << global_coo_index_in_block << ",new_global_row_index" << new_global_row_index << endl;
                // if(global_coo_index_in_block < 100){
                //     cout << "global_coo_index_in_block:" << global_coo_index_in_block << ",new_global_row_index:" << new_global_row_index << endl;
                // }
                write_to_array_with_data_type(coo_row_index, old_global_row_index->index_data_type, global_coo_index_in_block, new_global_row_index);

                // if(global_coo_index_in_block < 100){
                //     cout << "global_coo_index_in_block:" << global_coo_index_in_block << ",coo_row_index[global_coo_index_in_block]:" << read_from_array_with_data_type(coo_row_index, old_global_row_index->index_data_type, global_coo_index_in_block) << endl;
                // }

                // 写列号
                write_to_array_with_data_type(coo_col_index, old_global_col_index->index_data_type, global_coo_index_in_block, dim_2_inner_block_col_index_vec[new_local_row_index][index_in_this_row]);
                // 写函数
                write_double_to_array_with_data_type(coo_val, compressed_block->val_data_type, global_coo_index_in_block, dim_2_inner_block_val_index_vec[new_local_row_index][index_in_this_row]);
                global_coo_index_in_block++;
            }

            // 打印coo_row_index的前100个内容
            // print_arr_to_file_with_data_type(coo_row_index, old_global_row_index->index_data_type, 100, "/home/duzhen/spmv_builder/data_source/test0-1.log");

            // 用new_global_row_index为new_y_write_index赋值
            assert(new_global_row_index < new_y_write_index_vec.size());
            assert(new_local_row_index < two_direct_index_map[0].size());
            // 向对应位置写对应排序后的行对应的原行索引，需要从新索引到原索引的映射
            new_y_write_index_vec[new_global_row_index] = two_direct_index_map[0][new_local_row_index] + block_first_row_index;
        }

        // cout << "global_coo_index_in_block:" << global_coo_index_in_block << ",block_coo_end:" << block_coo_end << endl;
        assert(global_coo_index_in_block == block_coo_end + 1);

        // 打印row_index
        // print_arr_to_file_with_data_type(coo_row_index, old_global_row_index->index_data_type, old_global_row_index->length, "/home/duzhen/spmv_builder/data_source/test0-1.log");
    }

    // print_arr_to_file_with_data_type(old_global_row_index->index_arr, old_global_row_index->index_data_type, old_global_row_index->length, "/home/duzhen/spmv_builder/data_source/test0-1.log");

    // 创建一个新的索引，y_write_index
    index_of_compress_block_t *new_y_write_index = new index_of_compress_block_t();
    new_y_write_index->level_of_this_index = OTHERS;
    new_y_write_index->index_compressed_type = COO;
    // cout << "new_y_write_index_vec.size():" << new_y_write_index_vec.size() << ",old_global_row_index->length:" << old_global_row_index->length << endl;
    // new_y_write_index_vec和行的数量相等
    // assert(new_y_write_index_vec.size() == old_global_row_index->length);
    new_y_write_index->length = new_y_write_index_vec.size();
    // 索引的数据类型和原来保持一致
    new_y_write_index->index_data_type = old_global_row_index->index_data_type;
    // 索引的范围
    new_y_write_index->max_row_index = old_global_row_index->max_row_index;
    new_y_write_index->min_row_index = old_global_row_index->min_row_index;
    new_y_write_index->max_col_index = old_global_row_index->max_col_index;
    new_y_write_index->min_col_index = old_global_row_index->min_col_index;

    // 申请新的索引数组
    new_y_write_index->index_arr = malloc_arr(new_y_write_index->length, new_y_write_index->index_data_type);
    // 将新的索引拷贝进来，
    copy_unsigned_long_arr_to_others(&(new_y_write_index_vec[0]), new_y_write_index->index_arr, new_y_write_index->index_data_type, new_y_write_index->length);
    // 将索引写到数组中
    compressed_block->y_write_index.push_back(new_y_write_index);
}

// 一定非零元范围的行数量
vector<unsigned long> get_row_num_of_each_row_nnz_range(operator_manager_t *op_manager, vector<unsigned long> row_nnz_range)
{
    assert(op_manager != NULL && row_nnz_range.size() > 2);
    assert(op_manager->matrix != NULL);
    assert(row_nnz_range[0] == 0);

    vector<unsigned long> return_row_num_of_each_range(row_nnz_range.size() - 1);

    // 找出每一行的非零元数量
    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(op_manager->matrix->coo_row_index_cache, UNSIGNED_LONG, 0, op_manager->matrix->dense_row_number - 1, 0, op_manager->matrix->nnz - 1);

    // 找出对应nnz范围的行的数量
    for (unsigned long i = 0; i < nnz_of_each_row.size(); i++)
    {
        // 当前行的非零元数量
        unsigned long nnz_of_cur_row = nnz_of_each_row[i];

        bool is_found = false;

        // 查看处于哪个范围
        for (unsigned long j = 0; j < return_row_num_of_each_range.size(); j++)
        {
            if (nnz_of_cur_row >= row_nnz_range[j] && nnz_of_cur_row < row_nnz_range[j + 1])
            {
                return_row_num_of_each_range[j]++;
                is_found = true;
                break;
            }
        }

        if (is_found == false)
        {
            cout << "nnz_of_cur_row:" << nnz_of_cur_row << " not found" << endl;
        }

        assert(is_found);
    }

    // 加一下，然后看看总共的行数量
    unsigned long row_num_sum = 0;
    for (unsigned long i = 0; i < return_row_num_of_each_range.size(); i++)
    {
        row_num_sum = row_num_sum + return_row_num_of_each_range[i];
    }

    assert(row_num_sum == op_manager->matrix->dense_row_number);

    return return_row_num_of_each_range;
}

vector<unsigned long> get_nnz_of_each_row_in_spec_range(void *row_index_arr, data_type data_type_of_row_index_arr, unsigned long begin_row_bound, unsigned long end_row_bound, unsigned long global_coo_start, unsigned long global_coo_end)
{
    assert(row_index_arr != NULL);
    assert(data_type_of_row_index_arr == UNSIGNED_LONG || data_type_of_row_index_arr == UNSIGNED_INT || data_type_of_row_index_arr == UNSIGNED_SHORT || data_type_of_row_index_arr == UNSIGNED_CHAR);
    assert(end_row_bound >= begin_row_bound && global_coo_end >= global_coo_start);

    unsigned long total_row_num = end_row_bound - begin_row_bound + 1;

    // 申请一个数组存储每一行的非零元数量
    vector<unsigned long> nnz_of_each_row(total_row_num);

    unsigned int i;
    for (i = 0; i < total_row_num; i++)
    {
        nnz_of_each_row[i] = 0;
    }

    // 行号增序的检查
    unsigned long last_nz_row_index = 0;

    unsigned long cur_coo_index;
    for (cur_coo_index = global_coo_start; cur_coo_index <= global_coo_end; cur_coo_index++)
    {
        unsigned long cur_nz_row_index = read_from_array_with_data_type(row_index_arr, data_type_of_row_index_arr, cur_coo_index);

        // assert(last_nz_row_index <= cur_nz_row_index);

        if (last_nz_row_index > cur_nz_row_index)
        {
            cout << "last_nz_row_index:" << last_nz_row_index << ", "
                 << "cur_nz_row_index:" << cur_nz_row_index << endl;
            assert(false);
        }

        assert(cur_nz_row_index >= begin_row_bound && cur_nz_row_index <= end_row_bound);

        // 计算相对行号
        unsigned long local_cur_nz_row_index = cur_nz_row_index - begin_row_bound;
        // cout << "local_cur_nz_row_index:" << local_cur_nz_row_index << ",total_row_num:" << total_row_num << endl;
        assert(local_cur_nz_row_index < total_row_num);

        // 相对行的非零元数量+1
        nnz_of_each_row[local_cur_nz_row_index] = nnz_of_each_row[local_cur_nz_row_index] + 1;
        last_nz_row_index = cur_nz_row_index;
    }

    // for (unsigned long i = 0; i < )
    // {
        
    // }

    // 返回每一行的元素数量
    return nnz_of_each_row;
}

vector<unsigned long> get_nnz_of_each_row_in_spec_range(index_of_compress_block_t *old_global_row_index, unsigned long begin_row_bound, unsigned long end_row_bound, unsigned long global_coo_start, unsigned long global_coo_end)
{
    assert(old_global_row_index != NULL && old_global_row_index->type_of_index == ROW_INDEX && old_global_row_index->index_compressed_type == COO);
    assert(end_row_bound >= begin_row_bound && global_coo_end >= global_coo_start && global_coo_end < old_global_row_index->length);

    // cout << "end_row_bound:" << end_row_bound << ",begin_row_bound:" << begin_row_bound << endl;
    unsigned long total_row_num = end_row_bound - begin_row_bound + 1;

    // 申请一个数组存储每一行的非零元数量
    vector<unsigned long> nnz_of_each_row(total_row_num);

    unsigned int i;
    for (i = 0; i < total_row_num; i++)
    {
        nnz_of_each_row[i] = 0;
    }

    // 遍历所有的非零元
    unsigned long last_nz_row_index = 0;

    unsigned long cur_coo_index;
    for (cur_coo_index = global_coo_start; cur_coo_index <= global_coo_end; cur_coo_index++)
    {
        unsigned long cur_nz_row_index = read_from_array_with_data_type(old_global_row_index->index_arr, old_global_row_index->index_data_type, cur_coo_index);
        assert(last_nz_row_index <= cur_nz_row_index);
        assert(cur_nz_row_index >= begin_row_bound && cur_nz_row_index <= end_row_bound);

        // 计算相对行号
        unsigned long local_cur_nz_row_index = cur_nz_row_index - begin_row_bound;
        // cout << "local_cur_nz_row_index:" << local_cur_nz_row_index << ",total_row_num:" << total_row_num << endl;
        assert(local_cur_nz_row_index < total_row_num);

        // 相对行的非零元数量+1
        nnz_of_each_row[local_cur_nz_row_index] = nnz_of_each_row[local_cur_nz_row_index] + 1;

        last_nz_row_index = cur_nz_row_index;
    }

    // 返回每一行的元素数量
    return nnz_of_each_row;
}

vector<unsigned long> get_nnz_of_each_row_in_compressed_sub_matrix(compressed_block_t* compressed_sub_block)
{
    assert(compressed_sub_block != NULL);
    // 初始的行索引是存在的
    assert(compressed_sub_block->read_index.size() > 0 && compressed_sub_block->read_index[0] != NULL);
    assert(compressed_sub_block->read_index[0]->type_of_index == ROW_INDEX && compressed_sub_block->read_index[0]->index_arr != NULL);

    assert(compressed_sub_block->read_index[0]->max_row_index >= compressed_sub_block->read_index[0]->min_row_index);
    // 当前子块总的行数量
    unsigned long total_row_num_of_compressed_sub_matrix = compressed_sub_block->read_index[0]->max_row_index - compressed_sub_block->read_index[0]->min_row_index + 1;

    index_of_compress_block_t* row_index_in_compressed_block = compressed_sub_block->read_index[0];

    // 创建对应数量的数组
    vector<unsigned long> row_nnz_of_compress_block(total_row_num_of_compressed_sub_matrix);

    // 初始化这个数组
    for (unsigned long i = 0; i < row_nnz_of_compress_block.size(); i++)
    {
        row_nnz_of_compress_block[i] = 0;
    }

    assert(row_index_in_compressed_block->block_num == 0);
    unsigned long last_nz_row_index = 0;
    // 遍历所有的行索引
    for (unsigned long row_index_id = 0; row_index_id < row_index_in_compressed_block->length; row_index_id++)
    {
        unsigned long cur_nz_row_index = read_from_array_with_data_type(row_index_in_compressed_block->index_arr, row_index_in_compressed_block->index_data_type, row_index_id);
        assert(cur_nz_row_index >= last_nz_row_index);
        // 当前行号不能超过行非零元数组的容量
        assert(cur_nz_row_index < row_nnz_of_compress_block.size());

        row_nnz_of_compress_block[cur_nz_row_index]++;

        last_nz_row_index = cur_nz_row_index;
    }

    return row_nnz_of_compress_block;
}

// 冒泡排序，然后得出每一位在排序之后的新位置
// 先是新行号和老的行号的映射，用来写y_write_index，然后是老行号到新行号的映射，用来调换原来非零元的顺序
vector<vector<unsigned long>> index_map_after_sorting(vector<unsigned long> nnz_of_each_row)
{
    sizeof(nnz_of_each_row.size() > 0);

    vector<vector<unsigned long>> two_direct_index_map;

    // 排序之后没一个新位置和老位置之间的映射
    vector<unsigned long> element_new_to_old_index(nnz_of_each_row.size());
    // 老位置与新位置之间的映射
    vector<unsigned long> element_old_to_new_index(nnz_of_each_row.size());

    unsigned long i;
    for (i = 0; i < element_new_to_old_index.size(); i++)
    {
        element_new_to_old_index[i] = i;
    }

    // // 打印每一行非零元和数量
    // for(i = 0; i < nnz_of_each_row.size(); i++){
    //     // if(nnz_of_each_row[i] == 2){
    //     //     cout << i << endl;
    //     // }
    //     cout << nnz_of_each_row[i] << ",";
    // }
    // cout << endl;

    // exit(-1);

    // 用冒泡排序，从高到底排序，记录排序之后每个值原来的位置，主要是修改两个数组元素的位置，nnz_of_each_row和element_new_to_old_index
    assert(get_config()["SORT_THREAD_NUM"].as_integer() > 0);
    if (get_config()["SORT_THREAD_NUM"].as_integer() == 1)
    {
        for (i = 1; i < nnz_of_each_row.size(); i++)
        {
            // cout << "i:" << i << endl;
            unsigned long cur_element = nnz_of_each_row[i];
            unsigned long cur_element_index = element_new_to_old_index[i];

            // 用一个bool判断是否需要执行插入
            bool need_swap = false;

            // 元素要插入的位置
            unsigned long index_need_to_be_inserted;
            // 和之前的每一个值比较，之前都是排序好的
            unsigned long j;
            for (j = i - 1; j >= 0; j--)
            {
                // cout << "j:" << j << endl;
                assert(j >= 0 && j < nnz_of_each_row.size());

                // cout << "cur_element:" << cur_element << ",nnz_of_each_row[j]:" << nnz_of_each_row[j] << endl;
                if (cur_element <= nnz_of_each_row[j])
                {
                    // cur_element比之前的都小了，那就放在j+1的位置
                    index_need_to_be_inserted = j + 1;

                    if (j + 1 == i)
                    {
                        need_swap = false;
                    }
                    else
                    {
                        need_swap = true;
                    }

                    break;
                }

                // unsigned long都是正的，很难退出，需要在和第一个比较之后手动退出
                if (j == 0)
                {
                    // 能到达这里说明在比较的值比第一个值都大，需要换到数组头部
                    index_need_to_be_inserted = 0;
                    need_swap = true;
                    break;
                }
            }

            // cout << "need_swap:" << need_swap << endl;
            if (need_swap == true)
            {
                // 将index_need_to_be_inserted到i - 1的内容全部向后挪
                for (j = i - 1; j >= index_need_to_be_inserted; j--)
                {
                    nnz_of_each_row[j + 1] = nnz_of_each_row[j];
                    element_new_to_old_index[j + 1] = element_new_to_old_index[j];

                    // 如果j已经到0了就退出
                    if (j == 0)
                    {
                        break;
                    }
                }

                nnz_of_each_row[index_need_to_be_inserted] = cur_element;
                element_new_to_old_index[index_need_to_be_inserted] = cur_element_index;
            }
        }
    }
    else
    {
        // 这里引入多线程排序
    }

    // 检查排序的结果是不是正确
    for (i = 1; i < nnz_of_each_row.size(); i++)
    {
        assert(nnz_of_each_row[i - 1] >= nnz_of_each_row[i]);
    }

    // element_old_to_new_index的转化
    for (i = 0; i < element_new_to_old_index.size(); i++)
    {
        // cout << "element_new_to_old_index[i]:" << element_new_to_old_index[i] << ",element_old_to_new_index.size():" << element_old_to_new_index.size() << endl;
        assert(element_new_to_old_index[i] >= 0 && element_new_to_old_index[i] < element_old_to_new_index.size());
        element_old_to_new_index[element_new_to_old_index[i]] = i;
    }

    two_direct_index_map.push_back(element_new_to_old_index);
    two_direct_index_map.push_back(element_old_to_new_index);

    // 返回排序之后数组的原索引，
    return two_direct_index_map;
}

void compressed_block_sort(operator_manager_t *op_manager, compressed_block_t *compressed_block, vector<unsigned long> sort_block_index_arr)
{
    assert(op_manager != NULL && compressed_block != NULL && op_manager->matrix->is_sorted == false);
    compressed_block_sort(compressed_block, sort_block_index_arr);
}

void total_compressed_block_sort(operator_manager_t *op_manager, compressed_block_t *compressed_block)
{
    assert(op_manager != NULL && compressed_block != NULL && op_manager->matrix->is_sorted == false);

    total_compressed_block_sort(compressed_block);
}

// 对某一个子密集视图矩阵进行排序，
void total_compressed_block_sort(operator_manager_t *op_manager, unsigned long index_of_dense_block_table)
{
    assert(op_manager != NULL);

    sparse_struct_t *matrix = op_manager->matrix;
    // 查看选的子矩阵是不是表格里面有的
    assert(index_of_dense_block_table < matrix->block_coor_table.item_arr.size());

    // unsigned long row_num_in_dense_sub_block = matrix->block_coor_table.item_arr[index_of_dense_block_table]->max_dense_row_index - matrix->block_coor_table.item_arr[index_of_dense_block_table]->min_dense_row_index + 1;

    compressed_block_t *target_compressed_block = matrix->block_coor_table.item_arr[index_of_dense_block_table]->compressed_block_ptr;

    // 压缩子图排序之前不能padding
    assert(target_compressed_block->read_index[0]->max_row_index == matrix->block_coor_table.item_arr[index_of_dense_block_table]->max_dense_row_index);
    assert(target_compressed_block->read_index[0]->min_row_index == matrix->block_coor_table.item_arr[index_of_dense_block_table]->min_dense_row_index);

    total_compressed_block_sort(target_compressed_block);
}

// 在全局范围内进行一次排序，并且这个排序只能发生在线程块粒度的分块之前
void total_compressed_block_sort(compressed_block_t *compressed_block)
{
    assert(compressed_block != NULL && compressed_block->is_sorted == false && compressed_block->read_index.size() == 2);
    assert(compressed_block->y_write_index.size() == 0);

    index_of_compress_block_t *old_global_row_index = compressed_block->read_index[0];
    index_of_compress_block_t *old_global_col_index = compressed_block->read_index[1];

    assert(old_global_row_index != NULL && old_global_col_index != NULL && old_global_row_index->index_arr != NULL && old_global_col_index->index_arr != NULL);

    assert(old_global_row_index->length == old_global_col_index->length);
    unsigned long coo_num = old_global_row_index->length;

    // 这个数组的大小一开始每一位的内容与其索引是一样的，计算这个子块中的行数量，空行也加入考虑
    unsigned long row_num_of_this_compressed_block = old_global_row_index->max_row_index - old_global_row_index->min_row_index + 1;
    unsigned long coo_index_begin = 0;
    unsigned long coo_index_end = old_global_row_index->length - 1;

    // y_write_index中会增加一个行索引，用来做输出索引，这个行索引本质上是逻辑行号和实际行号的对应，这个行索引本质上也有对应的
    vector<unsigned long> new_y_write_index_vec(row_num_of_this_compressed_block);

    // new_y_write_index为一个压缩块专属，行索引从0开始计算。
    unsigned long i;
    for (i = 0; i < new_y_write_index_vec.size(); i++)
    {
        new_y_write_index_vec[i] = i;
    }

    // 使用的是二维数组，一行一个小数组，存储排序之后的全局结果
    vector<vector<unsigned long>> dim_2_inner_block_col_index_vec(row_num_of_this_compressed_block);
    vector<vector<double>> dim_2_inner_block_val_index_vec(row_num_of_this_compressed_block);

    // 遍历所有的非零元，执行排序
    // 找出每一行非零元数量
    vector<unsigned long> nnz_in_each_row = get_nnz_of_each_row_in_spec_range(old_global_row_index, 0, row_num_of_this_compressed_block - 1, coo_index_begin, coo_index_end);
    // 找出排序之后的索引，第一个是new to old，第二个是old to new
    vector<vector<unsigned long>> two_direct_index_map = index_map_after_sorting(nnz_in_each_row);

    assert(two_direct_index_map[0].size() == two_direct_index_map[1].size());

    // 遍历所有的非零元，找到其新的位置（two_direct_index_map[1]），然后存到dim2的两个数组里面
    unsigned long nz_index;
    for (nz_index = 0; nz_index < coo_num; nz_index++)
    {
        // 获取全局行号
        unsigned long global_row_index = read_from_array_with_data_type(old_global_row_index->index_arr, old_global_row_index->index_data_type, nz_index);
        // 全局列号
        unsigned long global_col_index = read_from_array_with_data_type(old_global_col_index->index_arr, old_global_col_index->index_data_type, nz_index);
        // 值
        double global_val = read_double_from_array_with_data_type(compressed_block->val_arr, compressed_block->val_data_type, nz_index);
        // 找出老的行号对应的新行号
        assert(global_row_index < two_direct_index_map[1].size());
        unsigned long new_row_index_cor_to_old = two_direct_index_map[1][global_row_index];

        // 新的col和val的二维缓冲区
        dim_2_inner_block_col_index_vec[new_row_index_cor_to_old].push_back(global_col_index);
        dim_2_inner_block_val_index_vec[new_row_index_cor_to_old].push_back(global_val);
    }

    // 大小检查
    unsigned long new_global_row_index;
    for (new_global_row_index = 0; new_global_row_index < row_num_of_this_compressed_block; new_global_row_index++)
    {
        // 对应的旧索引
        unsigned long old_row_index = two_direct_index_map[0][new_global_row_index];
        // 查看非零元数量是不是正确
        assert(nnz_in_each_row[old_row_index] == dim_2_inner_block_val_index_vec[new_global_row_index].size() && dim_2_inner_block_col_index_vec[new_global_row_index].size() == dim_2_inner_block_val_index_vec[new_global_row_index].size());

        // 查看是不是从头到小排序
        if (new_global_row_index < row_num_of_this_compressed_block - 1)
        {
            assert(dim_2_inner_block_val_index_vec[new_global_row_index].size() >= dim_2_inner_block_val_index_vec[new_global_row_index + 1].size());
        }
    }

    unsigned long global_coo_index = coo_index_begin;
    // 将排序之后的数据拷贝到val、col、row
    for (new_global_row_index = 0; new_global_row_index < row_num_of_this_compressed_block; new_global_row_index++)
    {
        unsigned long index_in_this_row;
        for (index_in_this_row = 0; index_in_this_row < dim_2_inner_block_col_index_vec[new_global_row_index].size(); index_in_this_row++)
        {
            // 看写的位置是不是超出了范围
            assert(global_coo_index >= coo_index_begin && global_coo_index <= coo_index_end);
            // 写行号
            write_to_array_with_data_type(old_global_row_index->index_arr, old_global_row_index->index_data_type, global_coo_index, new_global_row_index);
            // 写列号
            write_to_array_with_data_type(old_global_col_index->index_arr, old_global_col_index->index_data_type, global_coo_index, dim_2_inner_block_col_index_vec[new_global_row_index][index_in_this_row]);
            // 写函数
            write_double_to_array_with_data_type(compressed_block->val_arr, compressed_block->val_data_type, global_coo_index, dim_2_inner_block_val_index_vec[new_global_row_index][index_in_this_row]);
            global_coo_index++;
        }

        // 写y_write_index的输出
        assert(new_global_row_index < new_y_write_index_vec.size() && new_global_row_index < two_direct_index_map[0].size());
        new_y_write_index_vec[new_global_row_index] = two_direct_index_map[0][new_global_row_index];
    }

    assert(global_coo_index == coo_index_end + 1);

    // 创建一个新的写y的索引
    index_of_compress_block_t *new_y_write_index = new index_of_compress_block_t();
    new_y_write_index->level_of_this_index = OTHERS;
    new_y_write_index->index_compressed_type = COO;
    new_y_write_index->length = new_y_write_index_vec.size();
    new_y_write_index->index_data_type = old_global_row_index->index_data_type;

    new_y_write_index->max_row_index = old_global_row_index->max_row_index;
    new_y_write_index->min_row_index = old_global_row_index->min_row_index;
    new_y_write_index->max_col_index = old_global_row_index->max_col_index;
    new_y_write_index->min_col_index = old_global_row_index->min_col_index;

    // 申请新的索引数组
    new_y_write_index->index_arr = malloc_arr(new_y_write_index->length, new_y_write_index->index_data_type);
    // 将新的索引拷贝进来，
    copy_unsigned_long_arr_to_others(&(new_y_write_index_vec[0]), new_y_write_index->index_arr, new_y_write_index->index_data_type, new_y_write_index->length);
    // 将索引写到数组中
    compressed_block->y_write_index.push_back(new_y_write_index);
    compressed_block->is_sorted = true;
}

// 在密集矩阵视图下进行排序，并且重写所有的索引
void total_dense_block_sort(operator_manager_t *op_manager)
{
    assert(op_manager != NULL && op_manager->matrix != NULL);
    // 还没有经过压缩和分块
    assert(op_manager->matrix->coo_col_index_cache != NULL && op_manager->matrix->coo_row_index_cache != NULL && op_manager->matrix->coo_value_cache != NULL);
    assert(op_manager->matrix->block_coor_table.item_arr.size() == 0 && op_manager->matrix->is_sorted == false && op_manager->matrix->sorted_row_index == NULL);

    sparse_struct_t *matrix = op_manager->matrix;
    // 开始排序
    matrix->is_sorted = true;

    // 初始化sorted_row_index，按照行的数量来决定申请的空间大小和数据类型
    matrix->data_type_of_sorted_row_index = find_most_suitable_data_type(matrix->dense_row_number);
    matrix->sorted_row_index = malloc_arr(matrix->dense_row_number, matrix->data_type_of_sorted_row_index);

    // 找出每一行的非零元数量
    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);

    // 按照非零元数量将行从大到小进行排序，得到old to new和new to old两个方向
    cout << "begin sorting" << endl;
    vector<vector<unsigned long>> two_direct_index_map = index_map_after_sorting(nnz_of_each_row);
    cout << "end sorting" << endl;

    assert(two_direct_index_map[0].size() == two_direct_index_map[1].size());

    // 使用的是二维数组，一行一个小数组，存储排序之后的全局结果
    vector<vector<unsigned long>> dim_2_inner_block_col_index_vec(matrix->dense_row_number);
    vector<vector<double>> dim_2_inner_block_val_index_vec(matrix->dense_row_number);

    // 遍历所有的非零元，找到其新的位置（two_direct_index_map[1]），然后存到dim2的两个数组里面
    unsigned long nz_index;
    for (nz_index = 0; nz_index < matrix->nnz; nz_index++)
    {
        // 获取全局行号
        unsigned long global_row_index = read_from_array_with_data_type(matrix->coo_row_index_cache, UNSIGNED_LONG, nz_index);
        // 全局列号
        unsigned long global_col_index = read_from_array_with_data_type(matrix->coo_col_index_cache, UNSIGNED_LONG, nz_index);
        // 值
        double global_val = read_double_from_array_with_data_type(matrix->coo_value_cache, matrix->val_data_type, nz_index);
        // 找出老的行号对应的新行号
        assert(global_row_index < two_direct_index_map[1].size());
        unsigned long new_row_index_cor_to_old = two_direct_index_map[1][global_row_index];

        // 新的col和val的二维缓冲区
        dim_2_inner_block_col_index_vec[new_row_index_cor_to_old].push_back(global_col_index);
        dim_2_inner_block_val_index_vec[new_row_index_cor_to_old].push_back(global_val);
    }

    // 大小检查
    unsigned long new_global_row_index;
    for (new_global_row_index = 0; new_global_row_index < matrix->dense_row_number; new_global_row_index++)
    {
        // 对应的旧索引
        unsigned long old_row_index = two_direct_index_map[0][new_global_row_index];
        // 查看非零元数量是不是正确
        assert(nnz_of_each_row[old_row_index] == dim_2_inner_block_val_index_vec[new_global_row_index].size() && dim_2_inner_block_col_index_vec[new_global_row_index].size() == dim_2_inner_block_val_index_vec[new_global_row_index].size());

        // 查看是不是从头到小排序
        if (new_global_row_index < matrix->dense_row_number - 1)
        {
            assert(dim_2_inner_block_val_index_vec[new_global_row_index].size() >= dim_2_inner_block_val_index_vec[new_global_row_index + 1].size());
        }
    }

    unsigned long global_coo_index = 0;
    // 将排序之后的数据拷贝到val、col、row
    for (new_global_row_index = 0; new_global_row_index < matrix->dense_row_number; new_global_row_index++)
    {
        unsigned long index_in_this_row;
        for (index_in_this_row = 0; index_in_this_row < dim_2_inner_block_col_index_vec[new_global_row_index].size(); index_in_this_row++)
        {
            // 看写的位置是不是超出了范围
            assert(global_coo_index >= 0 && global_coo_index <= matrix->nnz - 1);
            // 写行号
            write_to_array_with_data_type(matrix->coo_row_index_cache, UNSIGNED_LONG, global_coo_index, new_global_row_index);
            // 写列号
            write_to_array_with_data_type(matrix->coo_col_index_cache, UNSIGNED_LONG, global_coo_index, dim_2_inner_block_col_index_vec[new_global_row_index][index_in_this_row]);
            // 写函数
            write_double_to_array_with_data_type(matrix->coo_value_cache, matrix->val_data_type, global_coo_index, dim_2_inner_block_val_index_vec[new_global_row_index][index_in_this_row]);
            global_coo_index++;
        }

        // 写y_write_index的输出
        assert(new_global_row_index < two_direct_index_map[0].size());
        // matrix->[new_global_row_index] = two_direct_index_map[0][new_global_row_index];
        write_to_array_with_data_type(matrix->sorted_row_index, matrix->data_type_of_sorted_row_index, new_global_row_index, two_direct_index_map[0][new_global_row_index]);
    }

    assert(global_coo_index == matrix->nnz);
}

void total_dense_block_sort(operator_manager_t *op_manager, vector<unsigned long> first_row_of_each_sort_band)
{
    assert(op_manager != NULL && op_manager->matrix != NULL);
    // 还没有经过压缩和分块
    assert(op_manager->matrix->coo_col_index_cache != NULL && op_manager->matrix->coo_row_index_cache != NULL && op_manager->matrix->coo_value_cache != NULL);
    assert(op_manager->matrix->block_coor_table.item_arr.size() == 0 && op_manager->matrix->is_sorted == false && op_manager->matrix->sorted_row_index == NULL);
    assert(first_row_of_each_sort_band.size() > 1 && first_row_of_each_sort_band[0] == 0);
    assert(first_row_of_each_sort_band[first_row_of_each_sort_band.size() - 1] < op_manager->matrix->dense_row_number);

    sparse_struct_t *matrix = op_manager->matrix;
    // 开始排序
    matrix->is_sorted = true;

    // 初始化sorted_row_index，按照行的数量来决定申请的空间大小和数据类型
    matrix->data_type_of_sorted_row_index = find_most_suitable_data_type(matrix->dense_row_number);
    matrix->sorted_row_index = malloc_arr(matrix->dense_row_number, matrix->data_type_of_sorted_row_index);

    // 获取每一行非零元的数量
    // 找出每一行的非零元数量
    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);

    // cout << nnz_of_each_row[90672] << endl;

    unsigned long coo_index_begin_of_cur_row_band = 0;

    // 遍历所有的块，并为其排序
    unsigned long sort_band_index;
    for (sort_band_index = 0; sort_band_index < first_row_of_each_sort_band.size(); sort_band_index++)
    {
        // 下一个块的起始位置
        unsigned long coo_index_begin_of_next_row_band = coo_index_begin_of_cur_row_band;

        // 如果当前条带是最后一个条带，下一个条带的coo起始位置是就用整个矩阵非零元的数量代替
        if (sort_band_index == first_row_of_each_sort_band.size() - 1)
        {
            coo_index_begin_of_next_row_band = matrix->nnz;
        }
        else
        {
            for (unsigned long i = first_row_of_each_sort_band[sort_band_index]; i < first_row_of_each_sort_band[sort_band_index + 1]; i++)
            {
                // 叠加当前条带的所有非零元
                coo_index_begin_of_next_row_band = coo_index_begin_of_next_row_band + nnz_of_each_row[i];
            }
        }

        // coo_index_begin_of_next_row_band和coo_index_begin_of_cur_row_band相等时，代表当前条带为空条带，跳过
        if (coo_index_begin_of_next_row_band != coo_index_begin_of_cur_row_band)
        {

            // 在当前条带内排序
            unsigned long coo_index_end_of_cur_row_band = coo_index_begin_of_next_row_band - 1;

            // 每一个条带的起始行号和结束行号
            unsigned long row_index_begin_in_band = first_row_of_each_sort_band[sort_band_index];

            unsigned long row_index_end_in_band;
            if (sort_band_index == first_row_of_each_sort_band.size() - 1)
            {
                row_index_end_in_band = matrix->dense_row_number - 1;
            }
            else
            {
                // 下一个条带的起始行倒退一行
                row_index_end_in_band = first_row_of_each_sort_band[sort_band_index + 1] - 1;
            }

            unsigned long row_num_of_this_band = row_index_end_in_band - row_index_begin_in_band + 1;
            unsigned long coo_num_of_this_band = coo_index_end_of_cur_row_band - coo_index_begin_of_cur_row_band + 1;

            // 临时申请数组，包含val和col数组，第一层的大小为行的数量
            vector<vector<unsigned long>> dim2_inner_band_col_index_vec(row_num_of_this_band);
            vector<vector<double>> dim2_inner_band_var_index_vec(row_num_of_this_band);

            // 低效的计算，重新找出当前band每一行的行数量
            vector<unsigned long> nnz_of_each_row_in_band = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, row_index_begin_in_band, row_index_end_in_band, coo_index_begin_of_cur_row_band, coo_index_end_of_cur_row_band);
            // 排序之后两个行号的对应
            vector<vector<unsigned long>> two_direct_index_map_in_band = index_map_after_sorting(nnz_of_each_row_in_band);

            assert(two_direct_index_map_in_band[0].size() == two_direct_index_map_in_band[1].size());
            // 将原来的非零元放到新的位置
            unsigned long nz_index;
            for (nz_index = 0; nz_index < coo_num_of_this_band; nz_index++)
            {
                // cout << "nz_index_inner_block:" << nz_index_inner_block << endl;
                // 全局的coo索引
                unsigned long global_nz_index = nz_index + coo_index_begin_of_cur_row_band;
                assert(global_nz_index < matrix->nnz);
                // 获取全局行号
                unsigned long global_row_index = read_from_array_with_data_type(matrix->coo_row_index_cache, UNSIGNED_LONG, global_nz_index);
                // 全局列号
                unsigned long global_col_index = read_from_array_with_data_type(matrix->coo_col_index_cache, UNSIGNED_LONG, global_nz_index);
                // 全局值
                double global_val = read_double_from_array_with_data_type(matrix->coo_value_cache, matrix->val_data_type, global_nz_index);
                // 获取局部行号
                unsigned long local_row_index = global_row_index - row_index_begin_in_band;
                assert(local_row_index < two_direct_index_map_in_band[1].size());
                // 获取局部行号的新行号，需要老行号与新行号之间的转换
                unsigned long new_row_local_index_cor_to_old = two_direct_index_map_in_band[1][local_row_index];
                // 在二维数组中的新位置存储列号和值
                dim2_inner_band_col_index_vec[new_row_local_index_cor_to_old].push_back(global_col_index);
                dim2_inner_band_var_index_vec[new_row_local_index_cor_to_old].push_back(global_val);
            }

            // 检查新的二维数组中每一行的大小是不是满足要求，一看是不是增序，二看每一行的非零元数量是不是满足要求
            unsigned long new_local_row_index;
            for (new_local_row_index = 0; new_local_row_index < row_num_of_this_band; new_local_row_index++)
            {
                // 对应的旧索引
                unsigned long old_local_row_index = two_direct_index_map_in_band[0][new_local_row_index];
                // 旧索引对应的行非零元数量和转换之后对应行非零元数量进行比较
                assert(nnz_of_each_row_in_band[old_local_row_index] == dim2_inner_band_var_index_vec[new_local_row_index].size() && dim2_inner_band_col_index_vec[new_local_row_index].size() == dim2_inner_band_var_index_vec[new_local_row_index].size());

                // 查看是不是从大到小排序
                if (new_local_row_index < row_num_of_this_band - 1)
                {
                    assert(dim2_inner_band_var_index_vec[new_local_row_index].size() >= dim2_inner_band_var_index_vec[new_local_row_index + 1].size());
                }
            }

            unsigned long global_coo_index_in_band = coo_index_begin_of_cur_row_band;
            // 将排序之后的条带拷贝进去，包含行列和对应的值的数组，以及旧行和新行的对应

            for (new_local_row_index = 0; new_local_row_index < row_num_of_this_band; new_local_row_index++)
            {
                unsigned long new_global_row_index = new_local_row_index + row_index_begin_in_band;

                // 遍历这一行
                unsigned long index_in_this_row;
                for (index_in_this_row = 0; index_in_this_row < dim2_inner_band_col_index_vec[new_local_row_index].size(); index_in_this_row++)
                {
                    assert(global_coo_index_in_band >= coo_index_begin_of_cur_row_band && global_coo_index_in_band <= coo_index_end_of_cur_row_band);
                    // 写行
                    write_to_array_with_data_type(matrix->coo_row_index_cache, UNSIGNED_LONG, global_coo_index_in_band, new_global_row_index);
                    // 写列
                    write_to_array_with_data_type(matrix->coo_col_index_cache, UNSIGNED_LONG, global_coo_index_in_band, dim2_inner_band_col_index_vec[new_local_row_index][index_in_this_row]);
                    // 写值
                    write_double_to_array_with_data_type(matrix->coo_value_cache, matrix->val_data_type, global_coo_index_in_band, dim2_inner_band_var_index_vec[new_local_row_index][index_in_this_row]);

                    global_coo_index_in_band++;
                }

                // 写y的真实位置
                assert(new_local_row_index < two_direct_index_map_in_band[0].size());
                assert(two_direct_index_map_in_band[0][new_local_row_index] + row_index_begin_in_band < matrix->dense_row_number);
                if (new_global_row_index >= matrix->dense_row_number)
                {
                    cout << "new_global_row_index:" << new_global_row_index << ",matrix->dense_row_number:" << matrix->dense_row_number << ",new_local_row_index:" << new_local_row_index << ",coo_index_begin_of_cur_row_band:" << coo_index_begin_of_cur_row_band << endl;
                    cout << "sort_band_index:" << sort_band_index << ",first_row_of_each_sort_band.size():" << first_row_of_each_sort_band.size() << endl;
                    cout << "error" << endl;
                    assert(false);
                }

                write_to_array_with_data_type(matrix->sorted_row_index, matrix->data_type_of_sorted_row_index, new_global_row_index, two_direct_index_map_in_band[0][new_local_row_index] + row_index_begin_in_band);
            }
        }

        // 获取下一个块的起始位置
        coo_index_begin_of_cur_row_band = coo_index_begin_of_next_row_band;
    }
}

void compressed_block_sort(operator_manager_t *op_manager, unsigned long index_of_dense_block_table, vector<unsigned long> sort_block_index_arr)
{
    assert(op_manager != NULL);
    sparse_struct_t *matrix = op_manager->matrix;
    assert(index_of_dense_block_table < matrix->block_coor_table.item_arr.size());
    compressed_block_t *target_compressed_block = matrix->block_coor_table.item_arr[index_of_dense_block_table]->compressed_block_ptr;
    compressed_block_sort(target_compressed_block, sort_block_index_arr);
}

// 对一个子块的某一行或者全局的某一行执行padding，如果是子块，行号就是相对索引
void dense_col_level_padding(operator_manager_t *op_manager, dense_block_table_item_t *sub_block, vector<unsigned long> row_index_vec, vector<unsigned long> padding_target_size_vec)
{
    assert(op_manager != NULL && op_manager->matrix != NULL);
    assert(row_index_vec.size() == padding_target_size_vec.size());

    if (sub_block != NULL)
    {
        assert(padding_target_size_vec.size() == (sub_block->max_dense_row_index - sub_block->min_dense_row_index + 1));
        cout << "sub matrix padding is not supported" << endl;
        assert(false);
        // index_of_compress_block_t *old_global_row_index = sub_block->compressed_block_ptr->read_index[0];

        // // 获取当前子行每一行的行非零元数量
        // vector<unsigned long> nnz_in_each_row_of_block = get_nnz_of_each_row_in_spec_range(old_global_row_index, 0, (sub_block->max_dense_row_index - sub_block->min_dense_row_index), 0, old_global_row_index->length - 1);

        // assert(padding_target_size_vec.size() == nnz_in_each_row_of_block.size());

        // // 当前遍历到的行，

        // // 对每一行执行列方向的padding操作

        // for (int i = 0; i < padding_target_size_vec.size(); i++)
        // {
        //     // 当前行的非零元数量
        //     unsigned long cur_row_nnz = nnz_in_each_row_of_block[i];
        //     unsigned long target_row_nnz = padding_target_size_vec[i];

        // }
    }
    else
    {
        // 针对全局某一行在列方向的padding，遍历每一个非零元，当前非零元所在的行的实际大小和目标大小的差距，然后在不改变列索引排列顺序的情况下均匀地补上非零元
        // 申请三个数组，存储新的坐标和值
        sparse_struct_t *matrix = op_manager->matrix;
        assert(matrix != NULL);
        // 获得每一行的非零元数量
        vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);

        // 将整个数组变为二维的，
        vector<vector<double>> val_dim2_vec = convert_double_arr_to_dim2_vec(matrix->coo_row_index_cache, matrix->coo_value_cache, matrix->val_data_type, matrix->nnz, matrix->dense_row_number);
        vector<vector<unsigned long>> col_dim2_vec = convert_unsigned_long_arr_to_dim2_vec(matrix->coo_row_index_cache, matrix->coo_col_index_cache, matrix->nnz, matrix->dense_row_number);

        // 遍历二维数组的每一行，然后执行padding，padding最后一个非零元的列索引和0
        for (unsigned long i = 0; i < row_index_vec.size(); i++)
        {
            // 当前要处理的行
            unsigned long cur_padding_row = row_index_vec[i];
            unsigned long cur_target_padding_size = padding_target_size_vec[i];

            // 首先目标肯定比现有的大小大
            if (cur_target_padding_size <= val_dim2_vec[cur_padding_row].size())
            {
                cout << "cur_target_padding_size:" << cur_target_padding_size << " val_dim2_vec[cur_padding_row].size():" << val_dim2_vec[cur_padding_row].size() << endl;
            }

            assert(cur_target_padding_size > val_dim2_vec[cur_padding_row].size());
            assert(val_dim2_vec[cur_padding_row].size() == col_dim2_vec[cur_padding_row].size());

            // 一直padding到目标大小
            while (cur_target_padding_size > val_dim2_vec[cur_padding_row].size())
            {
                unsigned long padding_col_index = 0;
                // padding在行的最后，padding是最后一列的列号
                if (col_dim2_vec[cur_padding_row].size() != 0)
                {
                    padding_col_index = col_dim2_vec[cur_padding_row][col_dim2_vec[cur_padding_row].size() - 1];
                }

                col_dim2_vec[cur_padding_row].push_back(padding_col_index);
                val_dim2_vec[cur_padding_row].push_back(0);
            }
        }

        // 查看新的数组的大小
        unsigned long new_nnz = 0;
        for (unsigned long i = 0; i < col_dim2_vec.size(); i++)
        {
            assert(col_dim2_vec[i].size() == val_dim2_vec[i].size());
            new_nnz = new_nnz + col_dim2_vec[i].size();
        }

        // 删除之前的数组
        delete_arr_with_data_type(matrix->coo_row_index_cache, UNSIGNED_LONG);
        delete_arr_with_data_type(matrix->coo_col_index_cache, UNSIGNED_LONG);
        delete_arr_with_data_type(matrix->coo_value_cache, matrix->val_data_type);

        // 申请新的数组
        matrix->coo_row_index_cache = (unsigned long *)malloc_arr(new_nnz, UNSIGNED_LONG);
        matrix->coo_col_index_cache = (unsigned long *)malloc_arr(new_nnz, UNSIGNED_LONG);
        matrix->coo_value_cache = malloc_arr(new_nnz, matrix->val_data_type);

        // 将二维的内容整合到一维数组中
        // 插入的位置
        unsigned long result_insert_position = 0;
        assert(col_dim2_vec.size() == matrix->dense_row_number);
        for (unsigned long i = 0; i < col_dim2_vec.size(); i++)
        {
            // i就是实际意义的行号
            assert(col_dim2_vec[i].size() == val_dim2_vec[i].size());
            for (unsigned long j = 0; j < col_dim2_vec[i].size(); j++)
            {
                matrix->coo_row_index_cache[result_insert_position] = i;
                matrix->coo_col_index_cache[result_insert_position] = col_dim2_vec[i][j];

                write_double_to_array_with_data_type(matrix->coo_value_cache, matrix->val_data_type, result_insert_position, val_dim2_vec[i][j]);

                result_insert_position++;
            }
        }

        matrix->nnz = new_nnz;

        // 最后触发一个检查，看看padding是否成功
        // 查看padding之后每一行的非零元数量
        vector<unsigned long> new_nnz_of_each_row = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);

        // 修改老的行非零元数量，然后对比一下
        for (unsigned long i = 0; i < row_index_vec.size(); i++)
        {
            assert(row_index_vec[i] < nnz_of_each_row.size());
            nnz_of_each_row[row_index_vec[i]] = padding_target_size_vec[i];
        }

        // 现在做对比
        assert(new_nnz_of_each_row.size() == nnz_of_each_row.size());
        // 对比
        for (unsigned long i = 0; i < new_nnz_of_each_row.size(); i++)
        {
            assert(new_nnz_of_each_row[i] == nnz_of_each_row[i]);
        }
    }
}

vector<vector<double>> convert_double_arr_to_dim2_vec(unsigned long *target_arr, void *source_arr, data_type type, unsigned long length, unsigned long dim_1_size)
{
    assert(target_arr != NULL && source_arr != NULL);
    vector<vector<double>> return_dim2_vec(dim_1_size);

    for (unsigned long i = 0; i < length; i++)
    {
        unsigned long cur_target_position = target_arr[i];
        // 要放的东西搞出来
        double cur_contain = read_double_from_array_with_data_type(source_arr, type, i);

        // 将内容放到对应行
        assert(cur_target_position < return_dim2_vec.size());
        return_dim2_vec[cur_target_position].push_back(cur_contain);
    }

    return return_dim2_vec;
}

vector<vector<unsigned long>> convert_unsigned_long_arr_to_dim2_vec(unsigned long *target_arr, unsigned long *source_arr, unsigned long length, unsigned long dim_1_size)
{
    assert(target_arr != NULL && source_arr != NULL);
    vector<vector<unsigned long>> return_dim2_vec(dim_1_size);

    // 遍历数组
    for (unsigned long i = 0; i < length; i++)
    {
        unsigned long cur_target_position = target_arr[i];
        // 要放的东西搞出来
        unsigned long cur_contain = source_arr[i];

        // 将内容放到对应行
        assert(cur_target_position < return_dim2_vec.size());

        if (return_dim2_vec[cur_target_position].size() > 0)
        {
            assert(cur_contain >= return_dim2_vec[cur_target_position][return_dim2_vec[cur_target_position].size() - 1]);
        }

        return_dim2_vec[cur_target_position].push_back(cur_contain);
    }

    return return_dim2_vec;
}

void total_row_level_padding_direct(operator_manager_t *op_manager, unsigned long target_size, unsigned padding_col_num, global_padding_position padding_type, unsigned long input_col_index)
{
    assert(op_manager != NULL && op_manager->matrix != NULL);
    sparse_struct_t *matrix = op_manager->matrix;
    // padding之前不允许排序
    assert(matrix->block_coor_table.item_arr.size() == 0 && matrix->is_sorted == false && matrix->sorted_row_index == NULL);
    assert(padding_type == TOP_PADDING || padding_type == END_PADDING);

    // 当前行数量
    unsigned long cur_dense_row_number = matrix->dense_row_number;
    assert(cur_dense_row_number < target_size);

    // 根据不同的插入位置，放到不同的位置
    if (padding_type == TOP_PADDING)
    {
        // 在头部执行一个padding
        // 目标行号
        unsigned long target_row_number = target_size;
        assert(target_row_number > cur_dense_row_number);

        // 需要增加的行的数量
        unsigned long new_row_number_need_to_be_add = target_row_number - cur_dense_row_number;

        unsigned long new_nnz = new_row_number_need_to_be_add * padding_col_num + matrix->nnz;

        cout << "matrix->nnz:" << matrix->nnz << " new_nnz:" << new_nnz << endl;

        // 重新申请几个更大的数组，并且执行拷贝
        unsigned long *new_coo_row_index_cache = new unsigned long[new_nnz];
        // memcpy_with_data_type(new_coo_row_index_cache, matrix->coo_row_index_cache, matrix->nnz, UNSIGNED_LONG);
        // delete[](matrix->coo_row_index_cache);

        unsigned long *new_coo_col_index_cache = new unsigned long[new_nnz];
        // memcpy_with_data_type(new_coo_col_index_cache, matrix->coo_col_index_cache, matrix->nnz, UNSIGNED_LONG);
        // delete[](matrix->coo_col_index_cache);

        // 值数组
        void *new_coo_value_cache = malloc_arr(new_nnz, matrix->val_data_type);
        // memcpy_with_data_type(new_coo_value_cache, matrix->coo_value_cache, matrix->nnz, matrix->val_data_type);
        // delete_arr_with_data_type((matrix->coo_value_cache), matrix->val_data_type);

        // 当前非零元插入的位置
        unsigned long nz_insert_index = 0;

        // 首先在一开始对每一行增加非零元
        for (unsigned long i = 0; i < new_row_number_need_to_be_add; i++)
        {
            // 一开始为每一行增加空元
            new_coo_row_index_cache[nz_insert_index] = i;
            new_coo_col_index_cache[nz_insert_index] = 0;
            write_double_to_array_with_data_type(new_coo_value_cache, matrix->val_data_type, nz_insert_index, 0);

            nz_insert_index++;
        }

        // 遍历原本的数组，改写行，将内容当道新的数组中
        for (unsigned long i = 0; i < matrix->nnz; i++)
        {
            new_coo_row_index_cache[nz_insert_index] = matrix->coo_row_index_cache[i] + new_row_number_need_to_be_add;
            assert(new_coo_row_index_cache[nz_insert_index] < target_row_number);
            new_coo_col_index_cache[nz_insert_index] = matrix->coo_col_index_cache[i];
            double cur_val = read_double_from_array_with_data_type(matrix->coo_value_cache, matrix->val_data_type, i);
            write_double_to_array_with_data_type(new_coo_value_cache, matrix->val_data_type, nz_insert_index, cur_val);

            nz_insert_index++;
        }

        assert(nz_insert_index == new_nnz);

        // 替换元数据和数组
        matrix->nnz = new_nnz;
        matrix->dense_row_number = target_row_number;

        delete_arr_with_data_type(matrix->coo_row_index_cache, UNSIGNED_LONG);
        delete_arr_with_data_type(matrix->coo_col_index_cache, UNSIGNED_LONG);
        delete_arr_with_data_type(matrix->coo_value_cache, matrix->val_data_type);

        matrix->coo_row_index_cache = new_coo_row_index_cache;
        matrix->coo_col_index_cache = new_coo_col_index_cache;
        matrix->coo_value_cache = new_coo_value_cache;
    }
    else if (padding_type == END_PADDING)
    {
        unsigned long target_row_number = target_size;

        // 需要增加的非零元数量，新增非零元每行一个，所以减一下就可以知道需要增加的非零元数量
        unsigned long new_nnz = (target_row_number - cur_dense_row_number) * padding_col_num + matrix->nnz;

        // 重新申请几个更大的数组，并且执行拷贝
        unsigned long *new_coo_row_index_cache = new unsigned long[new_nnz];
        memcpy_with_data_type(new_coo_row_index_cache, matrix->coo_row_index_cache, matrix->nnz, UNSIGNED_LONG);
        delete[](matrix->coo_row_index_cache);
        matrix->coo_row_index_cache = NULL;

        unsigned long *new_coo_col_index_cache = new unsigned long[new_nnz];
        memcpy_with_data_type(new_coo_col_index_cache, matrix->coo_col_index_cache, matrix->nnz, UNSIGNED_LONG);
        delete[](matrix->coo_col_index_cache);
        matrix->coo_col_index_cache = NULL;

        // 值数组
        void *new_coo_value_cache = malloc_arr(new_nnz, matrix->val_data_type);
        memcpy_with_data_type(new_coo_value_cache, matrix->coo_value_cache, matrix->nnz, matrix->val_data_type);
        delete_arr_with_data_type((matrix->coo_value_cache), matrix->val_data_type);
        matrix->coo_value_cache = NULL;

        // 从当前行号到目标行号一直填充非零元
        // 从一个位置开始增加元素
        unsigned long append_index = matrix->nnz;

        for (unsigned long i = cur_dense_row_number; i < target_row_number; i++)
        {
            // 进行一定数量的padding
            for (unsigned long j = 0; j < padding_col_num; j++)
            {
                new_coo_row_index_cache[append_index] = i;
                assert(input_col_index + j < op_manager->matrix->dense_col_number);
                new_coo_col_index_cache[append_index] = input_col_index + j;
                write_double_to_array_with_data_type(new_coo_value_cache, matrix->val_data_type, append_index, 0);
                append_index++;
            }
        }

        assert(append_index == new_nnz);

        // 替换元数据
        matrix->dense_row_number = target_row_number;
        matrix->nnz = new_nnz;

        // 替换三个数组
        matrix->coo_row_index_cache = new_coo_row_index_cache;
        matrix->coo_col_index_cache = new_coo_col_index_cache;
        matrix->coo_value_cache = new_coo_value_cache;
    }
    else
    {
        cout << "Illegal padding type" << endl;
        assert(false);
    }
}

void total_row_level_padding(operator_manager_t *op_manager, unsigned long multiple, global_padding_position padding_type, unsigned long input_col_index)
{
    // 必须保证还没有进行分块和排序
    assert(op_manager != NULL && op_manager->matrix != NULL);
    sparse_struct_t *matrix = op_manager->matrix;
    assert(matrix->block_coor_table.item_arr.size() == 0 && matrix->is_sorted == false && matrix->sorted_row_index == NULL);
    assert(padding_type == TOP_PADDING || padding_type == END_PADDING);

    // 当前行的数量
    unsigned long cur_dense_row_number = matrix->dense_row_number;

    // 如果已经返回倍数就直接返回
    if (cur_dense_row_number % multiple == 0)
    {
        // 直接返回
        return;
    }

    if (padding_type == END_PADDING)
    {
        // 查看现在的倍数
        unsigned long coef = cur_dense_row_number / multiple + 1;
        // 目标行号
        unsigned long target_row_number = multiple * coef;

        assert(target_row_number > cur_dense_row_number && target_row_number % multiple == 0);

        // 需要增加的非零元数量，新增非零元每行一个，所以减一下就可以知道需要增加的非零元数量
        unsigned long new_nnz = target_row_number - cur_dense_row_number + matrix->nnz;

        // 重新申请几个更大的数组，并且执行拷贝
        unsigned long *new_coo_row_index_cache = new unsigned long[new_nnz];
        memcpy_with_data_type(new_coo_row_index_cache, matrix->coo_row_index_cache, matrix->nnz, UNSIGNED_LONG);
        delete[](matrix->coo_row_index_cache);
        matrix->coo_row_index_cache = NULL;

        unsigned long *new_coo_col_index_cache = new unsigned long[new_nnz];
        memcpy_with_data_type(new_coo_col_index_cache, matrix->coo_col_index_cache, matrix->nnz, UNSIGNED_LONG);
        delete[](matrix->coo_col_index_cache);
        matrix->coo_col_index_cache = NULL;

        // 值数组
        void *new_coo_value_cache = malloc_arr(new_nnz, matrix->val_data_type);
        memcpy_with_data_type(new_coo_value_cache, matrix->coo_value_cache, matrix->nnz, matrix->val_data_type);
        delete_arr_with_data_type((matrix->coo_value_cache), matrix->val_data_type);
        matrix->coo_value_cache = NULL;

        // 从当前行号到目标行号一直填充非零元
        // 从一个位置开始增加元素
        unsigned long append_index = matrix->nnz;
        for (unsigned long i = cur_dense_row_number; i < target_row_number; i++)
        {
            new_coo_row_index_cache[append_index] = i;
            new_coo_col_index_cache[append_index] = input_col_index;
            write_double_to_array_with_data_type(new_coo_value_cache, matrix->val_data_type, append_index, 0);
            append_index++;
        }

        assert(append_index == new_nnz);

        // 替换元数据
        matrix->dense_row_number = target_row_number;
        matrix->nnz = new_nnz;

        // 替换三个数组
        matrix->coo_row_index_cache = new_coo_row_index_cache;
        matrix->coo_col_index_cache = new_coo_col_index_cache;
        matrix->coo_value_cache = new_coo_value_cache;
    }
    else if (padding_type == TOP_PADDING)
    {
        // 查看现在的倍数
        unsigned long coef = cur_dense_row_number / multiple + 1;
        // 目标行号
        unsigned long target_row_number = multiple * coef;
        assert(target_row_number > cur_dense_row_number && target_row_number % multiple == 0);

        // 需要增加的行的数量
        unsigned long new_row_number_need_to_be_add = target_row_number - cur_dense_row_number;

        unsigned long new_nnz = new_row_number_need_to_be_add + matrix->nnz;

        // 重新申请几个更大的数组，并且执行拷贝
        unsigned long *new_coo_row_index_cache = new unsigned long[new_nnz];
        unsigned long *new_coo_col_index_cache = new unsigned long[new_nnz];

        // 值数组
        void *new_coo_value_cache = malloc_arr(new_nnz, matrix->val_data_type);

        // 当前非零元插入的位置
        unsigned long nz_insert_index = 0;

        // 首先在一开始对每一行增加非零元
        for (unsigned long i = 0; i < new_row_number_need_to_be_add; i++)
        {
            // 一开始为每一行增加空元
            new_coo_row_index_cache[nz_insert_index] = i;
            new_coo_col_index_cache[nz_insert_index] = 0;
            write_double_to_array_with_data_type(new_coo_value_cache, matrix->val_data_type, nz_insert_index, 0);

            nz_insert_index++;
        }

        // 遍历原本的数组，改写行，将内容当道新的数组中
        for (unsigned long i = 0; i < matrix->nnz; i++)
        {
            new_coo_row_index_cache[nz_insert_index] = matrix->coo_row_index_cache[i] + new_row_number_need_to_be_add;
            assert(new_coo_row_index_cache[nz_insert_index] < target_row_number);
            new_coo_col_index_cache[nz_insert_index] = matrix->coo_col_index_cache[i];
            double cur_val = read_double_from_array_with_data_type(matrix->coo_value_cache, matrix->val_data_type, i);
            write_double_to_array_with_data_type(new_coo_value_cache, matrix->val_data_type, nz_insert_index, cur_val);

            nz_insert_index++;
        }

        assert(nz_insert_index == new_nnz);

        // 替换元数据和数组
        matrix->nnz = new_nnz;
        matrix->dense_row_number = target_row_number;

        delete_arr_with_data_type(matrix->coo_row_index_cache, UNSIGNED_LONG);
        delete_arr_with_data_type(matrix->coo_col_index_cache, UNSIGNED_LONG);
        delete_arr_with_data_type(matrix->coo_value_cache, matrix->val_data_type);

        matrix->coo_row_index_cache = new_coo_row_index_cache;
        matrix->coo_col_index_cache = new_coo_col_index_cache;
        matrix->coo_value_cache = new_coo_value_cache;
    }
}

vector<unsigned long> total_dense_block_coarse_sort(operator_manager_t *op_manager, vector<unsigned long> bin_row_nnz_low_bound)
{
    // 创造一个数组存储每一个桶的起始行位置，先从计数开始
    vector<unsigned long> row_begin_index_of_each_bin_after_sort;
    row_begin_index_of_each_bin_after_sort.push_back(0);

    // cout << "bin_row_nnz_low_bound.size():" << bin_row_nnz_low_bound.size() << ", bin_row_nnz_low_bound[0]:" << bin_row_nnz_low_bound[0] << endl;
    assert(bin_row_nnz_low_bound.size() > 0 && bin_row_nnz_low_bound[0] == 0);

    sparse_struct_t *matrix = op_manager->matrix;

    assert(matrix != NULL && matrix->is_sorted == false && matrix->sorted_row_index == NULL);

    // 获取每一行的非零元数量
    vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);

    // 遍历每一行的非零元数量，将非零元数量为0的行取出来
    // 用一个数组存储非零元是0的行，用来插到排序数组的末尾，保证其大小是可控的
    vector<unsigned long> zero_row_index;

    for (unsigned long i = 0; i < nnz_of_each_row.size(); i++)
    {
        if (nnz_of_each_row[i] == 0)
        {
            zero_row_index.push_back(i);
        }
    }

    // 记录
    assert(nnz_of_each_row.size() == matrix->dense_row_number);

    // 全是4？
    // for (unsigned long i = 0; i < 20; i++)
    // {
    // if (nnz_of_each_row[i] != 4)
    // {
    // cout << i << "," << nnz_of_each_row[i] << endl;
    // exit(-1);
    // }
    // }

    // exit(-1);

    // 桶的数量
    unsigned long bin_num = bin_row_nnz_low_bound.size();

    // 三个二维数组，分别存储排序之后行、列、值三个索引
    // col和val在排序之后也要交换位置
    vector<vector<unsigned long>> dim2_row_index_vec(bin_num);
    vector<vector<unsigned long>> dim2_col_index_vec(bin_num);
    vector<vector<double>> dim2_var_vec(bin_num);

    unsigned long last_bin_index = 0;

    // 记录删上一行的行号
    unsigned long last_row_index = 0;

    // 遍历所有的非零元，将数据转存到对应的桶内
    for (unsigned long i = 0; i < matrix->nnz; i++)
    {
        // 获取当前行的行号
        unsigned long cur_row_index = read_from_array_with_data_type(matrix->coo_row_index_cache, UNSIGNED_LONG, i);

        assert(last_row_index <= cur_row_index);

        last_row_index = cur_row_index;

        // 当前列号
        unsigned long cur_col_index = read_from_array_with_data_type(matrix->coo_col_index_cache, UNSIGNED_LONG, i);
        // 当前值
        double cur_val = read_double_from_array_with_data_type(matrix->coo_value_cache, matrix->val_data_type, i);

        // 当前行的非零元数量
        unsigned long nnz_of_cur_row = nnz_of_each_row[cur_row_index];

        if (nnz_of_cur_row == 0)
        {
            zero_row_index.push_back(cur_row_index);
            continue;
        }

        unsigned long cur_bin_index;

        // 查看当前非零元数量所对应的桶，
        for (unsigned long j = 0; j < bin_num; j++)
        {
            if (j != bin_num - 1)
            {
                // 查看属于哪个桶
                if (nnz_of_cur_row >= bin_row_nnz_low_bound[j] && nnz_of_cur_row < bin_row_nnz_low_bound[j + 1])
                {
                    cur_bin_index = j;
                    // 退出循环
                    break;
                }
            }
            else
            {
                // 如果前面的都没法满足，那就是放在最后一个
                assert(j == bin_num - 1);
                cur_bin_index = j;
                break;
            }
        }

        last_bin_index = cur_bin_index;

        // 将当前值放到特定桶中
        dim2_row_index_vec[cur_bin_index].push_back(cur_row_index);
        dim2_col_index_vec[cur_bin_index].push_back(cur_col_index);
        dim2_var_vec[cur_bin_index].push_back(cur_val);
    }

    // 检查一下，是不是所有的每一个桶的非零元数量加起来是总的非零元数量
    unsigned long nnz_sum_of_each_bin = 0;
    for (unsigned long i = 0; i < bin_num; i++)
    {
        // 加一下
        // 打印每一个桶的大小
        // cout << dim2_row_index_vec[i].size() << endl;
        nnz_sum_of_each_bin = nnz_sum_of_each_bin + dim2_row_index_vec[i].size();
    }

    assert(nnz_sum_of_each_bin == matrix->nnz);

    // 用一个数组来存储排序之后的原索引
    vector<unsigned long> row_origin_index;

    // 将数据展开成一维的，rowindex要重新排好，同时整理两个数组，row_begin_index_of_each_bin_after_sort
    unsigned long last_row_origin_index;
    unsigned long last_row_new_index;

    // 插入原数组的位置
    unsigned long insert_position = 0;

    // 看看是不是第一个非零元
    bool isFirst = true;

    // 降序排列
    for (int i = bin_num - 1; i >= 0; i--)
    {
        for (unsigned long j = 0; j < dim2_row_index_vec[i].size(); j++)
        {
            if (isFirst == true)
            {
                unsigned long cur_row_origin_index = dim2_row_index_vec[i][j];
                unsigned long cur_col_origin_index = dim2_col_index_vec[i][j];
                double cur_origin_val = dim2_var_vec[i][j];

                // 将数据插入原数组
                assert(insert_position < matrix->nnz);
                write_to_array_with_data_type(matrix->coo_row_index_cache, UNSIGNED_LONG, insert_position, 0);
                write_to_array_with_data_type(matrix->coo_col_index_cache, UNSIGNED_LONG, insert_position, cur_col_origin_index);
                write_double_to_array_with_data_type(matrix->coo_value_cache, matrix->val_data_type, insert_position, cur_origin_val);
                insert_position++;

                last_row_new_index = 0;
                last_row_origin_index = cur_row_origin_index;
                // 登记原行号
                // cout << "row_origin_index.size():" << row_origin_index.size() << " last_row_new_index:" << last_row_new_index << endl;

                assert(row_origin_index.size() == last_row_new_index);
                row_origin_index.push_back(cur_row_origin_index);

                isFirst = false;
            }
            else
            {
                // 不是第一个非零元
                unsigned long cur_row_origin_index = dim2_row_index_vec[i][j];
                unsigned long cur_col_origin_index = dim2_col_index_vec[i][j];
                double cur_origin_val = dim2_var_vec[i][j];

                // 先写两个
                assert(insert_position < matrix->nnz);
                write_to_array_with_data_type(matrix->coo_col_index_cache, UNSIGNED_LONG, insert_position, cur_col_origin_index);
                write_double_to_array_with_data_type(matrix->coo_value_cache, matrix->val_data_type, insert_position, cur_origin_val);

                // 查看行号是不是和之前的同一行，
                if (cur_row_origin_index != last_row_origin_index)
                {
                    //

                    last_row_new_index++;
                    last_row_origin_index = cur_row_origin_index;

                    // 写索引的对应
                    assert(row_origin_index.size() == last_row_new_index);
                    row_origin_index.push_back(cur_row_origin_index);

                    // 写行索引
                    write_to_array_with_data_type(matrix->coo_row_index_cache, UNSIGNED_LONG, insert_position, last_row_new_index);
                }
                else
                {
                    // assert(cur_row_origin_index > );
                    // 如果是同一行，那就写一样的行索引
                    write_to_array_with_data_type(matrix->coo_row_index_cache, UNSIGNED_LONG, insert_position, last_row_new_index);
                }
                insert_position++;
            }
        }

        // 注册下一个桶的第一个行索引
        // cout << last_row_new_index + 1 << endl;
        row_begin_index_of_each_bin_after_sort.push_back(last_row_new_index + 1);
    }

    assert(insert_position == matrix->nnz);

    // 将空行的内容写入映射的末尾
    for (unsigned long i = 0; i < zero_row_index.size(); i++)
    {
        unsigned long cur_zero_row_index = zero_row_index[i];

        row_origin_index.push_back(cur_zero_row_index);
    }

    // 如果有零行，需要将零行的数量累加到最后一个桶中，修成桶首行的最后一个索引
    if (zero_row_index.size() != 0)
    {
        row_begin_index_of_each_bin_after_sort[row_begin_index_of_each_bin_after_sort.size() - 1] = row_begin_index_of_each_bin_after_sort[row_begin_index_of_each_bin_after_sort.size() - 1] + zero_row_index.size();
    }

    assert(row_origin_index.size() == matrix->dense_row_number);

    // 修改密集矩阵的一些信息
    matrix->is_sorted = true;
    // 数据类型
    matrix->data_type_of_sorted_row_index = find_most_suitable_data_type(matrix->dense_row_number);
    matrix->sorted_row_index = malloc_arr(matrix->dense_row_number, matrix->data_type_of_sorted_row_index);
    copy_unsigned_long_arr_to_others(&(row_origin_index[0]), matrix->sorted_row_index, matrix->data_type_of_sorted_row_index, matrix->dense_row_number);

    // 返回每一个bin的起始行号，最后一位是所有行的数量，输出是CSR格式的
    assert(row_begin_index_of_each_bin_after_sort.size() == bin_num + 1);
    // cout << "row_begin_index_of_each_bin_after_sort[row_begin_index_of_each_bin_after_sort.size() - 1]:" << row_begin_index_of_each_bin_after_sort[row_begin_index_of_each_bin_after_sort.size() - 1] << endl;
    // cout << "row_begin_index_of_each_bin_after_sort[row_begin_index_of_each_bin_after_sort.size() - 2]:" << row_begin_index_of_each_bin_after_sort[row_begin_index_of_each_bin_after_sort.size() - 2] << endl;
    // cout << "matrix->dense_row_number:" << matrix->dense_row_number << endl;
    assert(row_begin_index_of_each_bin_after_sort[row_begin_index_of_each_bin_after_sort.size() - 1] == matrix->dense_row_number);
    return row_begin_index_of_each_bin_after_sort;
}

// 
void compress_block_end_block_multiple_padding(operator_manager_t *op_manager, unsigned long compressed_block_id, unsigned long multiple, unsigned long padding_row_length)
{
    // 保证已经被压缩，并且还没有执行分块操作，但是排序可能已经做了
    // 不用额外修改排序产生的元数据，因为padding产生的行不会有任何归约操作，所有不会对写回的行号执行任何读取的操作。
    // 在padding之后是可以排序的，padding前排序和padding后排序产生的子块行索引的数量是不一样的，在padding之后就不能排序了
    // 因为padding的行是假的行，会使得逻辑有一些复杂化（但是并不是不能处理）
    // 所以我们暂时有一个强制的规定，只能先排序后padding
    assert(op_manager != NULL && op_manager->matrix != NULL);
    assert(compressed_block_id < op_manager->matrix->block_coor_table.item_arr.size());
    assert(multiple > 0 && padding_row_length > 0);
    assert(op_manager->matrix->block_coor_table.item_arr[compressed_block_id] != NULL && op_manager->matrix->block_coor_table.item_arr[compressed_block_id]->compressed_block_ptr != NULL);

    dense_block_table_item_t* table_item = op_manager->matrix->block_coor_table.item_arr[compressed_block_id];
    compressed_block_t* compressed_block_ptr = op_manager->matrix->block_coor_table.item_arr[compressed_block_id]->compressed_block_ptr;

    assert(compressed_block_ptr->padding_val_arr == NULL && compressed_block_ptr->staggered_padding_val_arr == NULL);

    // 这里保证我们没有执行过分块，只有最基本的数据
    assert(compressed_block_ptr->read_index.size() == 2);

    // 保证没有在压缩视图上padding过
    assert(table_item->max_dense_row_index == compressed_block_ptr->read_index[0]->max_row_index && compressed_block_ptr->read_index[0]->max_row_index == compressed_block_ptr->read_index[1]->max_row_index);
    assert(table_item->min_dense_row_index == compressed_block_ptr->read_index[0]->min_row_index && compressed_block_ptr->read_index[0]->min_row_index == compressed_block_ptr->read_index[1]->min_row_index);

    // 当前压缩子图的行数量
    unsigned long row_num_of_compressed_block = table_item->max_dense_row_index - table_item->min_dense_row_index + 1;

    // 当前压缩子图的目标数量
    if (row_num_of_compressed_block % multiple == 0)
    {
        return;
    }

    unsigned long target_row_num_of_compressed_block = (row_num_of_compressed_block / multiple + 1) * multiple;
    
    // 需要额外增加额外行号
    unsigned long added_row_num = target_row_num_of_compressed_block - row_num_of_compressed_block;

    index_of_compress_block_t *row_index = compressed_block_ptr->read_index[0];
    index_of_compress_block_t *col_index = compressed_block_ptr->read_index[1];

    // 将索引拷贝到vector中
    vector<unsigned long> new_row_index_vec;
    vector<unsigned long> new_col_index_vec;
    vector<double> new_val_vec;

    assert(row_index->length == col_index->length && col_index->length == compressed_block_ptr->size);
    assert(row_index->index_arr != NULL && col_index->index_arr != NULL && compressed_block_ptr->val_arr != NULL);
    for (unsigned long i = 0; i < row_index->length; i++)
    {
        new_row_index_vec.push_back(read_from_array_with_data_type(row_index->index_arr, row_index->index_data_type, i));
        new_col_index_vec.push_back(read_from_array_with_data_type(col_index->index_arr, col_index->index_data_type, i));
        new_val_vec.push_back(read_double_from_array_with_data_type(compressed_block_ptr->val_arr, compressed_block_ptr->val_data_type, i));
    }

    // 首先是行号， 因为压缩子图中存的是相对行号，所以要从最大行号开始存
    // 外层增加行索引
    // 增加的索引比当前的最后一个索引要大
    assert(row_num_of_compressed_block > new_row_index_vec[new_row_index_vec.size() - 1]);
    for (unsigned long i = row_num_of_compressed_block; i < target_row_num_of_compressed_block; i++)
    {
        // 内部循环，用每一行增加一定数量的列，列号就全是0好了
        for (unsigned long j = 0; j < padding_row_length; j++)
        {
            new_row_index_vec.push_back(i);
            new_col_index_vec.push_back(0);
            new_val_vec.push_back(0);
        }
    }

    // 删除三个索引
    delete_arr_with_data_type(row_index->index_arr, row_index->index_data_type);
    delete_arr_with_data_type(col_index->index_arr, col_index->index_data_type);
    delete_arr_with_data_type(compressed_block_ptr->val_arr, compressed_block_ptr->val_data_type);

    // 检查
    assert((new_row_index_vec[new_row_index_vec.size() - 1] + 1) % multiple == 0);

    // 处理所有行号的索引
    row_index->length = new_row_index_vec.size();
    // 增加要padding的内容
    row_index->max_row_index = row_index->max_row_index + added_row_num;
    // 数据类型
    row_index->index_data_type = find_most_suitable_data_type(row_index->max_row_index - row_index->min_row_index + 1);
    // 申请对应的数组
    row_index->index_arr = malloc_arr(row_index->length, row_index->index_data_type);
    copy_unsigned_long_arr_to_others(&(new_row_index_vec[0]), row_index->index_arr, row_index->index_data_type, row_index->length);
    
    // 处理新的COO列号索引
    col_index->length = new_col_index_vec.size();
    col_index->max_row_index = col_index->max_row_index + added_row_num;
    assert(col_index->max_row_index == row_index->max_row_index);
    // 数据类型是不可能变化的
    col_index->index_arr = malloc_arr(col_index->length, col_index->index_data_type);
    copy_unsigned_long_arr_to_others(&(new_col_index_vec[0]), col_index->index_arr, col_index->index_data_type, col_index->length);

    // 处理新的值数组
    compressed_block_ptr->size = new_val_vec.size();
    compressed_block_ptr->val_arr = malloc_arr(compressed_block_ptr->size, compressed_block_ptr->val_data_type);
    copy_double_arr_to_others(&(new_val_vec[0]), compressed_block_ptr->val_arr, compressed_block_ptr->val_data_type, compressed_block_ptr->size);

    assert(table_item->max_dense_row_index < compressed_block_ptr->read_index[0]->max_row_index && compressed_block_ptr->read_index[0]->max_row_index == compressed_block_ptr->read_index[1]->max_row_index);
    assert(compressed_block_ptr->size == row_index->length && row_index->length == col_index->length);
}
