#include "graph_enumerate.hpp"
#include <assert.h>
#include <iostream>

unsigned long max_long_unsigned(unsigned long a, unsigned long b)
{
    if (a >= b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

// 窗口的前windows_size-1行做一个均值，与最后一个最比较，如果最后一个的大小在均值的div_rate的比率范围之外
// 输出数组中存的是一个分块的第一行的行索引
vector<unsigned long> neighbor_avg_diff_filter(vector<unsigned long> row_nnz, int windows_size, double div_rate)
{
    int top_padding_size = windows_size - 1;
    assert(windows_size >= 2);

    // 要分块的位置
    vector<unsigned long> return_div_position;

    // 上一个分块的位置
    unsigned long last_div_position = 0;

    // 遍历所有的窗口，这里是行索引是padding之后的行索引，会往上挪
    for (unsigned long row_id = 0; row_id < row_nnz.size(); row_id++)
    {
        // 将窗口之前的内容相加，并且算出平均值
        unsigned long row_nnz_sum = 0;

        // 实际加和的次数
        unsigned long sum_count = 0;
        
        // 这里的索引也是padding之后的索引
        for (unsigned long win_content_id = max_long_unsigned(last_div_position, row_id); win_content_id < row_id + windows_size - 1; win_content_id++)
        {
            if (win_content_id > top_padding_size)
            {
                assert((win_content_id - top_padding_size) < row_nnz.size());
            }

            row_nnz_sum = row_nnz_sum + get_val_from_vec_with_padding(&(row_nnz[0]), win_content_id, top_padding_size);
            sum_count++;
        }

        assert(sum_count > 0);

        // 平均值
        double row_nnz_avg = (double)row_nnz_sum / (double)(sum_count);

        unsigned long windows_last_row_id = row_id + windows_size - 1;

        assert(windows_last_row_id - top_padding_size < row_nnz.size());

        // 窗口最后一个的非零元数量
        unsigned long nnz_row_of_windows_last_row = get_val_from_vec_with_padding(&(row_nnz[0]), windows_last_row_id, top_padding_size);

        // 如果小于在div和1/div之外，那就是要划分
        if (nnz_row_of_windows_last_row < row_nnz_avg / div_rate || nnz_row_of_windows_last_row > row_nnz_avg * div_rate)
        {
            // if (row_id == 12)
            // {
            //     cout << "nnz_row_of_windows_last_row:" << nnz_row_of_windows_last_row << endl;
            //     cout << "row_nnz_avg:" << row_nnz_avg << endl;
            //     cout << "row_nnz_sum:" << row_nnz_sum << endl;
            //     cout << "sum_count:" << sum_count << endl;
            //     cout << "windows_last_row_id:" << windows_last_row_id << endl;
            //     cout << "last_div_position:" << last_div_position << endl;
            // }

            // 等级的是padding之前的行索引
            return_div_position.push_back(row_id);
            // 登记这次分块的位置，这个行号使用的是padding之后的行索引
            last_div_position = windows_last_row_id;
        }
    }

    // 最后肯定还有一个分块
    assert(return_div_position.size() > 0 && return_div_position[return_div_position.size() - 1] < row_nnz.size());

    // 在矩阵结束的位置有一个默认的分块点
    return_div_position.push_back(row_nnz.size());
    last_div_position = row_nnz.size();

    return return_div_position;
}

unsigned long get_val_from_vec_with_padding(unsigned long *arr, int read_index, int padding_size)
{
    assert(arr != NULL);

    if (read_index < padding_size)
    {
        return 0;
    }
    else
    {
        return arr[read_index - padding_size];
    }
}

vector<unsigned long> row_div_position_acc_to_exponential_increase_row_nnz_range(vector<unsigned long> row_nnz, unsigned long smallest_row_nnz_range, unsigned long nnz_range_expansion_rate)
{
    assert(row_nnz.size() > 0);
    assert(smallest_row_nnz_range > 0);

    vector<unsigned long> return_div_position;
    // 用一个变量来存储当前行的行非零元上界和下界，只要在这个界限之外，就代表需要需要开一个新的分块点了
    // 上界是不被包含的，下界是被包含的
    unsigned long row_index_low_bound_of_cur_window = 0;
    unsigned long row_index_high_bound_of_cur_window = smallest_row_nnz_range;

    unsigned long cur_row_nnz = row_nnz[0];
    return_div_position.push_back(0);

    // 用一个变量来存遍历的次数
    unsigned long iter_count = 0;

    // 找到对应子窗口，这里代表没有找到对应的窗口
    while (cur_row_nnz < row_index_low_bound_of_cur_window || cur_row_nnz >= row_index_high_bound_of_cur_window)
    {
        iter_count++;
        if (iter_count % 10000 == 0)
        {
            cout << "row_nnz_range_div_position: have loop for " << iter_count << " times" << endl;
        }

        // 在这里代表没有找到对应的行非零元范围的窗口，需要重新调整窗口位置
        row_index_low_bound_of_cur_window = row_index_high_bound_of_cur_window;
        row_index_high_bound_of_cur_window = row_index_high_bound_of_cur_window * nnz_range_expansion_rate;
    }

    // 在这里row_index_low_bound_of_cur_window和row_index_high_bound_of_cur_window得到了第一个窗口。
    // 遍历剩下的行，每当不在之前的区间之内就记录一个新的分块点
    for (unsigned long row_id = 1; row_id < row_nnz.size(); row_id++)
    {
        // 新的行非零元数量
        cur_row_nnz = row_nnz[row_id];

        if (cur_row_nnz >= row_index_low_bound_of_cur_window && cur_row_nnz < row_index_high_bound_of_cur_window)
        {
            // 这里代表当前行的窗口和上一行是一致的
            continue;
        }

        // 这里代表到达了新的行非零元窗口范围，记录为一个块的首行索引，然后然后找出当前行所在的区间
        return_div_position.push_back(row_id);
        row_index_low_bound_of_cur_window = 0;
        row_index_high_bound_of_cur_window = smallest_row_nnz_range;

        // 找到对应的新的非零元数量的窗口
        iter_count = 0;
        while (cur_row_nnz < row_index_low_bound_of_cur_window || cur_row_nnz >= row_index_high_bound_of_cur_window)
        {
            iter_count++;
            if (iter_count % 10000 == 0)
            {
                cout << "row_nnz_range_div_position: have loop for " << iter_count << " times" << endl;
            }

            // 在这里代表没有找到对应的行非零元范围的窗口，需要重新调整窗口位置
            row_index_low_bound_of_cur_window = row_index_high_bound_of_cur_window;
            row_index_high_bound_of_cur_window = row_index_high_bound_of_cur_window * nnz_range_expansion_rate;
        }

        // 到这里已经找到了非零元对应的区间
    }
    
    // 最后肯定还有一个分块
    assert(return_div_position.size() > 0 && return_div_position[return_div_position.size() - 1] < row_nnz.size());

    // 在矩阵结束的位置有一个默认的分块点
    return_div_position.push_back(row_nnz.size());

    return return_div_position;
}

vector<unsigned long> row_div_position_acc_to_exponential_increase_row_nnz_range(vector<unsigned long> row_nnz, unsigned long smallest_row_nnz_range, unsigned long highest_row_nnz_range, unsigned long nnz_range_expansion_rate)
{
    assert(row_nnz.size() > 0);
    assert(smallest_row_nnz_range > 0);
    assert(highest_row_nnz_range >= smallest_row_nnz_range);

    vector<unsigned long> return_div_position;
    // 用一个变量来存储当前行的行非零元上界和下界，只要在这个界限之外，就代表需要需要开一个新的分块点了
    // 上界是不被包含的，下界是被包含的
    unsigned long row_index_low_bound_of_cur_window = 0;
    unsigned long row_index_high_bound_of_cur_window = smallest_row_nnz_range;

    unsigned long cur_row_nnz = row_nnz[0];
    return_div_position.push_back(0);

    // 用一个变量来存遍历的次数
    unsigned long iter_count = 0;

    // 用一个数组来规定行非零元范围
    vector<unsigned long> row_nnz_range;
    row_nnz_range.push_back(0);

    unsigned long row_nnz_bound = smallest_row_nnz_range;

    while (row_nnz_bound <= highest_row_nnz_range)
    {
        row_nnz_range.push_back(row_nnz_bound);
        row_nnz_bound = row_nnz_bound * nnz_range_expansion_rate;
    }

    // 最大值会有一个强制的分界线，如果不存在的话
    if (row_nnz_range[row_nnz_range.size() - 1] != highest_row_nnz_range)
    {
        assert(row_nnz_range[row_nnz_range.size() - 1] < highest_row_nnz_range);
        row_nnz_range.push_back(highest_row_nnz_range);
    }
    
    // 查看第一个块所属的区间
    unsigned long last_range_id = 0;

    // 找到对应子窗口，这里代表没有找到对应的窗口
    for (unsigned long i = 0; i < row_nnz_range.size(); i++)
    {
        if (i < row_nnz_range.size() - 1)
        {
            if (cur_row_nnz >= row_nnz_range[i] && cur_row_nnz < row_nnz_range[i + 1])
            {
                last_range_id = i;
                break;
            }
        }
        else
        {
            // 在最后一个桶
            assert(cur_row_nnz >= highest_row_nnz_range);
            assert(i == row_nnz_range.size() - 1);

            last_range_id = i;
        }
    }

    // 剩下的行所在的范围
    for (unsigned long row_id = 1; row_id < row_nnz.size(); row_id++)
    {
        // 新的行非零元数量
        cur_row_nnz = row_nnz[row_id];

        // 查看当前行所属于的范围
        unsigned long cur_row_range_id = 0;
        
        // 找到对应子窗口，这里代表没有找到对应的窗口
        for (unsigned long i = 0; i < row_nnz_range.size(); i++)
        {
            if (i < row_nnz_range.size() - 1)
            {
                if (cur_row_nnz >= row_nnz_range[i] && cur_row_nnz < row_nnz_range[i + 1])
                {
                    cur_row_range_id = i;
                    break;
                }
            }
            else
            {
                // 在最后一个桶
                assert(cur_row_nnz >= highest_row_nnz_range);
                assert(i == row_nnz_range.size() - 1);

                cur_row_range_id = i;
            }
        }

        // 如果和之前不一样，就代表这里是一个新的划分点
        if (cur_row_range_id != last_range_id)
        {
            return_div_position.push_back(row_id);
            last_range_id = cur_row_range_id;
        }
    }
    
    // 最后肯定还有一个分块
    assert(return_div_position.size() > 0 && return_div_position[return_div_position.size() - 1] < row_nnz.size());

    // 在矩阵结束的位置有一个默认的分块点
    return_div_position.push_back(row_nnz.size());

    return return_div_position;
}

// 每个桶的
vector<unsigned long> bin_row_nnz_low_bound_of_fixed_granularity_coar_sort(vector<unsigned long> row_nnz, unsigned long granularity)
{
    assert(granularity > 0 && row_nnz.size() > 0);

    // 准备一个set，将每一行的非零元数量插入到set中，将行非零元从小到大排序
    set<unsigned long> row_nnz_set;
    vector<unsigned long> bin_nnz_low_bound;

    for (unsigned long i = 0; i < row_nnz.size(); i++)
    {
        row_nnz_set.insert(row_nnz[i]);
    }

    // cout << "bin_row_nnz_low_bound_of_fixed_granularity_coar_sort:" << row_nnz_set.size() << endl;

    // 遍历set中的内容，如果最小的非零元没有0，那么最小桶的非零元数量就是0
    unsigned long set_index = 0;
    for (auto it : row_nnz_set)
    {
        // 如果当前索引和粒度取模之后不是0，就代表，这个行非零元不是桶下界，跳过
        if (set_index % granularity != 0)
        {
            continue;
        }

        // 补一个0，第一个永远填0
        if (bin_nnz_low_bound.size() == 0)
        {
            bin_nnz_low_bound.push_back(0);
        }
        else
        {
            bin_nnz_low_bound.push_back(it);
        }

        set_index++;
    }
    
    assert(bin_nnz_low_bound.size() > 0);
    assert(bin_nnz_low_bound[0] == 0);
    
    return bin_nnz_low_bound;
}

// 子矩阵的列分块，最后看末尾多出多少，然后做不同的处理
vector<vector<unsigned int>> col_block_size_of_each_row(vector<unsigned long> sub_compressed_matrix_row_nnz, unsigned long max_fixed_col_block_size)
{
    assert(max_fixed_col_block_size > 0);

    vector<vector<unsigned int>> return_col_block_size_of_each_row;
    for (auto nnz_of_row : sub_compressed_matrix_row_nnz)
    {
        // 如果是0就跳过
        // 跳过空行
        if (nnz_of_row != 0)
        {
            // 创造一行的列分块
            vector<unsigned int> col_block_size;

            // 查看完整列块的数量
            unsigned long complete_col_block_num = nnz_of_row / max_fixed_col_block_size;

            for (unsigned long i = 0; i < complete_col_block_num; i++)
            {
                // 前面都加入完整列分块大小
                col_block_size.push_back(max_fixed_col_block_size);
            }

            // 查看最后不完整的部分
            unsigned int remain_col_block_size = nnz_of_row % max_fixed_col_block_size;

            // 如果不完整的部分是0，最后一个块会大一点，但是不影响分块的结果

            // 查看最后的部分
            if (remain_col_block_size >= max_fixed_col_block_size / 2)
            {
                // 最后开一个块
                col_block_size.push_back(max_fixed_col_block_size);
            }
            else
            {
                //和已有的块合并在一起
                // 查看有没有最后一个块
                if (col_block_size.size() == 0)
                {
                    // 加入一个完整的列块大小
                    col_block_size.push_back(max_fixed_col_block_size);
                }
                else
                {
                    // 最后一个列块的大小double一下，把尾巴的部分包含进来
                    col_block_size[col_block_size.size() - 1] = col_block_size[col_block_size.size() - 1] * 2;
                    assert(col_block_size[col_block_size.size() - 1] == max_fixed_col_block_size * 2);
                }
            }

            return_col_block_size_of_each_row.push_back(col_block_size);
        }
    }

    // 返回
    return return_col_block_size_of_each_row;
}

// 不合并尾部的边角料，仅仅执行分块
vector<vector<unsigned int>> col_block_size_of_each_row_without_block_merge(vector<unsigned long> sub_compressed_matrix_row_nnz, unsigned long max_fixed_col_block_size)
{
    assert(max_fixed_col_block_size > 0);

    vector<vector<unsigned int>> return_col_block_size_of_each_row;
    for (auto nnz_of_row : sub_compressed_matrix_row_nnz)
    {
        // 如果是0就跳过
        // 跳过空行
        if (nnz_of_row != 0)
        {
            // 创造一行的列分块
            vector<unsigned int> col_block_size;

            // 查看完整列块的数量
            unsigned long complete_col_block_num = nnz_of_row / max_fixed_col_block_size;

            for (unsigned long i = 0; i < complete_col_block_num; i++)
            {
                // 前面都加入完整列分块大小
                col_block_size.push_back(max_fixed_col_block_size);
            }

            // 加一个新的分块，处理可能多出的边角料
            col_block_size.push_back(max_fixed_col_block_size);

            return_col_block_size_of_each_row.push_back(col_block_size);
        }
    }

    // 返回
    return return_col_block_size_of_each_row;
}


// 返回
vector<unsigned long> row_block_size_of_a_sub_matrix_by_fixed_div(unsigned long sub_matrix_row_num, unsigned long fixed_sub_block_size)
{
    assert(sub_matrix_row_num > 0 && fixed_sub_block_size > 0);

    vector<unsigned long> return_row_block_size_of_a_sub_matrix;
    // 首先查看需要多少个完整的子块
    unsigned long complete_sub_block = sub_matrix_row_num / fixed_sub_block_size;

    for (unsigned long i = 0; i < complete_sub_block; i++)
    {
        return_row_block_size_of_a_sub_matrix.push_back(fixed_sub_block_size);
    }

    // 查看剩下的部分
    unsigned long remain_row_num = sub_matrix_row_num % fixed_sub_block_size;

    assert(remain_row_num < fixed_sub_block_size);

    if (remain_row_num > 0)
    {
        // 剩下的部分加入到行块大小的序列中
        return_row_block_size_of_a_sub_matrix.push_back(remain_row_num);
    }

    assert(return_row_block_size_of_a_sub_matrix.size() > 0);

    return return_row_block_size_of_a_sub_matrix;
}

vector<unsigned long> row_block_size_of_a_sub_matrix_by_nnz_low_bound(vector<unsigned long> row_nnz, unsigned long block_nnz_low_bound)
{
    assert(row_nnz.size() > 0 && block_nnz_low_bound > 0);

    // 遍历所有行非零元数量，每超过1024就将遍历过的已有的行划分为一个行条带
    unsigned long accu_row_sum = 0;
    unsigned long accu_nnz_sum = 0;

    // 每个块的行数量
    vector<unsigned long> return_block_row_num_vec;

    // 遍历每一行
    for (unsigned long i = 0; i < row_nnz.size(); i++)
    {
        unsigned long cur_row_nnz = row_nnz[i];

        accu_nnz_sum = accu_nnz_sum + cur_row_nnz;
        accu_row_sum++;

        if (accu_nnz_sum >= block_nnz_low_bound)
        {
            return_block_row_num_vec.push_back(accu_row_sum);
            accu_nnz_sum = 0;
            accu_row_sum = 0;
        }
    }

    // 如果剩下的部分，大于0，那就再把最后的部分做一个分块
    if (accu_row_sum > 0)
    {
        // 如果剩下行没有非零元了，那就和已有的块合并
        if (accu_nnz_sum == 0)
        {
            if (return_block_row_num_vec.size() > 0)
            {
                return_block_row_num_vec[return_block_row_num_vec.size() - 1] = return_block_row_num_vec[return_block_row_num_vec.size() - 1] + accu_row_sum;
            }
            else
            {
                // 整个压缩子矩阵都是空的   
            }
        }
        else
        {
            return_block_row_num_vec.push_back(accu_row_sum);
        }
    }

    return return_block_row_num_vec;
}

vector<vector<unsigned int>> row_block_size_of_each_sub_block_by_fixed_div(vector<unsigned long> each_sub_block_row_num, unsigned long fixed_sub_block_row_size)
{
    assert(each_sub_block_row_num.size() > 0 && fixed_sub_block_row_size > 0);
    
    // 返回的每个子块内部的行号
    vector<vector<unsigned int>> return_row_num_vec_of_each_sub_block;

    for (auto sub_block_total_row_num : each_sub_block_row_num)
    {
        // 新的数组，用来存储每一个子块行号
        vector<unsigned int> row_num_vec_of_this_sub_block;

        for (unsigned long i = 0; i < sub_block_total_row_num / fixed_sub_block_row_size; i++)
        {
            row_num_vec_of_this_sub_block.push_back(fixed_sub_block_row_size);
        }

        if (sub_block_total_row_num % fixed_sub_block_row_size != 0)
        {
            row_num_vec_of_this_sub_block.push_back(sub_block_total_row_num % fixed_sub_block_row_size);
        }

        // 将子块内的行号记录下来
        return_row_num_vec_of_each_sub_block.push_back(row_num_vec_of_this_sub_block);
    }

    assert(return_row_num_vec_of_each_sub_block.size() == each_sub_block_row_num.size());

    return return_row_num_vec_of_each_sub_block;
}

vector<unsigned long> col_div_of_TLB_global_fixed_col_size(unsigned long WLB_block_num, unsigned long global_TLB_col_size)
{
    assert(WLB_block_num > 0 && global_TLB_col_size > 0);
    
    vector<unsigned long> return_TLB_col_size;

    for (unsigned long i = 0; i < WLB_block_num; i++)
    {
        return_TLB_col_size.push_back(global_TLB_col_size);
    }

    return return_TLB_col_size;
}