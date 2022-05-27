#include "dataset_builder.hpp"
#include <assert.h>
#include <set>

dataset_builder_t get_dataset_builder(unsigned long max_row_index, unsigned long max_col_index, unsigned long nnz_in_row)
{
    dataset_builder_t return_builder;

    assert(nnz_in_row <= max_col_index + 1);

    return_builder.max_row_index = max_row_index;
    return_builder.max_col_index = max_col_index;
    return_builder.nnz_in_row = nnz_in_row;

    return return_builder;
}

vector<unsigned long> get_row_index_of_dataset_builder(dataset_builder_t builder)
{
    vector<unsigned long> return_vec;

    // 遍历每一行
    for (unsigned long i = 0; i <= builder.max_row_index; i++)
    {
        // 为每一行添加固定数量的行索引
        for (unsigned long j = 0; j < builder.nnz_in_row; j++)
        {
            return_vec.push_back(i);
        }
    }

    assert(return_vec[return_vec.size() - 1] == builder.max_row_index);

    return return_vec;
}

vector<unsigned long> get_col_index_of_dataset_builder(dataset_builder_t builder)
{
    srand((unsigned)time(NULL));
    vector<unsigned long> return_vec;

    // 处理随机数，保证每一行的非零元数量相同，并且按照从小到大排序
    // 遍历一行
    for (unsigned long row_id = 0; row_id <= builder.max_row_index; row_id++)
    {
        // 申请一个新的set，向set中填随机数
        set<unsigned long> col_index_of_this_row;

        while (col_index_of_this_row.size() < builder.nnz_in_row)
        {
            // 随机的纵坐标
            unsigned long col_index = rand() % (builder.max_col_index + 1);
            assert(col_index <= builder.max_col_index);
            col_index_of_this_row.insert(col_index);
        }

        assert(col_index_of_this_row.size() == builder.nnz_in_row);

        // 将数据放到vec中
        unsigned long last_col_index;
        for (set<unsigned long>::iterator it = col_index_of_this_row.begin(); it != col_index_of_this_row.end(); it++)
        {
            if (it != col_index_of_this_row.begin())
            {
                // 和前一个比较
                assert(last_col_index < *it);
            }
            else
            {
                last_col_index = *it;
            }
            
            return_vec.push_back(*it);
        }
    }

    // for (unsigned long i = 0; i < (builder.max_row_index + 1) * builder.nnz_in_row; i++)
    // {
    //     // 生成一个列坐标
    //     unsigned long col_index = rand() % (builder.max_col_index + 1);
    //     assert(col_index <= builder.max_col_index);
    //     return_vec.push_back(col_index);
    // }
    assert(return_vec.size() == (builder.max_row_index + 1) * builder.nnz_in_row);

    return return_vec;
}

vector<double> get_val_of_dataset_builder(dataset_builder_t builder)
{
    vector<double> return_vec;

    for (unsigned long i = 0; i < (builder.max_row_index + 1) * builder.nnz_in_row; i++)
    {
        return_vec.push_back(1);
    }

    return return_vec;
}

vector<float> get_float_val_of_dataset_builder(dataset_builder_t builder)
{
    vector<float> return_vec;

    for (unsigned long i = 0; i < (builder.max_row_index + 1) * builder.nnz_in_row; i++)
    {
        return_vec.push_back(1);
    }

    return return_vec;
}

vector<double> get_double_val_of_dataset_builder(dataset_builder_t builder)
{
    vector<double> return_vec;

    for (unsigned long i = 0; i < (builder.max_row_index + 1) * builder.nnz_in_row; i++)
    {
        return_vec.push_back(1);
    }

    return return_vec;
}