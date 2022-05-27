#include "struct.hpp"
#include <string>
#include <string.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <cassert>
#include <time.h>
#include <cstdlib>
#include "config.hpp"
using namespace std;

// 打印dense_block_table_item，输入item，是日志输出还是命令行输出
string convert_dense_block_table_item_to_string(dense_block_table_item *item)
{
    // 打印范围
    string block_range = "[[" + to_string(item->min_dense_row_index) + "," + to_string(item->max_dense_row_index) + "][" +
                         to_string(item->min_dense_col_index) + "," + to_string(item->max_dense_col_index) + "][" +
                         to_string(item->begin_coo_index) + "," + to_string(item->end_coo_index) + "]]";

    string return_str = block_range + "(";

    // 首先先打印所有的坐标
    int i;
    for (i = 0; i < item->block_coordinate.size(); i++)
    {
        return_str = return_str + to_string(item->block_coordinate[i]);

        if (i != item->block_coordinate.size() - 1)
        {
            return_str = return_str + ",";
        }
    }
    return_str = return_str + ")" + to_string((long)item->compressed_block_ptr);
    return return_str;
}

// 打印dense_block_table，输入table，是日志输出还是命令行输出
void print_dense_block_table(dense_block_table *table, bool if_log, string log_name)
{
    if (if_log == false)
    {
        // 打印到命令行
        cout << "====="
             << "block coordinate table:"
             << "=====" << endl;
        int i;

        dense_block_table_item_t **item_ptr_array = &(table->item_arr[0]);

        for (i = 0; i < table->item_arr.size(); i++)
        {
            cout << convert_dense_block_table_item_to_string(item_ptr_array[i]) << " | " << endl;
        }
        cout << "=====" << endl;
    }
    else
    {
        // 打印到日志文件
        ofstream outfile(log_name, ios::app);
        outfile << "====="
                << "block coordinate table:"
                << "=====" << endl;
        int i;
        for (i = 0; i < table->item_arr.size(); i++)
        {
            outfile << convert_dense_block_table_item_to_string(table->item_arr[i]) << " | " << endl;
        }
        outfile << "=====" << endl;

        outfile.close();
    }
}

void split(const std::string &s, std::vector<std::string> &sv, const char delim)
{
    sv.clear();
    std::istringstream iss(s);
    std::string temp;

    while (std::getline(iss, temp, delim))
    {
        sv.emplace_back(std::move(temp));
    }

    return;
}

// 将矩阵coo数据的文件取出到内存中
void get_matrix_index_and_val_from_file(string coo_file_name, vector<unsigned long> &row_index_vec, vector<unsigned long> &col_index_vec, vector<float> &float_val_vec, vector<double> &double_val_vec, data_type val_data_type, unsigned long &max_row_index, unsigned long &max_col_index)
{
    assert(val_data_type == FLOAT || val_data_type == DOUBLE);
    assert(row_index_vec.size() == 0 && col_index_vec.size() == 0 && float_val_vec.size() == 0 && double_val_vec.size() == 0);

    max_col_index = 0;
    max_row_index = 0;

    // 读文件
    char buf[1024];

    ifstream infile;
    infile.open(coo_file_name);

    bool dataset_first_line = true;

    if (infile.is_open())
    {
        while (infile.good() && !infile.eof())
        {
            string line_str;
            vector<string> sv;
            memset(buf, 0, 1024);
            infile.getline(buf, 1024);
            line_str = buf;

            // 碰到奇怪的输入就跳过
            if (isspace(line_str[0]) || line_str.empty())
            {
                continue;
            }

            split(line_str, sv);

            // 矩阵的规模，先是最大行号，然后是最大列号
            if (dataset_first_line == true)
            {
                // 矩阵的规模，因为索引从0开始，所以索引的取值范围应该-1
                max_row_index = atol(sv[0].c_str()) - 1;
                max_col_index = atol(sv[1].c_str()) - 1;
                dataset_first_line = false;
                continue;
            }

            // 佛罗里达矩阵先行索引，然后是列索引
            unsigned long row_index = atol(sv[0].c_str()) - 1;
            unsigned long col_index = atol(sv[1].c_str()) - 1;

            // 增序的比较
            if (row_index_vec.size() != 0)
            {
                assert(col_index_vec.size() != 0);
                assert(row_index >= row_index_vec[row_index_vec.size() - 1]);
            }

            row_index_vec.push_back(row_index);
            col_index_vec.push_back(col_index);

            if (val_data_type == DOUBLE)
            {
                double_val_vec.push_back(stod(sv[2].c_str()));
            }
            else if (val_data_type == FLOAT)
            {
                float_val_vec.push_back(stof(sv[2].c_str()));
            }
            else
            {
                printf("unexpected data type\n");
                assert(false);
            }
        }
    }
    else
    {
        cout << "get_matrix_index_and_val_from_file: cannot open file " << coo_file_name << endl;
        assert(false);
    }

    infile.close();

    cout << "finish read file" << endl;
}

// 规定读入的文件，然后决定是单精度还是双精度
// 弗洛里达矩阵是先列后行，第一行是矩阵的大小
sparse_struct_t *init_sparse_struct_by_coo_file(string coo_file_name, data_type value_data_type)
{
    // 首先准备三个vector，分别是行列索引的值
    vector<unsigned long> col_arr;
    vector<unsigned long> row_arr;
    vector<float> val_arr_float;
    vector<double> val_arr_double;

    // 在遍历的过程中找到行和列的最大值
    unsigned long col_index_max = 0;
    unsigned long row_index_max = 0;

    // 读文件
    char buf[1024];

    ifstream infile;
    infile.open(coo_file_name);

    bool dataset_first_line = true;

    if (infile.is_open())
    {
        // 读弗洛里达矩阵格式，第一行是矩阵规模，先是行数量，然后是列数量
        while (infile.good() && !infile.eof())
        {
            string line_str;
            vector<string> sv;
            memset(buf, 0, 1024);
            infile.getline(buf, 1024);
            line_str = buf;

            // 碰到奇怪的输入就跳过
            if (isspace(line_str[0]) || line_str.empty())
            {
                continue;
            }

            split(line_str, sv);

            // 矩阵的规模，先是最大行号，然后是最大列号
            if (dataset_first_line == true)
            {
                // 矩阵的规模，因为索引从0开始，所以索引的取值范围应该-1
                row_index_max = atol(sv[0].c_str()) - 1;
                col_index_max = atol(sv[1].c_str()) - 1;
                dataset_first_line = false;
                continue;
            }

            // 佛罗里达矩阵先行索引，然后是列索引
            unsigned long row_index = atol(sv[0].c_str()) - 1;
            unsigned long col_index = atol(sv[1].c_str()) - 1;

            // if (row_index > row_index_max)
            // {
            //     row_index_max = row_index;
            // }

            // if (col_index > col_index_max)
            // {
            //     col_index_max = col_index;
            // }

            // 增序的比较
            if (row_arr.size() != 0)
            {
                assert(col_arr.size() != 0);
                assert(row_index >= row_arr[row_arr.size() - 1]);
            }

            // 分别是行、列、值
            row_arr.push_back(row_index);
            col_arr.push_back(col_index);

            if (value_data_type == DOUBLE)
            {
                val_arr_double.push_back(stod(sv[2].c_str()));
            }
            else if (value_data_type == FLOAT)
            {
                val_arr_float.push_back(stof(sv[2].c_str()));
            }
            else
            {
                printf("unexpected data type\n");
                assert(false);
            }
        }
    }
    else
    {
        cout << "get_matrix_index_and_val_from_file: cannot open file " << coo_file_name << endl;
        assert(false);
    }

    infile.close();

    // 打印获得的元素
    // int i;
    // for(i = 0;  i < row_arr.size(); i++){

    //     cout << row_arr[i] << " " << col_arr[i] << " ";
    //     if(value_data_type == DOUBLE){
    //         cout << val_arr_double[i] << endl;
    //     }else if(value_data_type == FLOAT){
    //         cout << val_arr_float[i] << endl;
    //     }
    // }

    // return NULL;

    cout << "finish read file" << endl;

    // 初始化这个结构
    return init_sparse_struct_by_coo_vector(row_arr, col_arr, val_arr_float, val_arr_double,
                                            value_data_type, col_index_max, row_index_max);
}

sparse_struct_t *init_sparse_struct_by_coo_vector(vector<unsigned long> row_arr, vector<unsigned long> col_arr,
                                                  vector<float> val_arr_float, vector<double> val_arr_double, data_type value_data_type,
                                                  unsigned long col_index_max, unsigned long row_index_max)
{
    assert(value_data_type == DOUBLE || value_data_type == FLOAT);
    assert(col_index_max >= 0 && row_index_max >= 0);
    // cout << row_arr.size() << " " << col_arr.size() << endl;
    assert(row_arr.size() && col_arr.size());
    assert(row_arr.size() == val_arr_float.size() || row_arr.size() == val_arr_double.size());
    sparse_struct_t *return_struct = new sparse_struct_t();

    // 索引和数量差1
    return_struct->dense_row_number = row_index_max + 1;
    return_struct->dense_col_number = col_index_max + 1;

    return_struct->nnz = row_arr.size();
    return_struct->origin_nnz = return_struct->nnz;

    // 稀疏矩阵现在就一个分块，所以block_coor_table里面必然是空的
    return_struct->is_blocked = false;

    // 还没有被排序过
    return_struct->is_sorted = false;
    return_struct->sorted_row_index = NULL;

    // 在压缩视图中才产生的东西
    return_struct->compressed_block_arr = NULL;

    // 将数据拷贝到数组中，这里要进行值拷贝，所以用memcpy更好
    // return_struct->coo_row_index_cache = &row_arr[0];
    return_struct->coo_col_index_cache = new unsigned long[col_arr.size()];
    return_struct->coo_row_index_cache = new unsigned long[row_arr.size()];
    memcpy(return_struct->coo_row_index_cache, &row_arr[0], row_arr.size() * sizeof(unsigned long));
    memcpy(return_struct->coo_col_index_cache, &col_arr[0], col_arr.size() * sizeof(unsigned long));

    return_struct->val_data_type = value_data_type;

    if (return_struct->val_data_type == DOUBLE)
    {
        return_struct->coo_value_cache = new double[val_arr_double.size()];
        memcpy(return_struct->coo_value_cache, &val_arr_double[0], val_arr_double.size() * sizeof(double));
    }

    if (return_struct->val_data_type == FLOAT)
    {
        return_struct->coo_value_cache = new float[val_arr_float.size()];
        memcpy(return_struct->coo_value_cache, &val_arr_float[0], val_arr_float.size() * sizeof(float));
    }

    cout << "finish init A" << endl;

    // 然后是x的数据，这里的x暂时是0.9就好了
    return_struct->coo_x_cache.x_data_type = value_data_type;

    // 按照列数量申请x
    if (value_data_type == FLOAT)
    {
        return_struct->coo_x_cache.x_arr = new float[return_struct->dense_col_number];
        int i;
        for (i = 0; i < return_struct->dense_col_number; i++)
        {
            ((float *)return_struct->coo_x_cache.x_arr)[i] = 0.9;
        }
    }
    else
    {
        return_struct->coo_x_cache.x_arr = new double[return_struct->dense_col_number];
        int i;
        for (i = 0; i < return_struct->dense_col_number; i++)
        {
            ((double *)return_struct->coo_x_cache.x_arr)[i] = 0.9;
        }
    }

    cout << "finish init x" << endl;

    // 打印所有的数据，保证初始化
    // int i;
    // for(i = 0; i < return_struct->dense_col_number; i++){
    //     if(value_data_type == FLOAT){
    //         cout << ((float *)return_struct->coo_x_cache.x_arr)[i] << ",";
    //     }else{
    //         cout << ((double *)return_struct->coo_x_cache.x_arr)[i] << ",";
    //     }
    // }

    // int i;
    // for (i = 0; i < return_struct->dense_col_number; i++)
    // {
    //     cout << return_struct->coo_col_index_cache[i] << ",";
    //     cout << return_struct->coo_row_index_cache[i] << endl;
    // }

    return return_struct;
}

sparse_struct_t *val_copy_from_old_matrix_struct(sparse_struct_t *matrix)
{
    assert(matrix != NULL);

    sparse_struct_t *return_matrix = new sparse_struct_t();

    // 拷贝最外层的一些基本数据
    return_matrix->dense_row_number = matrix->dense_row_number;
    return_matrix->dense_col_number = matrix->dense_col_number;
    return_matrix->nnz = matrix->nnz;

    return_matrix->origin_nnz = matrix->origin_nnz;

    return_matrix->is_compressed = matrix->is_compressed;
    return_matrix->is_blocked = matrix->is_blocked;

    // 全局的排序索引
    if (matrix->sorted_row_index != NULL)
    {
        assert(matrix->is_sorted == true);

        // 拷贝全局排序索引
        return_matrix->sorted_row_index = val_copy_from_old_arr_with_data_type(matrix->sorted_row_index, matrix->dense_row_number, matrix->data_type_of_sorted_row_index);

        return_matrix->is_sorted = true;
        return_matrix->data_type_of_sorted_row_index = matrix->data_type_of_sorted_row_index;
    }
    else
    {
        assert(matrix->is_sorted == false);
        return_matrix->sorted_row_index = NULL;
        return_matrix->is_sorted = false;
    }

    // 一个被抛弃的指针
    assert(matrix->compressed_block_arr == NULL);

    // COO行索引、列索引、值数组肯定都是存在的
    assert(matrix->coo_value_cache != NULL && matrix->coo_row_index_cache != NULL && matrix->coo_col_index_cache != NULL);

    return_matrix->coo_row_index_cache = (unsigned long *)val_copy_from_old_arr_with_data_type(matrix->coo_row_index_cache, matrix->nnz, UNSIGNED_LONG);
    return_matrix->coo_col_index_cache = (unsigned long *)val_copy_from_old_arr_with_data_type(matrix->coo_col_index_cache, matrix->nnz, UNSIGNED_LONG);

    // 值数组
    return_matrix->val_data_type = matrix->val_data_type;
    return_matrix->coo_value_cache = val_copy_from_old_arr_with_data_type(matrix->coo_value_cache, matrix->nnz, matrix->val_data_type);

    // 如果存在X数组，那就拷贝X数组
    if (return_matrix->coo_x_cache.x_arr != NULL)
    {
        return_matrix->coo_x_cache.x_data_type = matrix->coo_x_cache.x_data_type;

        return_matrix->coo_x_cache.x_arr = val_copy_from_old_arr_with_data_type(return_matrix->coo_x_cache.x_arr, return_matrix->dense_col_number, matrix->coo_x_cache.x_data_type);
    }

    // 子块表
    for (int item_id = 0; item_id < matrix->block_coor_table.item_arr.size(); item_id++)
    {
        // 当前每个子块中都是有确定内容的
        assert(matrix->block_coor_table.item_arr[item_id] != NULL);

        // 首先拷贝对应的表项
        dense_block_table_item_t *new_item = new dense_block_table_item_t();
        dense_block_table_item_t *old_item = matrix->block_coor_table.item_arr[item_id];

        new_item->block_coordinate = old_item->block_coordinate;
        new_item->min_dense_row_index = old_item->min_dense_row_index;
        new_item->max_dense_row_index = old_item->max_dense_row_index;
        new_item->min_dense_col_index = old_item->min_dense_col_index;
        new_item->max_dense_col_index = old_item->max_dense_col_index;
        new_item->begin_coo_index = old_item->begin_coo_index;
        new_item->end_coo_index = old_item->end_coo_index;

        new_item->is_col_blocked = old_item->is_col_blocked;

        new_item->is_sorted = old_item->is_sorted;

        // 如果存在压缩视图，就要进行压缩视图的值拷贝
        if (old_item->compressed_block_ptr != NULL)
        {
            compressed_block_t *new_compressed_block_ptr = new compressed_block_t();
            compressed_block_t *old_compressed_block_ptr = old_item->compressed_block_ptr;

            new_compressed_block_ptr->is_sorted = old_compressed_block_ptr->is_sorted;
            new_compressed_block_ptr->share_row_with_other_block = old_compressed_block_ptr->share_row_with_other_block;
            new_compressed_block_ptr->share_row_with_other_warp = old_compressed_block_ptr->share_row_with_other_warp;
            new_compressed_block_ptr->share_row_with_other_thread = old_compressed_block_ptr->share_row_with_other_thread;

            // 第一个值数组是肯定存在的
            assert(old_compressed_block_ptr->val_arr != NULL);

            // 值数组
            new_compressed_block_ptr->size = old_compressed_block_ptr->size;
            new_compressed_block_ptr->val_data_type = old_compressed_block_ptr->val_data_type;

            new_compressed_block_ptr->val_arr = val_copy_from_old_arr_with_data_type(old_compressed_block_ptr->val_arr, old_compressed_block_ptr->size, old_compressed_block_ptr->val_data_type);

            // 如果剩下两个数组都存在，也要执行拷贝
            if (old_compressed_block_ptr->padding_val_arr != NULL)
            {
                new_compressed_block_ptr->padding_arr_size = old_compressed_block_ptr->padding_arr_size;
                new_compressed_block_ptr->padding_val_arr = val_copy_from_old_arr_with_data_type(old_compressed_block_ptr->padding_val_arr, old_compressed_block_ptr->padding_arr_size, old_compressed_block_ptr->val_data_type);
            }

            if (old_compressed_block_ptr->staggered_padding_val_arr != NULL)
            {
                new_compressed_block_ptr->staggered_padding_val_arr_size = old_compressed_block_ptr->staggered_padding_val_arr_size;
                new_compressed_block_ptr->staggered_padding_val_arr = val_copy_from_old_arr_with_data_type(old_compressed_block_ptr->staggered_padding_val_arr, old_compressed_block_ptr->staggered_padding_val_arr_size, old_compressed_block_ptr->val_data_type);
            }

            // 三个索引指针的数组
            // 第一个是读索引
            for (unsigned long read_index_id = 0; read_index_id < old_compressed_block_ptr->read_index.size(); read_index_id++)
            {
                assert(old_compressed_block_ptr->read_index[read_index_id] != NULL);

                index_of_compress_block_t *old_compress_block_index_ptr = old_compressed_block_ptr->read_index[read_index_id];
                index_of_compress_block_t *new_compress_block_index_ptr = new index_of_compress_block_t();

                new_compress_block_index_ptr->level_of_this_index = old_compress_block_index_ptr->level_of_this_index;
                new_compress_block_index_ptr->index_compressed_type = old_compress_block_index_ptr->index_compressed_type;
                new_compress_block_index_ptr->block_num = old_compress_block_index_ptr->block_num;

                // 索引可能是存在的
                if (old_compress_block_index_ptr->index_arr != NULL)
                {
                    new_compress_block_index_ptr->index_arr = val_copy_from_old_arr_with_data_type(old_compress_block_index_ptr->index_arr, old_compress_block_index_ptr->length, old_compress_block_index_ptr->index_data_type);
                }

                new_compress_block_index_ptr->length = old_compress_block_index_ptr->length;
                new_compress_block_index_ptr->index_data_type = old_compress_block_index_ptr->index_data_type;
                new_compress_block_index_ptr->type_of_index = old_compress_block_index_ptr->type_of_index;

                // is_sort_arr，应该不会被用，如果被使用也是在BLB层次被使用
                assert(old_compress_block_index_ptr->is_sort_arr == NULL);

                if (old_compress_block_index_ptr->is_sort_arr != NULL)
                {
                    assert(read_index_id == 2);
                    new_compress_block_index_ptr->is_sort_arr = (bool *)val_copy_from_old_arr_with_data_type(old_compress_block_index_ptr->is_sort_arr, old_compress_block_index_ptr->block_num, BOOL);
                }

                assert(new_compress_block_index_ptr->is_sort_arr == NULL);

                // 行首索引
                if (old_compress_block_index_ptr->index_of_the_first_row_arr != NULL)
                {
                    // 只在TLB、WLB、BLB三个层次
                    assert(read_index_id == 2 || read_index_id == 3 || read_index_id == 4);
                    new_compress_block_index_ptr->index_of_the_first_row_arr = val_copy_from_old_arr_with_data_type(old_compress_block_index_ptr->index_of_the_first_row_arr, old_compress_block_index_ptr->block_num, old_compress_block_index_ptr->data_type_of_index_of_the_first_row_arr);
                }

                new_compress_block_index_ptr->data_type_of_index_of_the_first_row_arr = old_compress_block_index_ptr->data_type_of_index_of_the_first_row_arr;

                // 每个块的行数量
                if (old_compress_block_index_ptr->row_number_of_block_arr != NULL)
                {
                    // 只在TLB、WLB、BLB三个层次
                    assert(read_index_id == 2 || read_index_id == 3 || read_index_id == 4);
                    new_compress_block_index_ptr->row_number_of_block_arr = val_copy_from_old_arr_with_data_type(old_compress_block_index_ptr->row_number_of_block_arr, old_compress_block_index_ptr->block_num, old_compress_block_index_ptr->data_type_of_row_number_of_block_arr);
                }

                new_compress_block_index_ptr->data_type_of_row_number_of_block_arr = old_compress_block_index_ptr->data_type_of_row_number_of_block_arr;

                // 一个可能不再被使用的归约信息
                if (old_compress_block_index_ptr->tmp_result_write_index_arr != NULL)
                {
                    assert(read_index_id == 3);
                    new_compress_block_index_ptr->tmp_result_write_index_arr = val_copy_from_old_arr_with_data_type(old_compress_block_index_ptr->tmp_result_write_index_arr, old_compress_block_index_ptr->block_num, old_compress_block_index_ptr->data_type_of_tmp_result_write_index_arr);
                }

                new_compress_block_index_ptr->data_type_of_tmp_result_write_index_arr = old_compress_block_index_ptr->data_type_of_tmp_result_write_index_arr;

                new_compress_block_index_ptr->max_row_index = old_compress_block_index_ptr->max_row_index;
                new_compress_block_index_ptr->min_row_index = old_compress_block_index_ptr->min_row_index;
                new_compress_block_index_ptr->max_col_index = old_compress_block_index_ptr->max_col_index;
                new_compress_block_index_ptr->min_col_index = old_compress_block_index_ptr->min_col_index;

                // 块的第一个非零元的索引
                if (old_compress_block_index_ptr->coo_begin_index_arr != NULL)
                {
                    // 只有BLB层次和WLB层次有这两个东西
                    if (old_compress_block_index_ptr->level_of_this_index == TBLOCK_LEVEL)
                    {
                        assert(read_index_id == 2);
                        new_compress_block_index_ptr->coo_begin_index_arr = val_copy_from_old_arr_with_data_type(old_compress_block_index_ptr->coo_begin_index_arr, old_compress_block_index_ptr->length, old_compress_block_index_ptr->data_type_of_coo_begin_index_arr);
                    }
                    else if (old_compress_block_index_ptr->level_of_this_index == WRAP_LEVEL)
                    {
                        assert(read_index_id == 3);
                        new_compress_block_index_ptr->coo_begin_index_arr = val_copy_from_old_arr_with_data_type(old_compress_block_index_ptr->coo_begin_index_arr, old_compress_block_index_ptr->block_num, old_compress_block_index_ptr->data_type_of_coo_begin_index_arr);
                    }
                    else
                    {
                        cout << "should not have coo_begin_index_arr" << endl;
                        assert(false);
                    }
                }

                new_compress_block_index_ptr->data_type_of_coo_begin_index_arr = old_compress_block_index_ptr->data_type_of_coo_begin_index_arr;

                // 每个块的非零元数量
                if (old_compress_block_index_ptr->coo_block_size_arr != NULL)
                {
                    assert(read_index_id == 3 || read_index_id == 4);
                    // 这个数组只在WLB和TLB层次有，数组的大小和WLB数量一致
                    unsigned long warp_block_num = old_compressed_block_ptr->read_index[3]->block_num;
                    new_compress_block_index_ptr->coo_block_size_arr = val_copy_from_old_arr_with_data_type(old_compress_block_index_ptr->coo_block_size_arr, warp_block_num, old_compress_block_index_ptr->data_type_of_coo_block_size_arr);
                }

                new_compress_block_index_ptr->data_type_of_coo_block_size_arr = old_compress_block_index_ptr->data_type_of_coo_block_size_arr;

                if (old_compress_block_index_ptr->child_tmp_row_csr_index_arr != NULL)
                {
                    assert(read_index_id == 2 || read_index_id == 3);
                    new_compress_block_index_ptr->child_tmp_row_csr_index_arr = val_copy_from_old_arr_with_data_type(old_compress_block_index_ptr->child_tmp_row_csr_index_arr, old_compress_block_index_ptr->size_of_child_tmp_row_csr_index, old_compress_block_index_ptr->data_type_of_child_tmp_row_csr_index);
                }

                // 数组大小和数据类型
                new_compress_block_index_ptr->data_type_of_child_tmp_row_csr_index = old_compress_block_index_ptr->data_type_of_child_tmp_row_csr_index;
                new_compress_block_index_ptr->size_of_child_tmp_row_csr_index = old_compress_block_index_ptr->size_of_child_tmp_row_csr_index;

                if (old_compress_block_index_ptr->begin_index_in_tmp_row_csr_arr_of_block != NULL)
                {
                    assert(read_index_id == 2 || read_index_id == 3);
                    new_compress_block_index_ptr->begin_index_in_tmp_row_csr_arr_of_block = val_copy_from_old_arr_with_data_type(old_compress_block_index_ptr->begin_index_in_tmp_row_csr_arr_of_block, old_compress_block_index_ptr->block_num, old_compress_block_index_ptr->data_type_of_begin_index_in_tmp_row_csr_arr_of_block);
                }

                new_compress_block_index_ptr->data_type_of_begin_index_in_tmp_row_csr_arr_of_block = old_compress_block_index_ptr->data_type_of_begin_index_in_tmp_row_csr_arr_of_block;

                // 将读指针写到对应的指针vec中
                new_compressed_block_ptr->read_index.push_back(new_compress_block_index_ptr);
            }

            // 写索引，排序会产生的索引，可能有一个
            assert(old_compressed_block_ptr->y_write_index.size() <= 1);

            for (unsigned long write_index_id = 0; write_index_id < old_compressed_block_ptr->y_write_index.size(); write_index_id++)
            {
                assert(old_compressed_block_ptr->is_sorted == true || old_compressed_block_ptr->read_index[2]->is_sort_arr != NULL);
                assert(old_compressed_block_ptr->y_write_index[write_index_id] != NULL);

                index_of_compress_block_t *old_y_write_index = old_compressed_block_ptr->y_write_index[write_index_id];
                index_of_compress_block_t *new_y_write_index = new index_of_compress_block_t();

                assert(old_y_write_index->index_arr != NULL);

                // 只有index_arr是存在的
                new_y_write_index->index_arr = val_copy_from_old_arr_with_data_type(old_y_write_index->index_arr, old_y_write_index->length, old_y_write_index->index_data_type);
            }

            // 归约索引reduce_help_csr是肯定不存在的
            assert(old_compressed_block_ptr->reduce_help_csr.size() == 0);

            // 执行对应的赋值
            new_item->compressed_block_ptr = new_compressed_block_ptr;
        }

        // 将对应表项记录下来
        return_matrix->block_coor_table.item_arr.push_back(new_item);
    }

    return return_matrix;
}

void output_struct_coo_to_file(sparse_struct_t *matrix_struct, string file_name)
{
    ofstream OsWrite(file_name, ios::out | ios::trunc);

    int i;
    for (i = 0; i < matrix_struct->nnz; i++)
    {
        // 打印
        OsWrite << matrix_struct->coo_row_index_cache[i] << " " << matrix_struct->coo_col_index_cache[i] << " ";

        if (matrix_struct->val_data_type == DOUBLE)
        {
            OsWrite << ((double *)matrix_struct->coo_value_cache)[i] << endl;
        }
        else
        {
            OsWrite << ((float *)matrix_struct->coo_value_cache)[i] << endl;
        }
    }

    OsWrite.close();
}

// 检查分块是不是正确的，主要是查对于coo索引范围的取值范围是不是对的
bool check_dense_block_div(sparse_struct_t *matrix_struct)
{
    // 所有的分块索引
    vector<dense_block_table_item_t *> table_item_arr = matrix_struct->block_coor_table.item_arr;

    // 遍历所有的块，一个个检查
    int i;
    for (i = 0; i < table_item_arr.size(); i++)
    {
        // 遍历所有的分块表
        dense_block_table_item_t *item = table_item_arr[i];

        // 查看是不是增序排列
        unsigned long last_row_index = matrix_struct->coo_row_index_cache[item->begin_coo_index];
        unsigned long last_col_index = matrix_struct->coo_col_index_cache[item->begin_coo_index];

        // 遍历所有coo非零元
        int j;
        for (j = item->begin_coo_index; j <= item->end_coo_index; j++)
        {
            // 查看是不是查看是不是换行了，查看列是不是增序排列
            if (last_row_index < matrix_struct->coo_row_index_cache[j])
            {
                // 换行了，重新计算一下最后一个列号
                last_col_index = matrix_struct->coo_col_index_cache[j];
            }

            if (last_col_index > matrix_struct->coo_col_index_cache[j])
            {
                cout << "col index is not sorted in increased order" << endl;
                return false;
            }

            if (last_row_index > matrix_struct->coo_row_index_cache[j])
            {
                cout << "row index is not sorted in increasing order" << endl;
                return false;
            }

            if (matrix_struct->coo_row_index_cache[j] < item->min_dense_row_index || matrix_struct->coo_row_index_cache[j] > item->max_dense_row_index)
            {
                cout << "row index is not in legal limits" << endl;
                return false;
            }

            if (matrix_struct->coo_col_index_cache[j] < item->min_dense_col_index || matrix_struct->coo_col_index_cache[j] > item->max_dense_col_index)
            {
                cout << "col index is not in legal limits" << endl;
                return false;
            }

            last_row_index = matrix_struct->coo_row_index_cache[j];
        }
    }

    return true;
}

// 将数据类型转化为大写字符串
string convert_data_type_to_string(data_type type)
{
    if (type == CHAR)
    {
        return "CHAR";
    }

    if (type == UNSIGNED_CHAR)
    {
        return "UNSIGNED_CHAR";
    }

    if (type == SHORT)
    {
        return "SHORT";
    }

    if (type == UNSIGNED_SHORT)
    {
        return "UNSIGNED_SHORT";
    }

    if (type == INT)
    {
        return "INT";
    }

    if (type == UNSIGNED_INT)
    {
        return "UNSIGNED_INT";
    }

    if (type == LONG)
    {
        return "LONG";
    }

    if (type == UNSIGNED_LONG)
    {
        return "UNSIGNED_LONG";
    }

    if (type == LONG_LONG)
    {
        return "LONG_LONG";
    }

    if (type == UNSIGNED_LONG_LONG)
    {
        return "UNSIGNED_LONG_LONG";
    }

    if (type == FLOAT)
    {
        return "FLOAT";
    }

    if (type == DOUBLE)
    {
        return "DOUBLE";
    }

    assert(false);
    return "";
}

// 打印数据类型
void print_data_type(data_type type)
{
    if (type == CHAR)
    {
        cout << "CHAR" << endl;
        return;
    }

    if (type == UNSIGNED_CHAR)
    {
        cout << "UNSIGNED_CHAR" << endl;
        return;
    }

    if (type == SHORT)
    {
        cout << "SHORT" << endl;
        return;
    }

    if (type == UNSIGNED_SHORT)
    {
        cout << "UNSIGNED_SHORT" << endl;
        return;
    }

    if (type == INT)
    {
        cout << "INT" << endl;
        return;
    }

    if (type == UNSIGNED_INT)
    {
        cout << "UNSIGNED_INT" << endl;
        return;
    }

    if (type == LONG)
    {
        cout << "LONG" << endl;
        return;
    }

    if (type == UNSIGNED_LONG)
    {
        cout << "UNSIGNED_LONG" << endl;
        return;
    }

    if (type == LONG_LONG)
    {
        cout << "LONG_LONG" << endl;
        return;
    }

    if (type == UNSIGNED_LONG_LONG)
    {
        cout << "UNSIGNED_LONG_LONG" << endl;
        return;
    }

    if (type == FLOAT)
    {
        cout << "FLOAT" << endl;
        return;
    }

    if (type == DOUBLE)
    {
        cout << "DOUBLE" << endl;
        return;
    }
}

string read_str_from_command_line(int argc,char **argv, int cmd_input_index)
{
    assert(cmd_input_index < argc);
    assert(argv != NULL && *argv != NULL);

    const char* str_ptr = argv[cmd_input_index];
    
    return str_ptr;
}

void memcpy_with_data_type(void *dest, void *source, unsigned long source_size, data_type type)
{
    assert(source_size > 0);

    if (type == UNSIGNED_CHAR)
    {
        memcpy(dest, source, sizeof(unsigned char) * source_size);
        return;
    }

    if (type == UNSIGNED_SHORT)
    {
        memcpy(dest, source, sizeof(unsigned short) * source_size);
        return;
    }

    if (type == UNSIGNED_INT)
    {
        memcpy(dest, source, sizeof(unsigned int) * source_size);
        return;
    }

    if (type == UNSIGNED_LONG)
    {
        memcpy(dest, source, sizeof(unsigned long) * source_size);
        return;
    }

    if (type == FLOAT)
    {
        memcpy(dest, source, sizeof(float) * source_size);
        return;
    }

    if (type == DOUBLE)
    {
        memcpy(dest, source, sizeof(double) * source_size);
        return;
    }

    if (type == BOOL)
    {
        memcpy(dest, source, sizeof(bool) * source_size);
        return;
    }

    assert(false);
}

void *val_copy_from_old_arr_with_data_type(void *source, unsigned long source_size, data_type type)
{
    assert(source != NULL && source_size > 0);

    // 只有四种类型 // 还有两种浮点类型
    assert(type == UNSIGNED_CHAR || type == UNSIGNED_INT ||
           type == UNSIGNED_SHORT || type == UNSIGNED_LONG ||
           type == DOUBLE || type == FLOAT || type == BOOL);

    // 申请一个数组
    void *return_arr_ptr = malloc_arr(source_size, type);

    // 将数组拷贝到新的数组中
    memcpy_with_data_type(return_arr_ptr, source, source_size, type);

    assert(return_arr_ptr != NULL);

    return return_arr_ptr;
}

void *malloc_arr(unsigned long length, data_type type_of_arr)
{
    // 只有四种类型 // 还有两种浮点类型
    assert(type_of_arr == UNSIGNED_CHAR || type_of_arr == UNSIGNED_INT ||
           type_of_arr == UNSIGNED_SHORT || type_of_arr == UNSIGNED_LONG ||
           type_of_arr == DOUBLE || type_of_arr == FLOAT || type_of_arr == BOOL ||
           type_of_arr == CHAR || type_of_arr == INT || type_of_arr == SHORT ||
           type_of_arr == LONG);

    assert(length > 0);

    // 申请数组
    if (type_of_arr == UNSIGNED_CHAR)
    {
        // cout << "123" << endl;
        return new unsigned char[length];
    }
    else if (type_of_arr == UNSIGNED_SHORT)
    {
        // cout << "1231" << endl;
        return new unsigned short[length];
    }
    else if (type_of_arr == UNSIGNED_INT)
    {
        // cout << "1232," << length << endl;
        unsigned int *return_ptr = new unsigned int[length];
        // cout << "1" << endl;
        return (void *)return_ptr;
    }
    else if (type_of_arr == DOUBLE)
    {
        // cout << "1233" << endl;
        return new double[length];
    }
    else if (type_of_arr == FLOAT)
    {
        // cout << "1234" << endl;
        return new float[length];
    }
    else if (type_of_arr == BOOL)
    {
        return new bool[length];
    }
    else if (type_of_arr == UNSIGNED_LONG)
    {
        // cout << "1235" << endl;
        return new unsigned long[length];
    }
    else if (type_of_arr == CHAR)
    {
        return new char[length];
    }
    else if (type_of_arr == SHORT)
    {
        return new short[length];
    }
    else if (type_of_arr == INT)
    {
        return new int[length];
    }
    else if (type_of_arr == LONG)
    {
        return new long[length];
    }

    assert(false);
    return NULL;
}

double read_double_from_array_with_data_type(void *arr, data_type type, unsigned long read_pos)
{
    if (type == DOUBLE)
    {
        double *output_arr = (double *)arr;
        return (double)(output_arr[read_pos]);
    }

    if (type == FLOAT)
    {
        float *output_arr = (float *)arr;
        return (double)(output_arr[read_pos]);
    }

    cout << "read_double_from_array_with_data_type: data type is not supported, type:" << type << endl;
    assert(false);

    return 0;
}

unsigned long read_from_array_with_data_type(void *arr, data_type type, unsigned long read_pos)
{
    if (type == UNSIGNED_LONG)
    {
        unsigned long *output_arr = (unsigned long *)arr;
        return (unsigned long)(output_arr[read_pos]);
    }

    if (type == UNSIGNED_INT)
    {
        unsigned int *output_arr = (unsigned int *)arr;
        return (unsigned int)(output_arr[read_pos]);
    }

    if (type == UNSIGNED_SHORT)
    {
        unsigned short *output_arr = (unsigned short *)arr;
        return (unsigned short)(output_arr[read_pos]);
    }

    if (type == UNSIGNED_CHAR)
    {
        unsigned char *output_arr = (unsigned char *)arr;
        return (unsigned char)(output_arr[read_pos]);
    }

    if (type == BOOL)
    {
        bool *output_arr = (bool *)arr;
        return (bool)(output_arr[read_pos]);
    }

    cout << "read_from_array_with_data_type: data type is not supported, type:" << type << endl;
    assert(false);

    return 0;
}




void write_to_array_with_data_type(void *arr, data_type type, unsigned long write_pos, unsigned long write_val)
{
    assert(type == UNSIGNED_LONG || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_CHAR || type == BOOL);
    
    // 处理bool类型
    if (type == BOOL)
    {
        bool* input_arr = (bool*)arr;
        input_arr[write_pos] = write_val;
    }
    
    if (type == UNSIGNED_LONG)
    {
        unsigned long *input_arr = (unsigned long *)arr;
        input_arr[write_pos] = write_val;
    }

    if (type == UNSIGNED_INT)
    {
        unsigned int *input_arr = (unsigned int *)arr;
        input_arr[write_pos] = write_val;
    }

    if (type == UNSIGNED_SHORT)
    {
        unsigned short *input_arr = (unsigned short *)arr;
        input_arr[write_pos] = write_val;
    }

    if (type == UNSIGNED_CHAR)
    {
        unsigned char *input_arr = (unsigned char *)arr;
        input_arr[write_pos] = write_val;
    }
}

// 想浮点数组中写数据
void write_double_to_array_with_data_type(void *arr, data_type type, unsigned long write_pos, double write_val)
{
    assert(type == DOUBLE || type == FLOAT);

    if (type == DOUBLE)
    {
        double *input_arr = (double *)arr;
        input_arr[write_pos] = write_val;
    }

    if (type == FLOAT)
    {
        float *input_arr = (float *)arr;
        input_arr[write_pos] = write_val;
    }
}

void copy_unsigned_long_arr_to_others_with_offset(unsigned long *source_arr, void *dest_ptr, data_type dest_type, unsigned long length, unsigned long dest_offset)
{
    assert(dest_type == UNSIGNED_LONG || dest_type == UNSIGNED_INT || dest_type == UNSIGNED_SHORT || dest_type == UNSIGNED_CHAR);

    if (dest_type == UNSIGNED_LONG)
    {
        unsigned long *dest_arr = (unsigned long *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            // 数据拷贝
            dest_arr[dest_offset + i] = (unsigned long)source_arr[i];
        }
    }

    if (dest_type == UNSIGNED_INT)
    {
        unsigned int *dest_arr = (unsigned int *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[dest_offset + i] = (unsigned long)source_arr[i];
        }
    }

    if (dest_type == UNSIGNED_SHORT)
    {
        unsigned short *dest_arr = (unsigned short *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[dest_offset + i] = (unsigned long)source_arr[i];
        }
    }

    if (dest_type == UNSIGNED_CHAR)
    {
        unsigned char *dest_arr = (unsigned char *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[dest_offset + i] = (unsigned long)source_arr[i];
        }
    }
}

// double数据拷贝
void copy_double_arr_to_others_with_offset(double *source_arr, void *dest_ptr, data_type dest_type, unsigned long length, unsigned long dest_offset)
{
    assert(dest_type == DOUBLE || dest_type == FLOAT);

    if (dest_type == DOUBLE)
    {
        double *dest_arr = (double *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[dest_offset + i] = (double)source_arr[i];
        }
    }

    if (dest_type == FLOAT)
    {
        float *dest_arr = (float *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[dest_offset + i] = (float)source_arr[i];
        }
    }
}

void copy_unsigned_long_arr_to_others(unsigned long *source_arr, void *dest_ptr, data_type type, unsigned long length)
{
    assert(type == UNSIGNED_LONG || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_CHAR);

    if (type == UNSIGNED_LONG)
    {
        unsigned long *dest_arr = (unsigned long *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            // 数据拷贝
            dest_arr[i] = (unsigned long)source_arr[i];
        }
    }

    if (type == UNSIGNED_INT)
    {
        unsigned int *dest_arr = (unsigned int *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[i] = (unsigned int)source_arr[i];
        }
    }

    if (type == UNSIGNED_SHORT)
    {
        unsigned short *dest_arr = (unsigned short *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[i] = (unsigned short)source_arr[i];
        }
    }

    if (type == UNSIGNED_CHAR)
    {
        unsigned char *dest_arr = (unsigned char *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[i] = (unsigned char)source_arr[i];
        }
    }
}

void copy_double_arr_to_others(double *source_arr, void *dest_ptr, data_type type, unsigned long length)
{
    // 将一个double类型的数组拷贝到其他类型的数组中
    assert(type == DOUBLE || type == FLOAT);

    if (type == DOUBLE)
    {
        double *dest_arr = (double *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[i] = (double)source_arr[i];
        }
    }

    if (type == FLOAT)
    {
        float *dest_arr = (float *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[i] = (float)source_arr[i];
        }
    }
}

void copy_unsigned_int_arr_to_others(unsigned int *source_arr, void *dest_ptr, data_type type, unsigned long length)
{
    assert(type == UNSIGNED_LONG || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_CHAR);

    if (type == UNSIGNED_LONG)
    {
        unsigned long *dest_arr = (unsigned long *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            // 数据拷贝
            dest_arr[i] = (unsigned long)source_arr[i];
        }
    }

    if (type == UNSIGNED_INT)
    {
        unsigned int *dest_arr = (unsigned int *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[i] = (unsigned int)source_arr[i];
        }
    }

    if (type == UNSIGNED_SHORT)
    {
        unsigned short *dest_arr = (unsigned short *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[i] = (unsigned short)source_arr[i];
        }
    }

    if (type == UNSIGNED_CHAR)
    {
        unsigned char *dest_arr = (unsigned char *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[i] = (unsigned char)source_arr[i];
        }
    }
}

//  unsigned long数组向其他类型数组的数据拷贝，并且还需要还需要减去同一个偏移量
void copy_unsigned_long_index_to_others(unsigned long *source_arr, void *dest_ptr, data_type type, unsigned long length, unsigned long base_index)
{
    if (type == UNSIGNED_LONG)
    {
        unsigned long *dest_arr = (unsigned long *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[i] = (unsigned long)(source_arr[i] - base_index);
        }
        return;
    }

    if (type == UNSIGNED_INT)
    {
        unsigned int *dest_arr = (unsigned int *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[i] = (unsigned int)(source_arr[i] - base_index);
        }
        return;
    }

    if (type == UNSIGNED_SHORT)
    {
        unsigned short *dest_arr = (unsigned short *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[i] = (unsigned short)(source_arr[i] - base_index);
        }
        return;
    }

    if (type == UNSIGNED_CHAR)
    {
        unsigned char *dest_arr = (unsigned char *)dest_ptr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            dest_arr[i] = (unsigned char)(source_arr[i] - base_index);
        }
        return;
    }
}

void print_compressed_block(sparse_struct_t *matrix_struct, string dir_name)
{
    // 创造一个

    // 块id
    unsigned long block_id = 0;

    // 遍历所有的稠密子块
    int i;
    for (i = 0; i < matrix_struct->block_coor_table.item_arr.size(); i++)
    {
        dense_block_table_item_t *item = matrix_struct->block_coor_table.item_arr[i];
        // 输出多个文件，包含一个元数据文件、多个索引的文件
        // 输出元数据文件
        string file_name = dir_name + "/block_" + to_string(i) + ".meta";
        ofstream metaWrite(file_name, ios::out | ios::trunc);

        metaWrite << "min_dense_row_index:" << item->min_dense_row_index << endl;
        metaWrite << "max_dense_row_index:" << item->max_dense_row_index << endl;
        metaWrite << "min_dense_col_index:" << item->min_dense_col_index << endl;
        metaWrite << "max_dense_col_index:" << item->max_dense_col_index << endl;
        metaWrite << "begin_coo_index:" << item->begin_coo_index << endl;
        metaWrite << "end_coo_index:" << item->end_coo_index << endl;

        // 打印坐标
        metaWrite << "coor:";

        int j;
        for (j = 0; j < item->block_coordinate.size(); j++)
        {
            metaWrite << item->block_coordinate[j];
            if (j < item->block_coordinate.size() - 1)
            {
                metaWrite << ",";
            }
        }

        metaWrite << endl;

        metaWrite.close();

        assert(item->compressed_block_ptr != NULL);
        compressed_block_t *block = item->compressed_block_ptr;
        // 遍历所有索引，分别创建文件打印，针对数据结构list的遍历
        int k;
        for (k = 0; k < block->read_index.size(); k++)
        {
            index_of_compress_block_t *index_ptr = (block->read_index)[k];

            // 创建新文件
            file_name = dir_name + "/block_" + to_string(i) + "_index_read_" + to_string(k);

            // 创建文件
            ofstream indexWrite(file_name, ios::out | ios::trunc);

            // 写不同数据类型的文件
            if (index_ptr->index_data_type == UNSIGNED_LONG)
            {
                unsigned long *output_index_arr = (unsigned long *)(index_ptr->index_arr);
                int l;
                for (l = 0; l < index_ptr->length; l++)
                {
                    indexWrite << output_index_arr[l] << endl;
                }
            }
            else if (index_ptr->index_data_type == UNSIGNED_INT)
            {
                unsigned int *output_index_arr = (unsigned int *)(index_ptr->index_arr);
                int l;
                for (l = 0; l < index_ptr->length; l++)
                {
                    indexWrite << output_index_arr[l] << endl;
                }
            }
            else if (index_ptr->index_data_type == UNSIGNED_SHORT)
            {
                unsigned short *output_index_arr = (unsigned short *)(index_ptr->index_arr);
                int l;
                for (l = 0; l < index_ptr->length; l++)
                {
                    indexWrite << output_index_arr[l] << endl;
                }
            }
            else if (index_ptr->index_data_type == UNSIGNED_CHAR)
            {
                unsigned char *output_index_arr = (unsigned char *)(index_ptr->index_arr);
                int l;
                for (l = 0; l < index_ptr->length; l++)
                {
                    indexWrite << output_index_arr[l] << endl;
                }
            }
            else
            {
                cout << "error" << endl;
                assert(false);
            }

            indexWrite.close();
        }
    }
}

void print_compressed_block_meta_index(index_of_compress_block_t *compressed_block_index, string file_name)
{
    // 现阶段输出三个数组、first_row，first_coo，row_number
    assert(compressed_block_index != NULL);
    assert(compressed_block_index->row_number_of_block_arr != NULL);
    assert(compressed_block_index->index_of_the_first_row_arr != NULL);
    assert(compressed_block_index->coo_begin_index_arr != NULL);
    ofstream fileWrite(file_name, ios::out | ios::trunc);

    // 打印三个数组的版本
    if (compressed_block_index->index_arr == NULL)
    {
        unsigned long i;
        for (i = 0; i < compressed_block_index->block_num; i++)
        {
            fileWrite << read_from_array_with_data_type(compressed_block_index->index_of_the_first_row_arr, compressed_block_index->data_type_of_index_of_the_first_row_arr, i) << "," << read_from_array_with_data_type(compressed_block_index->row_number_of_block_arr, compressed_block_index->data_type_of_row_number_of_block_arr, i) << "," << read_from_array_with_data_type(compressed_block_index->coo_begin_index_arr, compressed_block_index->data_type_of_coo_begin_index_arr, i) << endl;
        }

        if (compressed_block_index->level_of_this_index == TBLOCK_LEVEL)
        {
            fileWrite << ",," << read_from_array_with_data_type(compressed_block_index->coo_begin_index_arr, compressed_block_index->data_type_of_coo_begin_index_arr, i) << endl;
        }
    }
    else
    {
        assert(compressed_block_index->child_tmp_row_csr_index_arr != NULL && compressed_block_index->begin_index_in_tmp_row_csr_arr_of_block != NULL);
        unsigned long i;
        for (i = 0; i < compressed_block_index->block_num; i++)
        {
            fileWrite << read_from_array_with_data_type(compressed_block_index->index_of_the_first_row_arr, compressed_block_index->data_type_of_index_of_the_first_row_arr, i) << "," << read_from_array_with_data_type(compressed_block_index->row_number_of_block_arr, compressed_block_index->data_type_of_row_number_of_block_arr, i) << "," << read_from_array_with_data_type(compressed_block_index->coo_begin_index_arr, compressed_block_index->data_type_of_coo_begin_index_arr, i) << "," << read_from_array_with_data_type(compressed_block_index->index_arr, compressed_block_index->index_data_type, i) << "," << read_from_array_with_data_type(compressed_block_index->begin_index_in_tmp_row_csr_arr_of_block, compressed_block_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block, i) << endl;
        }

        if (compressed_block_index->level_of_this_index == TBLOCK_LEVEL)
        {
            fileWrite << ",," << read_from_array_with_data_type(compressed_block_index->coo_begin_index_arr, compressed_block_index->data_type_of_coo_begin_index_arr, i) << endl;
        }
    }

    fileWrite.close();
}

void delete_arr_with_data_type(void *arr, data_type type)
{
    // cout << convert_data_type_to_string(type) << endl;

    assert(type == UNSIGNED_CHAR || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_LONG || 
        type == FLOAT || type == DOUBLE || type == CHAR || type == SHORT || type == INT || type == LONG || type == BOOL);

    assert(arr != NULL);


    if (type == FLOAT)
    {
        float *del_arr = (float *)arr;
        delete[] del_arr;
        return;
    }

    if (type == DOUBLE)
    {
        double *del_arr = (double *)arr;
        delete[] del_arr;
        return;
    }

    if (type == UNSIGNED_LONG)
    {
        unsigned long *del_arr = (unsigned long *)arr;
        delete[] del_arr;
        return;
    }

    if (type == UNSIGNED_INT)
    {
        unsigned int *del_arr = (unsigned int *)arr;
        delete[] del_arr;
        return;
    }

    if (type == UNSIGNED_SHORT)
    {
        unsigned short *del_arr = (unsigned short *)arr;
        delete[] del_arr;
        return;
    }

    if (type == UNSIGNED_CHAR)
    {
        unsigned char *del_arr = (unsigned char *)arr;
        delete[] del_arr;
        return;
    }

    if (type == BOOL)
    {
        bool *del_arr = (bool *)arr;
        delete[] del_arr;
        return;
    }

    if (type == CHAR)
    {
        char *del_arr = (char *)arr;
        delete[] del_arr;
        return;
    }

    if (type == SHORT)
    {
        short *del_arr = (short *)arr;
        delete[] del_arr;
        return;
    }

    if (type == INT)
    {
        int *del_arr = (int *)arr;
        delete[] del_arr;
        return;
    }

    if (type == LONG)
    {
        long* del_arr = (long *)arr;
        delete[] del_arr;
        return;
    }
}

bool check_tblock_sep(compressed_block_t *compressed_block)
{
    assert(compressed_block != NULL);
    assert(compressed_block->read_index.size() >= 3);
    assert(compressed_block->read_index[2]->level_of_this_index == TBLOCK_LEVEL);
    assert(compressed_block->read_index[2]->index_compressed_type == CSR);

    index_of_compress_block_t *t_block = compressed_block->read_index[2];
    index_of_compress_block_t *global_row_index = compressed_block->read_index[0];

    assert(global_row_index->type_of_index == ROW_INDEX);
    assert(global_row_index->index_compressed_type == COO);

    // 遍历每一个块，做检查
    unsigned long i;
    for (i = 0; i < compressed_block->read_index[2]->block_num; i++)
    {
        // 每个块在coo的起始位置和结束位置
        unsigned long coo_begin = read_from_array_with_data_type(t_block->coo_begin_index_arr, t_block->data_type_of_coo_begin_index_arr, i);
        unsigned long coo_end = read_from_array_with_data_type(t_block->coo_begin_index_arr, t_block->data_type_of_coo_begin_index_arr, i + 1) - 1;
        unsigned long first_row = read_from_array_with_data_type(t_block->index_of_the_first_row_arr, t_block->data_type_of_index_of_the_first_row_arr, i);
        unsigned long row_num = read_from_array_with_data_type(t_block->row_number_of_block_arr, t_block->data_type_of_row_number_of_block_arr, i);

        unsigned long j;
        for (j = coo_begin; j <= coo_end; j++)
        {
            unsigned long cur_global_row_index = read_from_array_with_data_type(global_row_index->index_arr, global_row_index->index_data_type, j);

            // 检查范围是不是满足要求
            if (cur_global_row_index >= first_row && cur_global_row_index < first_row + row_num)
            {
                continue;
            }
            else
            {
                return false;
            }
        }
    }
    return true;
}

void print_arr_to_file_with_data_type(void *arr, data_type type, unsigned long length, string file_name)
{
    assert(type == UNSIGNED_CHAR || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_LONG || type == DOUBLE || type == FLOAT || type == BOOL);
    // 向文件中写文件
    ofstream arrWrite(file_name, ios::out | ios::trunc);

    cout << "file_name:" << file_name << endl;

    if (type == UNSIGNED_CHAR || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_LONG || type == BOOL)
    {
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            arrWrite << read_from_array_with_data_type(arr, type, i) << endl;
        }
    }
    else if (type == DOUBLE || type == FLOAT)
    {
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            arrWrite << read_double_from_array_with_data_type(arr, type, i) << endl;
        }
    }

    arrWrite.close();
}

void print_arr_with_data_type(void *arr, data_type type, unsigned long length)
{
    print_data_type(type);
    assert(type == UNSIGNED_CHAR || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_LONG || type == DOUBLE || type == FLOAT);

    cout << "array:[";

    if (type == DOUBLE)
    {
        double *out_arr = (double *)arr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            cout << out_arr[i] << ",";
        }

        cout << "]" << endl;
    }

    if (type == FLOAT)
    {
        float *out_arr = (float *)arr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            cout << out_arr[i] << ",";
        }

        cout << "]" << endl;
    }

    if (type == UNSIGNED_CHAR)
    {
        unsigned char *out_arr = (unsigned char *)arr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            cout << out_arr[i] << ",";
        }

        cout << "]" << endl;
    }

    if (type == UNSIGNED_SHORT)
    {
        unsigned short *out_arr = (unsigned short *)arr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            cout << out_arr[i] << ",";
        }

        cout << "]" << endl;
    }

    if (type == UNSIGNED_INT)
    {
        unsigned int *out_arr = (unsigned int *)arr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            cout << out_arr[i] << ",";
        }

        cout << "]" << endl;
    }

    if (type == UNSIGNED_LONG)
    {
        unsigned long *out_arr = (unsigned long *)arr;
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            cout << out_arr[i] << ",";
        }

        cout << "]" << endl;
    }
}

// 将数据和元数据写到数组中
unsigned long write_total_matrix_to_file(sparse_struct_t *matrix, string prefix_file_name)
{
    srand(time(0));
    unsigned long matrix_id = rand() + time(0) % 1000;

    // matrix被完全分过块
    assert(matrix != NULL && matrix->coo_col_index_cache != NULL && matrix->coo_row_index_cache != NULL && matrix->block_coor_table.item_arr.size() > 0 && matrix->coo_value_cache != NULL);
    // 遍历所有的分块，保证所有的密集视图都被充分分块
    for (unsigned long i = 0; i < matrix->block_coor_table.item_arr.size(); i++)
    {
        compressed_block_t *sub_block = matrix->block_coor_table.item_arr[i]->compressed_block_ptr;
        assert(sub_block != NULL && sub_block->read_index.size() == 7);
        assert((sub_block->is_sorted == true && sub_block->y_write_index.size() == 1) || (sub_block->is_sorted == false && sub_block->y_write_index.size() == 0));
        assert(sub_block->staggered_padding_val_arr != NULL);
    }

    prefix_file_name = prefix_file_name + "/" + to_string(matrix_id) + "_" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer());

    // 创建这个文件夹
    system(("mkdir " + prefix_file_name).c_str());

    // 创建一个文件来记录所有数组的长度
    ofstream arr_size_file(prefix_file_name + "/" + "size_of_each_arr", ios::out | ios::trunc);

    // 如果有全局的排序，存sorted_row_index
    if (matrix->is_sorted == true)
    {
        string sorted_row_index_filename = prefix_file_name + "/" + "sorted_row_index";
        // cout << "sorted_row_index_filename:" << sorted_row_index_filename << endl;
        arr_size_file << "size_of_sorted_row_index:" << matrix->dense_row_number << endl;
        print_arr_to_file_with_data_type(matrix->sorted_row_index, matrix->data_type_of_sorted_row_index, matrix->dense_row_number, sorted_row_index_filename);
    }

    // 遍历所有的分块，每个分块中的数组分一个文件来存储
    unsigned long dense_block_index;
    for (dense_block_index = 0; dense_block_index < matrix->block_coor_table.item_arr.size(); dense_block_index++)
    {
        // 不存x，如果经过了全局排序，那就存一下sorted_row_index数组
        string prefix_file_name_of_subblock = prefix_file_name + "/dense_block_" + to_string(dense_block_index);

        system(("mkdir " + prefix_file_name_of_subblock).c_str());

        // 当前的块的压缩视图的指针
        compressed_block_t *block_compressed_view = matrix->block_coor_table.item_arr[dense_block_index]->compressed_block_ptr;

        arr_size_file << "size_of_dense_" << dense_block_index << "_staggered_padding_val_arr:" << block_compressed_view->staggered_padding_val_arr_size << endl;

        // 写staggered_padding_val_arr到文件
        print_arr_to_file_with_data_type(block_compressed_view->staggered_padding_val_arr, block_compressed_view->val_data_type, block_compressed_view->staggered_padding_val_arr_size, prefix_file_name_of_subblock + "/" + "staggered_padding_val_arr");

        // 遍历所有的读索引
        assert(block_compressed_view->read_index.size() == 7);
        unsigned long index_of_read_index;
        for (index_of_read_index = 0; index_of_read_index < block_compressed_view->read_index.size(); index_of_read_index++)
        {
            // cout << "index_of_read_index:" << index_of_read_index << endl;
            // 读出一个索引
            index_of_compress_block_t *read_index = block_compressed_view->read_index[index_of_read_index];
            // 索引内容的前缀
            string prefix_file_name_of_index = prefix_file_name_of_subblock + "/" + "read_index_" + to_string(index_of_read_index);

            system(("mkdir " + prefix_file_name_of_index).c_str());
            // 几个索引内容的数组分别是行索引、列索引、block、warp、thread、padding col、padding stagger col
            // 首先是除了thread之外都有的index_arr数组
            // 只要不是空就写
            if (read_index->index_arr != NULL)
            {
                assert(index_of_read_index != 4);
                arr_size_file << "size_of_dense_" << dense_block_index << "_read_index_" << index_of_read_index << "_index_arr:" << read_index->length << endl;
                print_arr_to_file_with_data_type(read_index->index_arr, read_index->index_data_type, read_index->length, prefix_file_name_of_index + "/" + "index_arr");
            }

            // is_sort_arr，这个数组在最终的程序中可能不会用来，只是在是有一个块被排序的时候才会用到
            if (read_index->is_sort_arr != NULL)
            {
                assert(index_of_read_index == 2);
                arr_size_file << "size_of_dense_" << dense_block_index << "_read_index_" << index_of_read_index << "_is_sort_arr:" << read_index->block_num << endl;
                print_arr_to_file_with_data_type(read_index->is_sort_arr, BOOL, read_index->block_num, prefix_file_name_of_index + "/" + "is_sort_arr");
            }

            // index_of_the_first_row_arr
            if (read_index->index_of_the_first_row_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3 || index_of_read_index == 4);
                arr_size_file << "size_of_dense_" << dense_block_index << "_read_index_" << index_of_read_index << "_index_of_the_first_row_arr:" << read_index->block_num << endl;
                print_arr_to_file_with_data_type(read_index->index_of_the_first_row_arr, read_index->data_type_of_index_of_the_first_row_arr, read_index->block_num, prefix_file_name_of_index + "/" + "index_of_the_first_row_arr");
            }

            // row_number_of_block_arr
            if (read_index->row_number_of_block_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3 || index_of_read_index == 4);
                arr_size_file << "size_of_dense_" << dense_block_index << "_read_index_" << index_of_read_index << "_row_number_of_block_arr:" << read_index->block_num << endl;
                print_arr_to_file_with_data_type(read_index->row_number_of_block_arr, read_index->data_type_of_row_number_of_block_arr, read_index->block_num, prefix_file_name_of_index + "/" + "row_number_of_block_arr");
            }

            // tmp_result_write_index_arr，仅在warp列分块的时候是需要的，之后采用同步的归约的方式所必须的数组，因为当前层次会把所有的中间结果暂时存下来。
            // 在对仅行分块以及经过列分块的tblock的来说，如果要用全局同步的方式来归约不同tblock的结果，那么这个数组也是需要的，
            if (read_index->tmp_result_write_index_arr != NULL)
            {
                assert(index_of_read_index == 3);
                arr_size_file << "size_of_dense_" << dense_block_index << "_read_index_" << index_of_read_index << "_tmp_result_write_index_arr:" << read_index->block_num << endl;
                print_arr_to_file_with_data_type(read_index->tmp_result_write_index_arr, read_index->data_type_of_tmp_result_write_index_arr, read_index->block_num, prefix_file_name_of_index + "/" + "tmp_result_write_index_arr");
            }

            // coo_begin_index_arr
            if (read_index->coo_begin_index_arr != NULL)
            {
                // tblock层次和warp层次这两个数组的大小不一样
                if (read_index->level_of_this_index == TBLOCK_LEVEL)
                {
                    assert(index_of_read_index == 2);
                    arr_size_file << "size_of_dense_" << dense_block_index << "_read_index_" << index_of_read_index << "_coo_begin_index_arr:" << read_index->length << endl;
                    print_arr_to_file_with_data_type(read_index->coo_begin_index_arr, read_index->data_type_of_coo_begin_index_arr, read_index->length, prefix_file_name_of_index + "/" + "coo_begin_index_arr");
                }
                else if (read_index->level_of_this_index == WRAP_LEVEL)
                {
                    assert(index_of_read_index == 3);
                    arr_size_file << "size_of_dense_" << dense_block_index << "_read_index_" << index_of_read_index << "_coo_begin_index_arr:" << read_index->block_num << endl;
                    print_arr_to_file_with_data_type(read_index->coo_begin_index_arr, read_index->data_type_of_coo_begin_index_arr, read_index->block_num, prefix_file_name_of_index + "/" + "coo_begin_index_arr");
                }
                else
                {
                    cout << "should not have coo_begin_index_arr" << endl;
                    assert(false);
                }
            }

            // coo_block_size_arr
            if (read_index->coo_block_size_arr != NULL)
            {
                // 线程粒度和warp粒度的块存在这些东西，大小都是warp块的大小
                assert(index_of_read_index == 3 || index_of_read_index == 4);
                assert(block_compressed_view->read_index[3]->level_of_this_index == WRAP_LEVEL);
                unsigned long warp_block_num = block_compressed_view->read_index[3]->block_num;
                arr_size_file << "size_of_dense_" << dense_block_index << "_read_index_" << index_of_read_index << "_coo_block_size_arr:" << warp_block_num << endl;
                print_arr_to_file_with_data_type(read_index->coo_block_size_arr, read_index->data_type_of_coo_block_size_arr, warp_block_num, prefix_file_name_of_index + "/" + "coo_block_size_arr");
            }

            // child_tmp_row_csr_index_arr
            if (read_index->child_tmp_row_csr_index_arr != NULL)
            {
                // 只有tblock和warp层次需要有归约信息
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                arr_size_file << "size_of_dense_" << dense_block_index << "_read_index_" << index_of_read_index << "_child_tmp_row_csr_index_arr:" << read_index->size_of_child_tmp_row_csr_index << endl;
                print_arr_to_file_with_data_type(read_index->child_tmp_row_csr_index_arr, read_index->data_type_of_child_tmp_row_csr_index, read_index->size_of_child_tmp_row_csr_index, prefix_file_name_of_index + "/" + "child_tmp_row_csr_index_arr");
            }

            // begin_index_in_tmp_row_csr_arr_of_block
            if (read_index->begin_index_in_tmp_row_csr_arr_of_block != NULL)
            {
                // 每个tblock或者warp归约索引的偏移量，child数组就是所谓的“归约索引”，每个block和warp在执行全局同步的归约时，都需要找到自己的归约索引
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                arr_size_file << "size_of_dense_" << dense_block_index << "_read_index_" << index_of_read_index << "_begin_index_in_tmp_row_csr_arr_of_block:" << read_index->block_num << endl;
                print_arr_to_file_with_data_type(read_index->begin_index_in_tmp_row_csr_arr_of_block, read_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block, read_index->block_num, prefix_file_name_of_index + "/" + "begin_index_in_tmp_row_csr_arr_of_block");
            }
        }

        // 遍历所有的y write index
        // assert((block_compressed_view->is_sorted == false && block_compressed_view->y_write_index.size() == 0) || (block_compressed_view->is_sorted == true && block_compressed_view->y_write_index.size() > 0));

        unsigned long y_write_arr_index;
        for (y_write_arr_index = 0; y_write_arr_index < block_compressed_view->y_write_index.size(); y_write_arr_index++)
        {
            // 出现了部分或者整体的压缩
            assert(block_compressed_view->is_sorted == true || block_compressed_view->read_index[2]->is_sort_arr != NULL);

            string prefix_file_name_of_index = prefix_file_name_of_subblock + "/" + "y_write_arr_index_" + to_string(y_write_arr_index);

            system(("mkdir " + prefix_file_name_of_index).c_str());

            index_of_compress_block_t *cur_y_write_index = block_compressed_view->y_write_index[y_write_arr_index];
            // 打印index_arr即可
            assert(cur_y_write_index->index_arr != NULL);

            arr_size_file << "size_of_dense_" << dense_block_index << "_y_write_arr_index_" << y_write_arr_index << "_index_arr:" << cur_y_write_index->length << endl;
            // 将这个数组输出
            print_arr_to_file_with_data_type(cur_y_write_index->index_arr, cur_y_write_index->index_data_type, cur_y_write_index->length, prefix_file_name_of_index + "/" + "index_arr");
        }

        // 输出所有padding之后的非零元值
        assert(block_compressed_view->staggered_padding_val_arr != NULL);
        arr_size_file << "size_of_dense_" << dense_block_index << "_staggered_padding_val_arr:" << block_compressed_view->staggered_padding_val_arr_size << endl;
        print_arr_to_file_with_data_type(block_compressed_view->staggered_padding_val_arr, block_compressed_view->val_data_type, block_compressed_view->staggered_padding_val_arr_size, prefix_file_name_of_subblock + "/" + "staggered_padding_val_arr");
    }

    arr_size_file.close();

    return matrix_id;
}

void *read_arr_from_file_with_data_type(unsigned long length, data_type arr_data_type, string file_name)
{
    assert(length > 0);
    assert(arr_data_type == UNSIGNED_LONG || arr_data_type == UNSIGNED_INT || arr_data_type == UNSIGNED_SHORT || arr_data_type == UNSIGNED_CHAR ||
           arr_data_type == BOOL || arr_data_type == FLOAT || arr_data_type == DOUBLE);

    // 创建一个特定数据类型的数组
    void *arr_need_to_return = malloc_arr(length, arr_data_type);

    if (arr_data_type == UNSIGNED_LONG || arr_data_type == UNSIGNED_INT || arr_data_type == UNSIGNED_SHORT || arr_data_type == UNSIGNED_CHAR || arr_data_type == BOOL)
    {
        // 读文件
        char buf[1024];

        ifstream infile;
        infile.open(file_name);

        unsigned long cur_insert_index = 0;

        if (infile.is_open())
        {
            // 读弗洛里达矩阵格式，第一行是矩阵规模，先是行数量，然后是列数量
            while (infile.good() && !infile.eof())
            {
                string line_str;
                memset(buf, 0, 1024);
                infile.getline(buf, 1024);
                line_str = buf;

                // 碰到奇怪的输入就跳过
                if (isspace(line_str[0]) || line_str.empty())
                {
                    continue;
                }

                // 佛罗里达矩阵先行索引，然后是列索引
                unsigned long arr_val = atol(line_str.c_str());

                assert(cur_insert_index < length);
                write_to_array_with_data_type(arr_need_to_return, arr_data_type, cur_insert_index, arr_val);

                cur_insert_index++;
            }
        }

        infile.close();
        return arr_need_to_return;
    }
    else if (arr_data_type == DOUBLE || arr_data_type == FLOAT)
    {
        // 读文件
        char buf[1024];

        ifstream infile;
        infile.open(file_name);

        unsigned long cur_insert_index = 0;

        // 读弗洛里达矩阵格式，第一行是矩阵规模，先是行数量，然后是列数量
        while (infile.good() && !infile.eof())
        {
            string line_str;
            memset(buf, 0, 1024);
            infile.getline(buf, 1024);
            line_str = buf;

            // 碰到奇怪的输入就跳过
            if (isspace(line_str[0]) || line_str.empty())
            {
                continue;
            }

            // 佛罗里达矩阵先行索引，然后是列索引
            double arr_val = stod(line_str.c_str());

            assert(cur_insert_index < length);
            write_double_to_array_with_data_type(arr_need_to_return, arr_data_type, cur_insert_index, arr_val);

            cur_insert_index++;
        }

        infile.close();
        return arr_need_to_return;
    }

    return arr_need_to_return;
}

// void write_string_to_file(string file_name, string output_str){
//     ofstream outfile(file_name, ios::trunc);
//     outfile << output_str << endl;
// }

data_type find_most_suitable_data_type_by_bit_num(int bit_num)
{
    assert(bit_num > 0);

    if (bit_num <= sizeof(unsigned char) * 8)
    {
        return UNSIGNED_CHAR;
    }

    if (bit_num <= sizeof(unsigned short) * 8)
    {
        return UNSIGNED_SHORT;
    }

    if (bit_num <= sizeof(unsigned int) * 8)
    {
        return UNSIGNED_INT;
    }

    if (bit_num <= sizeof(unsigned long) * 8)
    {
        return UNSIGNED_LONG;
    }

    cout << "find_most_suitable_data_type_by_bit_num : bit num is to large," << bit_num << endl;
    assert(false);
}

int bit_num_of_data_type(data_type type)
{
    assert(type == UNSIGNED_LONG || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_CHAR);

    if (type == UNSIGNED_LONG)
    {
        return sizeof(unsigned long) * 8;
    }

    if (type == UNSIGNED_INT)
    {
        return sizeof(unsigned int) * 8;
    }

    if (type == UNSIGNED_SHORT)
    {
        return sizeof(unsigned short) * 8;
    }

    if (type == UNSIGNED_CHAR)
    {
        return sizeof(unsigned char) * 8;
    }

    cout << "bit_num_of_data_type: data type is not supported" << endl;
    assert(false);
}

// 不包含bool类型
// https://blog.csdn.net/Chaolei3/article/details/79118329
unsigned long get_max_of_a_integer_data_type(data_type type)
{
    // 必须是整型
    assert(type == CHAR || type == UNSIGNED_CHAR || type == SHORT || type == UNSIGNED_SHORT || type == INT ||
           type == UNSIGNED_INT || type == LONG || type == UNSIGNED_LONG || type == UNSIGNED_LONG_LONG || type == LONG_LONG);

    if (type == CHAR)
    {
        return CHAR_MAX;
    }

    if (type == UNSIGNED_CHAR)
    {
        return UCHAR_MAX;
    }

    if (type == SHORT)
    {
        return SHRT_MAX;
    }

    if (type == UNSIGNED_SHORT)
    {
        return USHRT_MAX;
    }

    if (type == INT)
    {
        return INT_MAX;
    }

    if (type == UNSIGNED_INT)
    {
        return UINT_MAX;
    }

    if (type == LONG)
    {
        return LONG_MAX;
    }

    if (type == UNSIGNED_LONG)
    {
        return ULONG_MAX;
    }

    if (type == LONG_LONG)
    {
        return LONG_LONG_MAX;
    }

    if (type == UNSIGNED_LONG_LONG)
    {
        return ULONG_LONG_MAX;
    }

    assert(false);
    return 0;
}

unsigned long get_min_of_a_integer_data_type(data_type type)
{
    assert(type == CHAR || type == UNSIGNED_CHAR || type == SHORT || type == UNSIGNED_SHORT || type == INT ||
           type == UNSIGNED_INT || type == LONG || type == UNSIGNED_LONG || type == UNSIGNED_LONG_LONG || type == LONG_LONG);

    if (type == CHAR)
    {
        return CHAR_MIN;
    }

    if (type == UNSIGNED_CHAR)
    {
        return 0;
    }

    if (type == SHORT)
    {
        return SHRT_MIN;
    }

    if (type == UNSIGNED_SHORT)
    {
        return 0;
    }

    if (type == INT)
    {
        return INT_MIN;
    }

    if (type == UNSIGNED_INT)
    {
        return 0;
    }

    if (type == LONG)
    {
        return LONG_MIN;
    }

    if (type == UNSIGNED_LONG)
    {
        return 0;
    }

    if (type == LONG_LONG)
    {
        return LONG_LONG_MIN;
    }

    if (type == UNSIGNED_LONG_LONG)
    {
        return 0;
    }

    assert(false);
    return 0;
}

double get_max_of_a_float_data_type(data_type type)
{
    assert(type == FLOAT || type == DOUBLE);

    if (type == FLOAT)
    {
        return FLT_MAX;
    }

    if (type == DOUBLE)
    {
        return DBL_MAX;
    }

    assert(false);
    return 0;
}

double get_min_of_a_float_data_type(data_type type)
{
    assert(type == FLOAT || type == DOUBLE);

    if (type == FLOAT)
    {
        return FLT_MIN;
    }

    if (type == DOUBLE)
    {
        return DBL_MIN;
    }

    assert(false);
    return 0;
}

data_type get_data_type_from_type_info(const type_info& input_info)
{
    if (input_info == typeid(char))
    {
        return CHAR;
    }
    
    if (input_info == typeid(unsigned char))
    {
        return UNSIGNED_CHAR;
    }

    if (input_info == typeid(short))
    {
        return SHORT;
    }

    if (input_info == typeid(unsigned short))
    {
        return UNSIGNED_SHORT;
    }

    if (input_info == typeid(int))
    {
        return INT;
    }

    if (input_info == typeid(unsigned int))
    {
        return UNSIGNED_INT;
    }

    if (input_info == typeid(long))
    {
        return LONG;
    }

    if (input_info == typeid(unsigned long))
    {
        return UNSIGNED_LONG;
    }

    if (input_info == typeid(bool))
    {
        return BOOL;
    }

    if (input_info == typeid(float))
    {
        return FLOAT;
    }

    if (input_info == typeid(double))
    {
        return DOUBLE;
    }

    assert(false);
}
