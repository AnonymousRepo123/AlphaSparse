#include <bits/stdc++.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <string.h>
#include <sys/time.h>
#include <vector>

using namespace std;


// 将COO文件读入，改变数组的内容
void get_coo_from_file(string file_name, vector<unsigned int> &row_index, vector<unsigned int> &col_index, vector<float> &val_arr, unsigned int& row_num, unsigned int& col_num, unsigned int& val_num)
{
    assert(row_index.size() == 0);
    assert(col_index.size() == 0);
    assert(val_arr.size() == 0);

    std::ifstream fin(file_name.c_str());
	if (!fin)
	{
		cout << "File Not found\n";
		exit(0);
	}

    while (fin.peek() == '%')
    {
        fin.ignore(2048, '\n');
    }

    fin >> row_num >> col_num >> val_num;

    // 读文件，主要是读行的和列
    for (unsigned int l = 0; l < val_num; l++)
    {
        // 将坐标和数值读出来
        long m, n;
        float value;

        // 假设每一行三个元素
        fin >> m >> n >> value;

        // 每一行三个元素
        // 增加一个元素
        assert(m > 0 && n > 0);
        assert(m <= row_num && n <= col_num);

        row_index.push_back(m - 1);
        col_index.push_back(n - 1);
        val_arr.push_back(1);
    }

    assert(val_arr.size() == val_num);

    fin.close();    
}

// 将COO合适转变为SELL格式，传入两个参数，一个是每一行的多少个线程的参数，一个是行分块的参数
bool convert_COO_2_SELL_P(vector<unsigned int> row_index, vector<unsigned int> col_index, vector<float> val_arr, unsigned int row_num, unsigned int col_num, unsigned int val_num, vector<unsigned int>& slice_row_off, vector<unsigned int>& new_col_index, vector<float>& new_val_arr, unsigned int& new_row_num, unsigned int& new_col_num, unsigned int& new_val_num, int thread_per_row, int row_per_slice)
{
    // 要返回的矩阵都是空的
    assert(slice_row_off.size() == 0);
    assert(new_col_index.size() == 0);
    assert(new_val_arr.size() == 0);
    assert(row_per_slice > 1);
    assert(thread_per_row > 1);

    // 条带偏移量一开始是0
    slice_row_off.push_back(0);

    // 首先是判断一共多少个slices
    unsigned int slices_num = row_num / row_per_slice;
    
    // 至少有一个slice
    if (row_num % row_per_slice != 0)
    {
        // 如果不能整除，就需要多一个条带
        slices_num = slices_num + 1;
    }

    // 当前的正在处理的slice id
    // 准备一个二维数组，用来存储当前条带被缓存的内容
    vector<vector<unsigned int>> row_index_slice_cache(row_per_slice);
    vector<vector<unsigned int>> col_index_slice_cache(row_per_slice);
    vector<vector<float>> val_arr_slice_cache(row_per_slice);

    // 遍历所有的元素
    assert(val_arr.size() == val_num);
    for (unsigned int i = 0; i < val_arr.size(); i++)
    {
        unsigned int cur_row_index = row_index[i];
        unsigned int cur_col_index = col_index[i];
        float cur_val = val_arr[i];

        // 获取当前非零元的slice编号
        unsigned int cur_slice_id = cur_row_index / row_per_slice;
        assert(cur_slice_id < slices_num);

        // 将内容放到cache中
        unsigned int row_index_inner_slice = cur_row_index % row_per_slice;

        row_index_slice_cache[row_index_inner_slice].push_back(cur_row_index);
        col_index_slice_cache[row_index_inner_slice].push_back(cur_col_index);
        val_arr_slice_cache[row_index_inner_slice].push_back(cur_val);
        
        // 判断当前非零元是不是一个slices的结尾
        bool is_end_of_cur_slice = false;

        if (i == (val_arr.size() - 1))
        {
            // 到达最后一个元素
            is_end_of_cur_slice = true;
        }
        else
        {
            // 如果不是最后一个元素，看下一个元素是不是在同一条带
            unsigned int next_row_index = row_index[i + 1];
            assert(next_row_index >= cur_row_index);
            // 下一个非零元的行条带编号
            unsigned int slice_of_next_row = next_row_index / row_per_slice;
            assert(slice_of_next_row < slices_num);
            assert(slice_of_next_row >= cur_slice_id);
            
            // 如果下一个行的条带索引高过这一行，那就说明当前行条带的最后一行
            if (slice_of_next_row > cur_slice_id)
            {
                is_end_of_cur_slice = true;                
            }
        }
        
        // 如果是条带的最后一行
        if (is_end_of_cur_slice == true)
        {
            // 这里整理和padding当前行
            // 查看当前slice的目标行长度
            unsigned int cur_slice_row_length = 0;

            // 遍历当前slice的每一行
            for (unsigned int row_in_slice = 0; row_in_slice < row_per_slice; row_in_slice++)
            {
                if (cur_slice_row_length < row_index_slice_cache[row_in_slice].size())
                {
                    cur_slice_row_length = row_index_slice_cache[row_in_slice].size();
                }
            }

            // 根据线程数量将计算行数量为对应的倍数
            if (cur_slice_row_length % thread_per_row != 0)
            {
                cur_slice_row_length = (cur_slice_row_length / thread_per_row + 1) * thread_per_row;
            }

            // 将输出拉直，并且同时处理padding
            for (unsigned int row_in_slice = 0; row_in_slice < row_per_slice; row_in_slice++)
            {
                assert(cur_slice_row_length >= row_index_slice_cache[row_in_slice].size());
                for (unsigned int nz_in_row = 0; nz_in_row < cur_slice_row_length; nz_in_row++)
                {
                    unsigned int padding_col_index = 0;
                    if (nz_in_row < row_index_slice_cache[row_in_slice].size())
                    {
                        new_col_index.push_back(row_index_slice_cache[row_in_slice][nz_in_row]);
                        padding_col_index = row_index_slice_cache[row_in_slice][nz_in_row];
                        new_val_arr.push_back(val_arr_slice_cache[row_in_slice][nz_in_row]);
                    }
                    else
                    {
                        new_col_index.push_back(padding_col_index);
                        new_val_arr.push_back(0);
                    }
                }
            }

            assert(slice_row_off.size() > 0);
            // 拉直之后在slice的行偏移量之后添加对应的东西，先将空的slice偏移量填入
            // 对于第n号条带来说，他的偏移量应该插入到row_off的n+1号位置，所以在他之前，row off的最后一位索引是n
            while (slice_row_off.size() - 1 < cur_slice_id)
            {
                slice_row_off.push_back(slice_row_off[slice_row_off.size() - 1]);
            }

            assert(slice_row_off.size() - 1 == cur_slice_id);

            // 最后一位偏移量会多出来一个，加一个当前条带的非零元数量
            slice_row_off.push_back(slice_row_off[slice_row_off.size() - 1] + row_per_slice * cur_slice_row_length);

            // 当最后完成一个slice的整理之后，将整理的cache清空
            for (unsigned int row_in_slice = 0; row_in_slice < row_per_slice; row_in_slice++)
            {
                vector<unsigned int>().swap(row_index_slice_cache[row_in_slice]);
                vector<unsigned int>().swap(col_index_slice_cache[row_in_slice]);
                vector<float>().swap(val_arr_slice_cache[row_in_slice]);

                assert(row_index_slice_cache[row_in_slice].size() == 0);
                assert(col_index_slice_cache[row_in_slice].size() == 0);
                assert(val_arr_slice_cache[row_in_slice].size() == 0);
            }
        }
    }

    // 遍历完之后，获得新的非零元数量
    new_val_num = slice_row_off[slice_row_off.size() - 1];
    new_row_num = row_num;
    new_col_num = col_num;

    assert(slice_row_off.size() == slices_num + 1);

    assert(new_val_num >= val_num);
    if (new_val_num / val_num > 4)
    {
        return false;
    }

    return true;
}

// 根究行索引进行排序
void sort_according_to_row_index(unsigned int row_num, unsigned int col_num, unsigned int val_num, vector<unsigned int>& row_index, vector<unsigned int>& col_index, vector<float>& val_arr)
{
    // 当前的数组的大小
    assert(row_index.size() == col_index.size());
    assert(row_index.size() == val_arr.size());
    assert(val_arr.size() == val_num);

    // 三个新的二维数组，分别重新处理行、列、值
    vector<vector<unsigned int>> dim2_row_index(row_num);
    vector<vector<unsigned int>> dim2_col_index(row_num);
    vector<vector<float>> dim2_val_arr(row_num);

    // 遍历所有的内容
    for (unsigned int i = 0; i < val_num; i++)
    {
        unsigned int cur_row_index = row_index[i];
        unsigned int cur_col_index = col_index[i];
        float cur_val = val_arr[i];

        assert(cur_row_index < row_num);
        assert(cur_col_index < col_num);
        // 将三个值分别放到对应的二维数组中
        dim2_row_index[cur_row_index].push_back(cur_row_index);
        dim2_col_index[cur_row_index].push_back(cur_col_index);
        dim2_val_arr[cur_row_index].push_back(cur_val);
    }

    // 创建新的返回数组
    vector<unsigned int> new_row_index;
    vector<unsigned int> new_col_index;
    vector<float> new_val_arr;
    
    // 遍历二维数组，合并到一维
    for (unsigned int i = 0; i < dim2_row_index.size(); i++)
    {
        // 当前行的第一维都被一样
        assert(dim2_row_index[i].size() == dim2_col_index[i].size());
        assert(dim2_row_index[i].size() == dim2_val_arr[i].size());
        for (unsigned int j = 0; j < dim2_row_index[i].size(); j++)
        {
            new_row_index.push_back(dim2_row_index[i][j]);
            new_col_index.push_back(dim2_col_index[i][j]);
            new_val_arr.push_back(dim2_val_arr[i][j]);
        }
    }
    
    row_index = new_row_index;
    col_index = new_col_index;
    val_arr = new_val_arr;
}

// 核函数



// 删除文件夹中所有目录：rm  -rf 'find . -type d'
int main()
{
    // 将一个文件读入
    string file_name = "/home/duzhen/matrix_suite/bone010/bone010.mtx";

    // COO格式的属性
    vector<unsigned int> row_index;
    vector<unsigned int> col_index;
    vector<float> val_arr;
    unsigned int row_num;
    unsigned int col_num;
    unsigned int val_num;

    get_coo_from_file(file_name, row_index, col_index, val_arr, row_num, col_num, val_num);

    sort_according_to_row_index(row_num, col_num, val_num, row_index, col_index, val_arr);

    vector<unsigned int> SELL_row_off;
    vector<unsigned int> SELL_col_index;
    vector<float> SELL_val_arr;
    unsigned int SELL_row_num;
    unsigned int SELL_col_num;
    unsigned int SELL_val_num;
    
    unsigned int thread_per_row = 2;
    unsigned int row_per_slice = 32;

    bool result = convert_COO_2_SELL_P(row_index, col_index, val_arr, row_num, col_num, val_num, SELL_row_off, SELL_col_index, SELL_val_arr, SELL_row_num, SELL_col_num, SELL_val_num, thread_per_row, row_per_slice);

    // cout << col_num << " " << row_num << " " << val_num;
    cout << result << endl;

    for (unsigned int i = 0; i < 10; i++)
    {
        cout << SELL_row_off[i] <<  ",";
    }

    cout << endl;

    for (unsigned int i = 0; i < 30; i++)
    {
        cout << SELL_col_index[i] << ",";
    }

    cout << endl;

    for (unsigned int i = 0; i < 30; i++)
    {
        cout << SELL_val_arr[i] << ",";
    }

    cout << endl;

    return 0;
}

