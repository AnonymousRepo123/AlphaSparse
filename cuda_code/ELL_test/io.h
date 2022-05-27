#ifndef IO_H
#define IO_H

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string.h>
#include <cuda.h>
#include <assert.h>
#include <sys/time.h>

using namespace std;

unsigned int *col_idx;
float *values;

unsigned int origin_nnz;

int dummy_var = 0;

struct timeval pre_start, pre_end;

static void conv(string data_file_name ,unsigned int &nnz, unsigned int &row_num, unsigned int &column_num, unsigned int &max_row_length, bool isData = true, int &nnz_avg = dummy_var, int &nnz_dev = dummy_var)
{
	string d;
	d = data_file_name;
	std::ifstream fin(d.c_str());
	if (!fin)
	{
		cout << "File Not found\n";
		exit(0);
	}

	//int row_length, column_length, nnz;

	// Ignore headers and comments:
	while (fin.peek() == '%')
		fin.ignore(2048, '\n');

	// Read defining parameters:
	fin >> row_num >> column_num >> nnz;

	// 在ELL中获取一个真正的nnz，真正的row_num

	origin_nnz = nnz;
	// Create your matrix:
	unsigned int *row, *column;
	float *coovalues;
	row = new unsigned int[nnz];
	column = new unsigned int[nnz];
	coovalues = new float[nnz];

	// values = (float *)malloc(sizeof(float) * nnz);
	// col_idx = (unsigned int*)malloc(sizeof(unsigned int) * nnz);
	
	// Read the data
	for (int l = 0; l < nnz; l++)
	{
		int m, n;
		float data;
		if (!isData)
			fin >> m >> n; // >> data;
		else
			fin >> m >> n >> data;
		
		row[l] = m - 1;
		column[l] = n - 1;
		if (!isData)
			coovalues[l] = rand() % 10 + 1;
		else
			coovalues[l] = data;
	}

	gettimeofday(&pre_start, NULL);

	// row colum coovalue是已经在数组中矩阵
	// max_row_length = coo2ell(row_num, nnz, coovalues, row, column, values, col_idx);
	// 遍历所有的非零元，找出最大行非零元，然后边padding边交错存储
	max_row_length = 0;

	// 用一个变量存储当前行的非零元
	unsigned int cur_row_nnz = 0;
	// 上一个非零元的行号
	unsigned int row_index_of_last_nz = 0;
	for (unsigned int i = 0; i < nnz; i++)
	{
		unsigned int cur_row_index = row[i];
		// unsigned int cur_col_index = column[i];

		if (cur_row_nnz != 0)
		{
			// 如果上一行的非零元已经被记录，要注意换行处
			if (cur_row_index == row_index_of_last_nz)
			{
				// 没有换行
				// cur_row_nnz++;
			}
			else
			{
				// 换行了，记录最大行非零元
				if (cur_row_nnz > max_row_length)
				{
					max_row_length = cur_row_nnz;
				}

				// 当前行的非零元数量更新为1
				cur_row_nnz = 0;
			}
		}

		cur_row_nnz++;
		row_index_of_last_nz = cur_row_index;
	}

	cout << "max_row_length:" << max_row_length << endl;

	// float ell_padding_rate = (max_row_length )

	// 获取当前行号最近的32倍数的
	// 这里最好不要对齐！
	if (row_num % 32 != 0)
	{
		row_num = ((row_num / 32) + 1) * 32;
	}

	assert(row_num % 32 == 0);

	// 新的总非零元数量
	unsigned int new_nnz = row_num * max_row_length;

	cout << "origin nnz:" << origin_nnz << " new nnz:" << new_nnz << endl;

	// padding率高于1.5就直接退出
	if (((float)new_nnz / origin_nnz) >= 5)
	{
		cout << "padding rate is higher than 5" << endl;
		exit(-1);
	}

	// 创建两个新的数组，按照新的非零元数量申请
	values = new float[new_nnz];
	col_idx = new unsigned int[new_nnz];

	// 初始化
	memset(&(values[0]), 0, new_nnz);
	memset(&(col_idx[0]), 0, new_nnz);

	// padding
	// 当前非零元在行内的索引
	unsigned int cur_index_inner_row = 0;
	for (unsigned int i = 0; i < nnz; i++)
	{
		// 获取行号列号和值
		unsigned int cur_row_index = row[i];
		unsigned int cur_col_index = column[i];
		float cur_value = coovalues[i];

		if (i == 0 || cur_row_index != row_index_of_last_nz)
		{
			if (i != 0)
			{
				assert(cur_row_index > row_index_of_last_nz);
			}

			// 当前非零元是一行的首个非零元
			cur_index_inner_row = 0;
			
			// 查看索引要放的目标位置
			unsigned int dest_index = cur_row_index + cur_index_inner_row * row_num;
			assert(dest_index < new_nnz);

			values[dest_index] = cur_value;
			col_idx[dest_index] = cur_col_index;

			row_index_of_last_nz = cur_row_index;
		}
		else
		{
			assert(cur_row_index == row_index_of_last_nz);
			cur_index_inner_row = cur_index_inner_row + 1;
			
			// 查看索引要放的目标位置
			unsigned int dest_index = cur_row_index + cur_index_inner_row * row_num;
			if (dest_index >= new_nnz)
			{
				cout << "dest_index:" << dest_index << " new_nnz:" << new_nnz << endl;
			}
			assert(dest_index < new_nnz);

			values[dest_index] = cur_value;
			col_idx[dest_index] = cur_col_index;

			row_index_of_last_nz = cur_row_index;
		}
	}

	nnz = new_nnz;

	delete[] row;
	delete[] column;
	delete[] coovalues;

	fin.close();
}

enum data_type
{
    CHAR,
    UNSIGNED_CHAR,
    SHORT,
    UNSIGNED_SHORT,
    INT,
    UNSIGNED_INT,
    LONG,
    UNSIGNED_LONG,
    LONG_LONG,
    UNSIGNED_LONG_LONG,
    FLOAT,
    DOUBLE,
    BOOL
};

unsigned long read_from_array_with_data_type(void *arr, data_type type, unsigned int read_pos)
{
    assert(type == UNSIGNED_LONG || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_CHAR || type == BOOL);

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

    cout << "error" << endl;
    exit(-1);
    return 0;
}

double read_double_from_array_with_data_type(void *arr, data_type type, unsigned int read_pos)
{
    assert(type == DOUBLE || type == FLOAT);

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

    return 0;
}

void print_arr_to_file_with_data_type(void *arr, data_type type, unsigned int length, string file_name)
{
    assert(type == UNSIGNED_CHAR || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_LONG || type == DOUBLE || type == FLOAT || type == BOOL);
    ofstream arrWrite(file_name, ios::out | ios::trunc);

    if (type == UNSIGNED_CHAR || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_LONG || type == BOOL)
    {
        unsigned int i;
        for (i = 0; i < length; i++)
        {
            arrWrite << read_from_array_with_data_type(arr, type, i) << endl;
        }
    }
    else if (type == DOUBLE || type == FLOAT)
    {
        unsigned int i;
        for (i = 0; i < length; i++)
        {
            arrWrite << read_double_from_array_with_data_type(arr, type, i) << endl;
        }
    }

    arrWrite.close();
}

#endif