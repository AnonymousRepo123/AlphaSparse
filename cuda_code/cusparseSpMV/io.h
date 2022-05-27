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

using namespace std;

// 列号和行偏移
int *col_idx = NULL;
int *row_off = NULL;
int *block_first_row_index = NULL;
float *values = NULL;


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

unsigned int read_from_array_with_data_type(void *arr, data_type type, unsigned int read_pos)
{
    assert(type == UNSIGNED_LONG || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_CHAR || type == BOOL);

    if (type == UNSIGNED_LONG)
    {
        unsigned int *output_arr = (unsigned int *)arr;
        return (unsigned int)(output_arr[read_pos]);
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

// 一个函数，为整个矩阵分块，非零元的数量不能超过1024，如果碰到一行特别长，长度超过1024，那就单独成块
unsigned int div_row_into_block(unsigned int *row_begin_off, unsigned int row_num)
{
	assert(row_begin_off != NULL);
	// 用一个vector记录每一个块的起始行号
	vector<unsigned int> block_first_row_index_vec;
	block_first_row_index_vec.push_back(0);
	
	// 记录已经积累的非零元数量
	unsigned int acc_nnz = 0;

	// 遍历所有的行偏移量
	for (int i = 0; i < row_num; i++)
	{
		// 查看当前行的非零元数量
		unsigned int cur_row_nnz = row_begin_off[i + 1] - row_begin_off[i];
		
		// bool cur_row_is_block_first_row = false;
		// 查看当前行是不是一个块的首行
		// if (cur_row_nnz >= 1024)
		// {
		// 	// 当前行的非零元数量大于1024，当前行就是一个块的第一行
		// 	if (i > block_first_row_index_vec[block_first_row_index_vec.size() - 1])
		// 	{
		// 		assert(i != 0);
		// 		block_first_row_index_vec.push_back(i);
		// 	}
		// 	else
		// 	{
		// 		// 只有首行会出现这个情况
		// 		assert(i == 0);
		// 	}

		// 	acc_nnz = 0;
		// }
		// else
		// {
			// 需要根据当前行非零元数量考虑换行问题
		if (cur_row_nnz + acc_nnz <= 1024)
		{
			// 继续累加，啥都不做
			acc_nnz = acc_nnz + cur_row_nnz;
		}
		else
		{
			// 累加之后超了，当前行是新快的第一行
			if (i > block_first_row_index_vec[block_first_row_index_vec.size() - 1])
			{
				assert(i != 0);
				block_first_row_index_vec.push_back(i);

				// 检查是不是只有一行，如果只有一行看看非零元数量超了吗
				unsigned int begin_row_index = block_first_row_index_vec[block_first_row_index_vec.size() - 2];
				unsigned int end_row_index = block_first_row_index_vec[block_first_row_index_vec.size() - 1];

				unsigned int begin_nz_index = row_begin_off[begin_row_index];
				unsigned int end_nz_index = row_begin_off[end_row_index];

				if (end_row_index - begin_row_index > 1 && end_nz_index - begin_nz_index > 1024)
				{
					cout << acc_nnz << "," << end_nz_index - begin_nz_index << endl;
					assert(end_nz_index - begin_nz_index <= 1024);
				}
			}
			else
			{
				// 只有第一位的时候回到这里
				assert(i == 0);
			}

			acc_nnz = cur_row_nnz;
		}
		// }
	}

	// 在末尾加一个所有行的数量，作为一个CSR的压缩的索引
	assert(row_num > block_first_row_index_vec[block_first_row_index_vec.size() - 1]);
	block_first_row_index_vec.push_back(row_num);
	
	// 将在vector中的内容放到申请的数组中
	unsigned int size_of_block_first_row_index = block_first_row_index_vec.size();

	// print_arr_to_file_with_data_type(&(block_first_row_index_vec[0]), UNSIGNED_LONG, size_of_block_first_row_index, "/home/duzhen/spmv_builder/data_source/test_result_6");

	// exit(-1);

	// 申请一个数组
	assert(block_first_row_index == NULL);
	block_first_row_index = new int[size_of_block_first_row_index];

	// 将向量中的数组拷贝进来
	memcpy(block_first_row_index, &(block_first_row_index_vec[0]), size_of_block_first_row_index * sizeof(unsigned int));

	// 这里做一个检查，检查包含多行的块的非零元数量
	for (int i = 0; i < size_of_block_first_row_index - 1; i++)
	{
		unsigned int begin_row_index = block_first_row_index[i];
		unsigned int end_row_index = block_first_row_index[i + 1];

		// 当前块所包含的非零元数量
		unsigned int begin_nz_index = row_begin_off[begin_row_index];
		unsigned int end_nz_index = row_begin_off[end_row_index];

		if (end_row_index - begin_row_index > 1)
		{
			assert(end_nz_index - begin_nz_index <= 1024);
		}
	}


	
	assert(size_of_block_first_row_index >= 2);
	// 实际块的数量少一位
	return (size_of_block_first_row_index - 1);
}



void sort(unsigned int *col_idx, float *a, int start, int end)
{
	int i, j;
	unsigned int it;
	float dt;

	// 将每一行的列冒泡排序，保证每一行的列索引自增，时间特别长
	for (i = end - 1; i > start; i--)
	{
		for (j = start; j < i; j++)
		{
			if (col_idx[j] > col_idx[j + 1])
			{
				if (a)
				{
					dt = a[j];
					a[j] = a[j + 1];
					a[j + 1] = dt;
				}
				it = col_idx[j];
				col_idx[j] = col_idx[j + 1];
				col_idx[j + 1] = it;
			}
		}
	}
}

void coo2csr(unsigned int row_length, unsigned int nnz, float *values, int *row, int *col,
			 float *csr_values, int *col_idx, int *row_start)
{
	int i, l;
	
	// 所有行全部初始化为0
	for (i = 0; i <= row_length; i++)
	{
		row_start[i] = 0;
	}

	/* determine row lengths */
	for (i = 0; i < nnz; i++)
	{
		// 记录每一行的长度
		row_start[row[i] + 1]++;
	}

	// for (i = 0; i < 10; i++)
	// {
	// 	cout << row[i] << endl;
	// }

	// exit(-1);

	for (i = 0; i < row_length; i++)
	{
		// 将每一行的长度向前累加，获得CSR索引
		row_start[i + 1] += row_start[i];
	}

	// for (i = 0; i < 10; i++)
	// {
	// 	cout << row_start[i] << endl;
	// }

	// exit(-1);

	/* go through the structure  once more. Fill in output matrix. */
	// 这里假设原来的数组是完全乱序的
	for (l = 0; l < nnz; l++)
	{
		i = row_start[row[l]];
		csr_values[i] = values[l];
		col_idx[i] = col[l];
		row_start[row[l]]++;
	}

	/* shift back row_start */
	for (i = row_length; i > 0; i--)
	{
		row_start[i] = row_start[i - 1];
	}

	row_start[0] = 0;

	cout << "sort" << endl;
	// 遍历每一行
	// for (i = 0; i < row_length; i++)
	// {
	// 	// cout << i << " " << row_start[i] << " " << row_start[i + 1] << endl;
	// 	sort(col_idx, csr_values, row_start[i], row_start[i + 1]);
	// }

	cout << "finish sort" << endl;
}

unsigned int dummy_var = 0;

struct timeval pre_start, pre_end;

static void conv(string data_file_name ,unsigned int &nnz, unsigned int &row_length, unsigned int &column_length, unsigned int &nnz_max, unsigned int &block_num, bool isData = true, unsigned int &nnz_avg = dummy_var, unsigned int &nnz_dev = dummy_var)
{
	string d;
	d = data_file_name;
	std::ifstream fin(d.c_str());
	if (!fin)
	{
		cout << "File Not found\n";
		exit(0);
	}

	//unsigned int row_length, column_length, nnz;

	// Ignore headers and comments:
	while (fin.peek() == '%')
		fin.ignore(2048, '\n');

	// Read defining parameters:
	fin >> row_length >> column_length >> nnz;

	// cout << row_length << endl;
	// cout << column_length << endl;
	// cout << nnz << endl;

	// unsigned int m, n;
	// float data;

	// fin >> m >> n >> data;

	// exit(-1);

	// Create your matrix:
	int *row, *column;
	float *coovalues;
	row = new int[nnz];
	column = new int[nnz];
	coovalues = new float[nnz];
	values = (float *)malloc(sizeof(float) * nnz);
	col_idx = (int *)malloc(sizeof(int) * nnz);
	row_off = (int *)malloc(sizeof(int) * (row_length + 1));
	std::fill(row, row + nnz, 0);
	std::fill(column, column + nnz, 0);
	std::fill(values, values + nnz, 0);

	// Read the data
	for (int l = 0; l < nnz; l++)
	{
		unsigned int m, n;
		float data;
		if (!isData)
			fin >> m >> n; // >> data;
		else
			fin >> m >> n >> data;

		assert(m >= 1 && n >= 1);
		assert(m <= row_length);
		assert(n <= column_length);

		row[l] = m - 1;
		column[l] = n - 1;
		if (!isData)
			coovalues[l] = rand() % 10 + 1;
		else
			coovalues[l] = data;
	}

	// gettimeofday(&pre_start, NULL);

	// cout << "get coo data" << endl;

	coo2csr(row_length, nnz, coovalues, row, column, values, col_idx, row_off);

	// cout << "get csr data" << endl;

	nnz_max = 0;

	unsigned int tot_nnz = 0, tot_nnz_square = 0;
	for (int i = 0; i < row_length - 1; i++)
	{
		unsigned int cur_nnz = row_off[i + 1] - row_off[i];
		tot_nnz += cur_nnz;
		tot_nnz_square += cur_nnz * cur_nnz;
		if (cur_nnz > nnz_max)
		{
			nnz_max = cur_nnz;
			// cout << "nnz_max_row:" << i << endl;
		}

	}

	tot_nnz += nnz - row_off[row_length - 1];
	tot_nnz_square += (nnz - row_off[row_length - 1]);

	if ((nnz - row_off[row_length - 1]) > nnz_max)
	{
		nnz_max = nnz - row_off[row_length - 1];
	}

	nnz_avg = tot_nnz / row_length;
	nnz_dev = (unsigned int)sqrt(tot_nnz_square / row_length - (nnz_avg * nnz_avg));

	row_off[row_length] = nnz;

	delete[] row;
	delete[] column;
	delete[] coovalues;

	fin.close();
	
	// cout << "begin div block" << endl;
	// 执行分块操作
	// block_num = div_row_into_block(row_off, row_length);
}


#endif