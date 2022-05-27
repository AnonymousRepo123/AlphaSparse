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

unsigned int *col_idx, *row_off;
float *values;

void sort(unsigned int *col_idx, float *a, int start, int end)
{
	int i, j, it;
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

void coo2csr(int row_length, int nnz, float *values, unsigned int *row, unsigned int *col,
			 float *csr_values, unsigned int *col_idx, unsigned int *row_start)
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

int dummy_var = 0;

// 预处理时间
struct timeval pre_start, pre_end;

static void conv(string data_file_name ,int &nnz, int &row_length, int &column_length, int &nnz_max, bool isData = true,
				 int &nnz_avg = dummy_var, int &nnz_dev = dummy_var)
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
	fin >> row_length >> column_length >> nnz;

	// cout << row_length << endl;
	// cout << column_length << endl;
	// cout << nnz << endl;

	// int m, n;
	// float data;

	// fin >> m >> n >> data;

	// exit(-1);

	// Create your matrix:
	unsigned int *row, *column;
	float *coovalues;
	row = new unsigned int[nnz];
	column = new unsigned int[nnz];
	coovalues = new float[nnz];
	values = (float *)malloc(sizeof(float) * nnz);
	col_idx = (unsigned int *)malloc(sizeof(unsigned int) * nnz);
	row_off = (unsigned int *)malloc(sizeof(unsigned int) * (row_length + 1));
	std::fill(row, row + nnz, 0);
	std::fill(column, column + nnz, 0);
	std::fill(values, values + nnz, 0);

	// Read the data
	for (int l = 0; l < nnz; l++)
	{
		int m, n;
		float data;
		if (!isData)
			fin >> m >> n; // >> data;
		else
			fin >> m >> n >> data;

		// if (l < 10)
		// {
		// 	cout << m << " " << n << endl;
		// }
		// else
		// {
		// 	assert(false);
		// }
		

		row[l] = m - 1;
		column[l] = n - 1;
		if (!isData)
			coovalues[l] = rand() % 10 + 1;
		else
			coovalues[l] = data;
	}

	cout << "get coo data" << endl;

	gettimeofday(&pre_start, NULL);

	coo2csr(row_length, nnz, coovalues, row, column, values, col_idx, row_off);

	cout << "get csr data" << endl;

	nnz_max = 0;

	int tot_nnz = 0, tot_nnz_square = 0;
	for (int i = 0; i < -1 + row_length; i++)
	{
		int cur_nnz = row_off[i + 1] - row_off[i];
		tot_nnz += cur_nnz;
		tot_nnz_square += cur_nnz * cur_nnz;
		if (cur_nnz > nnz_max)
			nnz_max = cur_nnz;
	}

	tot_nnz += nnz - row_off[row_length - 1];
	tot_nnz_square += (nnz - row_off[row_length - 1]);

	if ((nnz - row_off[row_length - 1]) > nnz_max)
	{
		nnz_max = nnz - row_off[row_length - 1];
	}

	nnz_avg = tot_nnz / row_length;
	nnz_dev = (int)sqrt(tot_nnz_square / row_length - (nnz_avg * nnz_avg));

	row_off[row_length] = nnz;

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

#endif