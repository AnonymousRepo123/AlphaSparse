#include <cuda_runtime.h>
#include <cstdlib>
#include <stdio.h>
#include <vector>
#include <assert.h>
#include <fstream>
#include <string.h>
#include <iostream>

#include <sys/time.h>

using namespace std;

enum compressed_block_index_type
{
    COO,
    CSR,
    ELL
};

enum index_type
{
    BLOCK_INDEX,
    ROW_INDEX,
    COL_INDEX
};

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

enum index_level
{
    TBLOCK_LEVEL,
    WRAP_LEVEL,
    THREAD_LEVEL,
    OTHERS
};

typedef struct index_of_compressed_block
{
    index_level level_of_this_index;
    compressed_block_index_type index_compressed_type;
    unsigned long block_num;
    void *index_arr;
    unsigned long length;
    data_type index_data_type;
    index_type type_of_index;

    bool *is_sort_arr;

    void *index_of_the_first_row_arr;
    data_type data_type_of_index_of_the_first_row_arr;

    void *row_number_of_block_arr;
    data_type data_type_of_row_number_of_block_arr;

    void *tmp_result_write_index_arr;
    data_type data_type_of_tmp_result_write_index_arr;

    unsigned long max_row_index;
    unsigned long min_row_index;
    unsigned long max_col_index;
    unsigned long min_col_index;

    void *coo_begin_index_arr;
    data_type data_type_of_coo_begin_index_arr;

    void *coo_block_size_arr;
    data_type data_type_of_coo_block_size_arr;

    void *child_tmp_row_csr_index_arr;

    data_type data_type_of_child_tmp_row_csr_index;

    unsigned long size_of_child_tmp_row_csr_index;

    void *begin_index_in_tmp_row_csr_arr_of_block;
    data_type data_type_of_begin_index_in_tmp_row_csr_arr_of_block;
} index_of_compress_block_t;


typedef struct compressed_block
{
    vector<index_of_compress_block_t *> read_index;
    vector<index_of_compress_block_t *> y_write_index;

    vector<index_of_compress_block_t *> reduce_help_csr;

    bool is_sorted;

    int size;
    data_type val_data_type;
    void *val_arr;

    int padding_arr_size;
    void *padding_val_arr;

    int staggered_padding_val_arr_size;
    void *staggered_padding_val_arr;
} compressed_block_t;

typedef struct dense_block_table_item
{
    vector<int> block_coordinate;

    unsigned long min_dense_row_index;
    unsigned long max_dense_row_index;

    unsigned long max_dense_col_index;
    unsigned long min_dense_col_index;

    unsigned long begin_coo_index;
    unsigned long end_coo_index;

    bool is_sorted;

    compressed_block_t *compressed_block_ptr;

} dense_block_table_item_t;

typedef struct dense_block_table
{
    vector<dense_block_table_item_t *> item_arr;
} dense_block_table_t;

typedef struct x_arr
{
    data_type x_data_type;
    void *x_arr;
} x_arr_t;

typedef struct sparse_struct
{
    unsigned long dense_row_number;
    unsigned long dense_col_number;
    unsigned long nnz;

    bool is_blocked;
    dense_block_table_t block_coor_table;

    bool is_sorted;
    data_type data_type_of_sorted_row_index;

    void *sorted_row_index;

    compressed_block *compressed_block_arr;

    unsigned long *coo_row_index_cache;
    unsigned long *coo_col_index_cache;

    data_type val_data_type;
    void *coo_value_cache;

    x_arr_t coo_x_cache;

} sparse_struct_t;

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

void *malloc_arr(unsigned long length, data_type type_of_arr)
{
    assert(type_of_arr == UNSIGNED_CHAR || type_of_arr == UNSIGNED_INT ||
           type_of_arr == UNSIGNED_SHORT || type_of_arr == UNSIGNED_LONG ||
           type_of_arr == DOUBLE || type_of_arr == FLOAT);

    assert(length > 0);

    if (type_of_arr == UNSIGNED_CHAR)
    {
        return new unsigned char[length];
    }
    else if (type_of_arr == UNSIGNED_SHORT)
    {
        return new unsigned short[length];
    }
    else if (type_of_arr == UNSIGNED_INT)
    {
        unsigned int *return_ptr = new unsigned int[length];
        return (void *)return_ptr;
    }
    else if (type_of_arr == DOUBLE)
    {
        return new double[length];
    }
    else if (type_of_arr == FLOAT)
    {
        return new float[length];
    }
    else
    {
        return new unsigned long[length];
    }
}

void write_to_array_with_data_type(void *arr, data_type type, unsigned long write_pos, unsigned long write_val)
{
    assert(type == UNSIGNED_LONG || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_CHAR);
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

void *read_arr_from_file_with_data_type(unsigned long length, data_type arr_data_type, string file_name)
{
    assert(length > 0);
    assert(arr_data_type == UNSIGNED_LONG || arr_data_type == UNSIGNED_INT || arr_data_type == UNSIGNED_SHORT || arr_data_type == UNSIGNED_CHAR ||
           arr_data_type == BOOL || arr_data_type == FLOAT || arr_data_type == DOUBLE);

    void *arr_need_to_return = malloc_arr(length, arr_data_type);

    unsigned long cur_insert_index = 0;

    if (arr_data_type == UNSIGNED_LONG || arr_data_type == UNSIGNED_INT || arr_data_type == UNSIGNED_SHORT || arr_data_type == UNSIGNED_CHAR || arr_data_type == BOOL)
    {
        char buf[1024];

        ifstream infile;
        infile.open(file_name);

        if (infile.is_open())
        {
            while (infile.good() && !infile.eof())
            {
                string line_str;
                memset(buf, 0, 1024);
                infile.getline(buf, 1024);
                line_str = buf;

                if (isspace(line_str[0]) || line_str.empty())
                {
                    continue;
                }

                unsigned long arr_val = atol(line_str.c_str());

                assert(cur_insert_index < length);
                write_to_array_with_data_type(arr_need_to_return, arr_data_type, cur_insert_index, arr_val);

                cur_insert_index++;
            }
        }
        
        assert(cur_insert_index == length);
        infile.close();
        return arr_need_to_return;
    }
    else if (arr_data_type == DOUBLE || arr_data_type == FLOAT)
    {
        char buf[1024];

        ifstream infile;
        infile.open(file_name);

        while (infile.good() && !infile.eof())
        {
            string line_str;
            memset(buf, 0, 1024);
            infile.getline(buf, 1024);
            line_str = buf;

            if (isspace(line_str[0]) || line_str.empty())
            {
                continue;
            }

            double arr_val = stod(line_str.c_str());

            assert(cur_insert_index < length);
            write_double_to_array_with_data_type(arr_need_to_return, arr_data_type, cur_insert_index, arr_val);

            cur_insert_index++;
        }
        
        assert(cur_insert_index == length);
        infile.close();
        return arr_need_to_return;
    }

    return arr_need_to_return;
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

void print_arr_to_file_with_data_type(void *arr, data_type type, unsigned long length, string file_name)
{
    assert(type == UNSIGNED_CHAR || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_LONG || type == DOUBLE || type == FLOAT || type == BOOL);
    ofstream arrWrite(file_name, ios::out | ios::trunc);

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

typedef struct compressed_dense_block_0
{


float* dense_0_val_arr;
unsigned long size_of_dense_0_val_arr = 4649600;

unsigned int* dense_0_col_index_arr;
unsigned long size_of_dense_0_col_index_arr = 4649600;
}compressed_dense_block_0_t;

compressed_dense_block_0_t* read_dense_block_0_from_file(string file_name_prefix)
{
compressed_dense_block_0_t *template_data = new compressed_dense_block_0_t();


template_data->dense_0_val_arr = (float *)read_arr_from_file_with_data_type(template_data->size_of_dense_0_val_arr, FLOAT, file_name_prefix + "/val_arr");

template_data->dense_0_col_index_arr = (unsigned int *)read_arr_from_file_with_data_type(template_data->size_of_dense_0_col_index_arr, UNSIGNED_INT, file_name_prefix + "/col_index_arr");
return template_data;
}


