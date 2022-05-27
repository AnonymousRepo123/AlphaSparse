#include <cuda_runtime.h>
#include <cstdlib>
#include <stdio.h>
#include <vector>
#include <assert.h>
#include <fstream>
#include <string.h>
#include <iostream>

using namespace std;

// 这个也是需要生成的文本的模板
// 压缩子块索引的类型
enum compressed_block_index_type
{
    COO,
    CSR,
    ELL
};

// 索引类型
enum index_type
{
    BLOCK_INDEX,
    ROW_INDEX,
    COL_INDEX
};

// 数据类型
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

// 索引的层次
enum index_level
{
    TBLOCK_LEVEL,
    WRAP_LEVEL,
    THREAD_LEVEL,
    OTHERS
};

// 一个子块压缩视图索引的基本结构
typedef struct index_of_compressed_block
{
    // 索引的层次
    index_level level_of_this_index;
    // 索引的类型，有COO型、CSR型、ELL型
    compressed_block_index_type index_compressed_type;
    // 包含的块的数量
    unsigned long block_num;
    // 用一个数组来存储索引本身，对于CSR来说索引的长度为block_num + 1
    void *index_arr = NULL;
    // 索引的长度在CSR的情况下是某一个层次的
    unsigned long length;
    // 索引的数据类型
    data_type index_data_type;
    // 索引的归属
    index_type type_of_index;

    // 用一个bool数组判断是不是排序过，用来之后做一些压缩，大小为block num
    bool *is_sort_arr = NULL;

    // 记录每一个块所包含的第一行的行号，数组大小是block_num
    void *index_of_the_first_row_arr = NULL;
    // 索引的数据类型
    data_type data_type_of_index_of_the_first_row_arr;

    // 每个块行的数量，数组大小是block_num
    void *row_number_of_block_arr = NULL;
    data_type data_type_of_row_number_of_block_arr;

    // 每个warp在block中归约的位置，在纵分块的时候就是必须有，这是仅在经过warp的纵分块中才有的数组
    void *tmp_result_write_index_arr;
    data_type data_type_of_tmp_result_write_index_arr;

    // 行号和列号的范围
    unsigned long max_row_index;
    unsigned long min_row_index;
    unsigned long max_col_index;
    unsigned long min_col_index;

    // 一个块的COO起始位置，这个矩阵的大小是block_num + 1，在warp层次时是block_num
    void *coo_begin_index_arr = NULL;
    data_type data_type_of_coo_begin_index_arr;

    // warp层次开始才有的数组，在warp层次，coo的其实位置只有block_num个，另外使用一个数组存储每个warp块的大小
    // 这是使用相对位置的代价
    void *coo_block_size_arr = NULL;
    data_type data_type_of_coo_block_size_arr;

    // 更低一层中间结果的行边界
    void *child_tmp_row_csr_index_arr = NULL;
    // 更低一层中间结果的行边界的数据类型
    data_type data_type_of_child_tmp_row_csr_index;
    // 这个数组的大小和每个块行的数量有关
    unsigned long size_of_child_tmp_row_csr_index;
    // 用一个和block数量相同的数组记录每个block的归约偏移的起始位置，因为每个block的行数量不一样所以归约之后的结果数量不一样，这个数组大小和block number一致，
    void *begin_index_in_tmp_row_csr_arr_of_block = NULL;
    data_type data_type_of_begin_index_in_tmp_row_csr_arr_of_block;
} index_of_compress_block_t;

// 一个子块的index分层结构
typedef struct compressed_block
{
    // 一共两种类型的索引，一种是用来读x和A的，一个是用来写y的(sort等改变实际索引值的时候会用到)
    // 索引之间都有一层一层的依赖关系，被依赖的索引通常长度更短，并且最底层一般就是列索引和值
    // 用一个数组来存储每一层读索引的指针
    vector<index_of_compress_block_t *> read_index;
    vector<index_of_compress_block_t *> y_write_index;

    // 在某一个维度的CSR，方便某一个层次的的规约
    vector<index_of_compress_block_t *> reduce_help_csr;

    // 判断这个压缩视图的子块是不是整体被压缩了
    bool is_sorted;

    // 用一个向量来存储值，要存储的是值的数量
    int size;
    data_type val_data_type;
    void *val_arr = NULL;

    // 一个数组，用来存储padding之后的数据
    int padding_arr_size;
    void *padding_val_arr = NULL;

    // 一个数组，用来存储交错存储的数据
    int staggered_padding_val_arr_size;
    void *staggered_padding_val_arr = NULL;

    // 可能需要一个值来存储每一个块的起始行号
} compressed_block_t;

typedef struct dense_block_table_item
{
    // 一个vector数组，代表块坐标
    vector<int> block_coordinate;

    // 在稠密视图上分块行和列的取值范围
    unsigned long min_dense_row_index;
    unsigned long max_dense_row_index;

    unsigned long max_dense_col_index;
    unsigned long min_dense_col_index;

    // 在压缩成一条数组的时候非零元的开始和结束位置
    unsigned long begin_coo_index;
    unsigned long end_coo_index;

    // 判断当前密集子块是不是行排序过
    bool is_sorted;

    // 指向压缩后的块结构
    compressed_block_t *compressed_block_ptr = NULL;

} dense_block_table_item_t;

// 一个表格，用来存所有的稀疏矩阵分块，表格的每一项包含一个二元组，二元组包含一个数组来存储每一个块的变长坐标，还有一个指针
typedef struct dense_block_table
{
    // 用一个dense_block_table_item指针的数组来存储
    vector<dense_block_table_item_t *> item_arr;
} dense_block_table_t;

// 用来存x
typedef struct x_arr
{
    data_type x_data_type;
    void *x_arr = NULL;
} x_arr_t;

typedef struct sparse_struct
{
    unsigned long dense_row_number;
    unsigned long dense_col_number;
    unsigned long nnz;

    // 判断是不是进行了分块
    bool is_blocked;
    // 针对稀疏矩阵分块，给出一个包含了二元组的数组，包含一个块坐标和块坐标对应的一个指针
    dense_block_table_t block_coor_table;

    // 一个全局的排序数组，只要一个部分进行了排序，就要加入一个全局的排序索引
    bool is_sorted;
    data_type data_type_of_sorted_row_index;
    // 排序产生的新索引，
    void *sorted_row_index = NULL;

    // 压缩矩阵视图子块的结构，一开始是一个COO的大矩阵，在稠密视图分块后变为多个COO的大矩阵
    // 只有在压缩操作的时候才会形成的东西
    compressed_block *compressed_block_arr = NULL;

    // 用一个数组来存储压缩前的coo数据，长度是nnz，一开始数据类型巨大，用来放下所有数据
    unsigned long *coo_row_index_cache = NULL;
    unsigned long *coo_col_index_cache = NULL;

    // 用一个值来存稀疏矩阵的精度
    data_type val_data_type;
    void *coo_value_cache = NULL;

    // 用一个数组放x
    x_arr_t coo_x_cache;

} sparse_struct_t;

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

void *malloc_arr(unsigned long length, data_type type_of_arr)
{
    // 只有四种类型 // 还有两种浮点类型
    assert(type_of_arr == UNSIGNED_CHAR || type_of_arr == UNSIGNED_INT ||
           type_of_arr == UNSIGNED_SHORT || type_of_arr == UNSIGNED_LONG ||
           type_of_arr == DOUBLE || type_of_arr == FLOAT);

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
    else
    {
        // cout << "1235" << endl;
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

// 从文件中读一个特定大小的数组，传出一个void的指针
void *read_arr_from_file_with_data_type(unsigned long length, data_type arr_data_type, string file_name)
{
    assert(length > 0);
    assert(arr_data_type == UNSIGNED_LONG || arr_data_type == UNSIGNED_INT || arr_data_type == UNSIGNED_SHORT || arr_data_type == UNSIGNED_CHAR ||
           arr_data_type == BOOL || arr_data_type == FLOAT || arr_data_type == DOUBLE);

    // 创建一个特定数据类型的数组
    void *arr_need_to_return = malloc_arr(length, arr_data_type);

    unsigned long cur_insert_index = 0;

    if (arr_data_type == UNSIGNED_LONG || arr_data_type == UNSIGNED_INT || arr_data_type == UNSIGNED_SHORT || arr_data_type == UNSIGNED_CHAR || arr_data_type == BOOL)
    {
        // 读文件
        char buf[1024];

        ifstream infile;
        infile.open(file_name);

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
        
        // cout << "cur_insert_index:" << cur_insert_index << ",length:" << length << endl;
        assert(cur_insert_index == length);
        infile.close();
        return arr_need_to_return;
    }
    else if (arr_data_type == DOUBLE || arr_data_type == FLOAT)
    {
        // 读文件
        char buf[1024];

        ifstream infile;
        infile.open(file_name);

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
        
        assert(cur_insert_index == length);
        infile.close();
        return arr_need_to_return;
    }

    return arr_need_to_return;
}

unsigned long read_from_array_with_data_type(void *arr, data_type type, unsigned long read_pos)
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
        return (unsigned long)(output_arr[read_pos]);
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

double read_double_from_array_with_data_type(void *arr, data_type type, unsigned long read_pos)
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

void print_arr_to_file_with_data_type(void *arr, data_type type, unsigned long length, string file_name)
{
    assert(type == UNSIGNED_CHAR || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_LONG || type == DOUBLE || type == FLOAT || type == BOOL);
    
    // 向文件中写文件
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

// 用一个结构来存储反序列化的所有数据，这里是所有压缩子块的信息，全部展开存储，因为每个子块的数据类型可能不一样
typedef struct compressed_matrix_content
{
    unsigned long size_of_staggered_padding_val_arr = 4118720;
    double *staggered_padding_val_arr;

    unsigned long size_of_read_index_0_index_arr = 4020731;
    unsigned int *read_index_0_index_arr;
    unsigned long size_of_read_index_1_index_arr = 4020731;
    unsigned int *read_index_1_index_arr;
    unsigned long size_of_read_index_2_index_arr = 466;
    unsigned int *read_index_2_index_arr;
    unsigned long size_of_read_index_3_index_arr = 14880;
    unsigned int *read_index_3_index_arr;
    unsigned long size_of_read_index_5_index_arr = 4118720;
    unsigned int *read_index_5_index_arr;
    unsigned long size_of_read_index_6_index_arr = 4118720;
    unsigned int *read_index_6_index_arr;

    // warp和tblock有需要有一套归约的索引，用来执行同步型归约
    unsigned long size_of_read_index_2_begin_index_in_tmp_row_csr_arr_of_block = 465;
    unsigned int *read_index_2_begin_index_in_tmp_row_csr_arr_of_block;
    unsigned long size_of_read_index_2_child_tmp_row_csr_index_arr = 930366;
    unsigned int *read_index_2_child_tmp_row_csr_index_arr;
    unsigned long size_of_read_index_3_begin_index_in_tmp_row_csr_arr_of_block = 14879;
    unsigned int *read_index_3_begin_index_in_tmp_row_csr_arr_of_block;
    unsigned long size_of_read_index_3_child_tmp_row_csr_index_arr = 944780;
    unsigned int *read_index_3_child_tmp_row_csr_index_arr;

    // warp和tblock都有每个块的coo begin
    unsigned long size_of_read_index_2_coo_begin_index_arr = 466;
    unsigned int *read_index_2_coo_begin_index_arr;
    unsigned long size_of_read_index_3_coo_begin_index_arr = 14879;
    unsigned int *read_index_3_coo_begin_index_arr;

    // warp和thread有coo size
    unsigned long size_of_read_index_3_coo_block_size_arr = 14879;
    unsigned int *read_index_3_coo_block_size_arr;
    unsigned long size_of_read_index_4_coo_block_size_arr = 14879;
    unsigned int *read_index_4_coo_block_size_arr;

    // warp、thread、tblock都有每个块的首行地址
    unsigned long size_of_read_index_2_index_of_the_first_row_arr = 465;
    unsigned int *read_index_2_index_of_the_first_row_arr;
    unsigned long size_of_read_index_3_index_of_the_first_row_arr = 14879;
    unsigned int *read_index_3_index_of_the_first_row_arr;
    unsigned long size_of_read_index_4_index_of_the_first_row_arr = 952224;
    unsigned int *read_index_4_index_of_the_first_row_arr;

    // warp、tblock有每个块的行数量
    unsigned long size_of_read_index_2_row_number_of_block_arr = 465;
    unsigned int *read_index_2_row_number_of_block_arr;
    unsigned long size_of_read_index_3_row_number_of_block_arr = 14879;
    unsigned int *read_index_3_row_number_of_block_arr;

    
} compressed_matrix_content_t;

// 存储所有的压缩子块
typedef struct all_compressed_block
{
    compressed_matrix_content_t * all_compressed_matrix_info;
    unsigned int size_of_sorted_row_index = 929901;
    unsigned int * sorted_row_index;
} all_compressed_block_t;

// 将一个目录下的矩阵反序列化
all_compressed_block_t *read_matrix_from_file(string file_name_prefix)
{
    all_compressed_block_t* total_matrix = new all_compressed_block_t();
    string file_name = "";

    file_name = file_name_prefix + "/" + "sorted_row_index";
    total_matrix->sorted_row_index = (unsigned int *)read_arr_from_file_with_data_type(total_matrix->size_of_sorted_row_index, UNSIGNED_INT, file_name);

    compressed_matrix_content_t* compressed_block = new compressed_matrix_content_t();

    string sub_block_file_prefix = file_name_prefix + "/dense_block_" + to_string(0);

    file_name = sub_block_file_prefix + "/" + "staggered_padding_val_arr";
    compressed_block->staggered_padding_val_arr = (double *)read_arr_from_file_with_data_type(compressed_block->size_of_staggered_padding_val_arr, DOUBLE, file_name);

    file_name = sub_block_file_prefix + "/read_index_0/" + "index_arr";
    compressed_block->read_index_0_index_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_0_index_arr, UNSIGNED_INT, file_name);

    file_name = sub_block_file_prefix + "/read_index_1/" + "index_arr";
    compressed_block->read_index_1_index_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_1_index_arr, UNSIGNED_INT, file_name);

    file_name = sub_block_file_prefix + "/read_index_2/" + "begin_index_in_tmp_row_csr_arr_of_block";
    compressed_block->read_index_2_begin_index_in_tmp_row_csr_arr_of_block = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_2_begin_index_in_tmp_row_csr_arr_of_block, UNSIGNED_INT, file_name);

    file_name = sub_block_file_prefix + "/read_index_2/" + "child_tmp_row_csr_index_arr";
    compressed_block->read_index_2_child_tmp_row_csr_index_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_2_child_tmp_row_csr_index_arr, UNSIGNED_INT, file_name);
    
    file_name = sub_block_file_prefix + "/read_index_2/" + "coo_begin_index_arr";
    compressed_block->read_index_2_coo_begin_index_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_2_coo_begin_index_arr, UNSIGNED_INT, file_name);

    file_name = sub_block_file_prefix + "/read_index_2/" + "index_arr";
    compressed_block->read_index_2_index_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_2_index_arr, UNSIGNED_INT, file_name);

    file_name = sub_block_file_prefix + "/read_index_2/" + "index_of_the_first_row_arr";
    compressed_block->read_index_2_index_of_the_first_row_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_2_index_of_the_first_row_arr, UNSIGNED_INT, file_name);

    file_name = sub_block_file_prefix + "/read_index_2/" + "row_number_of_block_arr";
    compressed_block->read_index_2_row_number_of_block_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_2_row_number_of_block_arr, UNSIGNED_INT, file_name);

    file_name = sub_block_file_prefix + "/read_index_3/" + "begin_index_in_tmp_row_csr_arr_of_block";
    compressed_block->read_index_3_begin_index_in_tmp_row_csr_arr_of_block = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_3_begin_index_in_tmp_row_csr_arr_of_block, UNSIGNED_INT, file_name);

    file_name = sub_block_file_prefix + "/read_index_3/" + "child_tmp_row_csr_index_arr";
    compressed_block->read_index_3_child_tmp_row_csr_index_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_3_child_tmp_row_csr_index_arr, UNSIGNED_INT, file_name);

    file_name = sub_block_file_prefix + "/read_index_3/" + "coo_begin_index_arr";
    compressed_block->read_index_3_coo_begin_index_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_3_coo_begin_index_arr, UNSIGNED_INT, file_name);

    file_name = sub_block_file_prefix + "/read_index_3/" + "coo_block_size_arr";
    compressed_block->read_index_3_coo_block_size_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_3_coo_block_size_arr, UNSIGNED_INT, file_name);

    file_name = sub_block_file_prefix + "/read_index_3/" + "index_arr";
    compressed_block->read_index_3_index_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_3_index_arr, UNSIGNED_INT, file_name);

    file_name = sub_block_file_prefix + "/read_index_3/" + "index_of_the_first_row_arr";
    compressed_block->read_index_3_index_of_the_first_row_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_3_index_of_the_first_row_arr, UNSIGNED_INT, file_name);

    file_name = sub_block_file_prefix + "/read_index_3/" + "row_number_of_block_arr";
    compressed_block->read_index_3_row_number_of_block_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_3_row_number_of_block_arr, UNSIGNED_INT, file_name);

    file_name = sub_block_file_prefix + "/read_index_4/" + "coo_block_size_arr";
    compressed_block->read_index_4_coo_block_size_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_4_coo_block_size_arr, UNSIGNED_INT, file_name);

    file_name = sub_block_file_prefix + "/read_index_4/" + "index_of_the_first_row_arr";
    compressed_block->read_index_4_index_of_the_first_row_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_4_index_of_the_first_row_arr, UNSIGNED_INT, file_name);

    file_name = sub_block_file_prefix + "/read_index_5/" + "index_arr";
    compressed_block->read_index_5_index_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_5_index_arr, UNSIGNED_INT, file_name);
    
    file_name = sub_block_file_prefix + "/read_index_6/" + "index_arr";
    compressed_block->read_index_6_index_arr = (unsigned int *)read_arr_from_file_with_data_type(compressed_block->size_of_read_index_6_index_arr, UNSIGNED_INT, file_name);

    total_matrix->all_compressed_matrix_info = compressed_block;

    return total_matrix;
}

