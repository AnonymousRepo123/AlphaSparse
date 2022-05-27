// 用一个结构来表达完整的数据结构
#ifndef STRUCT_H
#define STRUCT_H

#include <vector>
#include <string>
#include <list>
#include <sstream>
#include <algorithm>
#include <limits.h>
#include <float.h>
using namespace std;

// 压缩子块索引的类型
enum compressed_block_index_type
{
    COO,
    CSR,
    ELL,
    NO_INDEX
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
    BOOL,
    NONE_DATA_TYPE,
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

    // 用一个bool数组判断是不是排序过，用来之后做一些压缩，大小为block num，这个应该是被弃用的
    bool *is_sort_arr = NULL;

    // 记录每一个块所包含的第一行的行号，数组大小是block_num
    void *index_of_the_first_row_arr = NULL;
    // 索引的数据类型
    data_type data_type_of_index_of_the_first_row_arr;

    // 每个块行的数量，数组大小是block_num
    void *row_number_of_block_arr = NULL;
    data_type data_type_of_row_number_of_block_arr;

    // 每个warp在block中归约的位置，在纵分块的时候就是必须有，这是仅在经过warp的纵分块中才有的数组
    void *tmp_result_write_index_arr = NULL;
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

    // 判断这个压缩视图的子块是不是整体被压缩了，还没用上
    bool is_sorted = false;

    // 判断压缩子矩阵是不是和其他的子矩阵共享
    // 用一个bool表示线程块粒度的子块有没有和其他子块共享行
    // 用来决定最终的结果是原子加还是直接赋值
    bool share_row_with_other_block = false;

    // 用一个bool值表示warp块有没有和其他warp共享行，决定需不需要warp层次的归约
    bool share_row_with_other_warp = false;

    // 用一个bool值表示thread之间有没有共享行，以此决定是不是需要在thread层次进行归约
    bool share_row_with_other_thread = false;

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

    // 用一个bool来判断，当前密集视图的矩阵是不是在外部被列切块过
    bool is_col_blocked = false;

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

    // 最最开始的非零元数量
    unsigned long origin_nnz;

    // 查看是不是已经被压缩过了
    bool is_compressed = false;

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



// 打印dense_block_table_item，输入item，是日志输出还是命令行输出
string convert_dense_block_table_item_to_string(dense_block_table_item *item);

// 打印dense_block_table，输入table，是日志输出还是命令行输出
void print_dense_block_table(dense_block_table *table, bool if_log = false, string log_name = "");

// 用一个文件初始化sparse_struct，文件中的元素是COO的格式
sparse_struct_t *init_sparse_struct_by_coo_file(string coo_file_name, data_type value_data_type);

// 执行稀疏矩阵格式的值拷贝工作
sparse_struct_t *val_copy_from_old_matrix_struct(sparse_struct_t * matrix);

// 从文件中获得一个所有的行索引、列索引、值数组以及最大的列索引、行索引。
void get_matrix_index_and_val_from_file(string coo_file_name, vector<unsigned long> &row_index_vec, vector<unsigned long> &col_index_vec, vector<float> &float_val_vec, vector<double> &double_val_vec, data_type val_data_type, unsigned long& max_row_index, unsigned long& max_col_index);


// 用COO的三个数组来初始化sparse struct，并且给出最大值和最小值，用来获知矩阵大小
sparse_struct_t *init_sparse_struct_by_coo_vector(vector<unsigned long> row_arr, vector<unsigned long> col_arr,
                                                  vector<float> val_arr_float, vector<double> val_arr_double, data_type value_data_type,
                                                  unsigned long col_index_max, unsigned long row_index_max);

void split(const std::string &s, std::vector<std::string> &sv, const char delim = ' ');

// 将当前结构的coo输出到文件中
void output_struct_coo_to_file(sparse_struct_t *matrix_struct, string file_name);

// 检查dense分块是不是都是正确的
bool check_dense_block_div(sparse_struct_t *matrix_struct);

// 打印数据类型
void print_data_type(data_type type);

string convert_data_type_to_string(data_type type);

// 申请对应数据类型的数组
void *malloc_arr(unsigned long length, data_type type_of_arr);

// unsigned long数组向其他类型数组的数据拷贝，并且还需要还需要减去同一个偏移量
void copy_unsigned_long_index_to_others(unsigned long *source_arr, void *dest_ptr, data_type type, unsigned long length, unsigned long base_index);
void copy_unsigned_int_arr_to_others(unsigned int *source_arr, void *dest_ptr, data_type type, unsigned long length);
void copy_unsigned_long_arr_to_others(unsigned long *source_arr, void *dest_ptr, data_type type, unsigned long length);
void copy_double_arr_to_others(double *source_arr, void *dest_ptr, data_type type, unsigned long length);

// 数据拷贝到特定偏移量的一个位置
void copy_unsigned_long_arr_to_others_with_offset(unsigned long *source_arr, void *dest_ptr, data_type dest_type, unsigned long length, unsigned long dest_offset);
void copy_double_arr_to_others_with_offset(double *source_arr, void *dest_ptr, data_type dest_type, unsigned long length, unsigned long dest_offset);

// 向某一个数据类型的数组写数据
void write_to_array_with_data_type(void *arr, data_type type, unsigned long write_pos, unsigned long write_val);
void write_double_to_array_with_data_type(void *arr, data_type type, unsigned long write_pos, double write_val);

// 按照最大数据类型从不知道数据类型的数据中读出数据
unsigned long read_from_array_with_data_type(void *arr, data_type type, unsigned long read_pos);
// float和double的数组中付出
double read_double_from_array_with_data_type(void *arr, data_type type, unsigned long read_pos);

// 打印所有压缩视图，用一个文件夹来存储每一个输出文件
void print_compressed_block(sparse_struct_t *matrix_struct, string dir_name);

// 打印一个压缩之后的矩阵索引，输出到文件中
void print_compressed_block_meta_index(index_of_compress_block_t *compressed_block, string dir_name);

// 检查线程块层次分块的结果
bool check_tblock_sep(compressed_block_t *compressed_block);

// del一个数组
void delete_arr_with_data_type(void *arr, data_type type);

// 打印一个数组
void print_arr_with_data_type(void *arr, data_type type, unsigned long length);
void print_arr_to_file_with_data_type(void *arr, data_type type, unsigned long length, string file_name);

// 用一个函数所有块的数据打印出来，返回的是根文件夹的名字，本质上是一个id，再传入一个要将矩阵放的位置
unsigned long write_total_matrix_to_file(sparse_struct_t *matrix, string prefix_file_name);

// 从文件中读一个特定长度的数组
void *read_arr_from_file_with_data_type(unsigned long length, data_type arr_data_type, string file_name);

// 读命令行对应位的内容，然后输出为string
string read_str_from_command_line(int argc,char **argv, int cmd_input_index);

// 数组空间的拷贝，传入两个指针，新的新旧数组的大小，两个数组的数据类型一致
void memcpy_with_data_type(void *dest, void *source, unsigned long source_size, data_type type);

// 拷贝构造出一个新的、特定数据类型的数组
void* val_copy_from_old_arr_with_data_type(void *source, unsigned long source_size, data_type type);

// 根据数据的bit数量，判断数据类型
data_type find_most_suitable_data_type_by_bit_num(int bit_num);

// 数据类型的对应的位数
int bit_num_of_data_type(data_type type);

// 查看一个整型数据类型对应的最大值和最小值，首先整型
unsigned long get_max_of_a_integer_data_type(data_type type);
unsigned long get_min_of_a_integer_data_type(data_type type);

// 查看一个浮点数据类型对应的最大值和最小值
double get_max_of_a_float_data_type(data_type type);
double get_min_of_a_float_data_type(data_type type);

// 将一个数据类型的typeid转化为对应的对应的数据类型
data_type get_data_type_from_type_info(const type_info& input_info);

#endif
