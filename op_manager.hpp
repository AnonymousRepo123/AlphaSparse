#ifndef OP_MANAGER_H
#define OP_MANAGER_H
#include "struct.hpp"
#include <assert.h>

// 操作管理器，和一个具体的结构绑定
typedef struct operator_manager
{
    sparse_struct_t *matrix = NULL;
} operator_manager_t;

// 全局padding的位置
enum global_padding_position
{
    TOP_PADDING,
    END_PADDING
};

// 一个COO元素的三要素
typedef struct coo_element_float
{
    unsigned long row_index;
    unsigned long col_index;
    float val;
} coo_element_float_t;

// 一个COO元素的三要素
typedef struct coo_element_double
{
    unsigned long row_index;
    unsigned long col_index;
    double val;
} coo_element_double_t;

// 做一个矩阵分解的操作，主要执行长短行的分解，分解要在一开始，分解完之后矩阵的规模不变，会有大量的空行，并且产生多个operator_manager以及对应的子矩阵
// 原矩阵会被完全析构，新的矩阵不能执行任何指针拷贝
// 执行的是长短行分解，根据行非零元数量的取值范围，将矩阵分解为多个子矩阵
// 每个子矩阵的大小都和源矩阵一样，主要输入的参数是非零元数量的下界
vector<sparse_struct_t *> long_short_row_decomposition(sparse_struct_t *matrix_struct, vector<unsigned int> row_nnz_low_bound_of_sub_matrix);

// 定长行分块操作，分块的长度，要被分块的子块在表格中的指针，这个子块的索引最终会被删掉，然后换成多个新的索引
void fixed_len_row_div(sparse_struct_t *matrix_struct, dense_block_table_item_t *sub_block, int len);

inline void fixed_len_row_div(operator_manager_t *op_manager, dense_block_table_item_t *sub_block, int len)
{
    fixed_len_row_div(op_manager->matrix, sub_block, len);
}

// 将一个数组变为二维数组，第一个数组代表了第二个数组的目标位置，最后一个形参是输出的第一维的长度
vector<vector<unsigned long>> convert_unsigned_long_arr_to_dim2_vec(unsigned long *target_arr, unsigned long *source_arr, unsigned long length, unsigned long dim_1_size);
vector<vector<double>> convert_double_arr_to_dim2_vec(unsigned long *target_arr, void *source_arr, data_type type, unsigned long length, unsigned long dim_1_size);

// 进行变长的行分块，用一个数组存储每个块的行起始位置的CSR形式的压缩
void var_len_row_div(sparse_struct_t *matrix_struct, dense_block_table_item_t *sub_block, vector<unsigned long> block_first_row_csr_index);

inline void var_len_row_div(operator_manager_t *op_manager, dense_block_table_item_t *sub_block, vector<unsigned long> block_first_row_csr_index)
{
    assert(op_manager != NULL);
    var_len_row_div(op_manager->matrix, sub_block, block_first_row_csr_index);
}

void dense_col_level_padding(operator_manager_t *op_manager, dense_block_table_item_t *sub_block, vector<unsigned long> row_index_vec, vector<unsigned long> padding_target_size_vec);

// 定长列分块操作
void fixed_len_col_div(sparse_struct_t *matrix_struct, dense_block_table_item_t *sub_block, int len);

inline void fixed_len_col_div(operator_manager_t *op_manager, dense_block_table_item_t *sub_block, int len)
{
    assert(op_manager != NULL);
    fixed_len_col_div(op_manager->matrix, sub_block, len);
}

// 传入密集矩阵分块表的
inline void fixed_len_col_div(operator_manager_t *op_manager, unsigned long dense_block_table_index, int len)
{
    assert(op_manager != NULL);
    assert(op_manager->matrix->block_coor_table.item_arr.size() > dense_block_table_index);
    fixed_len_col_div(op_manager->matrix, op_manager->matrix->block_coor_table.item_arr[dense_block_table_index], len);
}

// 变长列分块操作
void var_len_col_div(sparse_struct_t *matrix_struct, dense_block_table_item_t *sub_block, vector<unsigned long> block_first_row_csr_index);

// 还有一些自定义性更强的分块，可以进一步展开

// 输入坐标范围，找出每一行的起始位置，用和后面一行相同的起始位置表示空行
vector<unsigned long> find_coo_row_index_range(sparse_struct_t *matrix_struct, unsigned long find_coo_begin,
                                               unsigned long find_coo_end);

// 在一定坐标范围内找出所有列索引符合要求的非零元
vector<coo_element_double_t> find_all_col_double_element(sparse_struct_t *matrix_struct, unsigned long find_coo_begin,
                                                         unsigned long find_coo_end, unsigned long col_index_begin,
                                                         unsigned long col_index_end);

vector<coo_element_float_t> find_all_col_float_element(sparse_struct_t *matrix_struct, unsigned long find_coo_begin,
                                                       unsigned long find_coo_end, unsigned long col_index_begin,
                                                       unsigned long col_index_end);

// 找出符合要求的非零元的指针版本，将三个数组分开存，传入三个数组的指针
void find_all_col_double_element(sparse_struct_t *matrix_struct, unsigned long find_coo_begin,
                                 unsigned long find_coo_end, unsigned long col_index_begin,
                                 unsigned long col_index_end, vector<unsigned long> *output_row_arr,
                                 vector<unsigned long> *output_col_arr, vector<double> *double_var_arr);

void find_all_col_float_element(sparse_struct_t *matrix_struct, unsigned long find_coo_begin,
                                unsigned long find_coo_end, unsigned long col_index_begin,
                                unsigned long col_index_end, vector<unsigned long> *output_row_arr,
                                vector<unsigned long> *output_col_arr, vector<float> *float_var_arr);

// 引入压缩操作，每个块共享值数组，但是每个块的索引都是块内独立，从0开始
void compress_dense_view(sparse_struct_t *matrix_struct);

inline void compress_dense_view(operator_manager_t *op_manager)
{
    compress_dense_view(op_manager->matrix);
}

// 根据数据的最大值判断合适的数据类型
data_type find_most_suitable_data_type(unsigned long max_index_number);

// 根据最大值和最小值判断合适的一个数组合适的数据类型
data_type find_most_suitable_data_type(unsigned long max_number, unsigned long min_number);

// 初始化一个操作管理器
void init_op_manager(operator_manager_t *op_manager, sparse_struct_t *matrix);

operator_manager_t *init_op_manager(sparse_struct_t *matrix);

void sep_thread_level_row_csr(operator_manager_t *op_manager, compressed_block_t *compressed_block, vector<unsigned int> block_size_arr);

// 直接在线程层次分块，不考虑和其他操作的依赖问题
void sep_thread_level_row_csr(compressed_block_t *compressed_block, vector<unsigned int> block_size_arr);

// 按照规定，只能一行一块
void sep_thread_level_row_csr(compressed_block_t *compressed_block);

// 按照线程块的粒度切分
void sep_tblock_level_row_csr(compressed_block_t *compressed_block, vector<unsigned int> block_size_arr);
// 执行一个缺省的BLB级别的分块
void default_sep_tblock_level_row_csr(compressed_block_t* compressed_block_ptr);
// 一行一个BLB分块，用来执行BLB列分块的准备
void one_row_sep_tblock_level_row_csr(compressed_block_t* compressed_block_ptr);

// 按照线程块的粒度切分
void sep_tblock_level_col_csr(compressed_block_t *compressed_block, vector<unsigned long> block_index_arr, vector<vector<unsigned int>> block_size_arr);
void sep_tblock_level_col_csr(compressed_block_t *compressed_block, vector<unsigned int> block_size_arr);


// 按照warp粒度行分块，只有部分的块需要进一步行分块，每次warp分块完都需要修改父块的很多东西
// 没有进一步分块就就把整个父块划分为一块
void sep_warp_level_row_csr(compressed_block_t *compressed_block, vector<unsigned long> block_index_arr, vector<vector<unsigned int>> row_block_size_arr);
// 一个缺省的WLB分块
void default_sep_warp_level_row_csr(compressed_block_t *compressed_block_ptr);
// 一行一个WLB分块，用来执行WLB列分块的准备
void one_row_sep_warp_level_row_csr(compressed_block_t *compressed_block_ptr);

// 在warp行分块的基础上再列分块
void sep_warp_level_col_csr(compressed_block_t *compressed_block, vector<unsigned long> block_index_arr, vector<vector<unsigned int>> row_block_size_arr);

// 按照ELL的方式，纵切割，对于没有安排纵切割的块，就默认采用行切割，纵切割的块大小是相等的，并且按照32的倍数padding，并且交错存储，这个块会引入三个新索引，其中两个是列索引，并且引入两个新的值数组
// 并且改变之前所有的warp和block索引
void sep_thread_level_col_ell_with_padding(compressed_block_t *compressed_block, vector<unsigned long> block_index_arr, vector<unsigned long> thread_col_block_size);

// 按照非非零元数量进行分块，不对齐的分块
void sep_thread_level_acc_to_nnz(compressed_block_t *compressed_block, unsigned long thread_level_block_size);

// 根据一段实际的非零元，找到每一行实际的thread块数量，在thread层次和warp层次同时padding，保证每个thread非零元数量相等的同时，一个warp的thread分块的数量是32的倍数，最激进的padding策略
// 参数是全局的行索引，以及实际非零元的上界和下界（包含上界和下界的值），返回每一行的thread块数量
vector<unsigned long> find_thread_block_num_of_each_line_after_padding_in_thread_and_warp_level(index_of_compress_block_t *old_global_row_index, unsigned long global_warp_first_row, unsigned long warp_row_num, unsigned long global_warp_coo_start, unsigned long global_warp_coo_end, unsigned long thread_level_block_size);

// 在一段coo格式的非零元中，找到最大的行非零元数量
unsigned long find_max_row_nnz_in_coo_sub_block(index_of_compress_block_t *old_global_row_index, unsigned long global_warp_first_row, unsigned long warp_row_num, unsigned long global_coo_start, unsigned long global_coo_end);

// 在一段coo格式的非零元中，找出一定范围内的行非零元数量，所有的上界和下界都是包含边界的
vector<unsigned long> get_nnz_of_each_row_in_spec_range(index_of_compress_block_t *old_global_row_index, unsigned long begin_row_bound, unsigned long end_row_bound, unsigned long global_coo_start, unsigned long global_coo_end);
// 直接传入数组的版本
vector<unsigned long> get_nnz_of_each_row_in_spec_range(void *row_index_arr, data_type data_type_of_row_index_arr, unsigned long begin_row_bound, unsigned long end_row_bound, unsigned long global_coo_start, unsigned long global_coo_end);
// 计算一个压缩子块每一行的非零元数量
vector<unsigned long> get_nnz_of_each_row_in_compressed_sub_matrix(compressed_block_t* compressed_sub_block);

// 计算一定区间内非零元的数量，传入coo的行索引
vector<unsigned long> get_row_num_of_each_row_nnz_range(operator_manager_t *op_manager, vector<unsigned long> row_nnz_range);

// 对每一行的coo数量进行排序，得出每一位在排序之后的位置
vector<vector<unsigned long>> index_map_after_sorting(vector<unsigned long> nnz_of_each_row);

// 进行sort操作，这个操作可以在全局、也可以在compressed情况下进行，还没实现
void dense_block_sort(operator_manager_t *op_manager, dense_block_table_item_t *sub_block, int len);

// 针对一个压缩视图子矩阵的排序，对某一个tblock进行排序，在现在的结构下，warp和thread内部的排序都不太必要。排序的意义是预处理时间和排序效果对应的负载均衡的tradeoff
void compressed_block_sort(compressed_block_t *compressed_block, vector<unsigned long> sort_block_index_arr);

void compressed_block_sort(operator_manager_t *op_manager, compressed_block_t *compressed_block, vector<unsigned long> sort_block_index_arr);

// 对某一个密集矩阵进行分块，传入的是索引号
void compressed_block_sort(operator_manager_t *op_manager, unsigned long index_of_dense_block_table, vector<unsigned long> sort_block_index_arr);

// 针对密集矩阵视图的某一个子块进行分块，也就是对某一个压缩视图进行排序
void total_compressed_block_sort(compressed_block_t *compressed_block);

// 针对某一个压缩视图执行排序
void total_compressed_block_sort(operator_manager_t *op_manager, unsigned long index_of_dense_block_table);

// 加一个带上op_manager的版本
void total_compressed_block_sort(operator_manager_t *op_manager, compressed_block_t *compressed_block);

// 针对整个矩阵进行分块，先得到一个整体的顺序，然后在全局准备一个sorted_index，
void total_dense_block_sort(operator_manager_t *op_manager);

// 按行分组之后进行排序，sell-x式的排序，因为不支持稠密矩阵视图进行排序之后，用这个来弥补一下
// 用一个数组存储排序条带的起始行号
void total_dense_block_sort(operator_manager_t *op_manager, vector<unsigned long> first_row_of_each_sort_band);

// 一个全局的行padding，将行padding到特定的倍数，填入的新值列索引为一个定值，填入的值为0
void total_row_level_padding(operator_manager_t *op_manager, unsigned long multiple, global_padding_position padding_type = END_PADDING, unsigned long input_col_index = 0);

// 一个全局的行padding，直接padding到目标大小。并且需要padding的非零元数量
void total_row_level_padding_direct(operator_manager_t *op_manager, unsigned long target_size, unsigned padding_col_num = 1, global_padding_position padding_type = END_PADDING, unsigned long input_col_index = 0);

// 全局的行padding，padding一定的数量、一定非零元个数的行
inline void total_row_level_padding_add(operator_manager_t *op_manager, unsigned long add_size, unsigned padding_col_num = 1, global_padding_position padding_type = END_PADDING, unsigned long input_col_index = 0)
{
    assert(op_manager != NULL && op_manager->matrix != NULL);

    sparse_struct_t *matrix = op_manager->matrix;

    // 并且现在还没有分过块
    assert(matrix->block_coor_table.item_arr.size() == 0 && matrix->is_sorted == false && matrix->sorted_row_index == NULL);
    assert(padding_type == TOP_PADDING || padding_type == END_PADDING);

    // 原本的行数量
    unsigned long old_row_num = matrix->dense_row_number;

    // 目标行数量
    unsigned long target_size = old_row_num + add_size;

    // 执行padding
    total_row_level_padding_direct(op_manager, target_size, padding_col_num, padding_type, input_col_index);
}

// 粗粒度排序，将数据大致分为几个桶，返回每个桶的首行索引，用CSR的方式压缩，最后一位是总行号，用一个数组存储每个桶的行非零元数量下界
vector<unsigned long> total_dense_block_coarse_sort(operator_manager_t *op_manager, vector<unsigned long> bin_row_nnz_low_bound);

// 对某一个压缩视图执行row end padding，这个过程会使得压缩视图的行数量和稠密视图的行数量不匹配，如果碰到排序的情况，对写回的行索引写一个随机的行号即可
// 为了支持row padding，需要对所有模板都执行一个修改。只有压缩子块的相对行号满足稠密视图分块的约束的才可以。所以所有的模板需要一个比较，当写结果的时候发现
// 当前结果的压缩子块相对行号小于稠密子块的行数量的时候，就不要写回，防止错误和数据的越界。
// 倒数第二个参数是需要将压缩子块的行数量补位multiple的倍数。主要可能是32，或者BLB等长分割的行条带行数量。最后一个参数是padding的行的非零元数量
void compress_block_end_block_multiple_padding(operator_manager_t *op_manager, unsigned long compressed_block_id, unsigned long multiple, unsigned long padding_row_length);

#endif