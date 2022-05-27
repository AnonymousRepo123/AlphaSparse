#ifndef GRAPH_ENUMERATE_H
#define GRAPH_ENUMERATE_H

#include "struct.hpp"
#include <vector>
#include <set>

unsigned long get_val_from_vec_with_padding(unsigned long* arr, int read_index, int padding_size);

// 用一个窗口来遍历所有的行非零元数量，在开头执行0 padding。并且分块所依据的比率是和平均值的行非零元大小有关系
// 当不排序的时候可以采用这一种切法
vector<unsigned long> neighbor_avg_diff_filter(vector<unsigned long> row_nnz, int windows_size, double div_rate);

// 将整个矩阵进行行切分，使得每个行条带的行非零元都在幂律增加的行非零元数量区间内，不要求已经经过行排序
vector<unsigned long> row_div_position_acc_to_exponential_increase_row_nnz_range(vector<unsigned long> row_nnz, unsigned long smallest_row_nnz_range, unsigned long nnz_range_expansion_rate);

vector<unsigned long> row_div_position_acc_to_exponential_increase_row_nnz_range(vector<unsigned long> row_nnz, unsigned long smallest_row_nnz_range, unsigned long highest_row_nnz_range, unsigned long nnz_range_expansion_rate);

// 根据行非零元数量执行粗粒度的排序，传入一个参数代表粒度大小，根据行非零元的顺序
vector<unsigned long> bin_row_nnz_low_bound_of_fixed_granularity_coar_sort(vector<unsigned long> row_nnz, unsigned long granularity);

// 用来帮助tblock列分块的函数，根据一个子块（一个非空行）的非零元数量来判断列分块的大小，这里采用的是固定大小的列分块，在固定长度的列分块大小和行边界的位置执行切分子矩阵的列分块，最后一个分块使用四舍五入的方式，
// 如果剩下的部分多于一半的max_fixed_col_block_size，那就增加一个纵分块，如果剩下的少于一半的max_fixed_col_block_size，那就让最后一个块大一点，可以把多出来的部分包进来
// 对于BLB的列分块来说，sub_compressed_matrix_row_nnz的大小是行的数量，返回的第一层数组大小是非空行的数量，
// 这个函数只能用来在BLB分块中使用
vector<vector<unsigned int>> col_block_size_of_each_row(vector<unsigned long> sub_compressed_matrix_row_nnz, unsigned long max_fixed_col_block_size);

// WLB版本的列分块，不执行合并操作
vector<vector<unsigned int>> col_block_size_of_each_row_without_block_merge(vector<unsigned long> sub_compressed_matrix_row_nnz, unsigned long max_fixed_col_block_size);

// 一个行切分的策略，按照子矩阵行数量的切分，切分出来的是等长的节点，除了最后一个行条带可能会少一点。
vector<unsigned long> row_block_size_of_a_sub_matrix_by_fixed_div(unsigned long sub_matrix_row_num, unsigned long fixed_sub_block_size);

// 一个行切分策略，按照每一个行条带最小非零元数量来分块，传入的是行非零元数量，只有最后一个行条带可能是不满足要求的
vector<unsigned long> row_block_size_of_a_sub_matrix_by_nnz_low_bound(vector<unsigned long> row_nnz, unsigned long block_nnz_low_bound);

// 将一个压缩子块的子块进一步行分块，传入的是每一个压缩子块的子块的行数量，each_sub_block_row_num的size和返回值第一层的数量相等
vector<vector<unsigned int>> row_block_size_of_each_sub_block_by_fixed_div(vector<unsigned long> each_sub_block_row_num, unsigned long fixed_sub_block_row_size);

// 全局完全一致的TLB列分块枚举策略
vector<unsigned long> col_div_of_TLB_global_fixed_col_size(unsigned long WLB_block_num, unsigned long global_TLB_col_size);

// 找到两个值更大的那一个
unsigned long max_long_unsigned(unsigned long a, unsigned long b);

// 图枚举，根据一个节点

#endif