#ifndef ARR_OPTIMIZATION_H
#define ARR_OPTIMIZATION_H

#include <string>
#include <assert.h>
#include "struct.hpp"
#include <iostream>

using namespace std;

// 压缩手段的类型，线性压缩和常量压缩，分支压缩
enum arr_compress_type
{
    NONE_COMPRESS,
    LINEAR_COMPRESS,
    CONSTANT_COMPRESS,
    BRANCH_COMPRESS,
    CYCLE_LINEAR_COMPRESS,
    CYCLE_INCREASE_COMPRESS
};

// 对于不同优化类型的数组的输出，这里包含了所有的优化内容
typedef struct linear_compress
{
    // 让索引乘一个系数就是数组的输出
    unsigned long coefficient = 1;
    unsigned long intercept = 0;
} linear_compress_t;

linear_compress_t *init_linear_compressor(void *source_arr, data_type type, unsigned long arr_size, bool need_checked = false);

string code_of_arr_read(linear_compress_t *compressor, string output_var_name, string index_var_name);

// 对于常值优化的处理，当一个数组中所有的值都一样的时候，将数组压缩成一个常量
typedef struct constant_compress
{
    // 常量
    unsigned long constant = 0;
} constant_compress_t;

constant_compress_t *init_constant_compressor(void *source_arr, data_type type, unsigned long arr_size, bool need_checked = false);

string code_of_arr_read(constant_compress_t *compressor, string output_var_name, string index_var_name);

// 分支压缩
typedef struct branch_compress
{
    // 上界和下界，是包含分界点的范围，以及在界内的常量
    vector<unsigned long> index_low_bound;
    vector<unsigned long> index_up_bound;
    vector<unsigned long> constant;
} branch_compress_t;

branch_compress_t *init_branch_compressor(void *source_arr, data_type type, unsigned long arr_size, bool need_checked = false);

string code_of_arr_read(branch_compress_t *compressor, string output_var_name, string index_var_name);

// 周期线性压缩，用来压缩一些相对索引，比如warp的非零元起始索引
typedef struct cycle_linear_compress
{
    // 周期大小，每个周期其实位置，斜率
    unsigned long cycle = 0;
    unsigned long coefficient = 1;
    unsigned long intercept = 0;
} cycle_linear_compress_t;

cycle_linear_compress_t *init_cycle_linear_compressor(void *source_arr, data_type type, unsigned long arr_size, unsigned long cycle_num, bool need_checked = false);

string code_of_arr_read(cycle_linear_compress_t *compressor, string output_var_name, string index_var_name);

// 周期累计压缩，每个周期加一定的数值，比如00112233，需要一个截距，一个周期之后对应值+1
typedef struct cycle_increase_compress
{
    // 周期大小和斜率
    unsigned long cycle = 0;
    unsigned long intercept = 0;
} cycle_increase_compress_t;

cycle_increase_compress_t *init_cycle_increase_compressor(void *source_arr, data_type type, unsigned long arr_size, bool need_checked = false);

string code_of_arr_read(cycle_increase_compress_t *compressor, string output_var_name, string index_var_name);

// 用一个函数析构一个压缩器的结构体
void delete_compressor_with_type(void* del_compressor, arr_compress_type type);

#endif