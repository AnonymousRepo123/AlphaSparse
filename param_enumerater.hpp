#ifndef PARAM_ENUMERATER_H
#define PARAM_ENUMERATER_H

#include "struct.hpp"
#include <assert.h>
#include <iostream>

// 参数只有long的整型参数和double的浮点参数，只要针对这两种情况考虑即可
// 整形参数
typedef struct integer_independ_param
{
    long* param_ptr;

    // 下一个参数值
    long next_param_val;
    
    // 下界上界和步长
    long param_low_bound;
    long param_up_bound;

    long step_size;
} integer_independ_param_t;

// 浮点型参数，
typedef struct float_independ_param
{
    double* param_ptr;

    // 下一个参数值
    double next_param_val;

    double param_low_bound;
    double param_up_bound;

    double step_size;
} float_independ_param_t;


// 参数之间的依赖，当一个参数的取值依赖于另外一个参数的取值时，还是需要两个参数的指针，两个参数的数据类型，还有一个函数指针、计算从一个参数计算另一个参数的方法
typedef struct single_dependency_param
{
    void* depend_param;
    void* param;

    // 被依赖的参数和当前参数的数据类型
    data_type depend_param_data_type;
    data_type param_data_type;

    // 前面的参数时需要被修改的数据，后面的参数是被依赖的数据
    void (*fun)(void*, void*, data_type, data_type);
} single_dependency_param_t;

// 还有二元依赖过滤器，需要两个参数的指针和数据类型，还有函数指针，将两个参数的指针和对应的数据类型都放到函数指针中
typedef struct binary_param_dependency_filter
{
    void* param1;
    void* param2;

    // 两个参数的数据类型
    data_type param1_data_type;
    data_type param2_data_type;

    // 两个参数的依赖关系，前一个是参数1，后一个是参数2
    bool (*fun)(void*, void*, data_type, data_type);
} binary_param_dependency_filter_t;

// 参数分为浮点类的参数和整形类的参数，分成两个数组，将一个double类型的数组接在long类型的参数数组后面

// 还有一个数组存所有依赖于其他参数的参数

// 还有一个数组一个依赖过滤器，为已经产生的

typedef struct param_enumerater
{
    // 两个数组分别存浮点参数和整型参数
    vector<integer_independ_param_t> integer_independ_param_vec;
    vector<float_independ_param_t> float_independ_param_vec;

    // 单一依赖的参数
    vector<single_dependency_param_t> single_dependency_param_vec;
    vector<binary_param_dependency_filter_t> binary_param_dependency_filter_vec;
} param_enumerater_t;


// 初始化一个参数
integer_independ_param_t init_integer_independ_param(long* param_ptr, long param_low_bound, long param_up_bound, long step_size);
float_independ_param_t init_float_independ_param(double* param_ptr, double param_low_bound, double param_up_bound, double step_size);

// 初始化一个依赖型参数
single_dependency_param_t init_single_dependency_param(void* param, void* depend_param, data_type param_data_type, data_type depend_param_data_type, void (*fun)(void*, void*, data_type, data_type));

// 初始化一个依赖过滤器
binary_param_dependency_filter_t init_binary_param_dependency_filter(void* param1, void* param2, data_type param1_data_type, data_type param2_data_type, bool (*fun)(void*, void*, data_type, data_type));

// 将参数往前挪一个，如果从尾部挪到头部了，返回true，提示已经完成了一轮
bool set_integer_independ_param_to_next(integer_independ_param_t* param_setter);
bool set_float_independ_param_to_next(float_independ_param_t* param_setter);

// 判断一个参数是不是已经调到尾部，当所有参数都在尾部的时候，说明整个参数调优的部分已经完成
bool is_end_of_integer_param_setter(integer_independ_param_t* param_setter);
bool is_end_of_float_param_setter(float_independ_param_t* param_setter);

// 找到下一个参数组合，也是没有尽头的，每次执行一个这个函数，先检查一下是不是到末尾了
// 返回的是bool，代表是不是已经结束了，如果结束了，这个时候参数会被修改，但是这个时候的参数组合和之前是有重复的，所以没有必要再运行了，结束了就范围true
bool set_param_combination_to_next(param_enumerater_t* param_enumerater);

// 查看整个枚举过程是不是结束了
bool is_end_of_param_combination(param_enumerater_t* param_enumerater);

// 向参数枚举器添加一个整型无依赖参数
void register_integer_independ_param_to_enumerater(param_enumerater_t* param_enumerater, long* param_ptr, long param_low_bound, long param_up_bound, long step_size);

// 向参数枚举器添加一个浮点无依赖参数
void register_float_independ_param_to_enumerater(param_enumerater_t* param_enumerater, double* param_ptr, double param_low_bound, double param_up_bound, double step_size);

// 向枚举器添加一个依赖于其他参数的参数
void register_single_dependency_param_to_enumerater(param_enumerater_t* param_enumerater, void* param, data_type param_data_type, void* depend_param, data_type depend_param_data_type, void (*fun)(void*, void*, data_type, data_type));

// 向枚举器添加一个过滤器
void register_binary_param_dependency_filter_to_enumerater(param_enumerater_t* param_enumerater, void* param1, data_type param1_data_type, void* param2, data_type param2_data_type, bool (*fun)(void*, void*, data_type, data_type));

// 用断言检查一个指针是不是已经重复的
void assert_check_repeat_param_ptr(param_enumerater_t* param_enumerater, void* param_ptr);

#endif