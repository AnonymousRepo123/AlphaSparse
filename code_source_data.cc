#include "code_source_data.hpp"
#include <typeinfo>
#include <assert.h>
#include<algorithm>

using namespace std;

// 关于通用数组类型的实现。
// universal_array::universal_array(vector<T> input_vec, data_type suggest_type, bool need_compress)
// {
//     assert(input_vec.size() > 0);

//     // 首先查看取值范围是不是和建议的数据类型吻合
//     if (suggest_type == BOOL)
//     {
//         assert(typeid(T).name() == typeid(bool).name());
//     }
//     else if (suggest_type == DOUBLE)
//     {
//         // 这里只是精度问题
//         assert(typeid(T).name() == typeid(double).name());
//     }
//     else if (suggest_type == FLOAT)
//     {
//         assert(typeid(T).name() == typeid(float).name());
//     }
//     else
//     {
//         // 这里是遇到整型的情况，需要处理可能的小数据类型问题
//         T max_val = *max_element(input_vec.begin(), input_vec.end());
//         T min_val = *min_element(input_vec.begin(), input_vec.end());

//         // 暂时只支持无符号
//         assert(min_val >= 0);

//         // 对于整数来说，如果没有设定自动压缩，就要看手动设定的数据类型是不是满足要求
//         // 获取数组的最大值和最小值
//         if (need_compress == false)
//         {
//             assert(max_val <= get_max_of_a_integer_data_type(suggest_type));
//             assert(min_val >= get_min_of_a_integer_data_type(suggest_type));
//         }
//         else
//         {
//             // 如果自带压缩，就直接找压缩的方法
//             if (min_val >= 0)
//             {
//                 suggest_type = find_most_suitable_data_type(max_val);
//             }
//             else
//             {
//                 suggest_type = find_most_suitable_data_type(max_val, min_val);
//             }
//         }
//     }

//     // 到这里只支持无符号、bool和浮点
//     assert(suggest_type == FLOAT || suggest_type == DOUBLE || suggest_type == UNSIGNED_CHAR || suggest_type == UNSIGNED_SHORT ||
//         suggest_type == UNSIGNED_INT || suggest_type == UNSIGNED_LONG || suggest_type == BOOL);

//     // 现在suggest_type中存着数据类型
//     this->type = suggest_type;
//     this->len = input_vec.size();

//     this->arr_ptr = malloc_arr(this->len, this->type);

//     assert(this->arr_ptr != NULL);

//     // 遍历所有的将向量中所有的非零元拷贝到新的数组中
//     for (unsigned long i = 0; i < input_vec.size(); i++)
//     {
//         // 浮点类型
//         if (this->type == FLOAT || this->type == DOUBLE)
//         {
//             assert((typeid(T).name() == typeid(float).name()) || (typeid(T).name() == typeid(double).name()));
//             write_double_to_array_with_data_type(this->arr_ptr, this->type, i, input_vec[i]);
//         }
//         else
//         {
//             write_to_array_with_data_type(this->arr_ptr, this->type, i, input_vec[i]);
//         }
//     }

//     cout << "universal_array::universal_array: the final data type is " << convert_data_type_to_string(this->type) << endl;
// }

// 利用引用的方式来初始化，不拷贝
universal_array::universal_array(void *input_arr_ptr, unsigned long len, data_type type, bool copy)
{
    assert(input_arr_ptr != NULL);
    assert(len > 0);
    // 支持浮点类型，无符号整型，bool类型
    assert(type == UNSIGNED_CHAR || type == UNSIGNED_SHORT || type == UNSIGNED_INT || type == UNSIGNED_LONG || type == BOOL ||
        type == FLOAT || type == DOUBLE);
    
    if (copy == false)
    {
        // 引用传值
        this->arr_ptr = input_arr_ptr;
        this->len = len;
        this->type = type;
    }
    else
    {
        // 申请新的空间，并且执行拷贝
        this->arr_ptr = val_copy_from_old_arr_with_data_type(input_arr_ptr, len, type);
        this->len = len;
        this->type = type;

        assert(this->arr_ptr != NULL);
    }
}

universal_array::~universal_array()
{
    assert(this->arr_ptr != NULL);

    // 析构
    delete_arr_with_data_type(this->arr_ptr, this->type);
}

void universal_array::write_integer_to_arr(unsigned long input_val, unsigned long input_index)
{
    // 不能是浮点类型
    assert(this->type != FLOAT && this->type != DOUBLE);
    assert(input_index < this->len);
    assert(this->arr_ptr != NULL);

    write_to_array_with_data_type(this->arr_ptr, this->type, input_index, input_val);
}

void universal_array::write_float_to_arr(double input_val, unsigned long input_index)
{
    // 必须是浮点类型
    assert(this->type == FLOAT || this->type == DOUBLE);
    assert(input_index < this->len);
    assert(this->arr_ptr != NULL);

    write_double_to_array_with_data_type(this->arr_ptr, this->type, input_index, input_val);
}

unsigned long universal_array::read_integer_from_arr(unsigned long read_index)
{
    // 不能是浮点类型
    assert(this->type != FLOAT && this->type != DOUBLE);
    assert(read_index < this->len);
    assert(this->arr_ptr != NULL);

    return read_from_array_with_data_type(this->arr_ptr, this->type, read_index);
}

double universal_array::read_float_from_arr(unsigned long read_index)
{
    // 必须是浮点类型
    assert(this->type == FLOAT || this->type == DOUBLE);
    assert(read_index < this->len);
    assert(this->arr_ptr != NULL);

    return read_double_from_array_with_data_type(this->arr_ptr, this->type, read_index);
}

// 将已有的数组进行压缩
void universal_array::compress_data_type()
{
    assert(this->type != FLOAT && this->type != DOUBLE && this->type != BOOL);
    assert(this->arr_ptr != NULL);

    // 找出当前数组的最大值
    unsigned long max_num = 0;

    // 遍历当前所有的元素
    for (unsigned long i = 0; i < this->len; i++)
    {
        unsigned long cur_num = this->read_integer_from_arr(i);

        if (cur_num > max_num)
        {
            max_num = cur_num;
        }
    }

    // 用最大值获取当前需要的数据类型
    data_type new_type = find_most_suitable_data_type(max_num);

    // 老的数组指针
    void* old_arr_ptr = this->arr_ptr;
    data_type old_type = this->type;

    // 新的指针
    this->arr_ptr = malloc_arr(this->len, new_type);
    this->type = new_type;

    // 老的数据拷贝到新的指针
    for (unsigned long i = 0; i < this->len; i++)
    {
        unsigned long old_num = read_from_array_with_data_type(old_arr_ptr, old_type, i);

        // 写到新的数组里面
        write_to_array_with_data_type(this->arr_ptr, this->type, i, old_num);
    }

    // 析构老的指针
    delete_arr_with_data_type(old_arr_ptr, old_type);
}


int var_code_position_type::global = 1;
int var_code_position_type::tblock = 2;
int var_code_position_type::warp = 3;
int var_code_position_type::thread = 4;
int var_code_position_type::col_index = 5;
int var_code_position_type::val = 6;
int var_code_position_type::none = 7;


data_source_item::data_source_item(string name, POS_TYPE position_type)
{
    this->name = name;
    this->position_type = position_type;
}

string constant_data_source_item::get_define_code()
{
    // 左右值都是存在的，位置和数据类型都没有问题
    assert(this->type_of_data_source == CONS_DATA_SOURCE);
    assert(this->data_type_of_constant_data_source != NONE_DATA_TYPE);
    assert(this->name != "" && this->right_value_name != "");
    assert(this->position_type != var_code_position_type::none);

    string return_str = "";

    return_str = return_str + convert_data_type_to_string(this->data_type_of_constant_data_source);
    return_str = return_str + " " + this->name + ";";
    
    return return_str;
}

string constant_data_source_item::get_assign_code()
{
    // 左右值都是存在的，位置和数据类型都没有问题
    assert(this->type_of_data_source == CONS_DATA_SOURCE);
    assert(this->data_type_of_constant_data_source != NONE_DATA_TYPE);
    assert(this->name != "" && this->right_value_name != "");
    assert(this->position_type != var_code_position_type::none);

    string return_str = "";

    return_str = return_str + this->name + " = " + this->right_value_name + ";";
    
    return return_str;
}



