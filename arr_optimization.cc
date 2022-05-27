#include "arr_optimization.hpp"
#include "config.hpp"

// 初始化一下，看看有没有压缩的可能性
// 这个数组必然是增序的
linear_compress_t *init_linear_compressor(void *source_arr, data_type type, unsigned long arr_size, bool need_checked)
{
    // cout << "arr_size:" << arr_size << ", source_arr:" << source_arr << endl;
    
    if (arr_size <= 1)
    {
        return NULL;
    }

    assert(arr_size > 1 && source_arr != NULL);

    // 首先查看前两位
    unsigned long first_content = read_from_array_with_data_type(source_arr, type, 0);
    unsigned long second_content = read_from_array_with_data_type(source_arr, type, 1);

    if (second_content < first_content)
    {
        return NULL;
    }
    // 两位之间的两位之间的相减就是斜率
    unsigned long tmp_coef = second_content - first_content;

    // 看看是不是需要检查
    if (need_checked == true)
    {
        // 遍历剩下的内容，看看是不是一样的步长递进
        for (unsigned long i = 0; i < arr_size - 1; i++)
        {
            first_content = read_from_array_with_data_type(source_arr, type, i);
            second_content = read_from_array_with_data_type(source_arr, type, i + 1);

            if (second_content < first_content)
            {
                return NULL;
            }

            if (second_content - first_content != tmp_coef)
            {
                cout << "linear_compress_t:can not pass the check in index:" << i << endl;
                return NULL;
            }
        }
    }

    linear_compress_t *return_compressor = new linear_compress_t();

    // 这里是通过了检查
    return_compressor->coefficient = tmp_coef;
    return_compressor->intercept = read_from_array_with_data_type(source_arr, type, 0);

    return return_compressor;
}

// 对变量进行赋值，传入两个变量，被赋值的变量，用来索引数组的变量
string code_of_arr_read(linear_compress_t *compressor, string output_var_name, string index_var_name)
{
    assert(compressor != NULL);

    string return_str = output_var_name + " = " + index_var_name;

    if (compressor->coefficient != 1)
    {
        return_str = return_str + " * " + to_string(compressor->coefficient);
    }

    // 根据截距是不是0来决定要不要包含截距
    if (compressor->intercept != 0)
    {
        return_str = return_str + " + " + to_string(compressor->intercept);
    }

    return return_str;
}

constant_compress_t *init_constant_compressor(void *source_arr, data_type type, unsigned long arr_size, bool need_checked)
{
    assert(source_arr != NULL && arr_size > 0);

    unsigned long first_content = read_from_array_with_data_type(source_arr, type, 0);

    if (need_checked == true)
    {
        for (unsigned long i = 0; i < arr_size; i++)
        {
            // 检查每一位是不是都一样
            if (read_from_array_with_data_type(source_arr, type, i) != first_content)
            {
                cout << "constant_compress_t:can not pass the check in index:" << i << endl;
                return NULL;
            }
        }
    }

    constant_compress_t *return_compressor = new constant_compress_t();

    // 通过了检查
    return_compressor->constant = first_content;

    return return_compressor;
}

// 赋值的代码
string code_of_arr_read(constant_compress_t *compressor, string output_var_name, string index_var_name)
{
    assert(compressor != NULL);
    string return_str = output_var_name + " = " + to_string(compressor->constant);
    return return_str;
}

branch_compress_t *init_branch_compressor(void *source_arr, data_type type, unsigned long arr_size, bool need_checked)
{
    assert(source_arr != NULL && arr_size > 2);

    vector<unsigned long> new_low_bound_vec;
    vector<unsigned long> new_up_bound_vec;
    vector<unsigned long> new_val_vec;

    // 用一开始那一个初始化
    new_low_bound_vec.push_back(0);
    new_val_vec.push_back(read_from_array_with_data_type(source_arr, type, 0));

    // 最多只能有三个分支
    for (unsigned long i = 0; i < arr_size; i++)
    {
        // 如果发现不一样的，就补上下界，并且定义新的值和上界
        unsigned long cur_val = read_from_array_with_data_type(source_arr, type, i);

        if (cur_val != new_val_vec[new_val_vec.size() - 1])
        {

            new_up_bound_vec.push_back(i - 1);
            // 设定新的下界和值
            new_low_bound_vec.push_back(i);
            new_val_vec.push_back(cur_val);

            if (new_up_bound_vec.size() >= get_config()["BRANCH_COMPRESS_MAX_SIZE"].as_integer())
            {
                cout << "branch_compress_t:too many branches at index:" << i << endl;

                cout << "[";

                for (unsigned j = 0; j < new_low_bound_vec.size(); j++)
                {
                    cout << new_low_bound_vec[j] << ",";
                }

                cout << "]" << endl;

                // cout << new_up_bound_vec << endl;
                return NULL;
            }
        }
    }

    assert(new_up_bound_vec.size() < get_config()["BRANCH_COMPRESS_MAX_SIZE"].as_integer() && new_low_bound_vec.size() <= get_config()["BRANCH_COMPRESS_MAX_SIZE"].as_integer() && new_val_vec.size() <= get_config()["BRANCH_COMPRESS_MAX_SIZE"].as_integer());

    // 用arr-size - 1 初始化上界
    new_up_bound_vec.push_back(arr_size - 1);

    // 初始化压缩元数据
    branch_compress_t *compressor = new branch_compress_t();

    compressor->index_low_bound = new_low_bound_vec;
    compressor->index_up_bound = new_up_bound_vec;
    compressor->constant = new_val_vec;

    return compressor;
}

// 分支压缩对应的代码
string code_of_arr_read(branch_compress_t *compressor, string output_var_name, string index_var_name)
{
    assert(compressor != NULL);
    assert(compressor->index_low_bound.size() <= get_config()["BRANCH_COMPRESS_MAX_SIZE"].as_integer() && compressor->index_up_bound.size() <= get_config()["BRANCH_COMPRESS_MAX_SIZE"].as_integer() && compressor->constant.size() <= get_config()["BRANCH_COMPRESS_MAX_SIZE"].as_integer());

    string return_str = "\n";

    // 遍历所有分支生成代码
    for (unsigned long branch_index = 0; branch_index < compressor->index_low_bound.size(); branch_index++)
    {
        unsigned long low_bound = compressor->index_low_bound[branch_index];
        unsigned long up_bound = compressor->index_up_bound[branch_index];
        unsigned long constant = compressor->constant[branch_index];

        assert(low_bound <= up_bound);

        // 如果上界和下界相等，就用等号，反之用大于小于号
        if (low_bound == up_bound)
        {
            return_str = return_str + "if(" + index_var_name + "==" + to_string(low_bound) + ")\n{\n";
        }
        else
        {
            return_str = return_str + "if(" + index_var_name + " <= " + to_string(up_bound) + " && " + index_var_name + " >= " + to_string(low_bound) + ")\n{\n";
        }

        // 这里填充一个赋值语句
        return_str = return_str + output_var_name + " = " + to_string(constant) + ";\n";

        // 根据是不是最后一个分支处理
        if (branch_index == compressor->index_low_bound.size() - 1)
        {
            return_str = return_str + "}\n";
        }
        else
        {
            return_str = return_str + "}else ";
        }
    }

    return return_str;
}

cycle_linear_compress_t *init_cycle_linear_compressor(void *source_arr, data_type type, unsigned long arr_size, unsigned long cycle_num, bool need_checked)
{
    assert(source_arr != NULL);

    // 循环的周期不满足要求就压缩失败
    if (cycle_num < arr_size)
    {

    }
    else
    {
        return NULL;
    }

    if (arr_size % cycle_num == 0)
    {

    }
    else
    {
        return NULL;
    }

    assert(cycle_num < arr_size);
    assert(arr_size % cycle_num == 0);

    if (cycle_num <= 1)
    {
        cout << "cycle_linear_compress_t:cycle_num <= 1" << endl;
        return NULL;
    }

    // 用第一个值来初始化截距
    unsigned long new_intercept = read_from_array_with_data_type(source_arr, type, 0);
    unsigned long new_coefficient = read_from_array_with_data_type(source_arr, type, 1) - read_from_array_with_data_type(source_arr, type, 0);

    // 查看是不是要检查
    if (need_checked == true)
    {
        // 如果需要检查，就需要查看是不是满足周期性，是不是每个周期的起始位置和斜率都一样
        // 遍历所有的非零元
        for (unsigned long i = 0; i < arr_size; i++)
        {
            // 获取当前数组的值
            unsigned long item_val = read_from_array_with_data_type(source_arr, type, i);

            // 查看当前索引和在一个周期中的位置
            unsigned long index_inner_cycle = i % cycle_num;

            if (index_inner_cycle != 0)
            {
                // 看看斜率是不是正确，这里要求斜率要是一个整数
                assert(item_val >= new_intercept && (item_val - new_intercept) % index_inner_cycle == 0);
            }

            if (index_inner_cycle != 0 && (item_val - new_intercept) / index_inner_cycle != new_coefficient)
            {
                // 这里代表不符合周期性斜率
                cout << "cycle_linear_compress_t:can not pass the check in index:" << i << endl;
                return NULL;
            }
        }
    }

    // 这里代表符合周期律，初始化并输出
    cycle_linear_compress_t *compressor = new cycle_linear_compress_t();
    compressor->cycle = cycle_num;
    compressor->coefficient = new_coefficient;
    compressor->intercept = new_intercept;

    return compressor;
}

// 打印对应优化后的代码
string code_of_arr_read(cycle_linear_compress_t *compressor, string output_var_name, string index_var_name)
{
    assert(compressor != NULL);

    // 先用索引取余，然后乘斜率，最后加上截距
    string return_str = output_var_name + " = (" + index_var_name + " % " + to_string(compressor->cycle) + ") * " + to_string(compressor->coefficient);

    if (compressor->intercept != 0)
    {
        return_str = return_str + " + " + to_string(compressor->intercept);
    }

    return return_str;
}

// 周期自增压缩器的初始化
cycle_increase_compress_t *init_cycle_increase_compressor(void *source_arr, data_type type, unsigned long arr_size, bool need_checked)
{
    assert(source_arr != NULL);

    // 如果数组的大小不够长，那就放弃
    if (arr_size <= 2)
    {
        cout << "init_cycle_increase_compressor: arr_size is not larger than 2, arr_size: " << arr_size << endl;
        return NULL;
    }

    // for (int i = arr_size - 10; i < arr_size; i++)
    // {
    //     cout << read_from_array_with_data_type(source_arr, type, i) << ",";
    // }

    // cout << endl;

    // 查看第一个元素出现了几次来确定周期
    unsigned long first_element = read_from_array_with_data_type(source_arr, type, 0);
    unsigned long new_cycle_num = 0;

    // 遍历整个数组，直到和first_element元素不一样
    for (unsigned long i = 0; i < arr_size; i++)
    {
        unsigned long cur_element = read_from_array_with_data_type(source_arr, type, i);
        // 如果不一样
        if (cur_element != first_element)
        {
            // 如果不是自增的，那就直接退出
            if (cur_element < first_element)
            {
                cout << "init_cycle_increase_compressor:arr is not self-increasing" << endl;
                return NULL;
            }
            // i就是周期大小
            // cout << new_cycle_num << endl;
            new_cycle_num = i;
            // exit(-1);
            break;
        }
    }

    assert(new_cycle_num > 0);

    // 这些压缩在空行的时候会失效

    // 完整的周期检查
    if (need_checked == true)
    {
        // 检查周期是否可以被完整的数组数量整除
        if (arr_size % new_cycle_num != 0)
        {
            cout << "init_cycle_increase_compressor: unvalid num" << endl;
            return NULL;
        }

        for (unsigned long i = 0; i < arr_size; i++)
        {
            unsigned long cur_element = read_from_array_with_data_type(source_arr, type, i);

            // 查看所属的周期
            unsigned long cycle_id = (unsigned long)(i / new_cycle_num);

            if (cur_element != first_element + cycle_id)
            {
                cout << "init_cycle_increase_compressor: cycle is not satisfied in index:" << i << endl;
                return NULL;
            }
        }
    }

    // 通过了检查
    cycle_increase_compress_t *compressor = new cycle_increase_compress_t();
    // 初始化截距和周期
    compressor->cycle = new_cycle_num;
    compressor->intercept = first_element;

    return compressor;
}

// 打印对应的压缩之后的代码
string code_of_arr_read(cycle_increase_compress_t *compressor, string output_var_name, string index_var_name)
{
    assert(compressor != NULL);

    string return_str = output_var_name + " = " + index_var_name + " / " + to_string(compressor->cycle);

    // 如果截距是0，就不加了
    if (compressor->intercept != 0)
    {
        // 加在后面
        return_str = return_str + " + " + to_string(compressor->intercept);
    }

    return return_str;
}

void delete_compressor_with_type(void* del_compressor, arr_compress_type type)
{
    assert(del_compressor != NULL);

    if (type == LINEAR_COMPRESS)
    {
        linear_compress_t* compressor = (linear_compress_t *)del_compressor;
        delete compressor;

        return;
    }

    if (type == CONSTANT_COMPRESS)
    {
        constant_compress_t* compressor = (constant_compress_t *)del_compressor;
        delete compressor;

        return;
    }

    if (type == BRANCH_COMPRESS)
    {
        branch_compress_t* compressor = (branch_compress_t *)del_compressor;
        delete compressor;

        return;
    }

    if (type == CYCLE_LINEAR_COMPRESS)
    {
        cycle_linear_compress_t* compressor = (cycle_linear_compress_t *)del_compressor;
        delete compressor;

        return;
    }

    if (type == CYCLE_INCREASE_COMPRESS)
    {
        cycle_increase_compress_t* compressor = (cycle_increase_compress_t *)del_compressor;
        delete compressor;

        return;
    }

    cout << "delete_compressor_with_type: compressor type is not supported" << endl;
    assert(false);
}