#include "param_enumerater.hpp"

integer_independ_param_t init_integer_independ_param(long* param_ptr, long param_low_bound, long param_up_bound, long step_size)
{
    // 初始化一个参数
    assert(param_ptr != NULL);
    assert(param_low_bound <= param_up_bound);
    assert(step_size >= 1);

    integer_independ_param_t param_setter;

    param_setter.param_ptr = param_ptr;
    param_setter.param_low_bound = param_low_bound;
    param_setter.param_up_bound = param_up_bound;
    param_setter.step_size = step_size;

    // 参数的初值比下界小一点点
    *(param_setter.param_ptr) = param_low_bound - 1;

    // 下界就是第一个值
    param_setter.next_param_val = param_low_bound;

    return param_setter;
}

float_independ_param_t init_float_independ_param(double* param_ptr, double param_low_bound, double param_up_bound, double step_size)
{
    // 初始化一个参数
    assert(param_ptr != NULL);
    assert(param_low_bound <= param_up_bound);
    assert(step_size > 0);

    float_independ_param_t param_setter;

    param_setter.param_ptr = param_ptr;
    param_setter.param_low_bound = param_low_bound;
    param_setter.param_up_bound = param_up_bound;
    param_setter.step_size = step_size;

    *(param_setter.param_ptr) = param_low_bound - 1;

    // 下界就是第一个值
    param_setter.next_param_val = param_low_bound;

    return param_setter;
}

single_dependency_param_t init_single_dependency_param(void* param, void* depend_param, data_type param_data_type, data_type depend_param_data_type, void (*fun)(void*, void*, data_type, data_type))
{
    assert(depend_param != NULL && param != NULL && fun != NULL);
    assert(param_data_type == DOUBLE || param_data_type == LONG);
    assert(depend_param_data_type == DOUBLE || depend_param_data_type == LONG);

    single_dependency_param_t param_setter;

    param_setter.param = param;
    param_setter.depend_param = depend_param;
    param_setter.param_data_type = param_data_type;
    param_setter.depend_param_data_type = depend_param_data_type;
    param_setter.fun = fun;

    return param_setter;
}

binary_param_dependency_filter_t init_binary_param_dependency_filter(void* param1, void* param2, data_type param1_data_type, data_type param2_data_type, bool (*fun)(void*, void*, data_type, data_type))
{
    assert(param1 != NULL && param2 != NULL && fun != NULL);
    assert(param1_data_type == DOUBLE || param1_data_type == LONG);
    assert(param2_data_type == DOUBLE || param2_data_type == LONG);

    binary_param_dependency_filter_t filter;
    
    filter.param1 = param1;
    filter.param1_data_type = param1_data_type;
    filter.param2 = param2;
    filter.param2_data_type = param2_data_type;
    filter.fun = fun;

    return filter;
}

bool set_integer_independ_param_to_next(integer_independ_param_t* param_setter)
{
    assert(param_setter != NULL);
    assert(param_setter->param_ptr != NULL);

    long old_param = *(param_setter->param_ptr);

    // 将下一个参数的值赋值到当前位置
    *(param_setter->param_ptr) = param_setter->next_param_val;

    // 将参数在现有的基础上处理
    param_setter->next_param_val = param_setter->next_param_val + param_setter->step_size;

    // 如果越界了，就回到原点
    if (param_setter->next_param_val > param_setter->param_up_bound)
    {
        param_setter->next_param_val = param_setter->param_low_bound;
    }

    // 确定返回值，如果参数赋值之后，值变小了，或者值没变，就代表到新的一轮了
    if (old_param >= *(param_setter->param_ptr))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool set_float_independ_param_to_next(float_independ_param_t* param_setter)
{
    assert(param_setter != NULL);
    assert(param_setter->param_ptr != NULL);
    
    double old_param = *(param_setter->param_ptr);

    // 将下个参数的值赋值到当前位置
    *(param_setter->param_ptr) = param_setter->next_param_val;

    // cout << "old_param:" << old_param << " " << "next_param_val:" << param_setter->next_param_val << endl;

    // 更新参数的下一个枚举
    param_setter->next_param_val = param_setter->next_param_val + param_setter->step_size;

    // 如果越界了，就回到原点
    if (param_setter->next_param_val > param_setter->param_up_bound)
    {
        param_setter->next_param_val = param_setter->param_low_bound;
    }

    // 确定返回值，如果参数赋值之后，值变小了，或者值没变，就代表到新的一轮了
    if (old_param >= *(param_setter->param_ptr))
    {
        return true;
    }
    else
    {
        return false;
    }
}

// 如果当前值正好等于上界，或者下一个枚举小于当前值，就代表到末尾了
bool is_end_of_integer_param_setter(integer_independ_param_t* param_setter)
{
    assert(param_setter != NULL);
    assert(param_setter->param_ptr != NULL);

    if (*(param_setter->param_ptr) == param_setter->param_up_bound || param_setter->next_param_val <= *(param_setter->param_ptr))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool is_end_of_float_param_setter(float_independ_param_t* param_setter)
{
    assert(param_setter != NULL);
    assert(param_setter->param_ptr != NULL);
    
    if (*(param_setter->param_ptr) == param_setter->param_up_bound || param_setter->next_param_val <= *(param_setter->param_ptr))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool is_end_of_param_combination(param_enumerater_t* param_enumerater)
{
    assert(param_enumerater != NULL);
    
    // 遍历所有的整型参数和所有的浮点型参数
    for (int i = 0; i < param_enumerater->integer_independ_param_vec.size(); i++)
    {
        if (is_end_of_integer_param_setter(&(param_enumerater->integer_independ_param_vec[i])) == false)
        {
            return false;
        }
    }

    for (int i = 0; i < param_enumerater->float_independ_param_vec.size(); i++)
    {
        if (is_end_of_float_param_setter(&(param_enumerater->float_independ_param_vec[i])) == false)
        {
            return false;
        }
    }

    return true;
}

bool set_param_combination_to_next(param_enumerater_t* param_enumerater)
{
    assert(param_enumerater != NULL);

    // 所有的参数数量
    int all_param_num = param_enumerater->integer_independ_param_vec.size() + param_enumerater->float_independ_param_vec.size();

    // cout << "all_param_num:" << all_param_num << endl;

    // 再找到下一个参数组合之前一直遍历
    while (is_end_of_param_combination(param_enumerater) == false)
    {
        for (int i = 0; i < all_param_num; i++)
        {
            // cout << "i:" << i << endl;
            if (i < param_enumerater->integer_independ_param_vec.size())
            {
                // 获得旧的参数，如果旧的参数小于下界，说明这是第一组参数，需要进位
                long old_param = *(param_enumerater->integer_independ_param_vec[i].param_ptr);

                // 查看是不是需要进位
                bool go_next_param_setter = set_integer_independ_param_to_next(&(param_enumerater->integer_independ_param_vec[i]));
                
                // 对于第一组参数来说，进位是必须的，给所有的参数赋值为下界
                if (old_param < param_enumerater->integer_independ_param_vec[i].param_low_bound)
                {
                    // 这是第一组参数
                    go_next_param_setter = true;
                }

                // 如果不需要进位，就退出循环
                if (go_next_param_setter == false)
                {
                    break;
                }
                // else
                // {
                //     if (i == 2)
                //     {
                //         cout << "进位" << endl;
                //     }
                // }
            }
            else
            {
                assert((i - param_enumerater->integer_independ_param_vec.size()) < param_enumerater->float_independ_param_vec.size());

                // 记录旧的参数
                double old_param = *(param_enumerater->float_independ_param_vec[i - param_enumerater->integer_independ_param_vec.size()].param_ptr);

                bool go_next_param_setter = set_float_independ_param_to_next(&(param_enumerater->float_independ_param_vec[i - param_enumerater->integer_independ_param_vec.size()]));
                
                if (old_param < param_enumerater->float_independ_param_vec[i - param_enumerater->integer_independ_param_vec.size()].param_low_bound)
                {
                    go_next_param_setter = true;
                }

                // 如果不进位，就退出
                if (go_next_param_setter == false)
                {
                    break;
                }
            }
        }

        // 这里找到了下一组无依赖参数，这里把依赖其他参数的参数计算出来
        for (int i = 0; i < param_enumerater->single_dependency_param_vec.size(); i++)
        {
            single_dependency_param_t* depend_param = &(param_enumerater->single_dependency_param_vec[i]);

            assert(depend_param->param != NULL && depend_param->depend_param != NULL && depend_param->fun != NULL);
            // 执行函数指针
            depend_param->fun(depend_param->param, depend_param->depend_param, depend_param->param_data_type, depend_param->depend_param_data_type);
        }

        // 检查所有的所有的依赖，如果依赖通过了就返回，不通过就执行下一轮
        bool pass_the_check = true;

        for (int i = 0; i < param_enumerater->binary_param_dependency_filter_vec.size(); i++)
        {
            binary_param_dependency_filter_t* filter = &(param_enumerater->binary_param_dependency_filter_vec[i]);
            assert(filter->fun != NULL && filter->param1 != NULL && filter->param2 != NULL);
            // 执行函数指针
            bool pass_the_filter = filter->fun(filter->param1, filter->param2, filter->param1_data_type, filter->param2_data_type);

            pass_the_check = pass_the_check && pass_the_filter;
        }

        // 通过检查就直接返回
        if (pass_the_check == true)
        {
            return false;
        }
    }

    // 如果到外面来了，说明搜索空间已经被遍历尽了，但是没有找出下一组参数组合
    return true;
}

void assert_check_repeat_param_ptr(param_enumerater_t* param_enumerater, void* param_ptr)
{
    assert(param_enumerater != NULL && param_ptr != NULL);
    
    // 检查当前指针在枚举器中有没有出现过
    for (int i = 0; i < param_enumerater->integer_independ_param_vec.size(); i++)
    {
        assert((void*)(param_enumerater->integer_independ_param_vec[i].param_ptr) != param_ptr);
    }

    for (int i = 0; i < param_enumerater->float_independ_param_vec.size(); i++)
    {
        assert((void*)(param_enumerater->float_independ_param_vec[i].param_ptr) != param_ptr);
    }

    for (int i = 0; i < param_enumerater->single_dependency_param_vec.size(); i++)
    {
        assert((void*)(param_enumerater->single_dependency_param_vec[i].param) != param_ptr);
    }    
}

void register_integer_independ_param_to_enumerater(param_enumerater_t* param_enumerater, long* param_ptr, long param_low_bound, long param_up_bound, long step_size)
{
    assert(param_enumerater != NULL && param_ptr != NULL);

    if (param_up_bound < param_low_bound)
    {
        cout << "param_up_bound:" << param_up_bound << ",param_low_bound:" << param_low_bound << endl;
    }

    if (step_size <= 0)
    {
        cout << "step_size:" << step_size << endl;
    }

    assert(param_up_bound >= param_low_bound && step_size > 0);
    
    
    assert_check_repeat_param_ptr(param_enumerater, param_ptr);

    // 生成一个参数设定器
    integer_independ_param_t param_setter = init_integer_independ_param(param_ptr, param_low_bound, param_up_bound, step_size);

    param_enumerater->integer_independ_param_vec.push_back(param_setter);
}

void register_float_independ_param_to_enumerater(param_enumerater_t* param_enumerater, double* param_ptr, double param_low_bound, double param_up_bound, double step_size)
{
    assert(param_enumerater != NULL && param_ptr != NULL);
    assert(param_up_bound >= param_low_bound && step_size > 0);

    assert_check_repeat_param_ptr(param_enumerater, param_ptr);

    float_independ_param_t param_setter = init_float_independ_param(param_ptr, param_low_bound, param_up_bound, step_size);

    param_enumerater->float_independ_param_vec.push_back(param_setter);
}

void register_single_dependency_param_to_enumerater(param_enumerater_t* param_enumerater, void* param, data_type param_data_type, void* depend_param, data_type depend_param_data_type, void (*fun)(void*, void*, data_type, data_type))
{
    assert(param_enumerater != NULL && param != NULL && depend_param != NULL && fun != NULL);
    assert(param_data_type == DOUBLE || param_data_type == LONG);
    assert(depend_param_data_type == DOUBLE || depend_param_data_type == LONG);

    // 要被修改的参数是之前不能出现过的
    assert_check_repeat_param_ptr(param_enumerater, param);

    single_dependency_param_t param_setter = init_single_dependency_param(param, depend_param, param_data_type, depend_param_data_type, fun);
    
    param_enumerater->single_dependency_param_vec.push_back(param_setter);
}

void register_binary_param_dependency_filter_to_enumerater(param_enumerater_t* param_enumerater, void* param1, data_type param1_data_type, void* param2, data_type param2_data_type, bool (*fun)(void*, void*, data_type, data_type))
{
    assert(param_enumerater != NULL && param1 != NULL && param2 != NULL);
    assert(param1_data_type == DOUBLE || param1_data_type == LONG);
    assert(param2_data_type == DOUBLE || param2_data_type == LONG);

    binary_param_dependency_filter_t filter = init_binary_param_dependency_filter(param1, param2, param1_data_type, param2_data_type, fun);

    param_enumerater->binary_param_dependency_filter_vec.push_back(filter);
}