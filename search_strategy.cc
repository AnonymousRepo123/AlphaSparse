#include "search_strategy.hpp"
#include <iostream>

using namespace std;

search_strategy_t init_search_strategy(unsigned long struggle_step_num, double total_allow_search_time)
{
    search_strategy_t return_time_strategy;
    
    return_time_strategy.remain_step_num = 10;

    // 执行对应的
    return_time_strategy.struggle_step_num = struggle_step_num;
    return_time_strategy.total_allow_search_time = total_allow_search_time;
    
    gettimeofday(&(return_time_strategy.begin_time), NULL);

    return_time_strategy.best_glops = 0;

    // 返回对应的执行
    return return_time_strategy;
}

bool continue_search(search_strategy_t* search_strategy_ptr, float gflops)
{
    assert(search_strategy_ptr != NULL);

    // 如果当前性能更好，刷新搜索的窗口，窗口大小为领先的大小*5，窗口大小不低于5，不高于25
    if (gflops > search_strategy_ptr->best_glops)
    {
        unsigned long new_remain_step_num = (gflops - search_strategy_ptr->best_glops) * 5;

        if (new_remain_step_num < 5)
        {
            new_remain_step_num = 5;
        }

        if (new_remain_step_num > 25)
        {
            new_remain_step_num = 25;
        }

        search_strategy_ptr->remain_step_num = new_remain_step_num;
        // 更新最大的性能
        search_strategy_ptr->best_glops = gflops;
    }
    else
    {
        // 如果还有剩余的remain_step_num，那就减一
        if (search_strategy_ptr->remain_step_num > 0)
        {
            search_strategy_ptr->remain_step_num = search_strategy_ptr->remain_step_num - 1;
        }
        else if (search_strategy_ptr->struggle_step_num > 0)
        {
            search_strategy_ptr->struggle_step_num = search_strategy_ptr->struggle_step_num - 1;
        }
        else
        {
            cout << "need to finish search" << endl;
            return false;
        }
    }
    
    // 计算当前的时间
    struct timeval end_time;

    gettimeofday(&(end_time), NULL);

    double time_use = (double)(end_time.tv_sec - search_strategy_ptr->begin_time.tv_sec) + (double)(end_time.tv_usec - search_strategy_ptr->begin_time.tv_usec)/1000000.0;

    cout << "continue_search: time_use:" << time_use << ", total_allow_search_time:" << search_strategy_ptr->total_allow_search_time << endl;

    if (time_use >= search_strategy_ptr->total_allow_search_time)
    {
        cout << "need to finish search" << endl;
        return false;
    }

    return true;
}

bool continue_search(search_strategy_t* search_strategy_ptr)
{
    assert(search_strategy_ptr != NULL);

    // 计算当前的时间
    struct timeval end_time;

    gettimeofday(&(end_time), NULL);

    double time_use = (double)(end_time.tv_sec - search_strategy_ptr->begin_time.tv_sec) + (double)(end_time.tv_usec - search_strategy_ptr->begin_time.tv_usec)/1000000.0;

    cout << "continue_search: time_use:" << time_use << ", total_allow_search_time:" << search_strategy_ptr->total_allow_search_time << endl;

    // 超时了
    if (time_use >= search_strategy_ptr->total_allow_search_time)
    {
        cout << "need to finish search" << endl;
        return false;
    }

    // 两个计数器都停了
    if (search_strategy_ptr->remain_step_num == 0 && search_strategy_ptr->struggle_step_num == 0)
    {
        cout << "need to finish search" << endl;
        return false;
    }

    return true;
}