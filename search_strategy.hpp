#ifndef SEARCH_STRATEGY_H
#define SEARCH_STRATEGY_H

// 实现一个类似于并行退火算法的算法

#include <sys/time.h>
#include <iostream>
#include <assert.h>

// 用一个变量存储还能进一步容许的步长
// 用一个变量存储起始的时间
// 用一个变量存储容许的最大搜索时间
// 用一个变量存储当前的最优性能
// 根据性能增加的幅度计算接下来要再尝试的步长，如果性能增加的少，再尝试的步长就少，性能增加的多，那接下里就多尝试。
// 可以为每一个骨架设计一些重新尝试的点，针对每一个骨架都需要重新尝试
// 如果在给出的步长窗口中没有搜出来，那就再给最后一个窗口，这个窗口会给一个“挣扎步长”，挣扎的总步长不能超过固定值。
typedef struct search_strategy
{
    // 根据最近一次提升的幅度，给出在提升之后进行搜索的窗口
    unsigned long remain_step_num;
    // 垂死挣扎的总步长，当搜索窗口走过之后，就会消耗“垂死挣扎步长”，当锤死挣扎步长消耗完之后，就停止搜索
    unsigned long struggle_step_num;
    // 起始时间
    struct timeval begin_time;
    // 能够接受的总时间秒数
    double total_allow_search_time;
    // 当前的最大非零元数量
    float best_glops;
} search_strategy_t;

// 初始化一个新的搜索策略
search_strategy_t init_search_strategy(unsigned long struggle_step_num, double total_allow_search_time);

// 用最近一次性能来更新要进行搜索的步数，返回是不是需要继续尝试，如果不用继续尝试了，就返回false
bool continue_search(search_strategy_t* search_strategy_ptr, float gflops);

// 直接查看是不是计数器和时间已经用完了
bool continue_search(search_strategy_t* search_strategy_ptr);

#endif