#ifndef MULTI_THREAD_FUNCTION_H
#define MULTI_THREAD_FUNCTION_H

#include <pthread.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <string.h>
#include <unistd.h>

// 多线程排序 https://blog.csdn.net/Return_nellen/article/details/79937320
typedef struct sort_meta_of_each_thread
{
    // 两个数组的指针
    unsigned long *old_index_arr = NULL;
    unsigned long *val_arr = NULL;
    unsigned long size_of_arr = 0;
    unsigned long l,r;
    int id;
}sort_meta_of_each_thread_t;

// 一个线程的排序，传入一定的元数据
void sort_in_thread(void* thread_meta);

// 对一段数组的排序，这里使用冒泡排序，快排的递归方式怕有风险
void sort(int left,int right, unsigned long *old_index_arr, unsigned long *val_arr, unsigned long size_of_arr);

// 排序
void merge(unsigned long left1, unsigned long right1, unsigned long left2, unsigned long right2, unsigned long *old_index_arr, unsigned long *val_arr, unsigned long size_of_arr);

// 要参与排序的两个数组，一个是每个非零元的在排序之前的索引，一个是数组的非零元
void multi_thread_sort(unsigned long * old_index_arr, unsigned long * val_arr, unsigned long size_of_arr, unsigned long thread_num);

#endif