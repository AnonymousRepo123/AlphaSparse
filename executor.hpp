#ifndef EXECUTOR_H
#define EXECUTOR_H

#include "code_builder.hpp"

// 给执行器自适应调整执行时间的功能

// 执行或部分执行一个code_builder，返回是不是执行成功的结果，参数是输出代码文件的目录
// 主要工作是输出文件、输出代码、编译代码，执行文件，找到输出的性能。
// 用一个变量决定要不要持久化数据，在调参的时候，不需要反复写元数据，用一个变量来控制是不是要写元数据
// 最后给一个缺省参数决定要不要自适应nnz，保证运行时间可以在8秒左右，先执行一个2000的遍历次数，然后根据执行的之间执行一个新的遍历次数，
bool part_execute_code_builder(code_builder_t* builder, vector<int> sub_matrix_id_vec, float& time_ms, float& gflops, string execute_path, string data_path, bool store_data);

bool part_execute_code_builder(code_builder_t* builder, vector<int> sub_matrix_id_vec, float& time_ms, float& gflops, string execute_path, string data_path, bool store_data, bool adaptive_repeat);

// 完全执行一个code_builder
bool execute_code_builder(code_builder_t* builder, float& time_ms, float& gflops, string execute_path, string data_path, bool store_data);

// 编译源文件，查看返回值是编译是不是能通过
bool compile_spmv_code(string execute_path);

// 执行代码，并且获得结果，用一个返回值判断是不是执行成功
bool execute_binary(string execute_path, float& total_execute_time, float& gflops);

#endif