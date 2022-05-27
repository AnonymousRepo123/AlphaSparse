#include <configor/json.hpp>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <config.hpp>

using namespace std;

// 这个配置管理器的头文件
// using namespace configor;

void print_param(int param = get_config()["KERNAL_REPEAT_TIME"].as_integer())
{
    cout << param << endl;
}

int main()
{
    // configor::json json_obj = get_config();

    // json_obj["FIXED_THREAD_COMBINE_SIZE"] = 32;
    // json_obj["FIXED_WARP_COMBINE_SIZE"] = 4;
    // json_obj["get_config()["SHARED_MEM_TOTAL_SIZE"].as_integer()"] = 48000;
    // json_obj["get_config()["DEFAULT_THREAD_BLOCK_NUM"].as_integer()"] = 120;
    // json_obj["get_config()["DEFAULT_THREAD_NUM_IN_BLOCK"].as_integer()"] = 512;
    // json_obj["MAX_TBLOCK_NUM"] = 65535;
    // json_obj["MAX_THREAD_NUM_IN_BLOCK"] = 1024;
    // json_obj["KERNAL_REPEAT_TIME"] = 1200;
    // json_obj["PADDING_RATE_UP_BOUND"] = 4;
    // json_obj["GFLOPS_UP_BOUND"] = 1000;
    // json_obj["get_config()["MAX_ROW_REDUCE_THREAD"].as_integer()"] = 999999;
    // json_obj["get_config()["ROOT_PATH_STR"].as_string()"] = "/home/duzhen/spmv_builder";
    // json_obj["get_config()["spmv_header_file"].as_string()"] = json_obj["get_config()["ROOT_PATH_STR"].as_string()"].as_string() + "/spmv_header_top.code";
    // json_obj["get_config()["HALF_MAX_ROW_REDUCE_THREAD"].as_integer()"] = 999998;
    // json_obj["get_config()["SORT_THREAD_NUM"].as_integer()"] = 1;
    // json_obj["get_config()["BRANCH_COMPRESS_MAX_SIZE"].as_integer()"] = 5;
    // json_obj["get_config()["DEFAULT_DEVICE_ID"].as_integer()"] = 0;
    
    // ofstream ofs("global_config.json");
    // cout << setw(4) << json_obj << endl;

    print_param();
}





