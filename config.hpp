#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <configor/json.hpp>
#include <fstream>
using namespace std;

#define CONFIG_FILE_NAME "global_config.json"

// 是不是要使用最终版本的输出
#define IDEAL_OUTPUT 1

configor::json get_config();
// 一个头文件，用来获取当前配置文件的json对象

// configor::json get_config()
// {
//     // 从外面读配置文件
//     configor::json return_json;
//     // 处理
//     ifstream ifs(CONFIG_FILE_NAME);

//     ifs >> return_json;

//     ifs.close();

//     return return_json;
// }

#endif