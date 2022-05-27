#include "config.hpp"

configor::json get_config()
{
    // 从外面读配置文件
    configor::json return_json;
    // 处理
    ifstream ifs(CONFIG_FILE_NAME);

    ifs >> return_json;

    ifs.close();

    return return_json;
}