// 机器学习数据集的收集，创建一个数据结构，包含了稠密视图的节点类型，
// 稠密视图参数策略类型，压缩视图节点类型，压缩视图参数策略类型，模板类型，最后还有一串从压缩子图开始的参数

#ifndef MACHINE_LEARNING_DATA_SET_COLLECTOR_H
#define MACHINE_LEARNING_DATA_SET_COLLECTOR_H

#include "struct.hpp"
#include "exe_graph.hpp"
#include "parameter_set_strategy.hpp"
#include "code_builder.hpp"
#include <memory>

using namespace std;


// 每一个条目的类型
class machine_learning_data_set_item
{
    // 包含一个几个类型，所有的数据类型的参数类型float
public:
    // 初始化一个item
    machine_learning_data_set_item(vector<exe_node_type> dense_graph_node_type_vec,
                                   vector<exe_node_param_set_strategy> dense_param_strategy_type_vec, vector<exe_node_type> compressed_graph_node_type_vec,
                                   vector<exe_node_param_set_strategy> compressed_param_strategy_type_vec, vector<float> all_param, 
                                   template_type type_of_template = NONE_TEMPLATE);
    
    // 将当前item打印出来
    string convert_item_to_string(bool just_param = false);

    // 将一个表项输出到文件中
    void append_item_to_file(string file_name);

private:
    // 稠密视图节点类型和策略
    vector<exe_node_type> dense_graph_node_type_vec;
    vector<exe_node_param_set_strategy> dense_param_strategy_type_vec;
    // 压缩视图的节点和策略
    vector<exe_node_type> compressed_graph_node_type_vec;
    vector<exe_node_param_set_strategy> compressed_param_strategy_type_vec;

    // 模板的类型
    template_type type_of_template = NONE_TEMPLATE;
    // 所有参数一字排开，是float类型的，最后一个参数是性能
    vector<float> all_param;
};

class machine_learning_data_set_collector
{
    // 维护一个表格
public: 
    // 构造函数，初始化要输出的文件
    machine_learning_data_set_collector(string file_name)
    {
        this->output_file_name = file_name;
    }

    // 向表格中增加一个表项
    void add_item_to_dataset(machine_learning_data_set_item item);

    // 传入具体的表项内容，来插入一个表项
    void add_item_to_dataset(vector<exe_node_type> dense_graph_node_type_vec,
                             vector<exe_node_param_set_strategy> dense_param_strategy_type_vec, vector<exe_node_type> compressed_graph_node_type_vec,
                             vector<exe_node_param_set_strategy> compressed_param_strategy_type_vec, vector<float> all_param,
                             template_type type_of_template = NONE_TEMPLATE);
    
    

    // 将整个表打印为字符串
    string convert_the_whole_dataset_to_string(bool just_param = false);

    // 将一个machine_learning_data_set_item的数组打印出来
    static string convert_item_vec_to_string(vector<machine_learning_data_set_item> data_set, bool just_param = false);

    // 传入dense阶段的节点和策略，将插入表项的dense阶段的节点类型和参数，类型好处理，参数需要额外实现一个抽取节点参数的函数
    // 这个时候针对下一个表项已经插入参数都是空的
    void insert_dense_stage_node_and_param_to_cur_item(exe_dense_sub_graph_t* dense_graph, param_strategy_of_sub_graph_t* dense_graph_param_strategy);
    // 直接传入类型和参数的数组的版本
    void insert_dense_stage_node_and_param_to_cur_item(vector<exe_node_type> dense_graph_node_type_vec, vector<exe_node_param_set_strategy> dense_param_strategy_type_vec, vector<float> param_vec);

    // 传入稠密阶段的节点和策略，并且插入部对应的参数，压缩阶段的参数和节点类型可能不是空的
    void insert_compressed_stage_node_and_param_to_cur_item(exe_compressed_sub_graph_t* compressed_graph, param_strategy_of_sub_graph_t* compressed_graph_param_strategy);
    // 直接传入类型和参数的数组
    void insert_compressed_stage_node_and_param_to_cur_item(vector<exe_node_type> compressed_graph_node_type_vec, vector<exe_node_param_set_strategy> compressed_param_strategy_type_vec, vector<float> param_vec);
    
    // 传入kernel阶段的节点和策略，并且插入对应的参数，加入一个参数代表是不是要将加入的表项输出到文件
    void insert_template_node_and_param_to_cur_item_and_add_to_dataset(template_type, vector<float> param_vec);

    // 清除下一个表项所有的积累值
    void clear_all_accu_info();

    // 清楚下一个表项所有的compressed阶段的积累
    void clear_compressed_accu_info();
    
    // 找出一张子表，将对应图结构的数据集提取出来
    // vector<machine_learning_data_set_item> 
    // 因为是递归的搜索，每个stage都会丢失之前stage的信息，所以需要用这个东西暂存下来
    vector<exe_node_type> accu_dense_graph_node_type_vec;
    vector<exe_node_param_set_strategy> accu_dense_param_strategy_type_vec;
    vector<float> accu_dense_param_vec;
    vector<exe_node_type> accu_compressed_sub_graph_node_type_vec;
    vector<exe_node_param_set_strategy> accu_compressed_sub_param_strategy_type_vec;
    vector<float> accu_compressed_param_vec;
    
private:
    vector<machine_learning_data_set_item> all_data_set;
    // 因为搜索的顺序是递归的，只有到templete阶段的时候才能得到性能数据。需要用一个变量来存储当前的item已经积攒的内容
    // Accumulated，当前积累的表项，会和template阶段的相关参数合并，并放到最终的表项中

    // 用一个变量存储如果要输出，那么输出的文件名
    string output_file_name = "";
};

// 声明一个空的，静态的中间数组指针
const static shared_ptr<machine_learning_data_set_collector> NULL_DATA_SET_COLLECTOR(nullptr);

#endif