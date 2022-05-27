#include "machine_learning_data_set_collector.hpp"

machine_learning_data_set_item::machine_learning_data_set_item(vector<exe_node_type> dense_graph_node_type_vec,
                                                               vector<exe_node_param_set_strategy> dense_param_strategy_type_vec, vector<exe_node_type> compressed_graph_node_type_vec,
                                                               vector<exe_node_param_set_strategy> compressed_param_strategy_type_vec, vector<float> all_param,
                                                               template_type type_of_template)
{
    assert(dense_graph_node_type_vec.size() == dense_param_strategy_type_vec.size());
    assert(compressed_param_strategy_type_vec.size() == compressed_param_strategy_type_vec.size());

    // 进行一个赋值
    this->dense_graph_node_type_vec = dense_graph_node_type_vec;
    this->dense_param_strategy_type_vec = dense_param_strategy_type_vec;
    this->compressed_graph_node_type_vec = compressed_graph_node_type_vec;
    this->compressed_param_strategy_type_vec = compressed_param_strategy_type_vec;
    this->type_of_template = type_of_template;
    this->all_param = all_param;
}

void machine_learning_data_set_collector::add_item_to_dataset(machine_learning_data_set_item item)
{
    this->all_data_set.push_back(item);
}

void machine_learning_data_set_collector::add_item_to_dataset(vector<exe_node_type> dense_graph_node_type_vec,
                                                              vector<exe_node_param_set_strategy> dense_param_strategy_type_vec, vector<exe_node_type> compressed_graph_node_type_vec,
                                                              vector<exe_node_param_set_strategy> compressed_param_strategy_type_vec, vector<float> all_param,
                                                              template_type type_of_template)
{
    // 创建一个新的表项
    machine_learning_data_set_item item(dense_graph_node_type_vec, dense_param_strategy_type_vec, compressed_graph_node_type_vec, compressed_param_strategy_type_vec, all_param, type_of_template);

    this->add_item_to_dataset(item);
}

void machine_learning_data_set_item::append_item_to_file(string file_name)
{
    ofstream write;
    
    cout << "machine_learning_data_set_item::append_item_to_file: write to " << file_name << endl;

    write.open(file_name, ios::app);

    write << this->convert_item_to_string() << endl;

    write.close();
}

string machine_learning_data_set_item::convert_item_to_string(bool just_param)
{
    assert(this->dense_graph_node_type_vec.size() == this->dense_param_strategy_type_vec.size());
    assert(this->compressed_graph_node_type_vec.size() == this->compressed_param_strategy_type_vec.size());
    // 每一个表里面总是要有点东西的
    assert(this->compressed_graph_node_type_vec.size() != 0 || this->dense_graph_node_type_vec.size() != 0 || 
        this->all_param.size() != 0 || this->type_of_template != NONE_TEMPLATE || this->all_param.size() != 0);

    string return_str = "";

    bool first_column = true;

    // 如果不是仅仅只有参数，那就需要一系列的节点类型
    if (just_param == false)
    {
        // 首先是dense节点的类型以及参数策略的类型
        for (unsigned long i = 0; i < this->dense_graph_node_type_vec.size(); i++)
        {
            if (first_column == true)
            {
                first_column = false;
            }
            else
            {
                return_str = return_str +  ",";
            }
            
            return_str = return_str + convert_exe_node_type_to_string(this->dense_graph_node_type_vec[i]);
            return_str = return_str + "," + convert_param_set_strategy_to_string(this->dense_param_strategy_type_vec[i]);
        }

        // 然后打印所有的压缩视图的节点和策略类型
        for (unsigned long i = 0; i < this->compressed_graph_node_type_vec.size(); i++)
        {
            if (first_column == true)
            {
                first_column = false;
            }
            else
            {
                return_str = return_str +  ",";
            }
            
            // 打印
            return_str = return_str + convert_exe_node_type_to_string(this->compressed_graph_node_type_vec[i]);
            return_str = return_str + "," + convert_param_set_strategy_to_string(this->compressed_param_strategy_type_vec[i]);
        }

        // 打印模板类型
        if (this->type_of_template != NONE_TEMPLATE)
        {
            if (first_column == true)
            {
                first_column = false;
            }
            else
            {
                return_str = return_str +  ",";
            }

            return_str = return_str + convert_template_type_to_string(this->type_of_template);
        }
    }

    // 打印剩余的所有参数
    for (unsigned long i = 0; i < this->all_param.size(); i++)
    {
        if (first_column == true)
        {
            first_column = false;
        }
        else
        {
            return_str = return_str +  ",";
        }

        return_str = return_str + to_string(this->all_param[i]);
    }

    return return_str;
}

string machine_learning_data_set_collector::convert_the_whole_dataset_to_string(bool just_param)
{
    // 打印当前的整个数据集的内容
    return machine_learning_data_set_collector::convert_item_vec_to_string(this->all_data_set, just_param);
}

string machine_learning_data_set_collector::convert_item_vec_to_string(vector<machine_learning_data_set_item> data_set, bool just_param)
{
    // 将所有的条目一个个打印出来
    string return_str = "";
    for (unsigned long i = 0; i < data_set.size(); i++)
    {
        return_str = return_str + data_set[i].convert_item_to_string(just_param);

        // 只要不是最后一个，那就那就要换行
        if (i != data_set.size() - 1)
        {
            return_str = return_str + "\n";
        }
    }

    return return_str;
}

void machine_learning_data_set_collector::insert_dense_stage_node_and_param_to_cur_item(vector<exe_node_type> dense_graph_node_type_vec, vector<exe_node_param_set_strategy> dense_param_strategy_type_vec, vector<float> param_vec)
{
    // 当前的所有累计数据必须都是空的
    assert(this->accu_dense_graph_node_type_vec.size() == 0);
    assert(this->accu_dense_param_strategy_type_vec.size() == 0);
    assert(this->accu_dense_param_vec.size() == 0);
    assert(this->accu_compressed_sub_graph_node_type_vec.size() == 0);
    assert(this->accu_compressed_sub_param_strategy_type_vec.size() == 0);
    assert(this->accu_compressed_param_vec.size() == 0);

    assert(dense_graph_node_type_vec.size() == dense_param_strategy_type_vec.size());

    // 对剩余的三个值进行赋值
    this->accu_dense_graph_node_type_vec = dense_graph_node_type_vec;
    this->accu_dense_param_strategy_type_vec = dense_param_strategy_type_vec;
    this->accu_dense_param_vec = param_vec;
}

void machine_learning_data_set_collector::insert_compressed_stage_node_and_param_to_cur_item(vector<exe_node_type> compressed_graph_node_type_vec, vector<exe_node_param_set_strategy> compressed_param_strategy_type_vec, vector<float> param_vec)
{
    // compressed阶段的数据必须是空的
    assert(this->accu_compressed_sub_param_strategy_type_vec.size() == 0);
    assert(this->accu_compressed_sub_graph_node_type_vec.size() == 0);
    assert(this->accu_compressed_param_vec.size() == 0);
    assert(this->accu_dense_graph_node_type_vec.size() == this->accu_dense_param_strategy_type_vec.size());

    assert(compressed_graph_node_type_vec.size() == compressed_param_strategy_type_vec.size());

    // 赋值
    this->accu_compressed_sub_graph_node_type_vec = compressed_graph_node_type_vec;
    this->accu_compressed_sub_param_strategy_type_vec = compressed_param_strategy_type_vec;
    this->accu_compressed_param_vec = param_vec;
}

void machine_learning_data_set_collector::insert_template_node_and_param_to_cur_item_and_add_to_dataset(template_type type_of_template, vector<float> param_vec)
{
    // cout << "insert_template_node_and_param_to_cur_item_and_add_to_dataset: into" << endl;
    // 已经积累的内容要满足要求
    assert(this->accu_dense_graph_node_type_vec.size() == this->accu_dense_param_strategy_type_vec.size());
    assert(this->accu_compressed_sub_graph_node_type_vec.size() == this->accu_compressed_sub_param_strategy_type_vec.size());
    assert(type_of_template != NONE_TEMPLATE);

    // 所有的参数合并起来
    vector<float> all_param_vec;

    for (unsigned long i = 0; i < this->accu_dense_param_vec.size(); i++)
    {
        all_param_vec.push_back(this->accu_dense_param_vec[i]);
    }

    for (unsigned long i = 0; i < this->accu_compressed_param_vec.size(); i++)
    {
        all_param_vec.push_back(this->accu_compressed_param_vec[i]);
    }

    // 将模板的参数加入进来
    for (unsigned long i = 0; i < param_vec.size(); i++)
    {
        all_param_vec.push_back(param_vec[i]);
    }

    // 将新的内容放到表格中
    this->add_item_to_dataset(this->accu_dense_graph_node_type_vec, this->accu_dense_param_strategy_type_vec, this->accu_compressed_sub_graph_node_type_vec,
        this->accu_compressed_sub_param_strategy_type_vec, all_param_vec, type_of_template);

    // 查看是不是需要输出到文件
    if (this->output_file_name != "")
    {
        assert(this->all_data_set.size() > 0);
        // 将表格中的最新内容输出到文件中
        machine_learning_data_set_item last_item = this->all_data_set[this->all_data_set.size() - 1];
        // 将最后一个表项的内容追加到文件中
        last_item.append_item_to_file(this->output_file_name);
    }
}

void machine_learning_data_set_collector::clear_all_accu_info()
{
    vector<exe_node_type> empty_node_type_vec;
    vector<exe_node_param_set_strategy> empty_node_strategy_vec;
    vector<float> empty_param_vec;

    this->accu_dense_graph_node_type_vec = empty_node_type_vec;
    this->accu_compressed_sub_graph_node_type_vec = empty_node_type_vec;
    
    this->accu_dense_param_strategy_type_vec = empty_node_strategy_vec;
    this->accu_compressed_sub_param_strategy_type_vec = empty_node_strategy_vec;

    this->accu_dense_param_vec = empty_param_vec;
    this->accu_compressed_param_vec = empty_param_vec;
}

void machine_learning_data_set_collector::clear_compressed_accu_info()
{
    vector<exe_node_type> empty_node_type_vec;
    vector<exe_node_param_set_strategy> empty_node_strategy_vec;
    vector<float> empty_param_vec;

    this->accu_compressed_sub_graph_node_type_vec = empty_node_type_vec;
    this->accu_compressed_sub_param_strategy_type_vec = empty_node_strategy_vec;
    this->accu_compressed_param_vec = empty_param_vec;
}

