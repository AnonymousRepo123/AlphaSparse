#include "executor.hpp"
#include <iterator>

using namespace std;

bool execute_binary(string execute_path, float& total_execute_time, float& gflops)
{
    // execute_path是执行目录
    // 删除执行目录可能存在结果文件
    string command = "cd " + execute_path + " && rm perf_result";
    system(command.c_str());

    // 执行编译
    bool is_compiler_success = compile_spmv_code(execute_path);
    
    if (is_compiler_success == false)
    {
        return false;
    }

    // 这里代表编译成功，成功之后就开始执行
    command = "cd " + execute_path + " && ./a.out";
    
    // 执行
    system(command.c_str());

    // 执行完之后查看结果文件是不是写出来了
    ifstream read_perf_result(execute_path + "/perf_result");

    // 有没有对应的文件
    if (!read_perf_result)
    {
        cout << "fail execution" << endl;
        return false;
    }

    float file_time;
    float file_gflops;

    // 获得时间
    read_perf_result >> file_time;
    read_perf_result >> file_gflops;

    // cout << file_time << " " << file_gflops << endl;
    
    read_perf_result.close();

    total_execute_time = file_time;
    gflops = file_gflops;

    // 如果gflops大于150，说明执行出现了问题
    if (file_gflops > get_config()["GFLOPS_UP_BOUND"].as_integer())
    {
        cout << "gflops is too high, maybe some mistake happened in kernal" << endl;
        return false;
    }

    return true;
}

bool compile_spmv_code(string execute_path)
{
    // 删除编译器的输出
    string command = "cd " + execute_path + " && rm compile_result";
    system(command.c_str());

    // 执行编译
    command = "cd " + execute_path + " && ./make_template.sh > compile_result 2>&1";
    system(command.c_str());

    // 读取编译结果中的内容，放在compile_result中
    ifstream in(execute_path + "/compile_result");
    istreambuf_iterator<char> begin(in);
    istreambuf_iterator<char> end;
    string compile_result(begin, end);

    string::size_type position = compile_result.find("error");

    // 查看编译的结果
    if (position != compile_result.npos)
    {
        // 编译有错误
        cout << "compile error!" << endl << compile_result << endl;
        assert(false);
        return false;
    }
    else
    {
        // 编译没有错误
        cout << "compile success" << endl;
    }

    ifstream fin(execute_path + "/a.out");

    if (!fin)
    {
        cout << "can't find compile result" << endl;
        return false;
    }

    fin.close();

    return true;
}

// 是不是要通过预执行来自定义循环的次数
bool part_execute_code_builder(code_builder_t* builder, vector<int> sub_matrix_id_vec, float& time_ms, float& gflops, string execute_path, string data_path, bool store_data, bool adaptive_repeat)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL && builder->op_manager->matrix != NULL);
    assert(sub_matrix_id_vec.size() > 0);
    assert(builder->op_manager->matrix->block_coor_table.item_arr.size() > 0);

    if (adaptive_repeat == false)
    {
        return part_execute_code_builder(builder, sub_matrix_id_vec, time_ms, gflops, execute_path, data_path, store_data);
    }
    else
    {
        // 这里代表需要自适应repeat，先预运行一次，然后再多次运行，先运行2000次
        // 每一个子图都是充分分块的，并且每一个子块的初始化了模板
        for (int i = 0; i < sub_matrix_id_vec.size(); i++)
        {
            int sub_matrix_id = sub_matrix_id_vec[i];
            
            assert(sub_matrix_id < builder->template_vec.size() && builder->template_vec.size() == builder->template_type_vec.size());
            assert(builder->template_vec[sub_matrix_id] != NULL);
            assert(sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());

            assert(builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id] != NULL);
            assert(builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

            // 获取压缩子图的指针
            compressed_block_t* compressed_block_ptr = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;
            assert(compressed_block_ptr->read_index.size() == 7);
        }

        // 将数据写到硬盘中
        if (store_data == true)
        {
            store_code_builder_data(builder, sub_matrix_id_vec, data_path);    
        }

        unsigned long pre_repeat_num = 2000;

        write_string_to_file(execute_path + "/template.h", build_header_file(builder, sub_matrix_id_vec));
        write_string_to_file(execute_path + "/template.cu", build_main_file(builder, sub_matrix_id_vec, pre_repeat_num));

        bool is_pre_success = execute_binary(execute_path, time_ms, gflops);

        if (is_pre_success == false)
        {
            return false;
        }

        unsigned long repeat_num = (8000.0 / time_ms) * pre_repeat_num;

        write_string_to_file(execute_path + "/template.h", build_header_file(builder, sub_matrix_id_vec));
        write_string_to_file(execute_path + "/template.cu", build_main_file(builder, sub_matrix_id_vec, repeat_num));

        return execute_binary(execute_path, time_ms, gflops);
    }
}

bool execute_code_builder(code_builder_t* builder, float& time_ms, float& gflops, string execute_path, string data_path, bool store_data)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL && builder->op_manager->matrix != NULL);
    assert(builder->op_manager->matrix->block_coor_table.item_arr.size() > 0);

    assert(builder->template_vec.size() == builder->template_type_vec.size() && builder->template_type_vec.size() == builder->op_manager->matrix->block_coor_table.item_arr.size());

    // 充分分块，存在模板
    for (unsigned long i = 0; i < builder->op_manager->matrix->block_coor_table.item_arr.size(); i++)
    {
        assert(builder->op_manager->matrix->block_coor_table.item_arr[i] != NULL);
        assert(builder->op_manager->matrix->block_coor_table.item_arr[i]->compressed_block_ptr != NULL);
        // 对应子块已经被完全分块
        assert(builder->op_manager->matrix->block_coor_table.item_arr[i]->compressed_block_ptr->read_index.size() == 7);
        // 对应的模板已经存在
        assert(builder->template_vec[i] != NULL && builder->template_type_vec[i] != NONE_TEMPLATE);
    }
    
    // 将数据写到硬盘中
    if (store_data == true)
    {
        store_code_builder_data(builder, data_path);    
    }

    // 先预执行一下
    unsigned long pre_repeat_num = 2000;

    write_string_to_file(execute_path + "/template.h", build_header_file(builder));
    write_string_to_file(execute_path + "/template.cu", build_main_file(builder, pre_repeat_num));

    bool is_pre_success = execute_binary(execute_path, time_ms, gflops);

    if (is_pre_success == false)
    {
        return false;
    }

    unsigned long repeat_num = (8000.0 / time_ms) * pre_repeat_num;

    write_string_to_file(execute_path + "/template.h", build_header_file(builder));
    write_string_to_file(execute_path + "/template.cu", build_main_file(builder, repeat_num));

    return execute_binary(execute_path, time_ms, gflops);
}

bool part_execute_code_builder(code_builder_t* builder, vector<int> sub_matrix_id_vec, float& time_ms, float& gflops, string execute_path, string data_path, bool store_data)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL && builder->op_manager->matrix != NULL);
    assert(sub_matrix_id_vec.size() > 0);
    assert(builder->op_manager->matrix->block_coor_table.item_arr.size() > 0);

    // 计算总非零元的数量
    unsigned long total_nnz = 0;

    // 每一个子图都是充分分块的，并且每一个子块的初始化了模板
    for (int i = 0; i < sub_matrix_id_vec.size(); i++)
    {
        int sub_matrix_id = sub_matrix_id_vec[i];
        
        assert(sub_matrix_id < builder->template_vec.size() && builder->template_vec.size() == builder->template_type_vec.size());
        assert(builder->template_vec[sub_matrix_id] != NULL);
        assert(sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());

        assert(builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id] != NULL);
        assert(builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);

        // 获取压缩子图的指针
        compressed_block_t* compressed_block_ptr = builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;
        assert(compressed_block_ptr->read_index.size() == 7);

        total_nnz = total_nnz + compressed_block_ptr->padding_arr_size;
    }

    // 将数据写到硬盘中
    if (store_data == true)
    {
        store_code_builder_data(builder, sub_matrix_id_vec, data_path);    
    }

    // 有一个目标要计算的nnz数量
    unsigned long target_nnz = 185200000000;

    unsigned long repeat_num = target_nnz / total_nnz;

    // 至少算两次
    if (repeat_num <= 1)
    {
        repeat_num = 2;
    }

    // 写代码文件
    write_string_to_file(execute_path + "/template.h", build_header_file(builder, sub_matrix_id_vec));
    write_string_to_file(execute_path + "/template.cu", build_main_file(builder, sub_matrix_id_vec, repeat_num));
    
    // 编译执行，执行结果引用传值
    return execute_binary(execute_path, time_ms, gflops);
}