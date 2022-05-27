#include <iostream>
#include "struct.hpp"
#include "op_manager.hpp"
#include <assert.h>
#include "code_builder.hpp"
#include <sys/time.h>
#include "arr_optimization.hpp"
#include "direct_atom_op.hpp"
#include "direct_atom_op_warp_compress.hpp"
#include "direct_atom_op_warp_block_compress.hpp"
#include "shared_memory_op.hpp"
#include "shared_memory_op_warp_compress.hpp"
#include "shared_memory_long_row_op.hpp"
#include "shared_memory_total_warp_reduce_op.hpp"
#include "direct_atom_total_warp_reduce_op.hpp"
#include "unaligned_warp_reduce_same_TLB_size_op.hpp"
#include "unaligned_warp_reduce_same_TLB_size_op_with_warp_reduce.hpp"
#include "dataset_builder.hpp"
#include "memory_garbage_manager.hpp"
#include "exe_graph.hpp"
#include "graph_enumerate.hpp"
#include "param_enumerater.hpp"
#include "default_auto_tuner.hpp"
#include "executor.hpp"
#include "search_strategy.hpp"
#include "code_source_data.hpp"
#include <memory>
#include <limits.h>
#include <float.h>
#include "machine_learning_data_set_collector.hpp"

using namespace std;

// int main()
// {
//     vector<unsigned long> vec;
//     vec.push_back(1);
//     vec.push_back(2);
//     vec.push_back(3);

//     shared_ptr<universal_array> arr_ptr = create_uni_arr_from_vec(vec);

//     cout << arr_ptr->read_integer_from_arr(1) << endl;
//     cout << convert_data_type_to_string(arr_ptr->get_data_type()) << endl;
//     // 压缩存储
//     arr_ptr->compress_data_type();
//     cout << arr_ptr->read_integer_from_arr(1) << endl;
//     cout << convert_data_type_to_string(arr_ptr->get_data_type()) << endl;
    
    
// }




















struct timeval pre_start, pre_end;

// 数据类型相同，划等号
void equal(void* param_ptr, void* depend_param_ptr, data_type param_data_type, data_type depend_param_data_type)
{
    assert(param_ptr != NULL && depend_param_ptr != NULL);
    assert(param_data_type == depend_param_data_type);
    assert(param_data_type == DOUBLE || param_data_type == LONG);

    if (param_data_type == DOUBLE)
    {
        *((double *)param_ptr) = *((double *)depend_param_ptr);
    }

    if (param_data_type == LONG)
    {
        *((long *)param_ptr) = *((long *)depend_param_ptr);
    }
}

// 要两个相等
bool same_param(void* param_ptr1, void* param_ptr2, data_type param_ptr1_data_type, data_type param_ptr2_data_type)
{
    assert(param_ptr1 != NULL && param_ptr2 != NULL && param_ptr1_data_type == param_ptr2_data_type);
    assert(param_ptr1_data_type == DOUBLE || param_ptr2_data_type == LONG);

    if (param_ptr1_data_type == DOUBLE)
    {
        return *((double *)param_ptr1) == *((double *)param_ptr2);
    }

    if (param_ptr2_data_type == LONG)
    {
        return *((long *)param_ptr1) == *((long *)param_ptr2);
    }
    
    assert(false);
    return false;
}

// int main()
// {
//     param_enumerater_t param_enumerater;

//     // 三个整形参数，三个double参数
//     vector<long> integer_param_vec;
//     vector<double> float_param_vec;

//     // 一个参数依赖于其他参数
//     long integer_param1;

//     for (int i = 0; i < 3; i++)
//     {
//         integer_param_vec.push_back(-2);
//         float_param_vec.push_back(-3);
//     }

//     for (int i = 0; i < integer_param_vec.size(); i++)
//     {
//         register_integer_independ_param_to_enumerater(&param_enumerater, &(integer_param_vec[i]), 25, 50, 10);
//     }

//     for (int i = 0; i < float_param_vec.size(); i++)
//     {
//         register_float_independ_param_to_enumerater(&param_enumerater, &(float_param_vec[i]), 1.5, 3.7, 0.5);
//     }

//     // 增加一个依赖，保证两个参数数值相等
//     register_single_dependency_param_to_enumerater(&param_enumerater, &integer_param1, LONG, &(integer_param_vec[0]), LONG, equal);

//     // 两个相等
//     register_binary_param_dependency_filter_to_enumerater(&param_enumerater, &(integer_param_vec[0]), LONG, &(integer_param_vec[1]), LONG, same_param);

//     while (set_param_combination_to_next(&param_enumerater) == false)
//     {
//         // 打印两个数组
//         cout << "[";

//         for (int i = 0; i < integer_param_vec.size(); i++)
//         {
//             cout << integer_param_vec[i] << ",";
//         }
//         cout << "]";

//         cout << "[";

//         for (int i = 0; i < float_param_vec.size(); i++)
//         {
//             cout << float_param_vec[i] << ",";
//         }

//         cout << "]" << integer_param1 << endl;
//     }
// }

// 

// direct_atom_template_warp_compress的自动调参
// int main ()
// {
//     dataset_builder_t data_builder = get_dataset_builder(600000, 59999, 4);

//     vector<double> none_vec;
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_vector(get_row_index_of_dataset_builder(data_builder), get_col_index_of_dataset_builder(data_builder), get_float_val_of_dataset_builder(data_builder), none_vec, FLOAT, 59999, 600000);

//     operator_manager_t* op_manager = init_op_manager(matrix);

//     // 直接压缩
//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     compress_block_end_block_multiple_padding(op_manager, 0, 512, 1);

//     unsigned int BLB_row_num = matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[0]->max_row_index - matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[0]->min_row_index + 1;

//     assert(BLB_row_num % 512 == 0);

//     vector<unsigned int> BLB_row_num_vec;

//     for (int i = 0; i < BLB_row_num / 512; i++)
//     {
//         BLB_row_num_vec.push_back(512);
//     }

//     sep_tblock_level_row_csr(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, BLB_row_num_vec);

//     // 放弃WLB分块
//     vector<unsigned long> BLB_id;
//     vector<vector<unsigned int>> WLB_row_size;
//     sep_warp_level_row_csr(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, BLB_id, WLB_row_size);

//     // thread级别行分块
//     vector<unsigned long> WLB_id;
//     vector<unsigned long> TLB_col_size;

//     sep_thread_level_col_ell_with_padding(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, WLB_id, TLB_col_size);

//     code_builder_t* builder = init_code_builder(op_manager);

//     direct_atom_template_warp_compress_t* new_template = init_direct_atom_template_warp_compress(builder, 0);

//     add_template_to_builder(builder, new_template, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 0);

//     // 执行压缩
//     try_all_compress(new_template);

//     // 对应位置选参数
//     float best_time;
//     float best_glops;

//     template_node_t node = find_best_param_of_direct_atom_template_warp_compress(builder, 0, best_time, best_glops);

//     cout << "best_time:" << best_time << endl;
//     cout << "best_glops:" << best_glops << endl;
//     cout << "tblock_num:" << ((direct_atom_template_warp_compress_node_param_t *)(node.template_param))->tblock_num << endl;
//     cout << "thread_num_in_block:" << ((direct_atom_template_warp_compress_node_param_t *)(node.template_param))->thread_num_in_block << endl;
// }

// int main()
// {
    // // 先创造一个矩阵
    // dataset_builder_t data_builder = get_dataset_builder(60000, 59999, 63);

    // vector<double> none_vec;
    // sparse_struct_t *matrix = init_sparse_struct_by_coo_vector(get_row_index_of_dataset_builder(data_builder), get_col_index_of_dataset_builder(data_builder), get_float_val_of_dataset_builder(data_builder), none_vec, FLOAT, 59999, 60000);

    // operator_manager_t* op_manager = init_op_manager(matrix);

    // compress_dense_view(op_manager);

    // print_dense_block_table(&(op_manager->matrix->block_coor_table));

    // // 初始化一个子图
    // exe_compressed_sub_graph_t* sub_graph = new exe_compressed_sub_graph_t();
    // param_strategy_of_sub_graph_t* param_strategy_sub_graph = new param_strategy_of_sub_graph_t();

    // compressed_tblock_level_row_div_evenly_param_strategy strategy1;
    // strategy1.block_row_num = 512;

    // add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(matrix, 0, sub_graph, param_strategy_sub_graph, COMPRESSED_TBLOCK_LEVEL_ROW_DIV, COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY, &strategy1);

    // execute_sub_matrix_exe_graph_with_param_strategy(matrix, 0, *sub_graph, *param_strategy_sub_graph);

    // del_strategy_of_param_strategy_node_in_sub_matrix(*param_strategy_sub_graph);
    // del_exe_node_param_of_compress_sub_matrix(*sub_graph);
    
    // delete sub_graph;
    // delete param_strategy_sub_graph;
// }


// 测试稠密矩阵的策略图和执行图
// int main()
// {
//     exe_dense_sub_graph_t dense_sub_graph;
//     param_strategy_of_sub_graph_t dense_sub_graph_param_strategy;

//     // 创造一个输入节点
//     dense_begin_memory_cache_input_file_direct_param_strategy_t begin_memory_cache_input_file_direct_param_strategy = get_begin_memory_cache_input_file_direct_param_strategy_from_coo_file("/home/duzhen/spmv_builder/data_source/rail4284.mtx.coo", FLOAT);

//     // 将输入节点加到图中
//     add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&dense_sub_graph, &dense_sub_graph_param_strategy, BEGIN_MEMORY_CACHE_INPUT_FILE, DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY, &begin_memory_cache_input_file_direct_param_strategy);

//     // // 创造一个排序节点
//     dense_row_coarse_sort_fixed_param_strategy_t row_coarse_sort_fixed_param_strategy;
//     row_coarse_sort_fixed_param_strategy.row_nnz_low_bound_step_size = 1;

//     // 将输入节点加到图中
//     add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&dense_sub_graph, &dense_sub_graph_param_strategy, DENSE_ROW_COARSE_SORT, DENSE_ROW_COARSE_SORT_FIXED_PARAM_STRATEGY, &row_coarse_sort_fixed_param_strategy);

//     // 创造一个分块节点

//     // 创造一个压缩节点
//     compress_none_param_strategy_t compress_param_strategy;
    
//     // 将输入节点加入到图中
//     add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&dense_sub_graph, &dense_sub_graph_param_strategy, COMPRESS, COMPRESS_NONE_PARAM_STRATEGY, &compress_param_strategy);

//     // 执行这两个节点
//     sparse_struct_t* matrix = execute_dense_matrix_exe_graph_with_param_strategy(dense_sub_graph, dense_sub_graph_param_strategy);

//     sparse_struct_t* copy_matrix = val_copy_from_old_matrix_struct(matrix);

//     print_dense_block_table(&(copy_matrix->block_coor_table));
// }

// ROW GROUP的处理
// int main()
// {
//     dense_begin_memory_cache_input_file_direct_param_strategy_t input_node_param_strategy = get_begin_memory_cache_input_file_direct_param_strategy_from_coo_file("/home/duzhen/spmv_builder/data_source/2D_27628_bjtcai.mtx.coo", FLOAT);

//     // 稠密视图的处理
//     exe_dense_sub_graph_t sub_graph;
//     param_strategy_of_sub_graph_t sub_graph_param_strategy;

//     add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&sub_graph, &sub_graph_param_strategy, BEGIN_MEMORY_CACHE_INPUT_FILE, DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY, &input_node_param_strategy);

//     compress_none_param_strategy_t compress_param_strategy;
//     add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&sub_graph, &sub_graph_param_strategy, COMPRESS, COMPRESS_NONE_PARAM_STRATEGY, &compress_param_strategy);

//     sparse_struct_t *matrix = execute_dense_matrix_exe_graph_with_param_strategy(&sub_graph, &sub_graph_param_strategy);

//     // 定一个优化路径骨架
//     exe_compressed_sub_graph_t sub_graph_skeleton;
//     // 定一个参数设定的骨架
//     param_strategy_of_sub_graph_t param_strategy_skeleton;

//     compressed_thread_level_col_div_fixed_param_strategy_t TLB_col_div_fixed_param_strategy;
//     TLB_col_div_fixed_param_strategy.col_block_nnz_num = 7;
//     add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_THREAD_LEVEL_COL_DIV, COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY, &TLB_col_div_fixed_param_strategy);

//     execute_sub_matrix_exe_graph_with_param_strategy(matrix, 0, &sub_graph_skeleton, &param_strategy_skeleton);
    
//     // 生成一个操作管理器
//     operator_manager_t* op_manager = init_op_manager(matrix);
//     // 生成一个代码生成器
//     code_builder_t* builder = init_code_builder(op_manager);

//     // 新的模板
//     direct_atom_template_warp_block_compress_t* new_template = init_direct_atom_template_warp_block_compress(builder, 0);

//     // 模板的参数
//     index_of_compress_block_t* TLB_index = builder->op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[4];

//     // 总线程数量
//     long total_thread_num = TLB_index->block_num;

//     // 总线程块数量
//     long tblock_num = total_thread_num / 128;

//     if (total_thread_num % 128 != 0)
//     {
//         tblock_num = tblock_num + 1;
//     }

//     new_template->thread_num_in_block = 128;
//     new_template->tblock_num = tblock_num;

//     try_all_compress(new_template);

//     add_template_to_builder(builder, new_template, DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS, 0);

//     float time;
//     float gflops;

//     bool is_success = execute_code_builder(builder, time, gflops, string(get_config()["ROOT_PATH_STR"].as_string()) + "/cuda_code", string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source", true);

//     cout << "gflops:" << gflops << endl;
// }

// int main()
// {
    
// }

// 执行SELL的BLB分块和CSR的处理
// int main()
// {
//     // 读文件
//     dense_begin_memory_cache_input_file_direct_param_strategy_t input_node_param_strategy = get_begin_memory_cache_input_file_direct_param_strategy_from_coo_file("/home/duzhen/spmv_builder/data_source/2D_27628_bjtcai.mtx.coo", FLOAT);
    
//     // 稠密视图的处理
//     exe_dense_sub_graph_t sub_graph;
//     param_strategy_of_sub_graph_t sub_graph_param_strategy;

//     add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&sub_graph, &sub_graph_param_strategy, BEGIN_MEMORY_CACHE_INPUT_FILE, DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY, &input_node_param_strategy);
    
//     // 稠密视图直接压缩
//     compress_none_param_strategy_t compress_param_strategy;
//     add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&sub_graph, &sub_graph_param_strategy, COMPRESS, COMPRESS_NONE_PARAM_STRATEGY, &compress_param_strategy);

//     // 执行稠密压缩
//     sparse_struct_t *matrix = execute_dense_matrix_exe_graph_with_param_strategy(&sub_graph, &sub_graph_param_strategy);

//     // 定一个优化路径骨架
//     exe_compressed_sub_graph_t sub_graph_skeleton;
//     // 定一个参数设定的骨架
//     param_strategy_of_sub_graph_t param_strategy_skeleton;

//     // 首先执行BLB64行分块
//     compressed_row_padding_direct_param_strategy_t row_padding_param_strategy;
//     compressed_tblock_level_row_div_evenly_param_strategy_t BLB_row_div_evenly_param_strategy;
//     compressed_thread_level_col_div_fixed_param_strategy_t TLB_col_div_fixed_param_strategy;

//     row_padding_param_strategy.multiply = 64;
//     row_padding_param_strategy.padding_row_length = 1;
//     BLB_row_div_evenly_param_strategy.block_row_num = 64;
//     TLB_col_div_fixed_param_strategy.col_block_nnz_num = 1;

//     add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_ROW_PADDING, COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY, &row_padding_param_strategy);
//     add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_TBLOCK_LEVEL_ROW_DIV, COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY, &BLB_row_div_evenly_param_strategy);
//     add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_THREAD_LEVEL_COL_DIV, COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY, &TLB_col_div_fixed_param_strategy);
    
//     execute_sub_matrix_exe_graph_with_param_strategy(matrix, 0, &sub_graph_skeleton, &param_strategy_skeleton);

//     // 生成一个操作管理器
//     operator_manager_t* op_manager = init_op_manager(matrix);
//     // 生成一个代码生成器
//     code_builder_t* builder = init_code_builder(op_manager);

//     shared_memory_template_warp_compress_t* new_template = init_shared_memory_template_warp_compress(builder, 0);

//     // BLB的索引
//     index_of_compress_block_t* BLB_index = builder->op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[2];

//     // BLB的数量
//     long tblock_num;
//     long thread_num_in_tblock;

//     // tblock数量和BLB数量相同
//     tblock_num = BLB_index->block_num;
    
//     if (tblock_num > MAX_TBLOCK_NUM - 1)
//     {
//         tblock_num = MAX_TBLOCK_NUM - 1; 
//     }

//     new_template->tblock_num = tblock_num;
//     new_template->thread_num_in_block = 256;
//     new_template->thread_num_of_row_reduce = 1;

//     try_all_compress(new_template);

//     add_template_to_builder(builder, new_template, SHARED_MEMORY_TEMPLATE_WARP_COMPRESS, 0);

//     float time;
//     float gflops;

//     bool is_success = execute_code_builder(builder, time, gflops, string(get_config()["ROOT_PATH_STR"].as_string()) + "/cuda_code", string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source", true);

//     cout << "gflops:" << gflops << endl;
// }

// 先执行SELL
// int main()
// {
//     //  
//     dense_begin_memory_cache_input_file_direct_param_strategy_t input_node_param_strategy = get_begin_memory_cache_input_file_direct_param_strategy_from_coo_file("/home/duzhen/spmv_builder/data_source/2D_27628_bjtcai.mtx.coo", FLOAT);
//     // 稠密视图的处理
//     exe_dense_sub_graph_t sub_graph;
//     param_strategy_of_sub_graph_t sub_graph_param_strategy;

//     // 将输入节点放到图中
//     add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&sub_graph, &sub_graph_param_strategy, BEGIN_MEMORY_CACHE_INPUT_FILE, DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY, &input_node_param_strategy);

//     // 放一个排序
//     dense_row_coarse_sort_fixed_param_strategy_t sort_param_strategy;
//     sort_param_strategy.row_nnz_low_bound_step_size = 1;

//     add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&sub_graph, &sub_graph_param_strategy, DENSE_ROW_COARSE_SORT, DENSE_ROW_COARSE_SORT_FIXED_PARAM_STRATEGY, &sort_param_strategy);

//     // 创造一个新的节点，执行一次压缩
//     compress_none_param_strategy_t compress_param_strategy;

//     add_a_exe_node_and_param_strategy_to_exe_dense_sub_graph(&sub_graph, &sub_graph_param_strategy, COMPRESS, COMPRESS_NONE_PARAM_STRATEGY, &compress_param_strategy);

//     // 执行对应的稠密子块调优
//     sparse_struct_t *matrix = execute_dense_matrix_exe_graph_with_param_strategy(&sub_graph, &sub_graph_param_strategy);

//     // 定一个优化路径骨架
//     exe_compressed_sub_graph_t sub_graph_skeleton;
//     // 定一个参数设定的骨架
//     param_strategy_of_sub_graph_t param_strategy_skeleton;

//     // 首先执行BLB64行分块
//     compressed_row_padding_direct_param_strategy_t row_padding_param_strategy;
//     compressed_tblock_level_row_div_evenly_param_strategy_t BLB_row_div_evenly_param_strategy;
//     compressed_thread_level_row_div_none_param_strategy TLB_row_div_none_param_strategy;

//     // 可以直接在策略里面初始化
//     row_padding_param_strategy.multiply = 64;
//     row_padding_param_strategy.padding_row_length = 1;

//     BLB_row_div_evenly_param_strategy.block_row_num = 64;

//     add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_ROW_PADDING, COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY, &row_padding_param_strategy);
//     add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_TBLOCK_LEVEL_ROW_DIV, COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY, &BLB_row_div_evenly_param_strategy);
//     add_a_exe_node_and_param_strategy_to_exe_compressed_sub_graph(&sub_graph_skeleton, &param_strategy_skeleton, COMPRESSED_THREAD_LEVEL_ROW_DIV, COMPRESSED_THREAD_LEVEL_ROW_DIV_NONE_PARAM_STRATEGY, &TLB_row_div_none_param_strategy);

//     // 执行压缩视图
//     execute_sub_matrix_exe_graph_with_param_strategy(matrix, 0, &sub_graph_skeleton, &param_strategy_skeleton);

//     // 加入一个模板
//     // 生成一个操作管理器
//     operator_manager_t* op_manager = init_op_manager(matrix);
//     // 生成一个代码生成器
//     code_builder_t* builder = init_code_builder(op_manager);
//     // 生成一个模板
//     direct_atom_template_warp_compress_t* new_template = init_direct_atom_template_warp_compress(builder, 0);

//     // BLB的索引
//     index_of_compress_block_t* BLB_index = builder->op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[2];

//     // BLB的数量
//     long tblock_num;
//     long thread_num_in_tblock;

//     // tblock数量和BLB数量相同
//     tblock_num = BLB_index->block_num;
    
//     if (tblock_num > MAX_TBLOCK_NUM - 1)
//     {
//         tblock_num = MAX_TBLOCK_NUM - 1; 
//     }

//     new_template->tblock_num = tblock_num;
//     new_template->thread_num_in_block = 64;

//     try_all_compress(new_template);

//     add_template_to_builder(builder, new_template, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 0);

//     float time;
//     float gflops;

//     bool is_success = execute_code_builder(builder, time, gflops, string(get_config()["ROOT_PATH_STR"].as_string()) + "/cuda_code", string(get_config()["ROOT_PATH_STR"].as_string()) + "/data_source", true);

//     cout << "gflops:" << gflops << endl;
// }

// 测试机器学习数据集
// int main()
// {
//     shared_ptr<machine_learning_data_set_collector> data_set_collector(new machine_learning_data_set_collector());
    
//     vector<exe_node_type> dense_graph_node_type_vec;
//     vector<exe_node_param_set_strategy> dense_param_strategy_type_vec;
//     vector<exe_node_type> compressed_graph_node_type_vec;
//     vector<exe_node_param_set_strategy> compressed_param_strategy_type_vec;
//     template_type type_of_template;
//     vector<float> all_param;

//     all_param.push_back(1);

//     dense_graph_node_type_vec.push_back(BEGIN_MEMORY_CACHE_INPUT_FILE);
//     dense_param_strategy_type_vec.push_back(DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY);

//     compressed_graph_node_type_vec.push_back(COMPRESSED_ROW_PADDING);
//     compressed_param_strategy_type_vec.push_back(COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY);

//     type_of_template = DIRECT_ATOM_TEMPLATE;

//     data_set_collector->add_item_to_dataset(dense_graph_node_type_vec, dense_param_strategy_type_vec, compressed_graph_node_type_vec, compressed_param_strategy_type_vec, all_param, type_of_template);

//     cout << data_set_collector->convert_the_whole_dataset_to_string() << endl;
// }


// 先创造一个矩阵
int main(int argc, char **argv)
{
    // 先创造一个矩阵
    // dataset_builder_t data_builder = get_dataset_builder(60000, 59999, 29);

    // exe_graph_t graph;

    // // 处理第一个节点
    // exe_begin_memory_cache_input_file_param_t begin_memory_cache_input_file_param;
    // begin_memory_cache_input_file_param.row_index_cache = get_row_index_of_dataset_builder(data_builder);
    // begin_memory_cache_input_file_param.col_index_cache = get_col_index_of_dataset_builder(data_builder);
    // begin_memory_cache_input_file_param.row_index_max = 60000;
    // begin_memory_cache_input_file_param.col_index_max = 59999;
    // begin_memory_cache_input_file_param.float_val_cache = get_float_val_of_dataset_builder(data_builder);
    // begin_memory_cache_input_file_param.val_data_type = FLOAT;

    // // 将节点放到图中
    // add_exe_begin_memory_cache_input_file_node_to_exe_graph(&graph, EXE_DENSE_SUB_GRAPH, begin_memory_cache_input_file_param, 0, GRAPH_END);
    
    // // 加一个行分块
    // exe_dense_row_div_param_t row_div_param;

    // row_div_param.dense_sub_block_id = 0;
    // row_div_param.row_div_position.push_back(0);
    // row_div_param.row_div_position.push_back(20000);
    // row_div_param.row_div_position.push_back(60001);

    // add_exe_dense_row_div_node_to_exe_graph(&graph, EXE_DENSE_SUB_GRAPH, row_div_param, 0, GRAPH_END);

    // 将数据取出来
    // exe_begin_memory_cache_input_file_param_t input_node_param = get_exe_begin_memory_cache_input_file_param_from_coo_file("/home/duzhen/spmv_builder/data_source/bcsstk36.mtx.coo", FLOAT);
    exe_begin_memory_cache_input_file_param_t input_node_param = get_exe_begin_memory_cache_input_file_param_from_coo_file(read_str_from_command_line(argc, argv, 1), FLOAT);
    
    shared_ptr<machine_learning_data_set_collector> data_set_collector(new machine_learning_data_set_collector(get_config()["ROOT_PATH_STR"].as_string() + string("/data_source/machine_learning_data_set")));

    // vector<unsigned long> row_nnz_vec = get_nnz_of_each_row_in_spec_range(&(input_node_param.row_index_cache[0]), UNSIGNED_LONG, 0, input_node_param.row_index_max, 0, input_node_param.row_index_cache.size() - 1);

    // print_arr_to_file_with_data_type(&(row_nnz_vec[0]), UNSIGNED_LONG, row_nnz_vec.size(), "/home/duzhen/spmv_builder/data_source/test_result_3");

    // exit(-1);
    // cout << check_begin_memory_cache_input_file(input_node_param) << endl;

    // cout << has_empty_line_in_begin_memory_cache_input_file(input_node_param) << endl;
    if (check_begin_memory_cache_input_file(input_node_param) == false)
    {
        cout << "input cannot pass the check" << endl;
        assert(false);
    }

    if (has_empty_line_in_begin_memory_cache_input_file(input_node_param) == true)
    {
        cout << "has empty line" << endl;
        assert(false);
    }


    if ((check_begin_memory_cache_input_file(input_node_param) == false) || (has_empty_line_in_begin_memory_cache_input_file(input_node_param) == true))
    {
        cout << "input cannot pass the check" << endl;
        assert(false);
    }

    // add_exe_begin_memory_cache_input_file_node_to_exe_graph(&graph, EXE_DENSE_SUB_GRAPH, input_node_param, 0, GRAPH_END);

    // 加入一个压缩节点
    // exe_compress_param_t compress_param;

    // add_exe_compress_node_to_exe_graph(&graph, EXE_DENSE_SUB_GRAPH, compress_param, 0, GRAPH_END);

    // 执行对应的稠密子图
    // execute_graph_dense_part(&graph);

    // 申请一个策略
    search_strategy_t search_strategy = init_search_strategy(10, 32400);
    // 申请一个机器学习数据集收集器
    // shared_ptr<machine_learning_data_set_collector> data_set_collector(new machine_learning_data_set_collector("/home/duzhen/spmv_builder/data_source/machine_learning_data_set"));

    float gflops;
    float time;

    // 执行对应的优化路径
    // compressed_sub_block_exe_graph_and_template_t sub_matrix_opt_path = find_best_path_of_compressed_sub_matrix(graph.dense_sub_graph, 0, time, gflops, &search_strategy);
    dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_t best_graph = find_best_graph_of_white_list_strategy(input_node_param, time, gflops, &search_strategy, data_set_collector);

    // 将执行的结果打印出来
    cout << "gflops:" << gflops << endl;
    cout << "time:" << time << endl;

    if (gflops != 0)
    {
        cout << convert_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template_to_string_safety(best_graph) << endl;;
    }
    else
    {
        cout << "error" << endl;
    }

    cout << "finish search" << endl;
    // 最后完整执行
    execute_dense_view_matrix_and_compressed_sub_block_exe_graph_and_template(&best_graph, gflops, time, 2);

    cout << "final_gflops:" << gflops << endl;
    cout << "final_time:" << time << endl;
}

// 测试从mem中读数据的
// int main()
// {
//     vector<float> float_val_vec;
//     vector<double> double_val_vec;
//     unsigned long max_col_index;
//     unsigned long max_row_index;
//     vector<unsigned long> col_index_vec;
//     vector<unsigned long> row_index_vec;

//     get_matrix_index_and_val_from_file("/home/duzhen/spmv_builder/data_source/bone010.mtx.coo", row_index_vec, col_index_vec, float_val_vec, double_val_vec, FLOAT, max_row_index, max_col_index);

//     exe_graph_t exe_graph;

//     exe_begin_memory_cache_input_file_param_t param;
//     param.col_index_cache = col_index_vec;
//     param.row_index_cache = row_index_vec;
//     param.float_val_cache = float_val_vec;
//     param.double_val_cache = double_val_vec;
//     param.col_index_max = max_col_index;
//     param.row_index_max = max_row_index;
//     param.val_data_type = FLOAT;

//     add_exe_begin_memory_cache_input_file_node_to_exe_graph(&exe_graph, EXE_DENSE_SUB_GRAPH, param, 0, GRAPH_END);

//     execute_graph_dense_part(&exe_graph);

//     compress_dense_view((&exe_graph)->op_manager);

//     print_dense_block_table(&(exe_graph.op_manager->matrix->block_coor_table));
    
    
// }

// 使用unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce
// int main()
// {
//     dataset_builder_t data_builder = get_dataset_builder(60000, 59999, 63);

//     vector<double> none_vec;
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_vector(get_row_index_of_dataset_builder(data_builder), get_col_index_of_dataset_builder(data_builder), get_float_val_of_dataset_builder(data_builder), none_vec, FLOAT, 59999, 60000);

//     operator_manager_t* op_manager = init_op_manager(matrix);

//     // 按照30000为一个条带执行行切分
//     vector<unsigned long> block_first_row_csr_index_vec;
//     block_first_row_csr_index_vec.push_back(0);
//     block_first_row_csr_index_vec.push_back(30000);
//     block_first_row_csr_index_vec.push_back(60001);

//     var_len_row_div(op_manager, NULL, block_first_row_csr_index_vec);

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // compress_block_end_block_multiple_padding(op_manager, 0, 64, 63);

//     compressed_block_t* compressed_block_ptr = op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr;

//     // 放弃线程块和warp级别的分块
//     unsigned long tblock_row_num = compressed_block_ptr->read_index[0]->max_row_index - compressed_block_ptr->read_index[0]->min_row_index + 1;

//     vector<unsigned int> block_row_num;

//     block_row_num.push_back(tblock_row_num);
    
//     sep_tblock_level_row_csr(compressed_block_ptr, block_row_num);

//     // 放弃warp的排序
//     vector<vector<unsigned int>> arr_of_row_block_size_arr;
//     vector<unsigned long> sep_block_id_arr;

//     sep_warp_level_row_csr(compressed_block_ptr, sep_block_id_arr, arr_of_row_block_size_arr);

//     sep_thread_level_acc_to_nnz(compressed_block_ptr, 4);

//     // 对应位置选参数
//     float best_time;
//     float best_glops;

//     set<template_type> template_set;
//     template_set.insert(UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE);
//     template_set.insert(UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE);

//     template_node_t template_node = find_best_template_node_of_specific_sub_matrix_from_template_set(matrix, 1, template_set, best_time, best_glops);

//     print_template_node(&template_node);
// }

// 使用shared_memory_total_warp_reduce_template
// int main()
// {
//     dataset_builder_t data_builder = get_dataset_builder(60000, 59999, 63);

//     vector<double> none_vec;
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_vector(get_row_index_of_dataset_builder(data_builder), get_col_index_of_dataset_builder(data_builder), get_float_val_of_dataset_builder(data_builder), none_vec, FLOAT, 59999, 60000);

//     operator_manager_t* op_manager = init_op_manager(matrix);

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     compress_block_end_block_multiple_padding(op_manager, 0, 64, 63);

//     // 放弃BLB分块
//     unsigned int BLB_row_num = matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[0]->max_row_index - matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[0]->min_row_index + 1;

//     assert(BLB_row_num % 64 == 0);

//     vector<unsigned int> BLB_row_num_vec;
//     for (unsigned long i = 0; i < BLB_row_num / 64; i++)
//     {
//         BLB_row_num_vec.push_back(64);
//     }

//     sep_tblock_level_row_csr(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, BLB_row_num_vec);

//     // WLB按照32纵分块，首先先执行一行一个WLB的行分块
//     vector<unsigned long> sep_block_id;
//     vector<vector<unsigned int>> spec_WLB_row_num_of_a_BLB;

//     compressed_block_t* compressed_block_view = matrix->block_coor_table.item_arr[0]->compressed_block_ptr;

//     // 遍历所有的BLB
//     assert(compressed_block_view->read_index[2]->block_num > 0);
//     index_of_compress_block_t* BLB_index = compressed_block_view->read_index[2];
//     assert(BLB_index->row_number_of_block_arr != NULL);
//     for (unsigned long BLB_id = 0; BLB_id < compressed_block_view->read_index[2]->block_num; BLB_id++)
//     {
//         // 获取当前BLB的行数量
//         unsigned long cur_BLB_row_num = read_from_array_with_data_type(BLB_index->row_number_of_block_arr, BLB_index->data_type_of_row_number_of_block_arr, BLB_id);
//         // 如果行数量大于1，那就需要进一步分块
//         if (cur_BLB_row_num > 1)
//         {
//             // 进一步分块
//             sep_block_id.push_back(BLB_id);
//             // 初始化一个进一步分块的WLB行号
//             vector<unsigned int> WLB_row_num;

//             for (unsigned long i = 0; i < cur_BLB_row_num; i++)
//             {
//                 WLB_row_num.push_back(1);
//             }

//             spec_WLB_row_num_of_a_BLB.push_back(WLB_row_num);
//         }
//     }

//     assert(sep_block_id.size() > 0 && spec_WLB_row_num_of_a_BLB.size() > 0);
//     // 这里执行一个WLB级别的行分块
//     sep_warp_level_row_csr(compressed_block_view, sep_block_id, spec_WLB_row_num_of_a_BLB);
    
//     assert(compressed_block_view->read_index.size() == 4);

//     index_of_compress_block_t* WLB_index = compressed_block_view->read_index[3];

//     // 申请一个数组，包含所有WLB的分块，然后执行列分块
//     vector<unsigned long> sep_WLB_id;
//     vector<vector<unsigned int>> sep_col_WLB_size;

//     for (unsigned long i = 0; i < WLB_index->block_num; i++)
//     {
//         vector<unsigned int> col_size;
//         sep_WLB_id.push_back(i);

//         col_size.push_back(32);
//         col_size.push_back(32);
//         col_size.push_back(32);

//         sep_col_WLB_size.push_back(col_size);
//     }

//     sep_warp_level_col_csr(compressed_block_view, sep_WLB_id, sep_col_WLB_size);
    
//     // 放弃TLB的分块每个TLB一个非零元
//     vector<unsigned long> sep_TLB_id;
//     vector<unsigned long> TLB_nnz;

//     for (unsigned long i = 0; i < WLB_index->block_num; i++)
//     {
//         sep_TLB_id.push_back(i);
//         TLB_nnz.push_back(1);
//     }

//     // 列分块
//     sep_thread_level_col_ell_with_padding(compressed_block_view, sep_TLB_id, TLB_nnz);

//     // 执行一个拷贝函数
//     sparse_struct_t* copy_matrix = val_copy_from_old_matrix_struct(matrix);

//     // 分别析构两个矩阵
//     memory_garbage_manager_t mem1;
//     memory_garbage_manager_t mem2;

//     delete_sparse_struct_t(&mem1, matrix);
//     delete_sparse_struct_t(&mem2, copy_matrix);

    // write_total_matrix_to_file(matrix, "/home/duzhen/spmv_builder/data_source");

    // write_total_matrix_to_file(copy_matrix, "/home/duzhen/spmv_builder/data_source");

    // code_builder_t* builder = init_code_builder(op_manager);

    // shared_memory_total_warp_reduce_template_t* new_template = init_shared_memory_total_warp_reduce_template(builder, 0);

    // // 写模板
    // add_template_to_builder(builder, new_template, SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE, 0);

    // // 执行压缩
    // try_all_compress(new_template);

    // // 对应位置选参数
    // float best_time;
    // float best_glops;

    // template_node_t node = find_best_param_of_shared_memory_total_warp_reduce_template(builder, 0, best_time, best_glops);

    // cout << "best_time:" << best_time << endl;
    // cout << "best_glops:" << best_glops << endl;
    // cout << "tblock_num:" << ((shared_memory_total_warp_reduce_template_node_param_t *)(node.template_param))->tblock_num << endl;
    // cout << "thread_num_in_block:" << ((shared_memory_total_warp_reduce_template_node_param_t *)(node.template_param))->thread_num_in_block << endl;
// }

// 使用shared_memory_templaye_warp_compress的自动调参
// int main()
// {
//     dataset_builder_t data_builder = get_dataset_builder(600000, 59999, 4);

//     vector<double> none_vec;
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_vector(get_row_index_of_dataset_builder(data_builder), get_col_index_of_dataset_builder(data_builder), get_float_val_of_dataset_builder(data_builder), none_vec, FLOAT, 59999, 600000);

//     operator_manager_t* op_manager = init_op_manager(matrix);

//     // 直接压缩
//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     compress_block_end_block_multiple_padding(op_manager, 0, 512, 1);

//     compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[0]->compressed_block_ptr;

//     unsigned int BLB_row_num = matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[0]->max_row_index - matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[0]->min_row_index + 1;

//     assert(BLB_row_num % 512 == 0);

//     vector<unsigned int> BLB_row_num_vec;

//     for (int i = 0; i < BLB_row_num / 512; i++)
//     {
//         BLB_row_num_vec.push_back(512);
//     }

//     sep_tblock_level_row_csr(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, BLB_row_num_vec);

//     // 放弃WLB分块
//     vector<unsigned long> BLB_id;
//     vector<vector<unsigned int>> WLB_row_size;
//     sep_warp_level_row_csr(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, BLB_id, WLB_row_size);

//     // thread级别行分块
//     vector<unsigned long> WLB_id;
//     vector<unsigned long> TLB_col_size;

//     unsigned long WLB_num = matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[3]->block_num;

//     for (unsigned long i = 0; i < WLB_num; i++)
//     {
//         WLB_id.push_back(i);
//         TLB_col_size.push_back(1);
//     }

//     sep_thread_level_col_ell_with_padding(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, WLB_id, TLB_col_size);

//     code_builder_t* builder = init_code_builder(op_manager);

//     shared_memory_template_warp_compress_t* new_template = init_shared_memory_template_warp_compress(builder, 0);

//     add_template_to_builder(builder, new_template, SHARED_MEMORY_TEMPLATE_WARP_COMPRESS, 0);

//     try_all_compress(new_template);

//     // new_template->tblock_num = 1172;
//     // new_template->thread_num_in_block = 416;
//     // new_template->thread_num_of_row_reduce = 1;

//     // store_code_builder_data(builder);

//     // write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     // write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));

//     // 对应位置选参数
//     float best_time;
//     float best_glops;

//     template_node_t node = find_best_param_of_shared_memory_template_warp_compress(builder, 0, best_time, best_glops);

//     cout << "best_time:" << best_time << endl;
//     cout << "best_glops:" << best_glops << endl;
//     cout << "tblock_num:" << ((shared_memory_template_warp_compress_node_param_t *)(node.template_param))->tblock_num << endl;
//     cout << "thread_num_in_block:" << ((shared_memory_template_warp_compress_node_param_t *)(node.template_param))->thread_num_in_block << endl;
//     cout << "thread_num_of_row_reduce:" << ((shared_memory_template_warp_compress_node_param_t *)(node.template_param))->thread_num_of_row_reduce << endl; 
// }

// 使用shared_memory_long_row_template的自动调参
// int main()
// {
//     dataset_builder_t data_builder = get_dataset_builder(6000, 59999, 512);
    
//     vector<double> none_vec;
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_vector(get_row_index_of_dataset_builder(data_builder), get_col_index_of_dataset_builder(data_builder), get_float_val_of_dataset_builder(data_builder), none_vec, FLOAT, 59999, 60000);

//     operator_manager_t* op_manager = init_op_manager(matrix);

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     unsigned int BLB_row_num = matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[0]->max_row_index - matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[0]->min_row_index + 1;
//     vector<unsigned int> BLB_row_num_vec;

//     for (unsigned long i = 0; i < BLB_row_num; i++)
//     {
//         BLB_row_num_vec.push_back(1);
//     }
    
//     sep_tblock_level_row_csr(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, BLB_row_num_vec);
    
//     // 放弃WLB分块
//     vector<unsigned long> sep_BLB_id;
//     vector<vector<unsigned int>> WLB_row_size_of_each_BLB;

//     sep_warp_level_row_csr(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, sep_BLB_id, WLB_row_size_of_each_BLB);

//     // TLB一个分块一个
//     unsigned long WLB_num = matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[3]->block_num;

//     // 放弃TLB的分块每个TLB一个非零元
//     vector<unsigned long> sep_TLB_id;
//     vector<unsigned long> TLB_nnz;

//     for (unsigned long i = 0; i < WLB_num; i++)
//     {
//         sep_TLB_id.push_back(i);
//         TLB_nnz.push_back(1);
//     }

//     // 列分块
//     sep_thread_level_col_ell_with_padding(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, sep_TLB_id, TLB_nnz);


//     code_builder_t* builder = init_code_builder(op_manager);

//     shared_memory_long_row_template_t* new_template = init_shared_memory_long_row_template(builder, 0);

    

//     // 写模板
//     add_template_to_builder(builder, new_template, SHARED_MEMORY_LONG_ROW_TEMPLATE, 0);

//     // 执行压缩
//     try_all_compress(new_template);

//     // 对应位置选参数
//     float best_time;
//     float best_glops;

    

//     template_node_t node = find_best_param_of_shared_memory_long_row_template(builder, 0, best_time, best_glops);

//     cout << "best_time:" << best_time << endl;
//     cout << "best_glops:" << best_glops << endl;
//     cout << "tblock_num:" << ((shared_memory_long_row_template_node_param_t *)(node.template_param))->tblock_num << endl;
//     cout << "thread_num_in_block:" << ((shared_memory_long_row_template_node_param_t *)(node.template_param))->thread_num_in_block << endl;
// }



// 使用direct_atom_total_warp_reduce_template的自动调参
// int main ()
// {
//     dataset_builder_t data_builder = get_dataset_builder(60000, 59999, 63);

//     vector<double> none_vec;
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_vector(get_row_index_of_dataset_builder(data_builder), get_col_index_of_dataset_builder(data_builder), get_float_val_of_dataset_builder(data_builder), none_vec, FLOAT, 59999, 60000);

//     operator_manager_t* op_manager = init_op_manager(matrix);

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 放弃BLB分块
//     unsigned int BLB_row_num = matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[0]->max_row_index - matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[0]->min_row_index + 1;
//     vector<unsigned int> BLB_row_num_vec;
//     BLB_row_num_vec.push_back(BLB_row_num);
//     sep_tblock_level_row_csr(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, BLB_row_num_vec);

//     // WLB按照32纵分块，首先先执行一行一个WLB的行分块
//     vector<unsigned long> sep_block_id;
//     vector<vector<unsigned int>> spec_WLB_row_num_of_a_BLB;

//     compressed_block_t* compressed_block_view = matrix->block_coor_table.item_arr[0]->compressed_block_ptr;

//     // 遍历所有的BLB
//     assert(compressed_block_view->read_index[2]->block_num > 0);
//     index_of_compress_block_t* BLB_index = compressed_block_view->read_index[2];
//     assert(BLB_index->row_number_of_block_arr != NULL);
//     for (unsigned long BLB_id = 0; BLB_id < compressed_block_view->read_index[2]->block_num; BLB_id++)
//     {
//         // 获取当前BLB的行数量
//         unsigned long cur_BLB_row_num = read_from_array_with_data_type(BLB_index->row_number_of_block_arr, BLB_index->data_type_of_row_number_of_block_arr, BLB_id);
//         // 如果行数量大于1，那就需要进一步分块
//         if (cur_BLB_row_num > 1)
//         {
//             // 进一步分块
//             sep_block_id.push_back(BLB_id);
//             // 初始化一个进一步分块的WLB行号
//             vector<unsigned int> WLB_row_num;

//             for (unsigned long i = 0; i < cur_BLB_row_num; i++)
//             {
//                 WLB_row_num.push_back(1);
//             }

//             spec_WLB_row_num_of_a_BLB.push_back(WLB_row_num);
//         }
//     }

//     assert(sep_block_id.size() > 0 && spec_WLB_row_num_of_a_BLB.size() > 0);
//     // 这里执行一个WLB级别的行分块
//     sep_warp_level_row_csr(compressed_block_view, sep_block_id, spec_WLB_row_num_of_a_BLB);
    
//     assert(compressed_block_view->read_index.size() == 4);

//     index_of_compress_block_t* WLB_index = compressed_block_view->read_index[3];

//     // 申请一个数组，包含所有WLB的分块，然后执行列分块
//     vector<unsigned long> sep_WLB_id;
//     vector<vector<unsigned int>> sep_col_WLB_size;

//     for (unsigned long i = 0; i < WLB_index->block_num; i++)
//     {
//         vector<unsigned int> col_size;
//         sep_WLB_id.push_back(i);

//         col_size.push_back(32);
//         col_size.push_back(32);
//         col_size.push_back(32);

//         sep_col_WLB_size.push_back(col_size);
//     }

//     sep_warp_level_col_csr(compressed_block_view, sep_WLB_id, sep_col_WLB_size);
    
//     // 放弃TLB的分块每个TLB一个非零元
//     vector<unsigned long> sep_TLB_id;
//     vector<unsigned long> TLB_nnz;

//     for (unsigned long i = 0; i < WLB_index->block_num; i++)
//     {
//         sep_TLB_id.push_back(i);
//         TLB_nnz.push_back(1);
//     }

//     // 列分块
//     sep_thread_level_col_ell_with_padding(compressed_block_view, sep_TLB_id, TLB_nnz);

//     code_builder_t* builder = init_code_builder(op_manager);

//     direct_atom_total_warp_reduce_template_t* new_template = init_direct_atom_total_warp_reduce_template(builder, 0);

//     // 写模板
//     add_template_to_builder(builder, new_template, DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE, 0);

//     // 执行压缩
//     try_all_compress(new_template);

//     // 对应位置选参数
//     float best_time;
//     float best_glops;

//     template_node_t node = find_best_param_of_direct_atom_total_warp_reduce_template(builder, 0, best_time, best_glops);

//     cout << "best_time:" << best_time << endl;
//     cout << "best_glops:" << best_glops << endl;
//     cout << "tblock_num:" << ((direct_atom_template_warp_block_compress_node_param_t *)(node.template_param))->tblock_num << endl;
//     cout << "thread_num_in_block:" << ((direct_atom_template_warp_block_compress_node_param_t *)(node.template_param))->thread_num_in_block << endl;
// }



// 测试direct_atom_template_warp_block_compress的自动调参
// int main ()
// {
//     dataset_builder_t data_builder = get_dataset_builder(600000, 59999, 4);

//     vector<double> none_vec;
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_vector(get_row_index_of_dataset_builder(data_builder), get_col_index_of_dataset_builder(data_builder), get_float_val_of_dataset_builder(data_builder), none_vec, FLOAT, 59999, 600000);

//     operator_manager_t* op_manager = init_op_manager(matrix);

//     // 直接压缩
//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     compress_block_end_block_multiple_padding(op_manager, 0, 32, 1);

//     // 放弃BLB分块
//     unsigned int BLB_row_num = matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[0]->max_row_index - matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[0]->min_row_index + 1;
//     vector<unsigned int> BLB_row_num_vec;
//     BLB_row_num_vec.push_back(BLB_row_num);
//     sep_tblock_level_row_csr(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, BLB_row_num_vec);

//     // 放弃WLB分块
//     vector<unsigned long> BLB_id;
//     vector<vector<unsigned int>> WLB_row_size;
//     sep_warp_level_row_csr(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, BLB_id, WLB_row_size);

//     // thread级别行分块
//     vector<unsigned long> WLB_id;
//     vector<unsigned long> TLB_col_size;

//     sep_thread_level_col_ell_with_padding(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, WLB_id, TLB_col_size);

//     code_builder_t* builder = init_code_builder(op_manager);

//     direct_atom_template_warp_block_compress_t* new_template = init_direct_atom_template_warp_block_compress(builder, 0);

//     // 写模板
//     add_template_to_builder(builder, new_template, DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS, 0);

//     // 执行压缩
//     try_all_compress(new_template);

//     // 对应位置选参数
//     float best_time;
//     float best_glops;

//     template_node_t node = find_best_param_of_direct_atom_template_warp_block_compress(builder, 0, best_time, best_glops);

//     cout << "best_time:" << best_time << endl;
//     cout << "best_glops:" << best_glops << endl;
//     cout << "tblock_num:" << ((direct_atom_template_warp_block_compress_node_param_t *)(node.template_param))->tblock_num << endl;
//     cout << "thread_num_in_block:" << ((direct_atom_template_warp_block_compress_node_param_t *)(node.template_param))->thread_num_in_block << endl;
// }

// int main()
// {
    // sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/bone010.mtx.coo", FLOAT);

    // exe_graph_t* exe_graph = new exe_graph_t();
    
    // exe_begin_input_file_param_t input_param;
    // input_param.input_file_name = "/home/duzhen/spmv_builder/data_source/bone010.mtx.coo";
    // input_param.val_data_type = FLOAT;

    // add_exe_begin_input_file_node_to_exe_graph(exe_graph, EXE_DENSE_SUB_GRAPH, input_param, 0, GRAPH_END);

    // vector<unsigned long> row_nnz = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);

    // vector<unsigned long> bin_low_bound = bin_row_nnz_low_bound_of_fixed_granularity_coar_sort(row_nnz, 1);

    // // 添加粗粒度排序节点
    // exe_dense_row_coarse_sort_param_t sort_param;
    // sort_param.bin_row_nnz_low_bound = bin_low_bound;

    // cout << dependence_of_exe_dense_row_coarse_sort_node(exe_graph, EXE_DENSE_SUB_GRAPH, sort_param, 0, GRAPH_END) << endl;

    // add_exe_dense_row_coarse_sort_node_to_exe_graph(exe_graph, EXE_DENSE_SUB_GRAPH, sort_param, 0, GRAPH_END);
    
    // exe_dense_row_div_param_t row_div_param;
    // row_div_param.dense_sub_block_id = 0;
    
    // row_div_param.row_div_position.push_back(0);
    
    // row_div_param.row_div_position.push_back(1000);
    
    // row_div_param.row_div_position.push_back(matrix->dense_row_number);

    // cout << dependence_of_exe_dense_row_div_node(exe_graph, EXE_DENSE_SUB_GRAPH, row_div_param, 0, GRAPH_END) << endl;

    // add_exe_dense_row_div_node_to_exe_graph(exe_graph, EXE_DENSE_SUB_GRAPH, row_div_param, 0, GRAPH_END);

    // // 再切分一次
    // exe_dense_fixed_col_div_param_t col_div_param;
    
    // col_div_param.dense_sub_block_id = 1;
    // col_div_param.fixed_col_block_size = 500000;

    // add_exe_dense_fixed_col_div_node_to_exe_graph(exe_graph, EXE_DENSE_SUB_GRAPH, col_div_param, 0, GRAPH_END);

//     // 压缩
//     exe_compress_param_t compress_param;

//     add_exe_compress_node_to_exe_graph(exe_graph, EXE_DENSE_SUB_GRAPH, compress_param, 0, GRAPH_END);

//     execute_graph_dense_part(exe_graph);
    
//     print_dense_block_table(&(exe_graph->op_manager->matrix->block_coor_table));

//     // 将密集子块的行非零元数量算出来，并且写到文件中
//     vector<unsigned long> sub_matrix_row_nnz = get_nnz_of_each_row_in_compressed_sub_matrix(exe_graph->op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr);

//     // print_arr_to_file_with_data_type(&(sub_matrix_row_nnz[0]), UNSIGNED_LONG, sub_matrix_row_nnz.size(), "/home/duzhen/spmv_builder/data_source/test_result_3");

//     // 执行一个压缩子图的padding
//     exe_compress_row_padding_param_t row_padding_param;

//     row_padding_param.multiply = 32;
//     row_padding_param.padding_row_length = 1;

//     add_exe_compress_row_padding_node_to_graph(exe_graph, EXE_COMPRESSED_SUB_GRAPH, row_padding_param, 0, GRAPH_END);
    
//     // 当前的全局行数量
//     unsigned long global_row_num = exe_graph->op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[0]->max_row_index - exe_graph->op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[0]->min_row_index + 1;
//     if (global_row_num % row_padding_param.multiply != 0)
//     {
//         global_row_num = (global_row_num / row_padding_param.multiply + 1) * row_padding_param.multiply;
//     }

//     assert(global_row_num % row_padding_param.multiply == 0);
//     // exit(-1);

//     // 增加一个BLB行切分
//     exe_compress_tblock_level_row_div_param_t BLB_row_div_param;
//     BLB_row_div_param.row_num_of_each_BLB.push_back(global_row_num);

//     add_exe_compress_BLB_row_div_node_to_exe_graph(exe_graph, EXE_COMPRESSED_SUB_GRAPH, BLB_row_div_param, 0, GRAPH_END);

//     // 增加一个列切分
//     // exe_compress_tblock_level_col_div_param_t BLB_col_div_param;
//     // BLB_col_div_param.col_block_nnz_num_of_each_BLB = col_block_size_of_each_row(sub_matrix_row_nnz, 1024);

//     // add_exe_compress_BLB_col_div_node_to_exe_graph(exe_graph, EXE_COMPRESSED_SUB_GRAPH, BLB_col_div_param, 0, GRAPH_END);

//     // 增加一个WLB的行切分
//     exe_compress_warp_level_row_div_param_t WLB_row_div_param;

//     WLB_row_div_param.row_num_of_each_WLB_in_BLB.push_back(row_block_size_of_a_sub_matrix_by_fixed_div(exe_graph->op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[0]->max_row_index - exe_graph->op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[0]->min_row_index + 1, 1));

//     add_exe_compress_WLB_row_div_node_to_exe_graph(exe_graph, EXE_COMPRESSED_SUB_GRAPH, WLB_row_div_param, 0, GRAPH_END);

//     // 增加一个WLB的列切分，采用的切分方法是，为每一个列切分
//     exe_compress_warp_level_col_div_param_t WLB_col_div_param;
    
//     // BLB分块已经被放弃，所以直接按照行非零元数量来切分
//     WLB_col_div_param.col_num_of_WLB_in_each_parent_row_block_or_BLB = col_block_size_of_each_row(sub_matrix_row_nnz, 32);

//     add_exe_compress_WLB_col_div_node_to_exe_graph(exe_graph, EXE_COMPRESSED_SUB_GRAPH, WLB_col_div_param, 0, GRAPH_END);

//     // 执行线程粒度的切分
//     exe_compress_thread_level_row_div_param_t TLB_row_div_param;
    
//     add_exe_compress_TLB_row_div_node_to_exe_graph(exe_graph, EXE_COMPRESSED_SUB_GRAPH, TLB_row_div_param, 0, GRAPH_END);

//     // // 执行线程粒度的列分块
//     // exe_compress_thread_level_col_div_param_t TLB_col_div_param;

//     // TLB_col_div_param.col_num_of_TLB_in_each_parent_block.push_back(1);

//     // add_exe_compress_TLB_col_div_node_to_exe_graph(exe_graph, EXE_COMPRESSED_SUB_GRAPH, TLB_col_div_param, 0, GRAPH_END);

//     // // 执行TLB的按照非零元数量的切分
//     // exe_compress_thread_level_nnz_div_param_t TLB_nnz_div_param;
//     // TLB_nnz_div_param.TLB_nnz_num = 4;

//     // add_exe_compress_thread_level_nnz_div_node_to_exe_graph(exe_graph, EXE_COMPRESSED_SUB_GRAPH, TLB_nnz_div_param, 0, GRAPH_END);

//     // 这里执行压缩子图
//     execute_graph_compress_part(exe_graph);

//     // 查看首行索引
//     print_arr_to_file_with_data_type(exe_graph->op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[4]->index_of_the_first_row_arr, exe_graph->op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[4]->data_type_of_index_of_the_first_row_arr, exe_graph->op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[4]->block_num, "/home/duzhen/spmv_builder/data_source/test_result_4");
// }


// int main()
// {
//     vector<int> vec;

//     vec.insert(vec.begin() + 1, 1);

//     cout << vec.size() << endl;
// }

// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/webbase-1M.mtx.coo", FLOAT);
    
//     operator_manager_t *op_manager = init_op_manager(matrix);

//     // 矩阵分解，0-128
//     vector<unsigned int> sub_matrix_low_bound;
//     sub_matrix_low_bound.push_back(0);
//     sub_matrix_low_bound.push_back(128);

//     // 老的矩阵会被析构
//     vector<sparse_struct_t *> matrx_vec = long_short_row_decomposition(matrix, sub_matrix_low_bound);

//     op_manager->matrix = matrx_vec[0];

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     op_manager->matrix = matrx_vec[1];

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

// }

// 测试warp相关的row padding
// int main()
// {
//     dataset_builder_t data_builder = get_dataset_builder(60000, 59999, 63);

//     vector<double> none_vec;
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_vector(get_row_index_of_dataset_builder(data_builder), get_col_index_of_dataset_builder(data_builder), get_float_val_of_dataset_builder(data_builder), none_vec, FLOAT, 59999, 60000);

//     // cout << matrix->dense_row_number << endl;

//     operator_manager_t* op_manager = init_op_manager(matrix);

//     // 按照30000为一个条带执行行切分
//     vector<unsigned long> block_first_row_csr_index_vec;
//     block_first_row_csr_index_vec.push_back(0);
//     block_first_row_csr_index_vec.push_back(30000);
//     block_first_row_csr_index_vec.push_back(60001);

//     var_len_row_div(op_manager, NULL, block_first_row_csr_index_vec);

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     {
//         // 第一个横向分两个块
//         unsigned long compressed_block_id = 0;
//         compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[compressed_block_id]->compressed_block_ptr;
        
//         unsigned long compressed_row_num = compressed_block_ptr->read_index[0]->max_row_index - compressed_block_ptr->read_index[0]->min_row_index + 1;

//         vector<unsigned int> BLB_row_size;

//         // 每512行一个行条带
//         for (unsigned long i = 0; i < compressed_row_num / 512; i++)
//         {
//             BLB_row_size.push_back(512);
//         }

//         if (compressed_row_num % 512 != 0)
//         {
//             BLB_row_size.push_back(compressed_row_num % 512);
//         }

//         sep_tblock_level_row_csr(compressed_block_ptr, BLB_row_size);

//         // warp不处理
//         vector<unsigned long> sep_BLB_id;
//         vector<vector<unsigned int>> WLB_row_size_of_each_BLB;

//         // 两个warp一行
//         for (unsigned long i = 0; i < BLB_row_size.size(); i++)
//         {
//             sep_BLB_id.push_back(i);

//             vector<unsigned int> WLB_row_size_of_cur_BLB;
            
//             for (unsigned long j = 0; j < BLB_row_size[i]; j++)
//             {
//                 WLB_row_size_of_cur_BLB.push_back(1);
//             }

//             WLB_row_size_of_each_BLB.push_back(WLB_row_size_of_cur_BLB);
//         }

//         sep_warp_level_row_csr(compressed_block_ptr, sep_BLB_id, WLB_row_size_of_each_BLB);

//         // 两个warp一行
//         unsigned long row_WLB_num = compressed_block_ptr->read_index[3]->block_num;

//         vector<unsigned long> col_sep_WLB_id;
//         vector<vector<unsigned int>> WLB_col_size;

//         for (unsigned long i = 0; i < row_WLB_num; i++)
//         {
//             vector<unsigned int> col_size_of_cur_WLB;
//             col_size_of_cur_WLB.push_back(32);
//             col_size_of_cur_WLB.push_back(32);

//             col_sep_WLB_id.push_back(i);
//             WLB_col_size.push_back(col_size_of_cur_WLB);
//         }

//         sep_warp_level_col_csr(compressed_block_ptr, col_sep_WLB_id, WLB_col_size);

//         // 一个TLB一个非零元
//         vector<unsigned long> sep_WLB_id;
//         vector<unsigned long> thread_col_size_of_each_WLB;

//         unsigned long WLB_num = compressed_block_ptr->read_index[3]->block_num;
        
//         for (unsigned long WLB_id = 0; WLB_id < WLB_num; WLB_id++)
//         {
//             sep_WLB_id.push_back(WLB_id);
//             thread_col_size_of_each_WLB.push_back(1);
//         }

//         sep_thread_level_col_ell_with_padding(compressed_block_ptr, sep_WLB_id, thread_col_size_of_each_WLB);
        
//         // 将TLB的首行索引打印出来
//         print_arr_to_file_with_data_type(compressed_block_ptr->read_index[4]->index_of_the_first_row_arr, compressed_block_ptr->read_index[4]->data_type_of_index_of_the_first_row_arr, compressed_block_ptr->read_index[4]->block_num, "/home/duzhen/spmv_builder/data_source/test_result_3");

//         cout << compressed_block_ptr->read_index[4]->block_num << endl;
//     }

//     {
//         unsigned long compressed_block_id = 1;
        
//         compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[compressed_block_id]->compressed_block_ptr;

//         compress_block_end_block_multiple_padding(op_manager, compressed_block_id, 512, 63);
        
//         unsigned long compressed_row_num = compressed_block_ptr->read_index[0]->max_row_index - compressed_block_ptr->read_index[0]->min_row_index + 1;

//         vector<unsigned int> BLB_row_size;

//         // 每512行一个行条带
//         for (unsigned long i = 0; i < compressed_row_num / 512; i++)
//         {
//             BLB_row_size.push_back(512);
//         }

//         if (compressed_row_num % 512 != 0)
//         {
//             BLB_row_size.push_back(compressed_row_num % 512);
//         }

//         sep_tblock_level_row_csr(compressed_block_ptr, BLB_row_size);

//         // warp不处理
//         vector<unsigned long> sep_BLB_id;
//         vector<vector<unsigned int>> WLB_row_size_of_each_BLB;

//         // 两个warp一行
//         for (unsigned long i = 0; i < BLB_row_size.size(); i++)
//         {
//             sep_BLB_id.push_back(i);

//             vector<unsigned int> WLB_row_size_of_cur_BLB;
            
//             for (unsigned long j = 0; j < BLB_row_size[i]; j++)
//             {
//                 WLB_row_size_of_cur_BLB.push_back(1);
//             }

//             WLB_row_size_of_each_BLB.push_back(WLB_row_size_of_cur_BLB);
//         }

//         sep_warp_level_row_csr(compressed_block_ptr, sep_BLB_id, WLB_row_size_of_each_BLB);

//         // 两个warp一行
//         unsigned long row_WLB_num = compressed_block_ptr->read_index[3]->block_num;

//         vector<unsigned long> col_sep_WLB_id;
//         vector<vector<unsigned int>> WLB_col_size;

//         for (unsigned long i = 0; i < row_WLB_num; i++)
//         {
//             vector<unsigned int> col_size_of_cur_WLB;
//             col_size_of_cur_WLB.push_back(32);
//             col_size_of_cur_WLB.push_back(32);

//             col_sep_WLB_id.push_back(i);
//             WLB_col_size.push_back(col_size_of_cur_WLB);
//         }

//         sep_warp_level_col_csr(compressed_block_ptr, col_sep_WLB_id, WLB_col_size);

//         // 一个TLB一个非零元
//         vector<unsigned long> sep_WLB_id;
//         vector<unsigned long> thread_col_size_of_each_WLB;

//         unsigned long WLB_num = compressed_block_ptr->read_index[3]->block_num;
        
//         for (unsigned long WLB_id = 0; WLB_id < WLB_num; WLB_id++)
//         {
//             sep_WLB_id.push_back(WLB_id);
//             thread_col_size_of_each_WLB.push_back(1);
//         }

//         sep_thread_level_col_ell_with_padding(compressed_block_ptr, sep_WLB_id, thread_col_size_of_each_WLB);
        
//         // 将TLB的首行索引打印出来
//         print_arr_to_file_with_data_type(compressed_block_ptr->read_index[4]->index_of_the_first_row_arr, compressed_block_ptr->read_index[4]->data_type_of_index_of_the_first_row_arr, compressed_block_ptr->read_index[4]->block_num, "/home/duzhen/spmv_builder/data_source/test_result_3");

//         cout << compressed_block_ptr->read_index[4]->block_num << endl;
//     }

//     // 存起来，执行对应的模板
//     code_builder_t* builder = init_code_builder(op_manager);
    
//     // 然后执行对应的
//     // 初始化一个模板
//     shared_memory_total_warp_reduce_template_t* new_template1 = init_shared_memory_total_warp_reduce_template(builder, 0);
    
//     shared_memory_total_warp_reduce_template_t* new_template2 = init_shared_memory_total_warp_reduce_template(builder, 1);
    

//     add_template_to_builder(builder, new_template1, SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE, 0);
//     add_template_to_builder(builder, new_template2, SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE, 1);
//     try_all_compress(new_template1);
//     try_all_compress(new_template2);
    
//     store_code_builder_data(builder);

//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));
// }

// 测试子块的row padding的shared memory
// int main()
// {
//     dataset_builder_t data_builder = get_dataset_builder(59999, 59999, 4);

//     vector<double> none_vec;
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_vector(get_row_index_of_dataset_builder(data_builder), get_col_index_of_dataset_builder(data_builder), get_float_val_of_dataset_builder(data_builder), none_vec, FLOAT, 59999, 59999);

//     // cout << matrix->dense_row_number << endl;

//     operator_manager_t* op_manager = init_op_manager(matrix);
    

//     // 按照30000为一个条带执行行切分
//     vector<unsigned long> block_first_row_csr_index_vec;
//     block_first_row_csr_index_vec.push_back(0);
//     block_first_row_csr_index_vec.push_back(30000);
//     block_first_row_csr_index_vec.push_back(60000);

//     var_len_row_div(op_manager, NULL, block_first_row_csr_index_vec);

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     {
//         // 对第一个块随便做一个ELL
//         unsigned long compressed_block_id = 0;
//         compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[compressed_block_id]->compressed_block_ptr;
        
//         unsigned long dense_row_num = matrix->block_coor_table.item_arr[compressed_block_id]->max_dense_row_index - matrix->block_coor_table.item_arr[compressed_block_id]->min_dense_row_index + 1;

//         vector<unsigned int> BLB_row_size;
        
//         for (unsigned long i = 0; i < dense_row_num / 512; i++)
//         {
//             BLB_row_size.push_back(512);
//         }

//         if (dense_row_num % 512 != 0)
//         {
//             BLB_row_size.push_back(dense_row_num % 512);
//         }

//         sep_tblock_level_row_csr(compressed_block_ptr, BLB_row_size);

//         // warp不处理
//         vector<unsigned long> sep_BLB_id;
//         vector<vector<unsigned int>> WLB_row_size_of_each_BLB;

//         sep_warp_level_row_csr(compressed_block_ptr, sep_BLB_id, WLB_row_size_of_each_BLB);
        
//         // 默认的行切分
//         vector<unsigned long> sep_WLB_id;
//         vector<unsigned long> thread_col_size_of_each_WLB;

//         // 查看warp的数量
//         unsigned long warp_num = compressed_block_ptr->read_index[3]->block_num;

//         for (unsigned long i = 0; i < warp_num; i++)
//         {
//             sep_WLB_id.push_back(i);
//             thread_col_size_of_each_WLB.push_back(1);
//         }

//         sep_thread_level_col_ell_with_padding(compressed_block_ptr, sep_WLB_id, thread_col_size_of_each_WLB);

//         // 将TLB的首行索引打印出来
//         print_arr_to_file_with_data_type(compressed_block_ptr->read_index[4]->index_of_the_first_row_arr, compressed_block_ptr->read_index[4]->data_type_of_index_of_the_first_row_arr, compressed_block_ptr->read_index[4]->block_num, "/home/duzhen/spmv_builder/data_source/test_result_3");

//         cout << compressed_block_ptr->read_index[4]->block_num << endl;
//     }

//     {
//         unsigned long compressed_block_id = 1;
//         compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[compressed_block_id]->compressed_block_ptr;

//         // 执行padding到32的倍数
//         compress_block_end_block_multiple_padding(op_manager, compressed_block_id, 512, 4);

//         unsigned long compressed_block_row_num = compressed_block_ptr->read_index[0]->max_row_index - compressed_block_ptr->read_index[0]->min_row_index + 1;
//         // cout << compressed_block_ptr->read_index[0]->max_row_index - compressed_block_ptr->read_index[0]->min_row_index + 1 << endl;

//         vector<unsigned int> BLB_row_size;
        
//         for (unsigned long i = 0; i < compressed_block_row_num / 512; i++)
//         {
//             BLB_row_size.push_back(512);
//         }

//         assert(compressed_block_row_num % 512 == 0);

//         sep_tblock_level_row_csr(compressed_block_ptr, BLB_row_size);

//         // warp不处理
//         vector<unsigned long> sep_BLB_id;
//         vector<vector<unsigned int>> WLB_row_size_of_each_BLB;

//         sep_warp_level_row_csr(compressed_block_ptr, sep_BLB_id, WLB_row_size_of_each_BLB);
        
//         // 默认的行切分
//         vector<unsigned long> sep_WLB_id;
//         vector<unsigned long> thread_col_size_of_each_WLB;

//         // 查看warp的数量
//         unsigned long warp_num = compressed_block_ptr->read_index[3]->block_num;

//         for (unsigned long i = 0; i < warp_num; i++)
//         {
//             sep_WLB_id.push_back(i);
//             thread_col_size_of_each_WLB.push_back(1);
//         }

//         sep_thread_level_col_ell_with_padding(compressed_block_ptr, sep_WLB_id, thread_col_size_of_each_WLB);
//         print_arr_to_file_with_data_type(compressed_block_ptr->read_index[4]->index_of_the_first_row_arr, compressed_block_ptr->read_index[4]->data_type_of_index_of_the_first_row_arr, compressed_block_ptr->read_index[4]->block_num, "/home/duzhen/spmv_builder/data_source/test_result_4");
//     }


//     // 存起来，执行对应的模板
//     code_builder_t* builder = init_code_builder(op_manager);

//     // 然后执行对应的
//     // 初始化一个模板
//     shared_memory_template_warp_compress_t* new_template1 = init_shared_memory_template_warp_compress(builder, 0);
//     shared_memory_template_warp_compress_t* new_template2 = init_shared_memory_template_warp_compress(builder, 1);
    
//     add_template_to_builder(builder, new_template1, SHARED_MEMORY_TEMPLATE_WARP_COMPRESS, 0);
//     add_template_to_builder(builder, new_template2, SHARED_MEMORY_TEMPLATE_WARP_COMPRESS, 1);
//     try_all_compress(new_template1);
//     try_all_compress(new_template2);
    
//     store_code_builder_data(builder);
    
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));
// }

// 测试子块的row padding操作atom thread的
// int main()
// {
//     dataset_builder_t data_builder = get_dataset_builder(59999, 59999, 4);

//     vector<double> none_vec;
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_vector(get_row_index_of_dataset_builder(data_builder), get_col_index_of_dataset_builder(data_builder), get_float_val_of_dataset_builder(data_builder), none_vec, FLOAT, 59999, 59999);

//     // cout << matrix->dense_row_number << endl;

//     operator_manager_t* op_manager = init_op_manager(matrix);
    

//     // 按照30000为一个条带执行行切分
//     vector<unsigned long> block_first_row_csr_index_vec;
//     block_first_row_csr_index_vec.push_back(0);
//     block_first_row_csr_index_vec.push_back(30000);
//     block_first_row_csr_index_vec.push_back(60000);

//     var_len_row_div(op_manager, NULL, block_first_row_csr_index_vec);

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     {
//         // 对第一个块随便做一个ELL
//         unsigned long compressed_block_id = 0;
//         compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[compressed_block_id]->compressed_block_ptr;
        
//         unsigned long dense_row_num = matrix->block_coor_table.item_arr[compressed_block_id]->max_dense_row_index - matrix->block_coor_table.item_arr[compressed_block_id]->min_dense_row_index + 1;

//         vector<unsigned int> BLB_row_size;
        
//         for (unsigned long i = 0; i < dense_row_num / 512; i++)
//         {
//             BLB_row_size.push_back(512);
//         }

//         if (dense_row_num % 512 != 0)
//         {
//             BLB_row_size.push_back(dense_row_num % 512);
//         }

//         sep_tblock_level_row_csr(compressed_block_ptr, BLB_row_size);

//         // warp不处理
//         vector<unsigned long> sep_BLB_id;
//         vector<vector<unsigned int>> WLB_row_size_of_each_BLB;

//         sep_warp_level_row_csr(compressed_block_ptr, sep_BLB_id, WLB_row_size_of_each_BLB);
        
//         // 默认的行切分
//         vector<unsigned long> sep_WLB_id;
//         vector<unsigned long> thread_col_size_of_each_WLB;

//         sep_thread_level_col_ell_with_padding(compressed_block_ptr, sep_WLB_id, thread_col_size_of_each_WLB);

//         // 将TLB的首行索引打印出来
//         print_arr_to_file_with_data_type(compressed_block_ptr->read_index[4]->index_of_the_first_row_arr, compressed_block_ptr->read_index[4]->data_type_of_index_of_the_first_row_arr, compressed_block_ptr->read_index[4]->block_num, "/home/duzhen/spmv_builder/data_source/test_result_3");

//         cout << compressed_block_ptr->read_index[4]->block_num << endl;
//     }

//     {
//         unsigned long compressed_block_id = 1;
//         compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[compressed_block_id]->compressed_block_ptr;

//         // 执行padding到32的倍数
//         compress_block_end_block_multiple_padding(op_manager, compressed_block_id, 512, 1);

//         unsigned long compressed_block_row_num = compressed_block_ptr->read_index[0]->max_row_index - compressed_block_ptr->read_index[0]->min_row_index + 1;
//         // cout << compressed_block_ptr->read_index[0]->max_row_index - compressed_block_ptr->read_index[0]->min_row_index + 1 << endl;

//         vector<unsigned int> BLB_row_size;
        
//         for (unsigned long i = 0; i < compressed_block_row_num / 512; i++)
//         {
//             BLB_row_size.push_back(512);
//         }

//         assert(compressed_block_row_num % 512 == 0);

//         sep_tblock_level_row_csr(compressed_block_ptr, BLB_row_size);

//         // warp不处理
//         vector<unsigned long> sep_BLB_id;
//         vector<vector<unsigned int>> WLB_row_size_of_each_BLB;

//         sep_warp_level_row_csr(compressed_block_ptr, sep_BLB_id, WLB_row_size_of_each_BLB);
        
//         // 默认的行切分
//         vector<unsigned long> sep_WLB_id;
//         vector<unsigned long> thread_col_size_of_each_WLB;

//         sep_thread_level_col_ell_with_padding(compressed_block_ptr, sep_WLB_id, thread_col_size_of_each_WLB);
//         print_arr_to_file_with_data_type(compressed_block_ptr->read_index[4]->index_of_the_first_row_arr, compressed_block_ptr->read_index[4]->data_type_of_index_of_the_first_row_arr, compressed_block_ptr->read_index[4]->block_num, "/home/duzhen/spmv_builder/data_source/test_result_4");
//     }


//     // 存起来，执行对应的模板
//     vector<int> sub_matrix_id;
//     sub_matrix_id.push_back(1);
    
//     code_builder_t* builder = init_code_builder(op_manager, sub_matrix_id);
    
//     // 然后执行对应的
//     // 初始化一个模板
//     // direct_atom_template_warp_compress_t* new_template1 = init_direct_atom_template_warp_compress(builder, 0);
//     direct_atom_template_warp_compress_t* new_template2 = init_direct_atom_template_warp_compress(builder, 1);

//     // add_template_to_builder(builder, new_template1, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 0);
//     add_template_to_builder(builder, new_template2, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 1);
//     // try_all_compress(new_template1);
//     try_all_compress(new_template2);

//     store_code_builder_data(builder, sub_matrix_id);

//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder, sub_matrix_id));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder, sub_matrix_id));

//     // 析构对应的模板
//     memory_garbage_manager_t mem_manager;
//     delete_template_without_matrix_with_type(&mem_manager, builder, 1);
// }



// int main()
// {
//     dataset_builder_t data_builder = get_dataset_builder(60000, 60000, 1024);

//     vector<double> none_vec;
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_vector(get_row_index_of_dataset_builder(data_builder), get_col_index_of_dataset_builder(data_builder), get_float_val_of_dataset_builder(data_builder), none_vec, FLOAT, 60000, 60000);

//     vector<unsigned long> row_nnz = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);

//     // vector<unsigned long> div_position_vec = row_nnz_range_div_position(row_nnz, 4, 8);
//     vector<vector<unsigned int>> col_block_size_vec = col_block_size_of_each_row(row_nnz, 1024);

//     for (auto col_block_size : col_block_size_vec[0])
//     {
//         cout << "col_block_size:" << col_block_size << endl;
//     }

//     // print_arr_to_file_with_data_type(&(row_nnz[0]), UNSIGNED_LONG, row_nnz.size(), "/home/duzhen/spmv_builder/data_source/test_result_3");
//     // print_arr_to_file_with_data_type(&(bin_low_bound[0]), UNSIGNED_LONG, bin_low_bound.size(), "/home/duzhen/spmv_builder/data_source/test_result_4");
    
//     return 0;
// }



// int main()
// {
//     dataset_builder_t data_builder = get_dataset_builder(60000, 60000, 60);

//     vector<double> none_vec;
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_vector(get_row_index_of_dataset_builder(data_builder), get_col_index_of_dataset_builder(data_builder), get_float_val_of_dataset_builder(data_builder), none_vec, FLOAT, 60000, 60000);

//     operator_manager_t *op_manager = init_op_manager(matrix);

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // exit(-1);

//     // 每16行执行一次行分块
//     compressed_block_t* cur_block = matrix->block_coor_table.item_arr[0]->compressed_block_ptr;

//     // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//     unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[0]->min_dense_row_index;
//     unsigned long block_end_row_index = matrix->block_coor_table.item_arr[0]->max_dense_row_index;

//     // 行数量
//     unsigned long row_num = block_end_row_index - block_begin_row_index + 1;

//     // 全局一个列分块
//     vector<unsigned int> block_row_num;

//     block_row_num.push_back(row_num);

//     sep_tblock_level_row_csr(cur_block, block_row_num);

//     vector<unsigned long> block_index_arr;
//     vector<vector<unsigned int>> row_block_size_arr;

//     // 搞一个默认的行切分（其实就是啥都不切），现在不支持warp直接列分块
//     sep_warp_level_row_csr(cur_block, block_index_arr, row_block_size_arr);
    
//     for (unsigned long i = 0; i < block_row_num.size(); i++)
//     {
//         // 列方向的分块，按照64非零元一个纵块来分块
//         // void sep_warp_level_col_csr(compressed_block_t *compressed_block, vector<unsigned long> block_index_arr, vector<vector<unsigned int>> col_block_size_arr)
//         // 当前块每一行分成16个子块
//         vector<unsigned int> sub_block_col_num;

//         // 每个子块的大小为64
//         for (unsigned long j = 0; j < 16; j++)
//         {
//             sub_block_col_num.push_back(64);
//         }
        
//         row_block_size_arr.push_back(sub_block_col_num);
//         block_index_arr.push_back(i);
//     }

//     // 一行一个warp
//     sep_warp_level_col_csr(cur_block, block_index_arr, row_block_size_arr);

//     // warp粒度的块的数量
//     unsigned long warp_num = cur_block->read_index[3]->block_num;

//     // assert(warp_num == matrix->dense_row_number * 16);
//     cout << "warp_num:" << warp_num << endl;
//     cout << "block_num:" << matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[2]->block_num << endl;

//     // 一个线程一个非零元
//     vector<unsigned long> futher_thread_block_vec;
//     vector<unsigned long> futher_thread_col_block_size;

//     for (unsigned long i = 0; i < warp_num; i++)
//     {
//         futher_thread_block_vec.push_back(i);
//         futher_thread_col_block_size.push_back(1);
//     }

//     sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);

    // set<template_type> type_vec = supported_template_of_sub_matrix(matrix, 0);

    // for (set<template_type>::iterator cur_temp_type_ptr = type_vec.begin(); cur_temp_type_ptr != type_vec.end(); cur_temp_type_ptr++)
    // {
    //     cout << convert_template_type_to_string(*cur_temp_type_ptr) << endl;
    // }
    
    // memory_garbage_manager_t* mem_manager = new memory_garbage_manager_t();

    // delete_sparse_struct_t(mem_manager, matrix);

    // delete mem_manager;
// }

// TODO：其他shared memory归约的改造(ok)，利用共享内存处理BLB首个非零元的索引。特定目标的矩阵生成（ok）。针对某一个密集子矩阵的padding操作（把所有的padding集合到一开始来处理这一步）。
// TODO：total warp reduce测试多线程归约(ok)。
// TODO：将warp上直接进行原子加的结果，线程粒度的块的大小是1，没有block层次的遍历。在warp粒度上执行元数据的广播（reduce）。
// int main()
// {
//     dataset_builder_t data_builder = get_dataset_builder(60000, 60000, 60);

//     vector<double> none_vec;
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_vector(get_row_index_of_dataset_builder(data_builder), get_col_index_of_dataset_builder(data_builder), get_float_val_of_dataset_builder(data_builder), none_vec, FLOAT, 60000, 60000);

//     operator_manager_t *op_manager = init_op_manager(matrix);

//     // if (matrix->dense_row_number % 32 != 0)
//     // {
//     //     // 要额外增加的行数量
//     //     unsigned long add_row_num = 32 - (matrix->dense_row_number % 32);

//     //     // 
//     //     total_row_level_padding_add(op_manager, add_row_num, 120);
//     // }

//     // cout << "matrix->dense_row_number:" << matrix->dense_row_number << endl;

//     // assert(matrix->dense_row_number % 32 == 0);

//     // 这里加一个分桶
//     // 做一个排序
//     vector<unsigned long> bin_nnz_range;
    
//     for (unsigned long i = 0; i < 500; i = i + 500)
//     {
//         bin_nnz_range.push_back(i);
//     }

//     vector<unsigned long> bin_first_row_vec = total_dense_block_coarse_sort(op_manager, bin_nnz_range);
    
//     // void total_row_level_padding_direct(operator_manager_t *op_manager, unsigned long target_size, unsigned padding_col_num, global_padding_position padding_type, unsigned long input_col_index)
//     // 按照目标方式补0
//     // total_row_level_padding(op_manager, block_row_num_size);

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // exit(-1);

//     // 每16行执行一次行分块
//     compressed_block_t* cur_block = matrix->block_coor_table.item_arr[0]->compressed_block_ptr;

//     // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//     unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[0]->min_dense_row_index;
//     unsigned long block_end_row_index = matrix->block_coor_table.item_arr[0]->max_dense_row_index;

//     // 行数量
//     unsigned long row_num = block_end_row_index - block_begin_row_index + 1;

//     // 全局一个列分块
//     vector<unsigned int> block_row_num;

//     block_row_num.push_back(row_num);

//     sep_tblock_level_row_csr(cur_block, block_row_num);

//     vector<unsigned long> block_index_arr;
//     vector<vector<unsigned int>> row_block_size_arr;

//     // 搞一个默认的行切分（其实就是啥都不切），现在不支持warp直接列分块
//     sep_warp_level_row_csr(cur_block, block_index_arr, row_block_size_arr);
    
//     for (unsigned long i = 0; i < block_row_num.size(); i++)
//     {
//         // 列方向的分块，按照64非零元一个纵块来分块
//         // void sep_warp_level_col_csr(compressed_block_t *compressed_block, vector<unsigned long> block_index_arr, vector<vector<unsigned int>> col_block_size_arr)
//         // 当前块每一行分成16个子块
//         vector<unsigned int> sub_block_col_num;

//         // 每个子块的大小为64
//         for (unsigned long j = 0; j < 16; j++)
//         {
//             sub_block_col_num.push_back(64);
//         }
        
//         row_block_size_arr.push_back(sub_block_col_num);
//         block_index_arr.push_back(i);
//     }

//     // 一行一个warp
//     sep_warp_level_col_csr(cur_block, block_index_arr, row_block_size_arr);

//     // warp粒度的块的数量
//     unsigned long warp_num = cur_block->read_index[3]->block_num;

//     // assert(warp_num == matrix->dense_row_number * 16);
//     cout << "warp_num:" << warp_num << endl;
//     cout << "block_num:" << matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[2]->block_num << endl;

//     // 一个线程一个非零元
//     vector<unsigned long> futher_thread_block_vec;
//     vector<unsigned long> futher_thread_col_block_size;

//     for (unsigned long i = 0; i < warp_num; i++)
//     {
//         futher_thread_block_vec.push_back(i);
//         futher_thread_col_block_size.push_back(1);
//     }

//     sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);

//     set<template_type> type_vec = supported_template_of_sub_matrix(matrix, 0);

//     for (set<template_type>::iterator cur_temp_type_ptr = type_vec.begin(); cur_temp_type_ptr != type_vec.end(); cur_temp_type_ptr++)
//     {
//         cout << convert_template_type_to_string(*cur_temp_type_ptr) << endl;
//     }

//     exit(-1);

//     code_builder_t* builder = init_code_builder(op_manager);

//     // 初始化一个模板
//     direct_atom_total_warp_reduce_template_t* new_template = init_direct_atom_total_warp_reduce_template(builder, 0);
//     // 设置归约的并行度
//     // set_row_reduce_thread_num(new_template, 4);

//     add_template_to_builder(builder, new_template, DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE, 0);
//     try_all_compress(new_template);

//     store_template_data(new_template, "/home/duzhen/spmv_builder/data_source");

//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));
// }

// Rucci，测试CSR5的方式
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/webbase-1M.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);

//     gettimeofday(&pre_start, NULL);

//     // 按照32的倍数padding
//     if (matrix->dense_row_number % 32 != 0)
//     {
//         total_row_level_padding(op_manager, 32);
//     }

//     // 执行一个分桶，产生一个分桶矩阵
//     vector<unsigned long> bin_nnz_range;
    
//     for (unsigned long i = 0; i < 500; i = i + 500)
//     {
//         bin_nnz_range.push_back(i);
//     }

//     vector<unsigned long> bin_first_row_vec = total_dense_block_coarse_sort(op_manager, bin_nnz_range);
    

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 三个层次的分块都用最简单的
//     compressed_block_t* cur_block = matrix->block_coor_table.item_arr[0]->compressed_block_ptr;
//     // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//     unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[0]->min_dense_row_index;
//     unsigned long block_end_row_index = matrix->block_coor_table.item_arr[0]->max_dense_row_index;

//     unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//     // 行切分，一行一个块
//     vector<unsigned int> block_row_num;

//     block_row_num.push_back(block_row_size);
    
//     sep_tblock_level_row_csr(cur_block, block_row_num);

//     // 放弃warp的排序
//     vector<vector<unsigned int>> arr_of_row_block_size_arr;
//     vector<unsigned long> sep_block_id_arr;

//     sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//     // 一个线程一个非零元
//     // vector<unsigned long> futher_thread_block_vec;
//     // vector<unsigned long> futher_thread_col_block_size;

//     sep_thread_level_acc_to_nnz(cur_block, 4);
    
//     // 列分块
//     // sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);

//     // 执行代码生成
//     code_builder_t* builder = init_code_builder(op_manager);
    
//     // write_total_matrix_to_file(matrix, "/home/duzhen/spmv_builder/data_source");

//     // exit(-1);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;
//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     // cout << supported_template_of_sub_matrix(op_manager->matrix, 0).size() << endl;

//     // 使用warp压缩型的模板
//     {
//         // 第一个桶使用的kernal
//         // 两个子矩阵
//         unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t* bin0_template = init_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce(builder, 0);

//         // exit(-1);
        

//         // 析构一个模板
//         // memory_garbage_manager_t mem_manager;

//         // delete_unaligned_warp_reduce_same_TLB_size_template(&mem_manager, bin0_template);

//         // exit(-1);

//         add_template_to_builder(builder, bin0_template, UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE, 0);

//         try_all_compress(bin0_template);
        
//         // 将block和warp层次的遍历全部去掉
//         // compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS, 0);

//         // direct_atom_template_warp_block_compress_t* compressed_template = (direct_atom_template_warp_block_compress_t*)builder->template_vec[0];

//         // compressed_template->thread_num_in_block = row_block_size;
//         // bin0_template->tblock_num = bin0_template->size_of_global_row_index_of_thread_level_block / bin0_template->thread_num_in_block;

//         // 压缩
//         // compress_global_row_index_of_thread_level_block(bin0_template);
//         // compress_block_begin_thread_index_offset(compressed_template);
//     }

    

//     store_code_builder_data(builder);

//     // 生成代码
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));
// }

// // Rucci，适用于ELL方式处理，也可以尝试warp reduce
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/Rucci1.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);

//     gettimeofday(&pre_start, NULL);

//     // 按照32的倍数padding
//     if (matrix->dense_row_number % 32 != 0)
//     {
//         total_row_level_padding(op_manager, 32);
//     }

//     // 执行一个分桶，产生一个分桶矩阵
//     vector<unsigned long> bin_nnz_range;
    
//     for (unsigned long i = 0; i < 500; i = i + 500)
//     {
//         bin_nnz_range.push_back(i);
//     }

//     vector<unsigned long> bin_first_row_vec = total_dense_block_coarse_sort(op_manager, bin_nnz_range);
    

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 三个层次的分块都用最简单的
//     compressed_block_t* cur_block = matrix->block_coor_table.item_arr[0]->compressed_block_ptr;
//     // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//     unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[0]->min_dense_row_index;
//     unsigned long block_end_row_index = matrix->block_coor_table.item_arr[0]->max_dense_row_index;

//     unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//     // 行切分，一行一个块
//     vector<unsigned int> block_row_num;

//     block_row_num.push_back(block_row_size);
    
//     sep_tblock_level_row_csr(cur_block, block_row_num);

//     // 放弃warp的排序
//     vector<vector<unsigned int>> arr_of_row_block_size_arr;
//     vector<unsigned long> sep_block_id_arr;

//     sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//     // 一个线程一个非零元
//     vector<unsigned long> futher_thread_block_vec;
//     vector<unsigned long> futher_thread_col_block_size;

//     // 列分块
//     sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);

//     // 执行代码生成
//     code_builder_t* builder = init_code_builder(op_manager);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;
//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     // 使用warp压缩型的模板
//     {
//         // 第一个桶使用的kernal
//         // 两个子矩阵
//         direct_atom_template_warp_compress_t* bin0_template = init_direct_atom_template_warp_compress(builder, 0);

//         add_template_to_builder(builder, bin0_template, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 0);
        
//         // 将block和warp层次的遍历全部去掉
//         // compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS, 0);

//         // direct_atom_template_warp_block_compress_t* compressed_template = (direct_atom_template_warp_block_compress_t*)builder->template_vec[0];

//         // compressed_template->thread_num_in_block = row_block_size;
//         bin0_template->tblock_num = bin0_template->size_of_global_row_index_of_thread_level_block / bin0_template->thread_num_in_block;

//         // 压缩
//         // compress_global_row_index_of_thread_level_block(bin0_template);
//         // compress_block_begin_thread_index_offset(compressed_template);
//     }

//     // store_code_builder_data(builder);

//     // 生成代码
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));
// }

// in-2004，有长行有短行
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/in-2004.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);
    
//     gettimeofday(&pre_start, NULL);

//     unsigned long long_row_min_size = 128;
//     unsigned long row_num_of_small_row_block = 512;
//     // 列分块的最小宽度
//     unsigned long col_block_min_width = 128;

//     // 不同
//     vector<unsigned long> row_nnz_range;
//     row_nnz_range.push_back(0);
//     row_nnz_range.push_back(long_row_min_size);
//     row_nnz_range.push_back(7754);

//     vector<unsigned long> row_num_of_each_range = get_row_num_of_each_row_nnz_range(op_manager, row_nnz_range);

//     for (unsigned long i = 0; i < row_num_of_each_range.size(); i++)
//     {
//         cout << row_num_of_each_range[i] << endl;
//     }

//     unsigned long row_num_of_short_row = row_num_of_each_range[0];
//     unsigned long row_num_of_long_row = row_num_of_each_range[1];

//     // 给短行做padding
//     if (row_num_of_short_row % row_num_of_small_row_block != 0)
//     {
//         row_num_of_short_row = (row_num_of_short_row / row_num_of_small_row_block + 1) * row_num_of_small_row_block;

//         // 执行一个padding
//         total_row_level_padding_direct(op_manager, row_num_of_short_row + row_num_of_long_row);
//     }

//     vector<unsigned long> bin_nnz_range;
    
//     for (unsigned long i = 0; i < 7754; i = i + 1)
//     {
//         bin_nnz_range.push_back(i);
//     }

//     vector<unsigned long> bin_first_row_vec = total_dense_block_coarse_sort(op_manager, bin_nnz_range);
    
//     vector<unsigned long> block_begin_row;
//     block_begin_row.push_back(0);
//     block_begin_row.push_back(row_num_of_long_row);
//     block_begin_row.push_back(row_num_of_short_row + row_num_of_long_row);

//     var_len_row_div(op_manager->matrix, NULL, block_begin_row);

//     vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);
    
//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));
// }

// bone010的优化，中等长度，相对均衡的矩阵，ELL的padding rate为16，CSR5性能非常高。
// 先尝试排序的SELL，然后尝试不排序的，之后可以尝试纵分块，然后使用共享内存的方式归约
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/bone010.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);

//     gettimeofday(&pre_start, NULL);

//     // 按照一定的长度处理执行条带分块
//     unsigned long row_block_size = 256;

//     // 执行padding
//     if (matrix->dense_row_number % row_block_size != 0)
//     {
//         total_row_level_padding(op_manager, row_block_size);
//     }

//     assert(op_manager->matrix->dense_row_number % row_block_size == 0);

//     // 做一个排序
//     vector<unsigned long> bin_nnz_range;
    
//     for (unsigned long i = 0; i < 45; i = i + 1)
//     {
//         bin_nnz_range.push_back(i);
//     }

//     vector<unsigned long> bin_first_row_vec = total_dense_block_coarse_sort(op_manager, bin_nnz_range);

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // block行分块
//     compressed_block_t* cur_block = matrix->block_coor_table.item_arr[0]->compressed_block_ptr;
//     // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//     unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[0]->min_dense_row_index;
//     unsigned long block_end_row_index = matrix->block_coor_table.item_arr[0]->max_dense_row_index;

//     unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//     // 行切分，一行一个块
//     vector<unsigned int> block_row_num;

//     for (unsigned long i = 0; i < block_row_size / row_block_size; i++)
//     {
//         block_row_num.push_back(row_block_size);
//     }
    
//     sep_tblock_level_row_csr(cur_block, block_row_num);

//     // 默认的warp和thread分块手段
//     // 放弃warp的排序
//     vector<vector<unsigned int>> arr_of_row_block_size_arr;
//     vector<unsigned long> sep_block_id_arr;

//     sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//     vector<unsigned long> futher_thread_block_vec;
//     vector<unsigned long> futher_thread_col_block_size;

//     // 列分块
//     sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);

//     // 执行代码生成
//     code_builder_t* builder = init_code_builder(op_manager);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;
//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);
    
//     // 生成模板
//     // 使用warp压缩型的模板
//     {
//         // 第一个桶使用的kernal
//         // 两个子矩阵
//         direct_atom_template_t* bin0_template = init_direct_atom_template(builder, 0);

//         add_template_to_builder(builder, bin0_template, DIRECT_ATOM_TEMPLATE, 0);
        
//         // 将block和warp层次的遍历全部去掉
//         compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 0);

//         direct_atom_template_warp_compress_t* compressed_template = (direct_atom_template_warp_compress_t*)builder->template_vec[0];

//         compressed_template->thread_num_in_block = row_block_size;
//         compressed_template->tblock_num = compressed_template->size_of_global_row_index_of_thread_level_block / compressed_template->thread_num_in_block;

//         // 压缩
//         compress_block_begin_thread_index_offset(compressed_template);
//         compress_global_row_index_of_thread_level_block(compressed_template);
//     }

//     store_code_builder_data(builder);

//     // 生成代码
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));
// }


// conf6_0-8x8-30的优化，也是类似于ELL的结构
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/conf6_0-8x8-30.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);

//     gettimeofday(&pre_start, NULL);

//     // 按照32的倍数padding
//     if (matrix->dense_row_number % 32 != 0)
//     {
//         total_row_level_padding(op_manager, 32);
//     }

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 三个层次的分块都用最简单的
//     compressed_block_t* cur_block = matrix->block_coor_table.item_arr[0]->compressed_block_ptr;
//     // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//     unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[0]->min_dense_row_index;
//     unsigned long block_end_row_index = matrix->block_coor_table.item_arr[0]->max_dense_row_index;

//     unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//     // 行切分，一行一个块
//     vector<unsigned int> block_row_num;

//     block_row_num.push_back(block_row_size);
    
//     sep_tblock_level_row_csr(cur_block, block_row_num);

//     // 放弃warp的排序
//     vector<vector<unsigned int>> arr_of_row_block_size_arr;
//     vector<unsigned long> sep_block_id_arr;

//     sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//     // 一个线程一个非零元
//     vector<unsigned long> futher_thread_block_vec;
//     vector<unsigned long> futher_thread_col_block_size;

//     // 列分块
//     sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);

//     // 执行代码生成
//     code_builder_t* builder = init_code_builder(op_manager);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;
//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     // 使用warp压缩型的模板
//     {
//         // 第一个桶使用的kernal
//         // 两个子矩阵
//         direct_atom_template_t* bin0_template = init_direct_atom_template(builder, 0);

//         add_template_to_builder(builder, bin0_template, DIRECT_ATOM_TEMPLATE, 0);
        
//         // 将block和warp层次的遍历全部去掉
//         compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS, 0);

//         direct_atom_template_warp_block_compress_t* compressed_template = (direct_atom_template_warp_block_compress_t*)builder->template_vec[0];

//         // compressed_template->thread_num_in_block = row_block_size;
//         compressed_template->tblock_num = compressed_template->size_of_global_row_index_of_thread_level_block / compressed_template->thread_num_in_block;

//         // 压缩
//         compress_global_row_index_of_thread_level_block(compressed_template);
//         // compress_block_begin_thread_index_offset(compressed_template);
//     }

//     store_code_builder_data(builder);

//     // 生成代码
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));
// }

// mc2depi，使用带padding的ELL的方式处理
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/mc2depi.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);

//     gettimeofday(&pre_start, NULL);

//     // 按照32的倍数padding
//     if (matrix->dense_row_number % 32 != 0)
//     {
//         total_row_level_padding(op_manager, 32);
//     }

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 三个层次的分块都用最简单的
//     compressed_block_t* cur_block = matrix->block_coor_table.item_arr[0]->compressed_block_ptr;
//     // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//     unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[0]->min_dense_row_index;
//     unsigned long block_end_row_index = matrix->block_coor_table.item_arr[0]->max_dense_row_index;

//     unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//     // 行切分，一行一个块
//     vector<unsigned int> block_row_num;

//     block_row_num.push_back(block_row_size);
    
//     sep_tblock_level_row_csr(cur_block, block_row_num);

//     // 放弃warp的排序
//     vector<vector<unsigned int>> arr_of_row_block_size_arr;
//     vector<unsigned long> sep_block_id_arr;

//     sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//     // 一个线程一个非零元
//     vector<unsigned long> futher_thread_block_vec;
//     vector<unsigned long> futher_thread_col_block_size;

//     // 列分块
//     sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);

//     // 执行代码生成
//     code_builder_t* builder = init_code_builder(op_manager);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;
//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     // 使用warp压缩型的模板
//     {
//         // 第一个桶使用的kernal
//         // 两个子矩阵
//         direct_atom_template_warp_compress_t* bin0_template = init_direct_atom_template_warp_compress(builder, 0);

//         add_template_to_builder(builder, bin0_template, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 0);
        
//         try_all_compress(bin0_template);

//         // memory_garbage_manager_t* mem_manager = new memory_garbage_manager_t();
//         // // 析构一下
//         // delete_direct_atom_template(mem_manager, bin0_template);

//         // print_all_register_ptr(mem_manager);

//         // delete mem_manager;

//         // exit(-1);
//     }

//     // memory_garbage_manager_t* mem_manager = new memory_garbage_manager_t();

//     // delete_code_builder(mem_manager, builder);

//     // exit(-1);

//     store_code_builder_data(builder);

//     // 生成代码
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));
// }

// consph的调优，排序后可以sell，最大行长度达到了63，在sell下可能有并行度不够的风险
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/consph.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);

//     gettimeofday(&pre_start, NULL);

//     // 行条带的宽度
//     unsigned long row_block_size = 128;

//     // 将行号padding为特定的倍数
//     if (matrix->dense_row_number % row_block_size != 0)
//     {
//         total_row_level_padding(op_manager, row_block_size);
//     }

//     assert(matrix->dense_row_number % row_block_size == 0);

//     vector<unsigned long> bin_nnz_range;
    
//     for (unsigned long i = 0; i < 67; i = i + 1)
//     {
//         bin_nnz_range.push_back(i);
//     }

//     vector<unsigned long> bin_first_row_vec = total_dense_block_coarse_sort(op_manager, bin_nnz_range);

//     // 压缩
//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));
    
//     compressed_block_t* cur_block = matrix->block_coor_table.item_arr[0]->compressed_block_ptr;

//     vector<unsigned int> block_row_num;

//     // 按照一定宽度行分块
//     for (unsigned long i = 0; i < matrix->dense_row_number / row_block_size; i++)
//     {
//         block_row_num.push_back(row_block_size);
//     }

//     sep_tblock_level_row_csr(cur_block, block_row_num);

//     // 放弃warp的排序
//     vector<vector<unsigned int>> arr_of_row_block_size_arr;
//     vector<unsigned long> sep_block_id_arr;

//     sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//     // 一个线程一个非零元
//     vector<unsigned long> futher_thread_block_vec;
//     vector<unsigned long> futher_thread_col_block_size;

    
//     // 列分块
//     sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);

    

//     // exit(-1);

//     // 执行代码生成
//     code_builder_t* builder = init_code_builder(op_manager);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;
//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     // 使用warp压缩型的模板
//     {
//         // 第一个桶使用的kernal
//         // 两个子矩阵
//         shared_memory_template_warp_compress_t* bin0_template = init_shared_memory_template_warp_compress(builder, 0);

//         add_template_to_builder(builder, bin0_template, SHARED_MEMORY_TEMPLATE_WARP_COMPRESS, 0);

//         try_all_compress(bin0_template);
        
//         // 将block和warp层次的遍历全部去掉

//         // compressed_template->thread_num_in_block = row_block_size;
//         // compressed_template->tblock_num = compressed_template->size_of_block_nz_begin_offset - 1;

//         // // 压缩
//         // compress_global_row_index_of_thread_level_block(compressed_template);
//         // compress_block_begin_thread_index_offset(compressed_template);
//     }

//     store_code_builder_data(builder);

//     // 生成代码
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));
// }

// ins2的优化
// 矩阵分为三个部分，第一个部分美行1-2个非零元的超短行子块，第二个部分是一个数万个行非零元的子块，第三个块行非零元为68
// 非零元数量 // 547003  607772  375710 1530485
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/ins2.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);

//     gettimeofday(&pre_start, NULL);

//     // 最长行的列快宽度
//     unsigned long col_width_of_super_long_row = 22400;
//     unsigned long col_width_of_long_row = 4096;
//     // 以及线程块的线程数量
//     unsigned long tblock_size_of_super_long_row = 1024;
//     unsigned long tblock_size_of_long_row = 64;

//     // // 中等长度的行的，线程粒度的块的宽度
//     // unsigned long mid_row_thread_col_size = 2;
//     // // 中等长度的行归约的并行度
//     // unsigned long mid_row_reduce_thread_num = 32;
//     // // 中等长度的行行条带的宽度
//     // unsigned long mid_row_block_size = 32;
    
//     // 获得每一行的非零元数量
//     vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);

//     cout << "nnz_of_each_row.size():" << nnz_of_each_row.size() << endl;

//     unsigned long bin0_first_row = 0;
//     unsigned long bin1_first_row = 0;
//     unsigned long bin2_first_row = 0;
//     // 遍历所有的非零元，找出找个梯度的不同非零元数量所对应的首行范围
//     for (unsigned long i = 0; i < nnz_of_each_row.size(); i++)
//     {
//         unsigned long row_nnz = nnz_of_each_row[i];
//         // 第一次遇到的大于100的行非零元长度就是1号块的首行
//         if (row_nnz > 100 && bin1_first_row == 0)
//         {
//             bin1_first_row = i;
//         }
        
//         if (row_nnz == 68 && bin2_first_row == 0)
//         {
//             bin2_first_row = i;
//         }
//     }

//     cout << "bin1_first_row:" << bin1_first_row << ",bin2_first_row:" << bin2_first_row << endl;

//     // 这里执行一个padding，头部一个padding，尾部一个padding
//     // 第一个桶的行数量
//     unsigned long bin0_row_num = bin1_first_row;
//     unsigned long bin2_row_num = matrix->dense_row_number - bin2_first_row;

//     // 计算要被padding的行数量，先做头部的padding
//     if (bin0_row_num % 32 != 0)
//     {
//         unsigned long add_row_num = ((bin0_row_num / 32) + 1) * 32 - bin0_row_num;
//         unsigned long target_row_num = matrix->dense_row_number + add_row_num;
        
//         cout << "matrix->dense_row_number:" << matrix->dense_row_number << " target_row_num:" << target_row_num << endl;
//         cout << "bin0_row_num:" << bin0_row_num << endl;
//         // 在头部padding
//         total_row_level_padding_direct(op_manager, target_row_num, 1, TOP_PADDING);
//     }

//     // 计算padding
//     // if (bin2_row_num % 32 != 0)
//     // {
//     //     unsigned long add_row_num = ((bin2_row_num / 32) + 1) * 32 - bin2_row_num;
//     //     unsigned long target_row_num = matrix->dense_row_number + add_row_num;

//     //     // 在尾部padding
//     //     total_row_level_padding_direct(op_manager, target_row_num, 1);
//     // }

//     // 获得每一行的非零元数量
//     vector<unsigned long> new_nnz_of_each_row = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);

//     cout << "new_nnz_of_each_row.size():" << new_nnz_of_each_row.size() << endl;

//     bin0_first_row = 0;
//     bin1_first_row = 0;
//     bin2_first_row = 0;
//     // 重新找出新的桶分界线
//     // 遍历所有的非零元，找出找个梯度的不同非零元数量所对应的首行范围
//     for (unsigned long i = 0; i < new_nnz_of_each_row.size(); i++)
//     {
//         unsigned long row_nnz = new_nnz_of_each_row[i];
//         // 第一次遇到的大于100的行非零元长度就是1号块的首行
//         if (row_nnz > 100 && bin1_first_row == 0)
//         {
//             bin1_first_row = i;
//         }
        
//         if (row_nnz == 68 && bin2_first_row == 0)
//         {
//             bin2_first_row = i;
//         }
//     }

//     cout << "bin1_first_row:" << bin1_first_row << ",bin2_first_row:" << bin2_first_row << endl;
    
//     assert(bin1_first_row % 32 == 0);
//     // assert(bin1_first_row % 32 == 0 && (matrix->dense_row_number - bin2_first_row) % 32 == 0);

//     // 分成三个快
//     vector<unsigned long> block_begin_row;
//     block_begin_row.push_back(0);
//     block_begin_row.push_back(bin1_first_row);
//     block_begin_row.push_back(bin2_first_row);
//     block_begin_row.push_back(op_manager->matrix->dense_row_number);

//     var_len_row_div(op_manager->matrix, NULL, block_begin_row);

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));
    
//     // 头块是全局ELL，中间一块是长行的模板
//     {
//         unsigned long bin_index = 0;
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[bin_index]->compressed_block_ptr;
//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[bin_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[bin_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         // 行切分，一行一个块
//         vector<unsigned int> block_row_num;

//         block_row_num.push_back(block_row_size);
        
//         sep_tblock_level_row_csr(cur_block, block_row_num);

//         // 放弃warp的排序
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 一个线程一个非零元
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         // 列分块
//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     // 第二个子块，
//     {
//         unsigned long bin_index = 1;
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[bin_index]->compressed_block_ptr;
//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[bin_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[bin_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;
//         // 行切分，一行一个块
//         vector<unsigned int> block_row_num;

//         for (unsigned long i = 0; i < block_row_size; i++)
//         {
//             block_row_num.push_back(1);
//         }
        
//         sep_tblock_level_row_csr(cur_block, block_row_num);

//         // 这里执行一个列切分，遍历每一行，第一行执行一个
//         vector<unsigned long> sub_block_index_vec;
        
//         // 行号的进一步划分
//         vector<vector<unsigned int>> col_block_size_vec;

//         for (unsigned long i = 0; i < block_row_size; i++)
//         {
//             // 当前行的行号
//             unsigned long cur_row_index = block_begin_row_index + i;
//             // 当前行的非零元数量
//             unsigned long nnz_of_this_row = new_nnz_of_each_row[cur_row_index];

//             cout << "nnz_of_this_row:" << nnz_of_this_row << endl;

//             vector<unsigned int> cur_row_col_block_size_vec;

//             if (nnz_of_this_row > 30000)
//             {
//                 sub_block_index_vec.push_back(i);

//                 // 按照长行的方式处理
//                 for (unsigned long col_block_index = 0; col_block_index < nnz_of_this_row / col_width_of_super_long_row + 1; col_block_index++)
//                 {
//                     cur_row_col_block_size_vec.push_back(col_width_of_super_long_row);
//                 }

//                 col_block_size_vec.push_back(cur_row_col_block_size_vec);
//             }
//             else
//             {
//                 sub_block_index_vec.push_back(i);

//                 // 按照长行的方式处理
//                 for (unsigned long col_block_index = 0; col_block_index < nnz_of_this_row / col_width_of_long_row + 1; col_block_index++)
//                 {
//                     cur_row_col_block_size_vec.push_back(col_width_of_long_row);
//                 }

//                 col_block_size_vec.push_back(cur_row_col_block_size_vec);
//             }
//         }

//         sep_tblock_level_col_csr(cur_block, sub_block_index_vec, col_block_size_vec);

//         // 放弃warp的排序
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 一个线程一个非零元
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;
//         for (unsigned long j = 0; j < warp_block_num; j++)
//         {
//             futher_thread_block_vec.push_back(j);
//             futher_thread_col_block_size.push_back(1);
//         }

//         // 列分块
//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     // 最后一个桶使用一行一个线程块的处理
//     {
//         unsigned long bin_index = 2;
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[bin_index]->compressed_block_ptr;
//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[bin_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[bin_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         // 行切分，一行一个块
//         vector<unsigned int> block_row_num;

//         // // 行切分，32行一个block块
//         // while (block_row_size >= mid_row_block_size)
//         // {
//         //     block_row_num.push_back(mid_row_block_size);
//         //     block_row_size = block_row_size - mid_row_block_size;
//         // }

//         // if (block_row_size > 0)
//         // {
//         //     block_row_num.push_back(block_row_size);
//         // }

//         for (unsigned i = 0; i < block_row_size; i++)
//         {
//             block_row_num.push_back(1);
//         }
        
//         sep_tblock_level_row_csr(cur_block, block_row_num);

//         // 放弃warp的排序
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // warp的数量
//         unsigned long warp_num = cur_block->read_index[3]->block_num;

//         // 一个线程一个非零元
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         for (unsigned long i = 0; i < warp_num; i++)
//         {
//             futher_thread_block_vec.push_back(i);
//             futher_thread_col_block_size.push_back(1);
//         }

//         // 列分块
//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     set<template_type> type_vec = supported_template_of_sub_matrix(matrix, 2);

//     for (set<template_type>::iterator cur_temp_type_ptr = type_vec.begin(); cur_temp_type_ptr != type_vec.end(); cur_temp_type_ptr++)
//     {
//         cout << convert_template_type_to_string(*cur_temp_type_ptr) << endl;
//     }

//     exit(-1);

//     // 执行代码生成
//     code_builder_t* builder = init_code_builder(op_manager);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;
//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     // bin0和2是ELL
//     {
//         // 第一个桶使用的kernal
//         // 两个子矩阵
//         direct_atom_template_t* bin0_template = init_direct_atom_template(builder, 0);

//         add_template_to_builder(builder, bin0_template, DIRECT_ATOM_TEMPLATE, 0);
        
//         // 将block和warp层次的遍历全部去掉
//         compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS, 0);

//         direct_atom_template_warp_block_compress_t* compressed_template = (direct_atom_template_warp_block_compress_t*)builder->template_vec[0];

//         compressed_template->tblock_num = compressed_template->size_of_global_row_index_of_thread_level_block / (compressed_template->thread_num_in_block);

//         // 行号和块号相同
//         compress_global_row_index_of_thread_level_block(compressed_template);
//     }

//     // bin1是shared memory
//     {
//         shared_memory_long_row_template_t* bin1_template = init_shared_memory_long_row_template(builder, 1);

//         bin1_template->thread_num_in_block = tblock_size_of_super_long_row;
//         bin1_template->tblock_num = bin1_template->size_of_row_index_of_block_level_block;

//         add_template_to_builder(builder, bin1_template, SHARED_MEMORY_LONG_ROW_TEMPLATE, 1);

//         // 执行行索引压缩
//         // compress_row_index_of_block_level_block(bin1_template);
//     }

//     // bin0和2是ELL
//     {
//         // 第一个桶使用的kernal
//         // 两个子矩阵
//         shared_memory_long_row_template_t* bin2_template = init_shared_memory_long_row_template(builder, 2);

//         add_template_to_builder(builder, bin2_template, SHARED_MEMORY_LONG_ROW_TEMPLATE, 2);
        
//         // 将block和warp层次的遍历全部去掉
//         // compress_template_in_builder(builder, SHARED_MEMORY_TEMPLATE_WARP_COMPRESS, 2);

//         // shared_memory_template_warp_compress_t* compressed_template = (shared_memory_template_warp_compress_t*)builder->template_vec[2];
//         bin2_template->thread_num_in_block = tblock_size_of_long_row;
//         bin2_template->tblock_num = bin2_template->size_of_row_index_of_block_level_block;

//         // 行号和块号相同
//         compress_row_index_of_block_level_block(bin2_template);
//     }

//     // exit(-1);

//     store_code_builder_data(builder);

//     // 生成代码
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));

//     return 0;
// }

// 7200
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/in-2004.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);
    
//     gettimeofday(&pre_start, NULL);

//     unsigned long long_row_min_size = 128;
//     unsigned long row_num_of_small_row_block = 128;
//     // 列分块的最小宽度
//     unsigned long col_block_min_width = 8000;

//     // 不同
//     vector<unsigned long> row_nnz_range;
//     row_nnz_range.push_back(0);
//     row_nnz_range.push_back(long_row_min_size);
//     row_nnz_range.push_back(7754);

//     vector<unsigned long> row_num_of_each_range = get_row_num_of_each_row_nnz_range(op_manager, row_nnz_range);

//     for (unsigned long i = 0; i < row_num_of_each_range.size(); i++)
//     {
//         cout << row_num_of_each_range[i] << endl;
//     }

//     unsigned long row_num_of_short_row = row_num_of_each_range[0];
//     unsigned long row_num_of_long_row = row_num_of_each_range[1];

//     // 给短行做padding
//     if (row_num_of_short_row % row_num_of_small_row_block != 0)
//     {
//         row_num_of_short_row = (row_num_of_short_row / row_num_of_small_row_block + 1) * row_num_of_small_row_block;

//         // 执行一个padding
//         total_row_level_padding_direct(op_manager, row_num_of_short_row + row_num_of_long_row);
//     }

//     vector<unsigned long> bin_nnz_range;
    
//     for (unsigned long i = 0; i < 4701; i = i + 1)
//     {
//         bin_nnz_range.push_back(i);
//     }

//     vector<unsigned long> bin_first_row_vec = total_dense_block_coarse_sort(op_manager, bin_nnz_range);
    
//     vector<unsigned long> block_begin_row;
//     block_begin_row.push_back(0);
//     block_begin_row.push_back(row_num_of_long_row);
//     block_begin_row.push_back(row_num_of_short_row + row_num_of_long_row);

//     var_len_row_div(op_manager->matrix, NULL, block_begin_row);

//     vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);
    
//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 首先执行第一个桶的分块
//     {
//         unsigned long bin_index = 0;
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[bin_index]->compressed_block_ptr;

//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[bin_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[bin_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         // 行切分，一行一个块
//         vector<unsigned int> block_row_num;

//         // 准备60个2048的列块
//         for (unsigned long i = 0; i < block_row_size; i++)
//         {
//             block_row_num.push_back(1);
//         }
        
//         sep_tblock_level_row_csr(cur_block, block_row_num);

//         // 执行列分块，遍历每一行
//         // 行号
//         vector<unsigned long> sub_block_index_vec;
        
//         // 行号的进一步划分
//         vector<vector<unsigned int>> col_block_size_vec;

//         // 执行列分块，每个块一分为二
//         for (unsigned long j = 0; j < block_row_size; j++)
//         {
//             // cout << "col div row " << j << endl;
//             // 当前列分块的大小
//             vector<unsigned int> cur_row_col_block_size_vec;

//             // 当前行的行号
//             unsigned long cur_row_index = block_begin_row_index + j;
//             // 当前行的非零元数量
//             unsigned long row_nnz = nnz_of_each_row[cur_row_index];

//             // 只要行非零元数量，大于两倍列分块数量，就一直执行分块，小于两倍的列分块数量时，就剩余的部分分为一块
//             while (row_nnz >= 2 * col_block_min_width)
//             {
//                 cur_row_col_block_size_vec.push_back(col_block_min_width);
//                 row_nnz = row_nnz - col_block_min_width;
//             }

//             assert(row_nnz > 0);
//             cur_row_col_block_size_vec.push_back(row_nnz);

//             sub_block_index_vec.push_back(j);
//             col_block_size_vec.push_back(cur_row_col_block_size_vec);
//         }
        
//         cout << "begin col block" << endl;
//         sep_tblock_level_col_csr(cur_block, sub_block_index_vec, col_block_size_vec);

//         // 放弃warp的排序
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 一个线程一个非零元
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;
//         for (unsigned long j = 0; j < warp_block_num; j++)
//         {
//             futher_thread_block_vec.push_back(j);
//             futher_thread_col_block_size.push_back(1);
//         }

//         // 列分块
//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     {
//         // 第二个密集子块是SELL
//         unsigned long cur_index = 1;

//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[cur_index]->compressed_block_ptr;
//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[cur_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[cur_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         assert(block_row_size % row_num_of_small_row_block == 0);

//         unsigned long block_num = block_row_size / row_num_of_small_row_block;

//         vector<unsigned int> row_number_of_block_arr;
        
//         for (unsigned long i = 0; i < block_num; i++)
//         {
//             row_number_of_block_arr.push_back(row_num_of_small_row_block);
//         }

//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);

//         // 不使用warp层次的分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // thread一行一个
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     // 执行代码生成
//     code_builder_t* builder = init_code_builder(op_manager);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;
//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     // 第一个桶使用的kernal
//     // 两个子矩阵
//     shared_memory_long_row_template_t* bin1_template = init_shared_memory_long_row_template(builder, 0);

//     bin1_template->tblock_num = bin1_template->size_of_row_index_of_block_level_block;
//     bin1_template->thread_num_in_block = 256;

//     add_template_to_builder(builder, bin1_template, SHARED_MEMORY_LONG_ROW_TEMPLATE, 0);

//     // 执行行索引压缩
//     compress_row_index_of_block_level_block(bin1_template);
//     // SELL
//     direct_atom_template_t* new_bin2_template = init_direct_atom_template(builder, 1);

//     new_bin2_template->tblock_num = new_bin2_template->size_of_block_nz_begin_offset;
//     new_bin2_template->thread_num_in_block = row_num_of_small_row_block;

//     add_template_to_builder(builder, new_bin2_template, DIRECT_ATOM_TEMPLATE, 1);

//     // 进行一个压缩，将warp级别的内容去掉
//     compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 1);

//     direct_atom_template_warp_compress_t* compressed_template = (direct_atom_template_warp_compress_t*)builder->template_vec[1];

//     compress_global_row_index_of_thread_level_block(compressed_template);
    
//     compress_block_begin_thread_index_offset(compressed_template);

//     store_code_builder_data(builder);

//     // 生成代码
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));

//     return 0;
// }


// webbase-1M，两个桶，长行一个桶，短行一个桶，有一个分界线。
// 长行需要被进一步切分
// 短行特别短，行的数量非常多，所以排序的导致的查行索引的开销会很大
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/webbase-1M.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);
    
//     gettimeofday(&pre_start, NULL);

//     unsigned long long_row_min_size = 128;
//     unsigned long row_num_of_small_row_block = 256;
//     // 列分块的最小宽度
//     unsigned long col_block_min_width = 512;

//     // 不同
//     vector<unsigned long> row_nnz_range;
//     row_nnz_range.push_back(0);
//     row_nnz_range.push_back(long_row_min_size);
//     row_nnz_range.push_back(4701);

//     vector<unsigned long> row_num_of_each_range = get_row_num_of_each_row_nnz_range(op_manager, row_nnz_range);

//     for (unsigned long i = 0; i < row_num_of_each_range.size(); i++)
//     {
//         cout << row_num_of_each_range[i] << endl;
//     }

//     unsigned long row_num_of_short_row = row_num_of_each_range[0];
//     unsigned long row_num_of_long_row = row_num_of_each_range[1];

//     // 给短行做padding
//     if (row_num_of_short_row % row_num_of_small_row_block != 0)
//     {
//         row_num_of_short_row = (row_num_of_short_row / row_num_of_small_row_block + 1) * row_num_of_small_row_block;

//         // 执行一个padding
//         total_row_level_padding_direct(op_manager, row_num_of_short_row + row_num_of_long_row);
//     }

//     vector<unsigned long> bin_nnz_range;
    
//     for (unsigned long i = 0; i < 4701; i = i + 1)
//     {
//         bin_nnz_range.push_back(i);
//     }

//     vector<unsigned long> bin_first_row_vec = total_dense_block_coarse_sort(op_manager, bin_nnz_range);
    
//     vector<unsigned long> block_begin_row;
//     block_begin_row.push_back(0);
//     block_begin_row.push_back(row_num_of_long_row);
//     block_begin_row.push_back(row_num_of_short_row + row_num_of_long_row);

//     var_len_row_div(op_manager->matrix, NULL, block_begin_row);

//     vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);
    
//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 首先执行第一个桶的分块
//     {
//         unsigned long bin_index = 0;
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[bin_index]->compressed_block_ptr;

//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[bin_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[bin_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         // 行切分，一行一个块
//         vector<unsigned int> block_row_num;

//         // 准备60个2048的列块
//         for (unsigned long i = 0; i < block_row_size; i++)
//         {
//             block_row_num.push_back(1);
//         }
        
//         sep_tblock_level_row_csr(cur_block, block_row_num);

//         // 执行列分块，遍历每一行
//         // 行号
//         vector<unsigned long> sub_block_index_vec;
        
//         // 行号的进一步划分
//         vector<vector<unsigned int>> col_block_size_vec;

//         // 执行列分块，每个块一分为二
//         for (unsigned long j = 0; j < block_row_size; j++)
//         {
//             // cout << "col div row " << j << endl;
//             // 当前列分块的大小
//             vector<unsigned int> cur_row_col_block_size_vec;

//             // 当前行的行号
//             unsigned long cur_row_index = block_begin_row_index + j;
//             // 当前行的非零元数量
//             unsigned long row_nnz = nnz_of_each_row[cur_row_index];

//             // 只要行非零元数量，大于两倍列分块数量，就一直执行分块，小于两倍的列分块数量时，就剩余的部分分为一块
//             while (row_nnz >= 2 * col_block_min_width)
//             {
//                 cur_row_col_block_size_vec.push_back(col_block_min_width);
//                 row_nnz = row_nnz - col_block_min_width;
//             }

//             assert(row_nnz > 0);
//             cur_row_col_block_size_vec.push_back(row_nnz);

//             sub_block_index_vec.push_back(j);
//             col_block_size_vec.push_back(cur_row_col_block_size_vec);
//         }
        
//         cout << "begin col block" << endl;
//         sep_tblock_level_col_csr(cur_block, sub_block_index_vec, col_block_size_vec);

//         // 放弃warp的排序
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 一个线程一个非零元
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;

//         for (unsigned long j = 0; j < warp_block_num; j++)
//         {
//             futher_thread_block_vec.push_back(j);
//             futher_thread_col_block_size.push_back(1);
//         }

//         // 列分块
//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     {
//         // 第二个密集子块使用一个密集的列分块
//         unsigned long cur_index = 1;

//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[cur_index]->compressed_block_ptr;
//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[cur_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[cur_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         assert(block_row_size % row_num_of_small_row_block == 0);

//         unsigned long block_num = block_row_size / row_num_of_small_row_block;

//         vector<unsigned int> row_number_of_block_arr;
        
//         for (unsigned long i = 0; i < block_num; i++)
//         {
//             row_number_of_block_arr.push_back(row_num_of_small_row_block);
//         }

//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);

//         // 不使用warp层次的分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         unsigned long warp_block_num = matrix->block_coor_table.item_arr[cur_index]->compressed_block_ptr->read_index[3]->block_num;

//         // thread一行一个
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         for (unsigned long i = 0; i < warp_block_num; i++)
//         {
//             // 当前块的非零元数量
//             unsigned long block_nnz = read_from_array_with_data_type(cur_block->read_index[3]->coo_block_size_arr, cur_block->read_index[3]->data_type_of_coo_block_size_arr, i);

//             unsigned long TLB_size = (block_nnz / row_num_of_small_row_block) / 3;
//             // unsigned long block_nnz = cur_block->read_index[3]->coo_block_size_arr;
//             if (TLB_size == 0)
//             {
//                 TLB_size = 1;
//             }

//             futher_thread_block_vec.push_back(i);
//             futher_thread_col_block_size.push_back(TLB_size);
//         }
        
//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     // 执行代码生成
//     code_builder_t* builder = init_code_builder(op_manager);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;
//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     // 第一个桶使用的kernal
//     // 两个子矩阵
//     shared_memory_long_row_template_t* bin1_template = init_shared_memory_long_row_template(builder, 0);

//     bin1_template->tblock_num = bin1_template->size_of_row_index_of_block_level_block;
//     bin1_template->thread_num_in_block = 256;

//     add_template_to_builder(builder, bin1_template, SHARED_MEMORY_LONG_ROW_TEMPLATE, 0);

//     // 执行行索引压缩
//     compress_row_index_of_block_level_block(bin1_template);
//     // SELL
//     // direct_atom_template_t* new_bin2_template = init_direct_atom_template(builder, 1);

//     // new_bin2_template->tblock_num = new_bin2_template->size_of_block_nz_begin_offset;
//     // new_bin2_template->thread_num_in_block = row_num_of_small_row_block;

//     // add_template_to_builder(builder, new_bin2_template, DIRECT_ATOM_TEMPLATE, 1);

//     // // 进行一个压缩，将warp级别的内容去掉
//     // compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 1);

//     // direct_atom_template_warp_compress_t* compressed_template = (direct_atom_template_warp_compress_t*)builder->template_vec[1];

//     // compress_global_row_index_of_thread_level_block(compressed_template);
    
//     // compress_block_begin_thread_index_offset(compressed_template);

//     // CSR adptive
//     shared_memory_template_warp_compress_t* new_bin2_template = init_shared_memory_template_warp_compress(builder, 1);

//     new_bin2_template->tblock_num = new_bin2_template->size_of_block_nz_begin_offset;
//     new_bin2_template->thread_num_in_block = 1024;

//     add_template_to_builder(builder, new_bin2_template, SHARED_MEMORY_TEMPLATE_WARP_COMPRESS, 1);

//     try_all_compress(new_bin2_template);

//     store_code_builder_data(builder);

//     // 生成代码
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));

//     return 0;
// }

// scircuit的优化，按照分桶后上下两个部分不同的处理方式。因为长行的数量太少了，又不长不短的，所以最终还是sell的效果好。
// 对于atomic的矩阵来说，可以搞一个将block级别的遍历压缩掉，但是保留warp级别的遍历的模板，从而适应细粒度的sell
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/scircuit.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);

//     gettimeofday(&pre_start, NULL);

//     // 以64为分界线将矩阵分为两个部分
//     unsigned long long_row_min_size = 32;
//     unsigned long row_num_of_small_row_block = 128;

//     // 首先执行查看不同桶行数量
//     vector<unsigned long> row_nnz_range;
//     row_nnz_range.push_back(0);
//     row_nnz_range.push_back(long_row_min_size);
//     row_nnz_range.push_back(355);

//     vector<unsigned long> row_num_of_each_range = get_row_num_of_each_row_nnz_range(op_manager, row_nnz_range);

//     for (unsigned long i = 0; i < row_num_of_each_range.size(); i++)
//     {
//         cout << row_num_of_each_range[i] << endl;
//     }

//     // exit(-1);

//     unsigned long row_num_of_short_row = row_num_of_each_range[0];
//     unsigned long row_num_of_long_row = row_num_of_each_range[1];

//     // 给短行做padding
//     if (row_num_of_short_row % row_num_of_small_row_block != 0)
//     {
//         row_num_of_short_row = (row_num_of_short_row / row_num_of_small_row_block + 1) * row_num_of_small_row_block;

//         // 执行一个padding
//         total_row_level_padding_direct(op_manager, row_num_of_short_row + row_num_of_long_row);
//     }

//     vector<unsigned long> bin_nnz_range;
    
//     for (unsigned long i = 0; i < 353; i = i + 1)
//     {
//         bin_nnz_range.push_back(i);
//     }
    
//     vector<unsigned long> bin_first_row_vec = total_dense_block_coarse_sort(op_manager, bin_nnz_range);

//     vector<unsigned long> block_begin_row;
//     block_begin_row.push_back(0);
//     block_begin_row.push_back(row_num_of_long_row);
//     block_begin_row.push_back(row_num_of_short_row + row_num_of_long_row);

//     var_len_row_div(op_manager->matrix, NULL, block_begin_row);
    
//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // exit(-1);

//     {
//         unsigned long bin_index = 0;
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[bin_index]->compressed_block_ptr;

//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[bin_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[bin_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         // 行切分，一行一个块
//         vector<unsigned int> block_row_num;

//         // 准备60个2048的列块
//         for (unsigned long i = 0; i < block_row_size; i++)
//         {
//             block_row_num.push_back(1);
//         }
        
//         sep_tblock_level_row_csr(cur_block, block_row_num);

//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 根据线程块中的线程数量来进行线程粒度的分块，
//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;

//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         for (unsigned long j = 0; j < warp_block_num; j++)
//         {
//             futher_thread_block_vec.push_back(j);
//             futher_thread_col_block_size.push_back(1);
//         }

//         // 列分块
//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     {
//         // 第二个密集子块是SELL
//         unsigned long cur_index = 1;

//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[cur_index]->compressed_block_ptr;
//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[cur_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[cur_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         assert(block_row_size % row_num_of_small_row_block == 0);

//         unsigned long block_num = block_row_size / row_num_of_small_row_block;

//         vector<unsigned int> row_number_of_block_arr;
        
//         for (unsigned long i = 0; i < block_num; i++)
//         {
//             row_number_of_block_arr.push_back(row_num_of_small_row_block);
//         }

//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);

//         // 不使用warp层次的分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // thread一行一个
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     code_builder_t* builder = init_code_builder(op_manager);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;
//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     // 两个子矩阵
//     shared_memory_long_row_template_t* bin1_template = init_shared_memory_long_row_template(builder, 0);

//     bin1_template->tblock_num = bin1_template->size_of_row_index_of_block_level_block;
//     bin1_template->thread_num_in_block = 64;

//     add_template_to_builder(builder, bin1_template, SHARED_MEMORY_LONG_ROW_TEMPLATE, 0);

//     // SELL
//     direct_atom_template_t* new_bin2_template = init_direct_atom_template(builder, 1);

//     new_bin2_template->tblock_num = new_bin2_template->size_of_block_nz_begin_offset;
//     new_bin2_template->thread_num_in_block = row_num_of_small_row_block;

//     add_template_to_builder(builder, new_bin2_template, DIRECT_ATOM_TEMPLATE, 1);

//     // 进行一个压缩，将warp级别的内容去掉
//     compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 1);

//     direct_atom_template_warp_compress_t* compressed_bin2_template = (direct_atom_template_warp_compress_t*)builder->template_vec[1];

//     compress_global_row_index_of_thread_level_block(compressed_bin2_template);
//     compress_block_begin_thread_index_offset(compressed_bin2_template);

//     store_code_builder_data(builder);

//     // 生成代码
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));

//     return 0;
// }


// mac_econ_fwd500的梳理
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/mac_econ_fwd500.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);

//     gettimeofday(&pre_start, NULL);

//     // 分桶
//     // 首先执行查看不同桶行数量
//     // vector<unsigned long> row_nnz_range;

//     // for (unsigned long i = 0; i < 46; i++)
//     // {
//     //     row_nnz_range.push_back(i);
//     // }

//     // vector<unsigned long> row_num_of_each_range = get_row_num_of_each_row_nnz_range(op_manager, row_nnz_range);

//     // cout << row_num_of_each_range.size() << endl;

//     // for (unsigned long i = 0; i < row_num_of_each_range.size(); i++)
//     // {
//     //     cout << row_num_of_each_range[i] << endl;
//     // }

//     // exit(-1);

//     // 全局SELL，SELL的宽度的参数
//     unsigned long row_block_row_num = 128;
    
//     // 执行一个行方向的padding
//     total_row_level_padding(op_manager, row_block_row_num);

//     // 执行一个粗粒度排序
//     // 按照8个非零元来分桶
//     vector<unsigned long> bin_nnz_range;
    
//     for (unsigned long i = 0; i < 45; i = i + 1)
//     {
//         bin_nnz_range.push_back(i);
//     }
    
//     vector<unsigned long> bin_first_row_vec = total_dense_block_coarse_sort(op_manager, bin_nnz_range);

//     // 压缩成两个子矩阵，密集子块的分块会自动忽略空块
//     compress_dense_view(op_manager);

//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 按照1024执行行分块
//     unsigned long bin_index = 0;
//     compressed_block_t* cur_block = matrix->block_coor_table.item_arr[bin_index]->compressed_block_ptr;

//     // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//     unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[bin_index]->min_dense_row_index;
//     unsigned long block_end_row_index = matrix->block_coor_table.item_arr[bin_index]->max_dense_row_index;

//     unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;
    
//     assert(block_row_size % row_block_row_num == 0);

//     // 执行block级别的行分块
//     vector<unsigned int> block_row_num_arr;

//     for (unsigned long i = 0; i < block_row_size / row_block_row_num; i++)
//     {
//         block_row_num_arr.push_back(row_block_row_num);
//     }

//     sep_tblock_level_row_csr(cur_block, block_row_num_arr);

//     // 放弃warp层次的分块
//     vector<vector<unsigned int>> arr_of_row_block_size_arr;
//     vector<unsigned long> sep_block_id_arr;

//     sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//     // 根据线程块中的线程数量来进行线程粒度的分块，
//     unsigned long warp_block_num = cur_block->read_index[3]->block_num;
//     assert(warp_block_num == cur_block->read_index[2]->block_num);

//     vector<unsigned long> futher_thread_block_vec;
//     vector<unsigned long> futher_thread_col_block_size;

//     sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);

//     code_builder_t* builder = init_code_builder(op_manager);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;
//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     direct_atom_template_t* new_template = init_direct_atom_template(builder, 0);
//     new_template->tblock_num = new_template->size_of_block_nz_begin_offset;
//     new_template->thread_num_in_block = row_block_row_num;

//     add_template_to_builder(builder, new_template, DIRECT_ATOM_TEMPLATE, 0);

//     // 进行一个压缩，将warp级别的内容去掉
//     compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 0);

//     direct_atom_template_warp_compress_t* compressed_template = (direct_atom_template_warp_compress_t*)builder->template_vec[0];

//     compress_global_row_index_of_thread_level_block(compressed_template);
//     compress_block_begin_thread_index_offset(compressed_template);

//     // 数据
//     store_code_builder_data(builder);

//     // 生成代码
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));

//     return 0;
// }


// cant，37以上和37一下非零元数量的变化不一样
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/cant.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);

//     gettimeofday(&pre_start, NULL);

//     // 全局SELL，SELL的宽度的参数
//     unsigned long row_block_row_num = 1024;
    
//     // 执行一个行方向的padding
//     total_row_level_padding(op_manager, row_block_row_num);
//     // 压缩成两个子矩阵，密集子块的分块会自动忽略空块
//     compress_dense_view(op_manager);

//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 按照1024执行行分块
//     unsigned long bin_index = 0;
//     compressed_block_t* cur_block = matrix->block_coor_table.item_arr[bin_index]->compressed_block_ptr;

//     // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//     unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[bin_index]->min_dense_row_index;
//     unsigned long block_end_row_index = matrix->block_coor_table.item_arr[bin_index]->max_dense_row_index;

//     unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;
    
//     assert(block_row_size % row_block_row_num == 0);

//     // 执行block级别的行分块
//     vector<unsigned int> block_row_num_arr;

//     for (unsigned long i = 0; i < block_row_size / row_block_row_num; i++)
//     {
//         block_row_num_arr.push_back(row_block_row_num);
//     }

//     sep_tblock_level_row_csr(cur_block, block_row_num_arr);

//     // 放弃warp层次的分块
//     vector<vector<unsigned int>> arr_of_row_block_size_arr;
//     vector<unsigned long> sep_block_id_arr;

//     sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//     // 根据线程块中的线程数量来进行线程粒度的分块，
//     unsigned long warp_block_num = cur_block->read_index[3]->block_num;
//     assert(warp_block_num == cur_block->read_index[2]->block_num);

//     vector<unsigned long> futher_thread_block_vec;
//     vector<unsigned long> futher_thread_col_block_size;

//     sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);

//     code_builder_t* builder = init_code_builder(op_manager);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;
//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     direct_atom_template_t* new_template = init_direct_atom_template(builder, 0);
//     new_template->tblock_num = new_template->size_of_block_nz_begin_offset;
//     new_template->thread_num_in_block = row_block_row_num;

//     add_template_to_builder(builder, new_template, DIRECT_ATOM_TEMPLATE, 0);

//     // 进行一个压缩，将warp级别的内容去掉
//     compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 0);

//     direct_atom_template_warp_compress_t* compressed_template = (direct_atom_template_warp_compress_t*)builder->template_vec[0];

//     compress_global_row_index_of_thread_level_block(compressed_template);
//     compress_block_begin_thread_index_offset(compressed_template);

//     // 数据
//     store_code_builder_data(builder);

//     // 生成代码
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));

//     return 0;
// }


// 对transient进行处理，分成三个桶，长桶用
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/transient.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);
    
//     gettimeofday(&pre_start, NULL);

//     // 长行和短行的分界线
//     unsigned long long_row_min_size = 128;
//     unsigned long mid_row_min_size = 8;
    
//     // 长行的纵向划分长度
//     unsigned long long_row_col_size = 8192;
//     // SELL的宽度
//     unsigned long row_num_of_small_row_block = 512;
//     unsigned long row_num_of_mid_row_block = 64;
    
//     // 首先执行查看不同桶行数量
//     vector<unsigned long> row_nnz_range;
//     row_nnz_range.push_back(0);
//     row_nnz_range.push_back(mid_row_min_size);
//     row_nnz_range.push_back(long_row_min_size);
//     row_nnz_range.push_back(60424);

//     vector<unsigned long> row_num_of_each_range = get_row_num_of_each_row_nnz_range(op_manager, row_nnz_range);
    
//     for (unsigned long i = 0; i < row_num_of_each_range.size(); i++)
//     {
//         cout << row_num_of_each_range[i] << endl;
//     }

//     unsigned long row_num_of_short_row = row_num_of_each_range[0];
//     unsigned long row_num_of_mid_row = row_num_of_each_range[1];
//     unsigned long row_num_of_long_row = row_num_of_each_range[2];

//     // 给短行的块做一个padding
//     if (row_num_of_short_row % row_num_of_small_row_block != 0)
//     {
//         row_num_of_short_row = (row_num_of_short_row / row_num_of_small_row_block + 1) * row_num_of_small_row_block;

//         // 执行一个padding
//         total_row_level_padding_direct(op_manager, row_num_of_short_row + row_num_of_long_row + row_num_of_mid_row);
//     }

//     // 给中行的块做一个padding
//     if (row_num_of_mid_row % row_num_of_mid_row_block != 0)
//     {
//         row_num_of_mid_row = (row_num_of_mid_row / row_num_of_mid_row_block + 1) * row_num_of_mid_row_block;

//         // 执行一个padding
//         total_row_level_padding_direct(op_manager, row_num_of_short_row + row_num_of_long_row + row_num_of_mid_row, mid_row_min_size);
//     }

//     // cout << "op_manager->matrix->dense_row_number % row_num_of_block:" << op_manager->matrix->dense_row_number % row_num_of_block << endl;
//     // assert(op_manager->matrix->dense_row_number % row_num_of_block == row_num_of_long_row % row_num_of_block);
//     cout << "row_num_of_short_row:" << row_num_of_short_row << " row_num_of_mid_row:" << row_num_of_mid_row << " row_num_of_long_row:" << row_num_of_long_row << endl;

//     // exit(-1);
//     // 执行一个分桶
//     vector<unsigned long> bin_nnz_range;
    
//     for (unsigned long i = 0; i < 60424; i = i + 4)
//     {
//         bin_nnz_range.push_back(i);
//     }
    
//     vector<unsigned long> bin_first_row_vec = total_dense_block_coarse_sort(op_manager, bin_nnz_range);
    
//     // 分桶
//     // 行分块，掐头去尾，分成三个快
//     vector<unsigned long> block_begin_row;
//     block_begin_row.push_back(0);
//     block_begin_row.push_back(row_num_of_long_row);
//     block_begin_row.push_back(row_num_of_mid_row + row_num_of_long_row);
//     block_begin_row.push_back(row_num_of_short_row + row_num_of_mid_row + row_num_of_long_row);

//     var_len_row_div(op_manager->matrix, NULL, block_begin_row);

//     // 压缩成两个子矩阵，密集子块的分块会自动忽略空块
//     compress_dense_view(op_manager);

//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // exit(-1);

//     {
//         unsigned long bin_index = 0;
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[bin_index]->compressed_block_ptr;

//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[bin_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[bin_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         // 列切分，遍历两列
//         vector<unsigned int> block_size_arr;

//         // 准备60个2048的列块
//         for (unsigned long i = 0; i < 100; i++)
//         {
//             block_size_arr.push_back(long_row_col_size);
//         }
        
//         sep_tblock_level_col_csr(cur_block, block_size_arr);

//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 根据线程块中的线程数量来进行线程粒度的分块，
//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;

//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         for (unsigned long j = 0; j < warp_block_num; j++)
//         {
//             futher_thread_block_vec.push_back(j);
//             futher_thread_col_block_size.push_back(1);
//         }

//         // 列分块
//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     {
//         // 第二个密集子块是SELL
//         unsigned long cur_index = 1;

//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[cur_index]->compressed_block_ptr;
//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[cur_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[cur_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         assert(block_row_size % row_num_of_mid_row_block == 0);

//         unsigned long block_num = block_row_size / row_num_of_mid_row_block;

//         vector<unsigned int> row_number_of_block_arr;
        
//         for (unsigned long i = 0; i < block_num; i++)
//         {
//             row_number_of_block_arr.push_back(row_num_of_mid_row_block);
//         }

//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);

//         // 不使用warp层次的分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // thread一行一个
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     {
//         // 第二个密集子块是SELL
//         unsigned long cur_index = 2;

//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[cur_index]->compressed_block_ptr;
//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[cur_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[cur_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         assert(block_row_size % row_num_of_small_row_block == 0);

//         unsigned long block_num = block_row_size / row_num_of_small_row_block;

//         vector<unsigned int> row_number_of_block_arr;
        
//         for (unsigned long i = 0; i < block_num; i++)
//         {
//             row_number_of_block_arr.push_back(row_num_of_small_row_block);
//         }

//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);

//         // 不使用warp层次的分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // thread一行一个
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     code_builder_t* builder = init_code_builder(op_manager);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;
//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);
    
//     // exit(-1);

//     // 两个子矩阵
//     shared_memory_long_row_template_t* bin1_template = init_shared_memory_long_row_template(builder, 0);

//     bin1_template->tblock_num = bin1_template->size_of_row_index_of_block_level_block;
//     bin1_template->thread_num_in_block = long_row_min_size;
//     // bin1_template->thread_num_in_block = 128;

//     // compress_row_index_of_block_level_block(bin1_template);

//     add_template_to_builder(builder, bin1_template, SHARED_MEMORY_LONG_ROW_TEMPLATE, 0);

//     // SELL
//     direct_atom_template_t* new_bin2_template = init_direct_atom_template(builder, 1);

//     new_bin2_template->tblock_num = new_bin2_template->size_of_block_nz_begin_offset;
//     new_bin2_template->thread_num_in_block = row_num_of_mid_row_block;

//     add_template_to_builder(builder, new_bin2_template, DIRECT_ATOM_TEMPLATE, 1);

//     // 进行一个压缩，将warp级别的内容去掉
//     compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 1);

//     direct_atom_template_warp_compress_t* compressed_bin2_template = (direct_atom_template_warp_compress_t*)builder->template_vec[1];

//     compress_global_row_index_of_thread_level_block(compressed_bin2_template);
//     compress_block_begin_thread_index_offset(compressed_bin2_template);

//     // SELL
//     direct_atom_template_t* new_bin3_template = init_direct_atom_template(builder, 2);
//     new_bin3_template->tblock_num = new_bin3_template->size_of_block_nz_begin_offset;
//     new_bin3_template->thread_num_in_block = row_num_of_small_row_block;

//     add_template_to_builder(builder, new_bin3_template, DIRECT_ATOM_TEMPLATE, 2);

//     // 进行一个压缩，将warp级别的内容去掉
//     compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 2);

//     direct_atom_template_warp_compress_t* compressed_bin3_template = (direct_atom_template_warp_compress_t*)builder->template_vec[2];

//     compress_global_row_index_of_thread_level_block(compressed_bin3_template);
//     compress_block_begin_thread_index_offset(compressed_bin3_template);

//     cout << "new_nnz:" << op_manager->matrix->nnz << " origin_nnz:" << op_manager->matrix->origin_nnz << endl;

//     // exit(-1);
//     // 数据
//     store_code_builder_data(builder);

//     // 生成代码
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));

//     return 0;
// }

// 对eu-2005进行处理，还是使用分桶+sell的方式
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/eu-2005.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);
//     gettimeofday(&pre_start, NULL);

//     // 长行的分界线
//     unsigned long long_row_min_size = 256;

//     vector<unsigned long> row_nnz_range;
//     row_nnz_range.push_back(0);
//     row_nnz_range.push_back(1);
//     row_nnz_range.push_back(long_row_min_size);
//     row_nnz_range.push_back(10000);

//     // 短行的行条带宽度
//     unsigned long small_row_block_row_num = 256;

//     // 增加一个新的函数，计算不同范围的行数量
//     vector<unsigned long> row_num_of_each_range = get_row_num_of_each_row_nnz_range(op_manager, row_nnz_range);

//     // 1-256的行数，
//     unsigned long row_num_of_small_row = row_num_of_each_range[1];
//     unsigned long target_row_num_of_small_row = row_num_of_small_row;

//     if (row_num_of_small_row % small_row_block_row_num != 0)
//     {
//         target_row_num_of_small_row = (row_num_of_small_row / small_row_block_row_num + 1) * small_row_block_row_num;
//     }

//     if (target_row_num_of_small_row != row_num_of_small_row)
//     {
//         total_row_level_padding_direct(op_manager, row_num_of_each_range[0] + target_row_num_of_small_row + row_num_of_each_range[2]);
//     }

//     assert((op_manager->matrix->dense_row_number - row_num_of_each_range[0] - row_num_of_each_range[2]) % small_row_block_row_num == 0);

//     // 按照8个非零元来分桶
//     vector<unsigned long> bin_nnz_range;

//     bin_nnz_range.push_back(0);
    
//     for (unsigned long i = 1; i < 6985; i = i + 2)
//     {
//         bin_nnz_range.push_back(i);
//     }
    
//     vector<unsigned long> bin_first_row_vec = total_dense_block_coarse_sort(op_manager, bin_nnz_range);

//     // 行分块，掐头去尾，分成三个快
//     vector<unsigned long> block_begin_row;
//     block_begin_row.push_back(0);
//     block_begin_row.push_back(row_num_of_each_range[2]);
//     block_begin_row.push_back(row_num_of_each_range[2] + target_row_num_of_small_row);
//     assert(row_num_of_each_range[2] + target_row_num_of_small_row + row_num_of_each_range[0] == matrix->dense_row_number);
//     block_begin_row.push_back(row_num_of_each_range[2] + target_row_num_of_small_row + row_num_of_each_range[0]);

//     var_len_row_div(op_manager->matrix, NULL, block_begin_row);

//     // 压缩成两个子矩阵，密集子块的分块会自动忽略空块
//     compress_dense_view(op_manager);

//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     //针对第一个子块，一行一块
//     {
//         unsigned long cur_index = 0;

//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[cur_index]->compressed_block_ptr;
//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[cur_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[cur_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;
        
//         // block块的数量，用来处理分块的方法，一行一块
//         unsigned long row_num_of_each_block = 1;
//         vector<unsigned int> row_number_of_block_arr;
        
//         // 每个块一行
//         for (unsigned long j = 0; j < block_row_size; j++)
//         {
//             row_number_of_block_arr.push_back(1);
//         }

//         // 执行行分块
//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);
        
//         // 忽略warp分块
//         // 在warp层次不分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 根据线程块中的线程数量来进行线程粒度的分块，
//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;
//         assert(warp_block_num == row_number_of_block_arr.size());
        
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         assert(warp_block_num == block_row_size);

//         for (unsigned long j = 0; j < warp_block_num; j++)
//         {
//             futher_thread_block_vec.push_back(j);
//             futher_thread_col_block_size.push_back(1);
//         }

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     // 第二个块是行分块，按照一定宽度的行条带分块
//     {
//         unsigned long cur_index = 1;

//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[cur_index]->compressed_block_ptr;
//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[cur_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[cur_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         assert(block_row_size % small_row_block_row_num == 0);

//         unsigned long block_num = block_row_size / small_row_block_row_num;

//         vector<unsigned int> row_number_of_block_arr;
        
//         for (unsigned long i = 0; i < block_num; i++)
//         {
//             row_number_of_block_arr.push_back(small_row_block_row_num);
//         }

//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);

//         // 不使用warp层次的分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // thread一行一个
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     code_builder_t* builder = init_code_builder(op_manager);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;
//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     exit(-1);

//     // 两个子矩阵
//     shared_memory_long_row_template_t* bin1_template = init_shared_memory_long_row_template(builder, 0);

//     bin1_template->tblock_num = bin1_template->size_of_row_index_of_block_level_block;
//     bin1_template->thread_num_in_block = long_row_min_size;

//     compress_row_index_of_block_level_block(bin1_template);

//     add_template_to_builder(builder, bin1_template, SHARED_MEMORY_LONG_ROW_TEMPLATE, 0);

//     // SELL
//     direct_atom_template_t* new_bin2_template = init_direct_atom_template(builder, 1);

//     new_bin2_template->tblock_num = new_bin2_template->size_of_block_nz_begin_offset;
//     new_bin2_template->thread_num_in_block = small_row_block_row_num;

//     add_template_to_builder(builder, new_bin2_template, DIRECT_ATOM_TEMPLATE, 1);

//     // 进行一个压缩，将warp级别的内容去掉
//     compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 1);

//     direct_atom_template_warp_compress_t* compressed_bin2_template = (direct_atom_template_warp_compress_t*)builder->template_vec[1];

//     compress_global_row_index_of_thread_level_block(compressed_bin2_template);
//     compress_block_begin_thread_index_offset(compressed_bin2_template);

//     // 数据
//     store_code_builder_data(builder);

//     // 生成代码
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));

//     return 0;
// }




// 对dc2处理，先分桶，然后长行一行一个block，短行用SELL
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/dc2.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);
    
//     gettimeofday(&pre_start, NULL);

//     // 让行号减去两行之后成为32的倍数
//     unsigned long total_row_num = matrix->dense_row_number;
    
//     unsigned long padding_target;
//     // 最近的32的倍数
//     if (total_row_num % 32 == 2)
//     {
//         // 不用额外padding
//     }
//     else
//     {
//         // 最近的32余2的行数量
//         padding_target = (total_row_num / 32 + 1) * 32 + 2;
//     }

//     total_row_level_padding_direct(op_manager, padding_target);

//     assert(op_manager->matrix->dense_row_number % 32 == 2);

//     // 粗粒度排序
//     vector<unsigned long> bin_nnz_range;

//     for (unsigned long i = 0; i < 288; i = i + 2)
//     {
//         bin_nnz_range.push_back(i);
//     }

//     bin_nnz_range.push_back(300);

//     // 分桶
//     vector<unsigned long> bin_first_row_vec = total_dense_block_coarse_sort(op_manager, bin_nnz_range);

//     // 分行块
//     vector<unsigned long> block_begin_row;
//     block_begin_row.push_back(bin_first_row_vec[0]);
//     block_begin_row.push_back(bin_first_row_vec[1]);
//     block_begin_row.push_back(matrix->dense_row_number);

//     // 进行行切分，前两行进行一个切分，执行针对长行的操作，剩下的执行一个SELL
//     var_len_row_div(op_manager->matrix, NULL, block_begin_row);
    
//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 第一个bin一行中的block数量
//     {
//         // 用一个变量存储一个block的宽度
//         unsigned long block_col_size = 4096;
//         unsigned long bin_index = 0;
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[bin_index]->compressed_block_ptr;

//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[bin_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[bin_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         // vector<unsigned int> row_number_of_block_arr;
//         // // 一行一个block
//         // for (unsigned long i = 0; i < block_row_size; i++)
//         // {
//         //     row_number_of_block_arr.push_back(1);
//         // }

//         // sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);

//         // 列切分，遍历两列
//         vector<unsigned int> block_size_arr;

//         // 准备60个2048的列块
//         for (unsigned long i = 0; i < 60; i++)
//         {
//             block_size_arr.push_back(block_col_size);
//         }
        
//         sep_tblock_level_col_csr(cur_block, block_size_arr);

//         // 不使用warp层次的分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 根据线程块中的线程数量来进行线程粒度的分块，
//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;
//         // assert(warp_block_num == row_number_of_block_arr.size());
        
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         // assert(warp_block_num == block_row_size);

//         for (unsigned long j = 0; j < warp_block_num; j++)
//         {
//             futher_thread_block_vec.push_back(j);
//             futher_thread_col_block_size.push_back(1);
//         }

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     // cout << "1" << endl;

//     // 第二个桶行条带的宽度
//     unsigned long bin_2_row_size = 128;

//     {
//         unsigned long bin_index = 1;
        
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[bin_index]->compressed_block_ptr;

//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[bin_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[bin_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         assert(block_row_size % bin_2_row_size == 0);

//         unsigned long block_num = block_row_size / bin_2_row_size;

//         vector<unsigned int> row_number_of_block_arr;
        
//         for (unsigned long i = 0; i < block_num; i++)
//         {
//             row_number_of_block_arr.push_back(bin_2_row_size);
//         }

//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);

//         // 不使用warp层次的分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // thread一行一个
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     // 第一个是长行，第二个是SELL
//     code_builder_t* builder = init_code_builder(op_manager);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;

//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     shared_memory_long_row_template_t* bin1_template = init_shared_memory_long_row_template(builder, 0);

//     bin1_template->tblock_num = bin1_template->size_of_row_index_of_block_level_block;
//     bin1_template->thread_num_in_block = 512;

//     add_template_to_builder(builder, bin1_template, SHARED_MEMORY_LONG_ROW_TEMPLATE, 0);

//     // ELL用原子加的方式
//     direct_atom_template_t* new_bin2_template = init_direct_atom_template(builder, 1);

//     new_bin2_template->tblock_num = new_bin2_template->size_of_block_nz_begin_offset;
//     new_bin2_template->thread_num_in_block = bin_2_row_size;

//     add_template_to_builder(builder, new_bin2_template, DIRECT_ATOM_TEMPLATE, 1);

//     // 进行一个压缩，将warp级别的内容去掉
//     compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 1);

//     direct_atom_template_warp_compress_t* compressed_bin2_template = (direct_atom_template_warp_compress_t*)builder->template_vec[1];

//     compress_global_row_index_of_thread_level_block(compressed_bin2_template);
//     compress_block_begin_thread_index_offset(compressed_bin2_template);

//     // 写数据，然后创造模板
//     store_code_builder_data(builder);

//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));

//     return 0;
// }


// 对sls生成ELL格式
// int main()
// {
//     sparse_struct_t *matrix_struct = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/sls.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix_struct);
//     gettimeofday(&pre_start, NULL);
    
//     // 全局32倍数的padding
//     total_row_level_padding(op_manager, 32);

//     // 全局ELL
//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 执行分块操作，将整个矩阵就分成一个block块和一个warp块
//     unsigned long min_row_index = op_manager->matrix->block_coor_table.item_arr[0]->min_dense_row_index;
//     unsigned long max_row_index = op_manager->matrix->block_coor_table.item_arr[0]->max_dense_row_index;

//     unsigned long block_row_size = max_row_index - min_row_index + 1;

//     vector<unsigned int> row_number_of_block_arr;
//     row_number_of_block_arr.push_back(block_row_size);
//     sep_tblock_level_row_csr(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr, row_number_of_block_arr);

//     // warp级别的行分块，这个块完全变成空的，放弃分块
//     vector<vector<unsigned int>> arr_of_row_block_size_arr;
//     vector<unsigned long> sep_block_id_arr;

//     sep_warp_level_row_csr(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr, sep_block_id_arr, arr_of_row_block_size_arr);

//     // 线程粒度按行分块
//     unsigned long warp_level_block_num = op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[3]->block_num;
//     vector<unsigned long> futher_thread_block_vec;
//     vector<unsigned long> futher_thread_col_block_size;

//     sep_thread_level_col_ell_with_padding(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr, futher_thread_block_vec, futher_thread_col_block_size);

//     // 创建代码生成器
//     code_builder_t* builder = init_code_builder(op_manager);

//     direct_atom_template_t* new_template = init_direct_atom_template(builder, 0);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;

//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     add_template_to_builder(builder, new_template, DIRECT_ATOM_TEMPLATE, 0);

//     compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS, 0);

//     direct_atom_template_warp_block_compress_t* compressed_template = (direct_atom_template_warp_block_compress_t*)builder->template_vec[0];

//     compress_global_row_index_of_thread_level_block(compressed_template);

//     store_code_builder_data(builder);

//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));

//     return 0;
// }


// 对Hardesty2生成ELL格式
// int main()
// {
//     sparse_struct_t *matrix_struct = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/mid_size_matrix/Hardesty2.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix_struct);
    
//     gettimeofday(&pre_start, NULL);
    
//     // 进行全局padding，使得行的数量是32的倍数，从而保证
//     total_row_level_padding(op_manager, 32);
    
//     // 将数据分为两个桶，0-3 3-6 0\3\6
//     // vector<unsigned long> bin_nnz_range;

//     // bin_nnz_range.push_back(0);
//     // bin_nnz_range.push_back(3);
//     // bin_nnz_range.push_back(6);

//     // // 这里做一个全局排序的处理
//     // total_dense_block_coarse_sort(op_manager, bin_nnz_range);

//     // print_arr_to_file_with_data_type(op_manager->matrix->sorted_row_index, op_manager->matrix->data_type_of_sorted_row_index, op_manager->matrix->dense_row_number, "/home/duzhen/spmv_builder/data_source/test_result_3");
//     // print_arr_to_file_with_data_type(op_manager->matrix->coo_row_index_cache, UNSIGNED_LONG, op_manager->matrix->nnz, "/home/duzhen/spmv_builder/data_source/test_result_4");


//     // 在稠密矩阵上不做切分，直接压缩
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));
//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 执行分块操作，将整个矩阵就分成一个block块和一个warp块
//     unsigned long min_row_index = op_manager->matrix->block_coor_table.item_arr[0]->min_dense_row_index;
//     unsigned long max_row_index = op_manager->matrix->block_coor_table.item_arr[0]->max_dense_row_index;

//     unsigned long block_row_size = max_row_index - min_row_index + 1;

//     vector<unsigned int> row_number_of_block_arr;
//     row_number_of_block_arr.push_back(block_row_size);
//     sep_tblock_level_row_csr(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr, row_number_of_block_arr);

//     // warp级别的行分块，这个块完全变成空的，放弃分块
//     vector<vector<unsigned int>> arr_of_row_block_size_arr;
//     vector<unsigned long> sep_block_id_arr;

//     sep_warp_level_row_csr(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr, sep_block_id_arr, arr_of_row_block_size_arr);

//     // 线程粒度按行分块
//     unsigned long warp_level_block_num = op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[3]->block_num;
//     vector<unsigned long> futher_thread_block_vec;
//     vector<unsigned long> futher_thread_col_block_size;

//     sep_thread_level_col_ell_with_padding(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr, futher_thread_block_vec, futher_thread_col_block_size);

//     // 创建代码生成器
//     code_builder_t* builder = init_code_builder(op_manager);

//     direct_atom_template_t* new_template = init_direct_atom_template(builder, 0);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;

//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     add_template_to_builder(builder, new_template, DIRECT_ATOM_TEMPLATE, 0);

//     compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS, 0);

//     direct_atom_template_warp_block_compress_t* compressed_template = (direct_atom_template_warp_block_compress_t*)builder->template_vec[0];

//     compress_global_row_index_of_thread_level_block(compressed_template);

//     store_code_builder_data(builder);

//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));

//     return 0;
// }

// 处理pdb1HYS
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/pdb1HYS.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);

//     gettimeofday(&pre_start, NULL);

//     unsigned long row_num_of_each_block = 64;

//     // 每个block占用256行,并且按照最长行的长度padding。在头部加上行，
//     total_row_level_padding(op_manager, row_num_of_each_block, TOP_PADDING);

//     // 直接压缩
//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 分块
//     unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[0]->min_dense_row_index;
//     unsigned long block_end_row_index = matrix->block_coor_table.item_arr[0]->max_dense_row_index;

//     assert((block_end_row_index - block_begin_row_index + 1) % row_num_of_each_block == 0);
//     unsigned long block_row_size = (block_end_row_index - block_begin_row_index + 1) / row_num_of_each_block;

//     // block块的数量，用来处理分块的方法，一行一块
//     vector<unsigned int> row_number_of_block_arr;

//     // 每个块一行
//     for (unsigned long j = 0; j < block_row_size; j++)
//     {
//         row_number_of_block_arr.push_back(row_num_of_each_block);
//     }

//     sep_tblock_level_row_csr(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, row_number_of_block_arr);

//     // warp不分块
//     vector<vector<unsigned int>> arr_of_row_block_size_arr;
//     vector<unsigned long> sep_block_id_arr;

//     sep_warp_level_row_csr(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, sep_block_id_arr, arr_of_row_block_size_arr);

//     // thread不分块，默认一行一个thread
//     vector<unsigned long> futher_thread_block_vec;
//     vector<unsigned long> futher_thread_col_block_size;
//     sep_thread_level_col_ell_with_padding(matrix->block_coor_table.item_arr[0]->compressed_block_ptr, futher_thread_block_vec, futher_thread_col_block_size);

//     // 用原子加的方式处理
//     code_builder_t *builder = init_code_builder(op_manager);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;

//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     direct_atom_template_t* old_template = init_direct_atom_template(builder, 0);
//     add_template_to_builder(builder, old_template, DIRECT_ATOM_TEMPLATE, 0);

//     // 将warp层次压缩
//     compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 0);
//     direct_atom_template_warp_compress_t* new_template = (direct_atom_template_warp_compress_t*)builder->template_vec[0];

//     compress_global_row_index_of_thread_level_block(new_template);
//     compress_block_begin_thread_index_offset(new_template);

//     new_template->tblock_num = new_template->size_of_thread_block_size_in_block;
//     new_template->thread_num_in_block = row_num_of_each_block;

//     store_code_builder_data(builder);

//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));
// }





// 生成一个合适的格式处理rail4284，性能很好的版本
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/rail4284.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);
    
//     gettimeofday(&pre_start, NULL);
//     // 分桶
//     vector<unsigned long> bin_nnz_range;

//     bin_nnz_range.push_back(0);
//     bin_nnz_range.push_back(32);
//     bin_nnz_range.push_back(256);
//     bin_nnz_range.push_back(2048);

//     // // 分桶
//     vector<unsigned long> bin_first_row_vec = total_dense_block_coarse_sort(op_manager, bin_nnz_range);

//     // // 获取每一行的非零元数量
//     // // 获取矩阵中每一行的非零元数量
//     vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);


//     // 先进行稠密行分块
//     var_len_row_div(op_manager->matrix, NULL, bin_first_row_vec);

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 对于第一个块特殊处理
//     for (unsigned long i = 0; i < 2; i++)
//     {
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[i]->compressed_block_ptr;
//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[i]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[i]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;
        
//         // block块的数量，用来处理分块的方法，一行一块
//         unsigned long row_num_of_each_block = 1;
//         vector<unsigned int> row_number_of_block_arr;
        
//         // 每个块一行
//         for (unsigned long j = 0; j < block_row_size; j++)
//         {
//             row_number_of_block_arr.push_back(1);
//         }

//         // 执行行分块
//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);
        
//         // 忽略warp分块
//         // 在warp层次不分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 根据线程块中的线程数量来进行线程粒度的分块，
//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;
//         assert(warp_block_num == row_number_of_block_arr.size());
        
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         assert(warp_block_num == block_row_size);

//         for (unsigned long j = 0; j < warp_block_num; j++)
//         {
//             futher_thread_block_vec.push_back(j);
//             futher_thread_col_block_size.push_back(1);
//         }

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     {
//         unsigned long sub_matrix = 2;
        
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[sub_matrix]->compressed_block_ptr;

//         // 三号子矩阵，使用1024个线程，每行使用32个线程执行规约，那么一个block最好负责32行的内容
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[sub_matrix]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[sub_matrix]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         vector<unsigned int> row_number_of_block_arr;
//         // 非零元数量不能超过2048，遍历所有行，并执行一个行分块
//         unsigned long acc_block_nz_num = 0;

//         // 累计的行数量
//         unsigned long acc_block_row_num = 0;

//         for (unsigned long i = block_begin_row_index; i <= block_end_row_index; i++)
//         {
//             unsigned long cur_row_nz_num = nnz_of_each_row[i];
//             // 当前行块的非零元数量不能超过2048
//             if (acc_block_nz_num + cur_row_nz_num < 2048)
//             {
//                 acc_block_nz_num = acc_block_nz_num + cur_row_nz_num;
//                 acc_block_row_num++;
//             }
//             else
//             {
//                 // 超过2048就将已有的写入
//                 assert(acc_block_row_num > 0);
//                 row_number_of_block_arr.push_back(acc_block_row_num);
//                 acc_block_nz_num = cur_row_nz_num;
//                 acc_block_row_num = 1;
//             }
//         }

//         // 如果还有剩余的，要加到最后
//         assert(acc_block_row_num > 0);
    
//         row_number_of_block_arr.push_back(acc_block_row_num);

//         // 执行行分块
//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);
        
//         // 放弃warp分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 每个线程一个非零元
//         // 线程层次的分块，一个块一个
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         // 遍历所有warp层次的块
//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;
//         assert(warp_block_num == row_number_of_block_arr.size());

        
//         // 看看现成内是不是有归约的空间
//         for (unsigned long i = 0; i < warp_block_num; i++)
//         {
//             futher_thread_block_vec.push_back(i);
//             futher_thread_col_block_size.push_back(1);
//         }

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     // 最后一个块，只有228行，3321个非零元，使用一定数量的block执行处理，使用最普通的CSR，adptive，一个block128行
//     {
//         unsigned long sub_matrix = 3;
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[sub_matrix]->compressed_block_ptr;

//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[sub_matrix]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[sub_matrix]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         // 行条带的划分
//         unsigned long row_num_of_each_block = 128;

//         vector<unsigned int> row_number_of_block_arr;

//         while (block_row_size > row_num_of_each_block)
//         {
//             row_number_of_block_arr.push_back(row_num_of_each_block);
//             block_row_size = block_row_size - row_num_of_each_block;
//         }

//         if (block_row_size > 0)
//         {
//             row_number_of_block_arr.push_back(block_row_size);
//         }

//         // 执行行分块
//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);
        
//         // 放弃warp分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 每个线程一个非零元
//         // 线程层次的分块，一个块一个
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         // 遍历所有warp层次的块
//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;
//         assert(warp_block_num == row_number_of_block_arr.size());

//         for (unsigned long i = 0; i < warp_block_num; i++)
//         {
//             futher_thread_block_vec.push_back(i);
//             futher_thread_col_block_size.push_back(1);
//         }

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     code_builder_t *builder = init_code_builder(op_manager);

//     gettimeofday(&pre_end, NULL);

//     double pre_timeuse = 1000000 * (pre_end.tv_sec - pre_start.tv_sec) + pre_end.tv_usec - pre_start.tv_usec;

//     printf("pre_process_time=%fms\n", pre_timeuse / 1000.0);

//     exit(-1);

//     for (int i = 0; i < 2; i++)
//     {
//         shared_memory_long_row_template_t* new_template = init_shared_memory_long_row_template(builder, i);
//         add_template_to_builder(builder, new_template, SHARED_MEMORY_LONG_ROW_TEMPLATE, i);
        
//         // 除了线程数量不一样，其他都一样
//         if (i == 0)
//         {
//             new_template->thread_num_in_block = 1024;
//         }

//         if (i == 1)
//         {
//             new_template->thread_num_in_block = 256;
//         }

//         if (i == 2)
//         {
//             new_template->thread_num_in_block = 256;
//         }
//         if (i == 3)
//         {
//             new_template->thread_num_in_block = 128;
//         }

//         if (i == 4)
//         {
//             new_template->thread_num_in_block = 64;
//         }

//         new_template->tblock_num = matrix->block_coor_table.item_arr[i]->compressed_block_ptr->read_index[3]->block_num;

//         // 进行压缩，主要是行号的压缩
//         compress_row_index_of_block_level_block(new_template);
//     }

//     // 所有的都用共享内存的初始化
//     for (int i = 2; i < 4; i++)
//     {
//         shared_memory_template_t *old_template = init_shared_memory_template(builder, i);

//         add_template_to_builder(builder, old_template, SHARED_MEMORY_TEMPLATE, i);
//         // 所有的都压缩
//         compress_template_in_builder(builder, SHARED_MEMORY_TEMPLATE_WARP_COMPRESS, i);

//         shared_memory_template_warp_compress_t* new_template = (shared_memory_template_warp_compress_t *)builder->template_vec[i];

//         // 声明计算资源
//         if (i == 0)
//         {
//             new_template->thread_num_in_block = 1024;
//             new_template->tblock_num = matrix->block_coor_table.item_arr[i]->compressed_block_ptr->read_index[3]->block_num;
//             set_row_reduce_thread_num(new_template, get_config()["MAX_ROW_REDUCE_THREAD"].as_integer());

//             // 进行压缩，主要是第一个非零元索引的压缩
//             compress_block_first_row_index(new_template);
//             compress_block_begin_thread_index_offset(new_template);
//             compress_row_offset_in_thread_tmp_result(new_template);
//         }
//         else if (i == 1)
//         {
//             new_template->thread_num_in_block = 512;
//             new_template->tblock_num = matrix->block_coor_table.item_arr[i]->compressed_block_ptr->read_index[3]->block_num;
//             set_row_reduce_thread_num(new_template, get_config()["MAX_ROW_REDUCE_THREAD"].as_integer());
//             compress_block_first_row_index(new_template);
//             compress_block_begin_thread_index_offset(new_template);
//         }
//         else if (i == 2)
//         {
//             // print_dense_block_table(&(op_manager->matrix->block_coor_table));
//             new_template->thread_num_in_block = 1024;
//             new_template->tblock_num = matrix->block_coor_table.item_arr[i]->compressed_block_ptr->read_index[3]->block_num;
//             set_row_reduce_thread_num(new_template, 32);
//             compress_thread_block_size_in_block(new_template);
//         }
//         else if (i == 3)
//         {
//             new_template->thread_num_in_block = 1024;
//             new_template->tblock_num = matrix->block_coor_table.item_arr[i]->compressed_block_ptr->read_index[3]->block_num;
//             set_row_reduce_thread_num(new_template, 1);
//             compress_thread_block_size_in_block(new_template);
//         }
//     }

//     store_code_builder_data(builder);
    

//     // 生成对应的模板
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));
// }

// 对于长行来说，一行有两个block的
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/rail4284.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);
    
//     // 分桶
//     vector<unsigned long> bin_nnz_range;

//     bin_nnz_range.push_back(0);
//     bin_nnz_range.push_back(32);
//     bin_nnz_range.push_back(256);
//     bin_nnz_range.push_back(8192);

//     vector<unsigned long> bin_first_row_vec = total_dense_block_coarse_sort(op_manager, bin_nnz_range);

//     vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);

//     var_len_row_div(op_manager->matrix, NULL, bin_first_row_vec);

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 对于第一个块来说，两个线程块处理一行
//     {
//         unsigned long bin_index = 0;
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[bin_index]->compressed_block_ptr;

//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[bin_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[bin_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         // 执行一个列分块，保证一行两个分块，先按照一行一块执行一个切分，
//         // block块的数量，用来处理分块的方法，一行一块
//         unsigned long row_num_of_each_block = 1;
//         vector<unsigned int> row_number_of_block_arr;
        
//         // 每个块一行
//         for (unsigned long j = 0; j < block_row_size; j++)
//         {
//             row_number_of_block_arr.push_back(1);
//         }

//         // 执行行分块
//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);

//         // cout << 1 << endl;
//         assert(cur_block->read_index[2]->block_num == block_row_size);

//         // 行号
//         vector<unsigned long> sub_block_index_vec;
        
//         // 行号的进一步划分
//         vector<vector<unsigned int>> col_block_size_vec;

//         // 执行列分块，每个块一分为二
//         for (unsigned long j = 0; j < block_row_size; j++)
//         {
//             // cout << "col div row " << j << endl;
//             // 当前列分块的大小
//             vector<unsigned int> cur_row_col_block_size_vec;

//             // 当前行的行号
//             unsigned long cur_row_index = block_begin_row_index + j;
//             // 当前行的非零元数量
//             unsigned long row_nnz = nnz_of_each_row[cur_row_index];

//             // 分成两个块
//             unsigned int nnz_of_first_col = row_nnz / 2;

//             cur_row_col_block_size_vec.push_back(nnz_of_first_col);
//             cur_row_col_block_size_vec.push_back(row_nnz - nnz_of_first_col);

//             sub_block_index_vec.push_back(j);
//             col_block_size_vec.push_back(cur_row_col_block_size_vec);

            
//         }
        
//         cout << "begin col block" << endl;
//         sep_tblock_level_col_csr(cur_block, sub_block_index_vec, col_block_size_vec);

//         assert(cur_block->read_index[2]->block_num == block_row_size * 2);

//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 根据线程块中的线程数量来进行线程粒度的分块，
//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;
//         assert(warp_block_num == row_number_of_block_arr.size() * 2);

//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         assert(warp_block_num == block_row_size * 2);

//         for (unsigned long j = 0; j < warp_block_num; j++)
//         {
//             futher_thread_block_vec.push_back(j);
//             futher_thread_col_block_size.push_back(1);
//         }

//         // 列分块
//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     // 第二个块一行一个
//     {
//         unsigned long bin_index = 1;
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[bin_index]->compressed_block_ptr;
//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[bin_index]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[bin_index]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;
        
//         // block块的数量，用来处理分块的方法，一行一块
//         unsigned long row_num_of_each_block = 1;
//         vector<unsigned int> row_number_of_block_arr;
        
//         // 每个块一行
//         for (unsigned long j = 0; j < block_row_size; j++)
//         {
//             row_number_of_block_arr.push_back(1);
//         }

//         // 执行行分块
//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);
        
//         // 忽略warp分块
//         // 在warp层次不分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 根据线程块中的线程数量来进行线程粒度的分块，
//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;
//         assert(warp_block_num == row_number_of_block_arr.size());
        
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         assert(warp_block_num == block_row_size);

//         for (unsigned long j = 0; j < warp_block_num; j++)
//         {
//             futher_thread_block_vec.push_back(j);
//             futher_thread_col_block_size.push_back(1);
//         }

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     {
//         unsigned long sub_matrix = 2;
        
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[sub_matrix]->compressed_block_ptr;

//         // 三号子矩阵，使用1024个线程，每行使用32个线程执行规约，那么一个block最好负责32行的内容
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[sub_matrix]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[sub_matrix]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         vector<unsigned int> row_number_of_block_arr;
//         // 非零元数量不能超过2048，遍历所有行，并执行一个行分块
//         unsigned long acc_block_nz_num = 0;

//         // 累计的行数量
//         unsigned long acc_block_row_num = 0;

//         for (unsigned long i = block_begin_row_index; i <= block_end_row_index; i++)
//         {
//             unsigned long cur_row_nz_num = nnz_of_each_row[i];
//             // 当前行块的非零元数量不能超过2048
//             if (acc_block_nz_num + cur_row_nz_num < 2048)
//             {
//                 acc_block_nz_num = acc_block_nz_num + cur_row_nz_num;
//                 acc_block_row_num++;
//             }
//             else
//             {
//                 // 超过2048就将已有的写入
//                 assert(acc_block_row_num > 0);
//                 row_number_of_block_arr.push_back(acc_block_row_num);
//                 acc_block_nz_num = cur_row_nz_num;
//                 acc_block_row_num = 1;
//             }
//         }

//         // 如果还有剩余的，要加到最后
//         assert(acc_block_row_num > 0);
    
//         row_number_of_block_arr.push_back(acc_block_row_num);

//         // 执行行分块
//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);
        
//         // 放弃warp分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 每个线程一个非零元
//         // 线程层次的分块，一个块一个
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         // 遍历所有warp层次的块
//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;
//         assert(warp_block_num == row_number_of_block_arr.size());

        
//         // 看看现成内是不是有归约的空间
//         for (unsigned long i = 0; i < warp_block_num; i++)
//         {
//             futher_thread_block_vec.push_back(i);
//             futher_thread_col_block_size.push_back(1);
//         }

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     // 最后一个块，只有228行，3321个非零元，使用一定数量的block执行处理，使用最普通的CSR，adptive，一个block128行
//     {
//         unsigned long sub_matrix = 3;
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[sub_matrix]->compressed_block_ptr;

//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[sub_matrix]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[sub_matrix]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         // 行条带的划分
//         unsigned long row_num_of_each_block = 128;

//         vector<unsigned int> row_number_of_block_arr;

//         while (block_row_size > row_num_of_each_block)
//         {
//             row_number_of_block_arr.push_back(row_num_of_each_block);
//             block_row_size = block_row_size - row_num_of_each_block;
//         }

//         if (block_row_size > 0)
//         {
//             row_number_of_block_arr.push_back(block_row_size);
//         }

//         // 执行行分块
//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);
        
//         // 放弃warp分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 每个线程一个非零元
//         // 线程层次的分块，一个块一个
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         // 遍历所有warp层次的块
//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;
//         assert(warp_block_num == row_number_of_block_arr.size());

//         for (unsigned long i = 0; i < warp_block_num; i++)
//         {
//             futher_thread_block_vec.push_back(i);
//             futher_thread_col_block_size.push_back(1);
//         }

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     code_builder_t *builder = init_code_builder(op_manager);
    
//     for (int i = 0; i < 2; i++)
//     {
//         shared_memory_long_row_template_t* new_template = init_shared_memory_long_row_template(builder, i);
//         add_template_to_builder(builder, new_template, SHARED_MEMORY_LONG_ROW_TEMPLATE, i);
        
//         // 除了线程数量不一样，其他都一样
//         if (i == 0)
//         {
//             new_template->thread_num_in_block = 1024;
//         }

//         if (i == 1)
//         {
//             new_template->thread_num_in_block = 256;
//         }

//         new_template->tblock_num = matrix->block_coor_table.item_arr[i]->compressed_block_ptr->read_index[2]->block_num;

//         // 进行压缩，主要是行号的压缩
//         if (i == 0)
//         {
//             // cout << 1 << endl;
//             compress_row_index_of_block_level_block(new_template, true, CYCLE_INCREASE_COMPRESS);
//             // cout << 2 << endl;
//         }
//         else
//         {
//             compress_row_index_of_block_level_block(new_template);   
//         }
        
//     }

//     for (int i = 2; i < 4; i++)
//     {
//         shared_memory_template_t *old_template = init_shared_memory_template(builder, i);

//         add_template_to_builder(builder, old_template, SHARED_MEMORY_TEMPLATE, i);
//         // 所有的都压缩
//         compress_template_in_builder(builder, SHARED_MEMORY_TEMPLATE_WARP_COMPRESS, i);

//         shared_memory_template_warp_compress_t* new_template = (shared_memory_template_warp_compress_t *)builder->template_vec[i];

//         // 声明计算资源
//         if (i == 0)
//         {
//             new_template->thread_num_in_block = 1024;
//             new_template->tblock_num = matrix->block_coor_table.item_arr[i]->compressed_block_ptr->read_index[3]->block_num;
//             set_row_reduce_thread_num(new_template, get_config()["MAX_ROW_REDUCE_THREAD"].as_integer());

//             // 进行压缩，主要是第一个非零元索引的压缩
//             compress_block_first_row_index(new_template);
//             compress_block_begin_thread_index_offset(new_template);
//             compress_row_offset_in_thread_tmp_result(new_template);
//         }
//         else if (i == 1)
//         {
//             new_template->thread_num_in_block = 256;
//             new_template->tblock_num = matrix->block_coor_table.item_arr[i]->compressed_block_ptr->read_index[3]->block_num;
//             set_row_reduce_thread_num(new_template, get_config()["MAX_ROW_REDUCE_THREAD"].as_integer());
//             compress_block_first_row_index(new_template);
//             compress_block_begin_thread_index_offset(new_template);
//         }
//         else if (i == 2)
//         {
//             // print_dense_block_table(&(op_manager->matrix->block_coor_table));
//             new_template->thread_num_in_block = 1024;
//             new_template->tblock_num = matrix->block_coor_table.item_arr[i]->compressed_block_ptr->read_index[3]->block_num;
//             set_row_reduce_thread_num(new_template, 32);
//             compress_thread_block_size_in_block(new_template);
//         }
//         else if (i == 3)
//         {
//             new_template->thread_num_in_block = 1024;
//             new_template->tblock_num = matrix->block_coor_table.item_arr[i]->compressed_block_ptr->read_index[3]->block_num;
//             set_row_reduce_thread_num(new_template, 1);
//             compress_thread_block_size_in_block(new_template);
//         }
//     }

//     store_code_builder_data(builder);

//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));
// }

// 桶更少的rail4284
// int main()
// {
//     sparse_struct_t *matrix = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/rail4284.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix);
    
//     // 分桶
//     vector<unsigned long> bin_nnz_range;

//     // 按照线程块大小的4倍，使得补零率不超过25%
//     bin_nnz_range.push_back(0);
//     bin_nnz_range.push_back(32);
//     bin_nnz_range.push_back(1024);
//     // bin_nnz_range.push_back(2048);
//     // bin_nnz_range.push_back(4906);

//     // 分桶
//     vector<unsigned long> bin_first_row_vec = total_dense_block_coarse_sort(op_manager, bin_nnz_range);

//     // 在这里对一个分块执行列方向上的padding
//     // void dense_col_level_padding(operator_manager_t *op_manager, dense_block_table_item_t *sub_block, vector<unsigned long> row_index_vec, vector<unsigned long> padding_target_size_vec);
//     // 两个数组，压缩每一行
//     vector<unsigned long> row_index_vec;
//     vector<unsigned long> padding_target_size_vec;

//     // 获取每一行的非零元数量
//     // 获取矩阵中每一行的非零元数量
//     vector<unsigned long> nnz_of_each_row = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);

//     // 遍历前三个条带，给出每一行的目标非零元数量
//     for (unsigned long i = 0; i < 1; i++)
//     {
//         unsigned long target_size_rate;
//         if (i == 0)
//         {
//             target_size_rate = 1024;
//         }

//         // if (i == 1)
//         // {
//         //     target_size_rate = 512;
//         // }

//         // if (i == 2)
//         // {
//         //     target_size_rate = 256;
//         // }

//         // 遍历每一行，补零为特定的倍数
//         unsigned long sub_block_min_row_index = bin_first_row_vec[i];
//         unsigned long sub_block_max_row_index = bin_first_row_vec[i + 1] - 1;

//         // 向两个数组中加入内容
//         for (unsigned long j = sub_block_min_row_index; j <= sub_block_max_row_index; j++)
//         {

//             // 当前行非零元数量
//             assert(j < nnz_of_each_row.size());
//             unsigned long nz_of_this_row = nnz_of_each_row[j];

//             if (nz_of_this_row % target_size_rate != 0)
//             {
//                 // 只有不是整除的时候才需要padding
//                 nz_of_this_row = (nz_of_this_row / target_size_rate + 1) * target_size_rate;
//                 row_index_vec.push_back(j);
//                 padding_target_size_vec.push_back(nz_of_this_row);
//             }
//         }
//     }

//     // 执行列padding
//     dense_col_level_padding(op_manager, NULL, row_index_vec, padding_target_size_vec);

//     // cout << "finish bin, bin size:" << endl;
//     // for (int i = 0; i < bin_first_row_vec.size(); i++)
//     // {
//     //     cout << bin_first_row_vec[i] << endl;
//     // }


//     // 先进行稠密行分块
//     var_len_row_div(op_manager->matrix, NULL, bin_first_row_vec);

//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     nnz_of_each_row = get_nnz_of_each_row_in_spec_range(matrix->coo_row_index_cache, UNSIGNED_LONG, 0, matrix->dense_row_number - 1, 0, matrix->nnz - 1);

//     // 首先处理前三个块
//     for (unsigned long i = 0; i < 1; i++)
//     {
//         // 线程块中线程的数量
//         unsigned long thread_num_in_block = 0;
        
//         // 前三个块的行线程数量分别是1024、512、256
//         if (i == 0)
//         {
//             thread_num_in_block = 1024;
//         }

//         // if (i == 1)
//         // {
//         //     thread_num_in_block = 512;
//         // }

//         // if (i == 2)
//         // {
//         //     thread_num_in_block = 256;
//         // }

//         // 当前子矩阵
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[i]->compressed_block_ptr;

//         // 遍历子矩阵的每一行，分别处理，查看子矩阵的行索引范围
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[i]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[i]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;
        
//         // block块的数量，用来处理分块的方法，一行一块
//         unsigned long row_num_of_each_block = 1;
//         vector<unsigned int> row_number_of_block_arr;
        
//         // 每个块一行
//         for (unsigned long j = 0; j < block_row_size; j++)
//         {
//             row_number_of_block_arr.push_back(1);
//         }

//         // 执行行分块
//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);
        
//         // 忽略warp分块
//         // 在warp层次不分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         assert(thread_num_in_block > 0);

//         // 根据线程块中的线程数量来进行线程粒度的分块，
//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;
//         assert(warp_block_num == row_number_of_block_arr.size());
        
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         assert(warp_block_num == block_row_size);

//         for (unsigned long j = 0; j < warp_block_num; j++)
//         {
//             futher_thread_block_vec.push_back(j);

//             // 当前块的全局行号
//             unsigned long global_row_index_of_this_block = block_begin_row_index + j;

//             assert(global_row_index_of_this_block >= block_begin_row_index && global_row_index_of_this_block <= block_end_row_index);

//             // 通过全局行号获得当前行的非零元数量
//             assert(global_row_index_of_this_block < nnz_of_each_row.size());
//             unsigned long nnz_of_this_row_block = nnz_of_each_row[global_row_index_of_this_block];

//             unsigned long nnz_of_thread_level_block;
//             // 查看当前线程粒度的块非零元的数量
//             if (nnz_of_this_row_block % thread_num_in_block == 0)
//             {
//                 // 可以整除，非零元数量直接除
//                 nnz_of_thread_level_block = nnz_of_this_row_block / thread_num_in_block;
//             }
//             else
//             {
//                 // 不可能不能整除
//                 assert(false);
//                 // nnz_of_thread_level_block  = nnz_of_this_row_block / thread_num_in_block + 1;
//             }

//             // cout << "nnz_of_this_row_block:" << nnz_of_this_row_block << " nnz_of_thread_level_block:" << nnz_of_thread_level_block << endl;
//             // exit(-1);

//             futher_thread_col_block_size.push_back(nnz_of_thread_level_block);
//         }

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     // exit(-1);

//     {
//         // 1号子矩阵
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[1]->compressed_block_ptr;

//         // 三号子矩阵，使用1024个线程，每行使用32个线程执行规约，那么一个block最好负责32行的内容
//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[1]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[1]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         vector<unsigned int> row_number_of_block_arr;
//         // 非零元数量不能超过2048，遍历所有行，并执行一个行分块
//         unsigned long acc_block_nz_num = 0;

//         // 累计的行数量
//         unsigned long acc_block_row_num = 0;

//         for (unsigned long i = block_begin_row_index; i <= block_end_row_index; i++)
//         {
//             unsigned long cur_row_nz_num = nnz_of_each_row[i];
//             // 当前行块的非零元数量不能超过2048
//             if (acc_block_nz_num + cur_row_nz_num < 2048)
//             {
//                 acc_block_nz_num = acc_block_nz_num + cur_row_nz_num;
//                 acc_block_row_num++;
//             }
//             else
//             {
//                 // 超过2048就将已有的写入
//                 assert(acc_block_row_num > 0);
//                 row_number_of_block_arr.push_back(acc_block_row_num);
//                 acc_block_nz_num = cur_row_nz_num;
//                 acc_block_row_num = 1;
//             }
//         }

//         // 如果还有剩余的，要加到最后
//         assert(acc_block_row_num > 0);
    
//         row_number_of_block_arr.push_back(acc_block_row_num);
        

//         // unsigned long row_num_of_each_block = 32;

        

//         // while (block_row_size > row_num_of_each_block)
//         // {
//         //     row_number_of_block_arr.push_back(row_num_of_each_block);
//         //     block_row_size = block_row_size - row_num_of_each_block;
//         // }

//         // // 塞剩下的，如果剩下的还有
//         // if (block_row_size > 0)
//         // {
//         //     row_number_of_block_arr.push_back(block_row_size);
//         // }

//         // 执行行分块
//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);
        
//         // 放弃warp分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 每个线程一个非零元
//         // 线程层次的分块，一个块一个
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         // 遍历所有warp层次的块
//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;
//         assert(warp_block_num == row_number_of_block_arr.size());

        
//         // 看看现成内是不是有归约的空间
//         for (unsigned long i = 0; i < warp_block_num; i++)
//         {
//             futher_thread_block_vec.push_back(i);
//             futher_thread_col_block_size.push_back(1);
//         }

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     // 最后一个块，只有228行，3321个非零元，使用一定数量的block执行处理，使用最普通的CSR，adptive，一个block128行
//     {
//         compressed_block_t* cur_block = matrix->block_coor_table.item_arr[2]->compressed_block_ptr;

//         unsigned long block_begin_row_index = matrix->block_coor_table.item_arr[2]->min_dense_row_index;
//         unsigned long block_end_row_index = matrix->block_coor_table.item_arr[2]->max_dense_row_index;

//         unsigned long block_row_size = block_end_row_index - block_begin_row_index + 1;

//         // 行条带的划分
//         unsigned long row_num_of_each_block = 128;

//         vector<unsigned int> row_number_of_block_arr;

//         while (block_row_size > row_num_of_each_block)
//         {
//             row_number_of_block_arr.push_back(row_num_of_each_block);
//             block_row_size = block_row_size - row_num_of_each_block;
//         }

//         if (block_row_size > 0)
//         {
//             row_number_of_block_arr.push_back(block_row_size);
//         }

//         // 执行行分块
//         sep_tblock_level_row_csr(cur_block, row_number_of_block_arr);
        
//         // 放弃warp分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         sep_warp_level_row_csr(cur_block, sep_block_id_arr, arr_of_row_block_size_arr);

//         // 每个线程一个非零元
//         // 线程层次的分块，一个块一个
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;

//         // 遍历所有warp层次的块
//         unsigned long warp_block_num = cur_block->read_index[3]->block_num;
//         assert(warp_block_num == row_number_of_block_arr.size());

//         for (unsigned long i = 0; i < warp_block_num; i++)
//         {
//             futher_thread_block_vec.push_back(i);
//             futher_thread_col_block_size.push_back(1);
//         }

//         sep_thread_level_col_ell_with_padding(cur_block, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     code_builder_t *builder = init_code_builder(op_manager);

//     // 所有的都用共享内存的初始化
//     for (int i = 0; i < 3; i++)
//     {
//         shared_memory_template_t *old_template = init_shared_memory_template(builder, i);
//         add_template_to_builder(builder, old_template, SHARED_MEMORY_TEMPLATE, i);
//         // 所有的都压缩
//         compress_template_in_builder(builder, SHARED_MEMORY_TEMPLATE_WARP_COMPRESS, i);

//         shared_memory_template_warp_compress_t* new_template = (shared_memory_template_warp_compress_t *)builder->template_vec[i];

//         // 声明计算资源
//         if (i == 0)
//         {
//             new_template->thread_num_in_block = 1024;
//             new_template->tblock_num = 660;
//             set_row_reduce_thread_num(new_template, get_config()["HALF_MAX_ROW_REDUCE_THREAD"].as_integer());

//             // 进行压缩，主要是第一个非零元索引的压缩
//             compress_block_first_row_index(new_template);
//             compress_block_begin_thread_index_offset(new_template);
//             compress_row_offset_in_thread_tmp_result(new_template);
//         }
//         // else if (i == 1)
//         // {
//         //     new_template->thread_num_in_block = 512;
//         //     new_template->tblock_num = 829;
//         //     set_row_reduce_thread_num(new_template, get_config()["HALF_MAX_ROW_REDUCE_THREAD"].as_integer());
//         //     compress_block_first_row_index(new_template);
//         //     compress_block_begin_thread_index_offset(new_template);
//         // }
//         // else if (i == 2)
//         // {
//         //     new_template->thread_num_in_block = 256;
//         //     new_template->tblock_num = 768;
//         //     set_row_reduce_thread_num(new_template, get_config()["HALF_MAX_ROW_REDUCE_THREAD"].as_integer());
//         //     compress_block_first_row_index(new_template);
//         //     compress_block_begin_thread_index_offset(new_template);
//         // }
//         else if (i == 1)
//         {
//             // print_dense_block_table(&(op_manager->matrix->block_coor_table));
//             new_template->thread_num_in_block = 1024;
//             new_template->tblock_num = matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[3]->block_num;
//             set_row_reduce_thread_num(new_template, 32);
//             compress_thread_block_size_in_block(new_template);
//         }
//         else if (i == 2)
//         {
//             new_template->thread_num_in_block = 1024;
//             new_template->tblock_num = matrix->block_coor_table.item_arr[2]->compressed_block_ptr->read_index[3]->block_num;
//             set_row_reduce_thread_num(new_template, 1);
//             compress_thread_block_size_in_block(new_template);
//         }
//     }

//     store_code_builder_data(builder);

//     // 生成对应的模板
//     // cout << 1 << endl;
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));
// }





// 生成CSR stream格式，每256一个条带，不做warp层次的分块，一个非零元一个thread
// int main()
// {
//     sparse_struct_t *matrix_struct = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/mid_size_matrix/Hardesty2.mtx.coo", FLOAT);
//     operator_manager_t *op_manager = init_op_manager(matrix_struct);
//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // tblock级别的分块，256行一块
//     unsigned long min_row_index = op_manager->matrix->block_coor_table.item_arr[0]->min_dense_row_index;
//     unsigned long max_row_index = op_manager->matrix->block_coor_table.item_arr[0]->max_dense_row_index;

//     unsigned long block_row_size = max_row_index - min_row_index + 1;

//     cout << "block_row_size:" << block_row_size << endl;

//     // 每个block 128行，保证共享内存不会超支
//     unsigned long row_num_of_each_block = 256;

//     vector<unsigned int> row_number_of_block_arr;

//     while (block_row_size > row_num_of_each_block)
//     {
//         row_number_of_block_arr.push_back(row_num_of_each_block);
//         block_row_size = block_row_size - row_num_of_each_block;
//     }

//     // 塞剩下的
//     assert(block_row_size > 0);
//     row_number_of_block_arr.push_back(block_row_size);
    
//     // 执行实际的分块
//     sep_tblock_level_row_csr(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr, row_number_of_block_arr);

//     // 在warp层次不分块
//     vector<vector<unsigned int>> arr_of_row_block_size_arr;
//     vector<unsigned long> sep_block_id_arr;

//     sep_warp_level_row_csr(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr, sep_block_id_arr, arr_of_row_block_size_arr);

//     // 线程层次的分块，一个块一个
//     vector<unsigned long> futher_thread_block_vec;
//     vector<unsigned long> futher_thread_col_block_size;

//     // 遍历所有warp层次的块
//     unsigned long warp_block_num = op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[3]->block_num;
//     assert(warp_block_num == row_number_of_block_arr.size());

//     for (unsigned long i = 0; i < warp_block_num; i++)
//     {
//         futher_thread_block_vec.push_back(i);
//         futher_thread_col_block_size.push_back(1);
//     }

//     sep_thread_level_col_ell_with_padding(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr, futher_thread_block_vec, futher_thread_col_block_size);

//     // 创造一个shared memory的模板，将warp级别的遍历压缩
//     code_builder_t *builder = init_code_builder(op_manager);

//     shared_memory_template_t *old_template = init_shared_memory_template(builder, 0);
//     add_template_to_builder(builder, old_template, SHARED_MEMORY_TEMPLATE, 0);

//     // compress_template_in_builder(builder, SHARED_MEMORY_TEMPLATE_WARP_COMPRESS, 0);
//     // shared_memory_template_warp_compress_t* new_template = (shared_memory_template_warp_compress_t *)builder->template_vec[0];

//     // 可以设置归约过程的并行度
//     store_code_builder_data(builder);
//     set_row_reduce_thread_num(old_template, 2);
    
//     // 生成代码
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));
// }

// 生成sell格式，按照4096的粒度的排序，按照512行一个block的粒度行分块
// int main()
// {
//     sparse_struct_t *matrix_struct = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/mid_size_matrix/Hardesty2.mtx.coo", DOUBLE);
//     operator_manager_t *op_manager = init_op_manager(matrix_struct);

//     // 进行全局padding，使得行的数量是1024的倍数，从而保证排序的粒度是对齐的
//     total_row_level_padding(op_manager, 4096);

//     // 在稠密矩阵上不做切分，直接压缩
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));
//     compress_dense_view(op_manager);
//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 分块操作，每512一个block
//     unsigned long min_row_index = op_manager->matrix->block_coor_table.item_arr[0]->min_dense_row_index;
//     unsigned long max_row_index = op_manager->matrix->block_coor_table.item_arr[0]->max_dense_row_index;

//     unsigned long block_row_size = max_row_index - min_row_index + 1;

//     cout << "block_row_size:" << block_row_size << endl;

//     // 每个block 512行
//     unsigned long row_num_of_each_block = 512;

//     vector<unsigned int> row_number_of_block_arr;

//     while (block_row_size > row_num_of_each_block)
//     {
//         row_number_of_block_arr.push_back(row_num_of_each_block);
//         block_row_size = block_row_size - row_num_of_each_block;
//     }

//     // 塞剩下的
//     assert(block_row_size > 0);
//     row_number_of_block_arr.push_back(block_row_size);
    
//     sep_tblock_level_row_csr(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr, row_number_of_block_arr);

//     // 在warp层次不分块
//     vector<vector<unsigned int>> arr_of_row_block_size_arr;
//     vector<unsigned long> sep_block_id_arr;

//     sep_warp_level_row_csr(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr, sep_block_id_arr, arr_of_row_block_size_arr);

//     // 线程层次的分块
//     vector<unsigned long> futher_thread_block_vec;
//     vector<unsigned long> futher_thread_col_block_size;

//     sep_thread_level_col_ell_with_padding(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr, futher_thread_block_vec, futher_thread_col_block_size);
    
//     // 创建代码生成器
//     code_builder_t* builder = init_code_builder(op_manager);

//     direct_atom_template_t* new_template = init_direct_atom_template(builder, 0);

//     add_template_to_builder(builder, new_template, DIRECT_ATOM_TEMPLATE, 0);

//     compress_template_in_builder(builder, DIRECT_ATOM_TEMPLATE_WARP_COMPRESS, 0);

//     direct_atom_template_warp_compress_t* compressed_template = (direct_atom_template_warp_compress_t*)builder->template_vec[0];

//     compress_global_row_index_of_thread_level_block(compressed_template);
//     compress_block_begin_thread_index_offset(compressed_template);

//     store_code_builder_data(builder);

//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/Hardesty2_compare/sell_template2.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/Hardesty2_compare/sell_template2.cu", build_main_file(builder));

//     return 0;
// }


// int main()
// {
//     // cout << sizeof(bool) << endl;

//     // return 0;

//     // void* input_arr_ptr = read_arr_from_file_with_data_type(94, UNSIGNED_INT, "/home/duzhen/spmv_builder/data_source/1282012747/dense_block_0/read_index_2/index_arr");

//     // cout << read_from_array_with_data_type(input_arr_ptr, UNSIGNED_INT, 2) << endl;

//     // return 0;

//     // dense_block_table_t *table = new dense_block_table_t();
//     // dense_block_table_item_t *item = new dense_block_table_item_t();
//     // item->block_coordinate.push_back(1);
//     // table->item_arr.push_back(item);
//     // print_dense_block_table(table, false, "");

//     sparse_struct_t *matrix_struct = init_sparse_struct_by_coo_file("/home/duzhen/spmv_builder/data_source/Hardesty2.mtx.coo", DOUBLE);

//     operator_manager_t *op_manager = init_op_manager(matrix_struct);

//     // 全局padding，按照1024的倍数来padding，保证激活后续优化
//     total_row_level_padding(op_manager, 1024);

//     // cout << "op_manager->matrix->dense_row_number:" << op_manager->matrix->dense_row_number << endl;

//     // print_arr_to_file_with_data_type(matrix_struct->coo_col_index_cache, UNSIGNED_LONG, matrix_struct->nnz, "/home/duzhen/spmv_builder/data_source/test0-3.log");

//     vector<unsigned long> first_row_of_each_sort_band;
//     first_row_of_each_sort_band.push_back(0);

//     // 每个条带

//     // 每个条带的大小为2048
//     unsigned long row_sort_band = 2048;
//     while (first_row_of_each_sort_band[first_row_of_each_sort_band.size() - 1] + row_sort_band < matrix_struct->dense_row_number)
//     {
//         first_row_of_each_sort_band.push_back(first_row_of_each_sort_band[first_row_of_each_sort_band.size() - 1] + row_sort_band);
//     }

//     // 全局进行一个排序
//     total_dense_block_sort(op_manager, first_row_of_each_sort_band);

//     // print_arr_to_file_with_data_type(matrix_struct->coo_row_index_cache, UNSIGNED_LONG, matrix_struct->nnz, "/home/duzhen/spmv_builder/data_source/test0-6.log");

//     // print_arr_to_file_with_data_type(matrix_struct->sorted_row_index, matrix_struct->data_type_of_sorted_row_index, matrix_struct->dense_row_number, "/home/duzhen/spmv_builder/data_source/test0-7.log");

//     // print_arr_to_file_with_data_type(matrix_struct->sorted_row_index, );

//     // fixed_len_col_div(op_manager, NULL, 200000);

//     // print_dense_block_table(&(matrix_struct->block_coor_table));

//     // // output_struct_coo_to_file(matrix_struct, "/home/duzhen/spmv_builder/data_source/test");

//     // 将矩阵切为两份，一份500000行
//     fixed_len_row_div(op_manager, NULL, 500736);

//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // exit(-1);

//     // fixed_len_row_div(op_manager, op_manager->matrix->block_coor_table.item_arr[0], 200000);
//     // fixed_len_row_div(matrix_struct, NULL, 100000);

//     // 这里做一个检查
//     // cout << "check:" << check_dense_block_div(op_manager->matrix) << endl;

//     compress_dense_view(op_manager);

//     // print_arr_to_file_with_data_type(op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[1]->index_arr, op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[1]->index_data_type, op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[1]->length, "/home/duzhen/spmv_builder/data_source/test0-3.log");

//     // 直接压缩整个子块
//     // total_compressed_block_sort(op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr);

//     // print_arr_to_file_with_data_type(op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[1]->index_arr, op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[1]->index_data_type, op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[1]->length, "/home/duzhen/spmv_builder/data_source/test0-4.log");

//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // exit(-1);

//     // print_compressed_block(matrix_struct, "/home/duzhen/spmv_builder/data_source");

//     // 按照同样的策略分配所有的块
//     for (unsigned long dense_id = 0; dense_id < op_manager->matrix->block_coor_table.item_arr.size(); dense_id++)
//     {
//         // for(unsigned long dense_id = 0; dense_id < 1; dense_id++){
//         // 行分块，每个块10000行
//         unsigned long min_row_index = op_manager->matrix->block_coor_table.item_arr[dense_id]->min_dense_row_index;
//         unsigned long max_row_index = op_manager->matrix->block_coor_table.item_arr[dense_id]->max_dense_row_index;

//         unsigned long block_row_size = max_row_index - min_row_index + 1;

//         cout << "block_row_size:" << block_row_size << endl;

//         // 每个block 1024行
//         unsigned long row_num_of_each_block = 1024;

//         vector<unsigned int> row_number_of_block_arr;

//         while (block_row_size > row_num_of_each_block)
//         {
//             row_number_of_block_arr.push_back(row_num_of_each_block);
//             block_row_size = block_row_size - row_num_of_each_block;
//         }

//         // 塞剩下的
//         assert(block_row_size > 0);
//         row_number_of_block_arr.push_back(block_row_size);

//         // cout << row_number_of_block_arr.size() << endl;
//         cout << "begin block level sep" << endl;
//         sep_tblock_level_row_csr(op_manager->matrix->block_coor_table.item_arr[dense_id]->compressed_block_ptr, row_number_of_block_arr);

//         // 剩下的部分列分块
//         vector<vector<unsigned int>> arr_of_row_block_size_arr;
//         vector<unsigned long> sep_block_id_arr;

//         compressed_block_t *this_compressed_block = op_manager->matrix->block_coor_table.item_arr[dense_id]->compressed_block_ptr;
//         index_of_compress_block_t *block_level_index = this_compressed_block->read_index[2];
//         unsigned long row_num_of_each_warp_block = 32;
//         // 遍历所有的tblock粒度的块，分别进一步按照63行一块的方式进行分块
//         for (unsigned long i = 0; i < this_compressed_block->read_index[2]->block_num; i++)
//         {
//             // cout << "i:" << i << endl;
//             vector<unsigned int> row_block_size_arr;
//             // 这一块的行数量
//             unsigned long block_row_size = read_from_array_with_data_type(block_level_index->row_number_of_block_arr, block_level_index->data_type_of_row_number_of_block_arr, i);
//             // cout << "block_row_size:" << block_row_size << endl;
//             // 对当前block进行一系列warp分块
//             while (block_row_size > row_num_of_each_warp_block)
//             {
//                 row_block_size_arr.push_back(row_num_of_each_warp_block);
//                 block_row_size = block_row_size - row_num_of_each_warp_block;
//             }

//             if (block_row_size != 0)
//             {
//                 // 剩下的部分放到数组中
//                 row_block_size_arr.push_back(block_row_size);
//             }

//             // cout << row_block_size_arr.size() << endl;
//             // 记录当前块要被分块的信息
//             arr_of_row_block_size_arr.push_back(row_block_size_arr);
//             sep_block_id_arr.push_back(i);
//         }

//         cout << arr_of_row_block_size_arr.size() << endl;

//         sep_warp_level_row_csr(op_manager->matrix->block_coor_table.item_arr[dense_id]->compressed_block_ptr, sep_block_id_arr, arr_of_row_block_size_arr);

//         // thread层次的分块，对所有的块都按照长度为5进行分块，首先查看warp的数量
//         // 看起来像一个全局ELL分块
//         unsigned long warp_level_block_num = op_manager->matrix->block_coor_table.item_arr[dense_id]->compressed_block_ptr->read_index[3]->block_num;
//         vector<unsigned long> futher_thread_block_vec;
//         vector<unsigned long> futher_thread_col_block_size;
        
//         for (unsigned long warp_level_block_id = 0; warp_level_block_id < warp_level_block_num; warp_level_block_id++)
//         {
//             futher_thread_block_vec.push_back(warp_level_block_id);
//             futher_thread_col_block_size.push_back(5);
//         }

//         sep_thread_level_col_ell_with_padding(op_manager->matrix->block_coor_table.item_arr[dense_id]->compressed_block_ptr, futher_thread_block_vec, futher_thread_col_block_size);
//     }

//     print_dense_block_table(&(op_manager->matrix->block_coor_table));

//     // 查看第一个块的warp
//     // print_arr_to_file_with_data_type(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[3]->index_arr, op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[3]->index_data_type, op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[3]->length, "/home/duzhen/spmv_builder/data_source/test0-6.log");

//     // exit(-1);

//     // cout << "size in block level:" << op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[2]->block_num << endl;

//     // exit(-1);

//     // 用一个数组存储需要排序的子块
//     // vector<unsigned long> block_index_need_to_be_sorted;
//     // block_index_need_to_be_sorted.push_back(0);
//     // block_index_need_to_be_sorted.push_back(1);

//     // compressed_block_sort(op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr, block_index_need_to_be_sorted);

//     // 打印原行索引
//     // print_arr_to_file_with_data_type(op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->y_write_index[0]->index_arr, op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->y_write_index[0]->index_data_type, op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->y_write_index[0]->length, "/home/duzhen/spmv_builder/data_source/test0-1.log");

//     // 将第一个块分块之后的结果打印出来
//     // print_compressed_block_meta_index(op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[2], "/home/duzhen/spmv_builder/data_source/test0-1.log");

//     // exit(-1);

//     // row_num_of_each_block = 63;
//     // block_row_size = 10000;
//     // while (block_row_size > row_num_of_each_block)
//     // {
//     //     row_block_size_arr.push_back(row_num_of_each_block);
//     //     block_row_size = block_row_size - row_num_of_each_block;
//     // }

//     // if (block_row_size != 0)
//     // {
//     //     // 剩下的部分放到数组中
//     //     row_block_size_arr.push_back(block_row_size);
//     // }

//     // 所有的warp都63行分为一块
//     // arr_of_row_block_size_arr.push_back(row_block_size_arr);
//     // arr_of_row_block_size_arr.push_back(row_block_size_arr);
//     // arr_of_row_block_size_arr.push_back(row_block_size_arr);
//     // arr_of_row_block_size_arr.push_back(row_block_size_arr);
//     // arr_of_row_block_size_arr.push_back(row_block_size_arr);
//     // arr_of_row_block_size_arr.push_back(row_block_size_arr);
//     // arr_of_row_block_size_arr.push_back(row_block_size_arr);
//     // sep_block_id_arr.push_back(0);
//     // sep_block_id_arr.push_back(1);
//     // sep_block_id_arr.push_back(2);
//     // sep_block_id_arr.push_back(3);
//     // sep_block_id_arr.push_back(4);
//     // sep_block_id_arr.push_back(5);

//     // sep_block_id_arr.push_back(6);

//     // return 0;

//     // // 进行列分块，分三块，大小分别是1、2、3
//     // vector<vector<unsigned int>> arr_of_col_block_size_arr;
//     // vector<unsigned int> col_block_size_arr;
//     // vector<unsigned long> sep_block_id_arr2;

//     // col_block_size_arr.push_back(10);
//     // col_block_size_arr.push_back(2);
//     // // col_block_size_arr.push_back(3);

//     // arr_of_col_block_size_arr.push_back(col_block_size_arr);
//     // arr_of_col_block_size_arr.push_back(col_block_size_arr);

//     // sep_block_id_arr2.push_back(0);

//     // // sep_warp_level_col_csr(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr, sep_block_id_arr2, arr_of_col_block_size_arr);

//     // print_compressed_block_meta_index(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[2], "/home/duzhen/spmv_builder/data_source/test4-0.log");
//     // print_compressed_block_meta_index(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[3], "/home/duzhen/spmv_builder/data_source/test5-0.log");

//     // 打印coo size
//     // print_arr_to_file_with_data_type(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[3]->coo_block_size_arr, op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[3]->data_type_of_coo_block_size_arr, op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[3]->block_num, "/home/duzhen/spmv_builder/data_source/test5-1.log");

//     // exit(-1);

//     // print_arr_with_data_type(op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[3]->tmp_result_write_index_arr, op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[3]->data_type_of_tmp_result_write_index_arr, op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[3]->block_num);
//     // print_arr_with_data_type(op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[3]->coo_block_size_arr, op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[3]->data_type_of_coo_block_size_arr, op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[3]->block_num);

//     // 先全部尝试行分块
//     // 搞两个空白的数组
//     // vector<unsigned long> futher_thread_block_vec;
//     // futher_thread_block_vec.push_back(0);
//     // vector<unsigned long> futher_thread_col_block_size;
//     // futher_thread_col_block_size.push_back(5);

//     // sep_thread_level_col_ell_with_padding(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr, futher_thread_block_vec, futher_thread_col_block_size);
//     // cout << "finish" << endl;
//     // 打印thread的一些内容
//     // print_arr_to_file_with_data_type(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[4]->index_of_the_first_row_arr, op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[4]->data_type_of_index_of_the_first_row_arr, op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[4]->block_num, "/home/duzhen/spmv_builder/data_source/test6.log");
//     // print_arr_to_file_with_data_type(op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[3]->index_arr, op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[3]->index_data_type, op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->read_index[3]->length, "/home/duzhen/spmv_builder/data_source/test7.log");
//     // print_arr_with_data_type(op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[4]->index_of_the_first_row_arr, op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[4]->data_type_of_index_of_the_first_row_arr, op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[4]->block_num);

//     // 密集矩阵不分块，10000行一块，一系列函数140
//     // write_total_matrix_to_file(op_manager->matrix, "/home/duzhen/spmv_builder/data_source");

//     // exit(-1);

//     // // 矩阵是每行4个元素，列分块按照1、2、3
//     // // 针对第一个块做处理
//     // vector<vector<unsigned int>> arr_of_block_col_size_arr;
//     // vector<unsigned int> first_block_col_size_arr;

//     // first_block_col_size_arr.push_back(1);
//     // first_block_col_size_arr.push_back(2);
//     // first_block_col_size_arr.push_back(3);

//     // arr_of_block_col_size_arr.push_back(first_block_col_size_arr);

//     // // 只处理第一个块
//     // vector<unsigned long> block_index_arr;

//     // block_index_arr.push_back(18);

//     // // sep_tblock_level_col_csr(op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr, block_index_arr, arr_of_block_col_size_arr);
//     // sep_tblock_level_col_csr(op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr, first_block_col_size_arr);

//     // // 打印分块的结果
//     // print_compressed_block_meta_index(op_manager->matrix->block_coor_table.item_arr[1]->compressed_block_ptr->read_index[2], "/home/duzhen/spmv_builder/data_source/test2.log");

//     // vector<unsigned int> null_arr;

//     // seq_thread_level_csr(matrix_struct->block_coor_table.item_arr[0]->compressed_block_ptr, null_arr);

//     // print_compressed_block(matrix_struct, "/home/duzhen/spmv_builder/data_source");
//     // fixed_len_col_div(matrix_struct, matrix_struct->block_coor_table.item_arr[8], 10000);

//     // print_dense_block_table(&(matrix_struct->block_coor_table));

//     // cout << (check_dense_block_div(matrix_struct) == true) << endl;

//     // cout << sizeof(unsigned char) << endl;
//     // cout << sizeof(unsigned short) << endl;
//     // cout << sizeof(unsigned int) << endl;
//     // cout << sizeof(unsigned long) << endl;

//     // return 1;

//     code_builder_t *builder = init_code_builder(op_manager);

//     // cout << op_manager->matrix->block_coor_table.item_arr[0]->compressed_block_ptr->reduce_help_csr.size() << endl;
//     for (unsigned long dense_id = 0; dense_id < op_manager->matrix->block_coor_table.item_arr.size(); dense_id++)
//     {
//         // store_template_data(new_template, "/home/duzhen/spmv_builder/data_source");
//         if (dense_id == 0)
//         {
//             shared_memory_template_t *new_template = init_shared_memory_template(builder, dense_id);
//             add_template_to_builder(builder, new_template, SHARED_MEMORY_TEMPLATE, dense_id);
//         }
//         else
//         {
//             shared_memory_template_t *new_template = init_shared_memory_template(builder, dense_id);
//             add_template_to_builder(builder, new_template, SHARED_MEMORY_TEMPLATE, dense_id);
//         }
//     }

//     cout << "finish init" << endl;

//     // 做一些优化
//     compress_template_in_builder(builder, SHARED_MEMORY_TEMPLATE_WARP_COMPRESS, 0);
//     shared_memory_template_warp_compress_t* new_template = (shared_memory_template_warp_compress_t *)builder->template_vec[0];

//     // 将内容打印出来
//     store_code_builder_data(builder);

//     compress_block_first_row_index(new_template);
//     compress_thread_block_size_in_block(new_template);
    
//     compress_block_nz_begin_offset(new_template);

//     // compress_global_row_index_of_thread_level_block(new_template);

//     // 将warp的thread数组做一些压缩
//     // compress_warp_begin_thread_index_offset(new_template);
//     // 将warp的thread size做一些压缩
//     // compress_thread_block_size_in_warp(new_template);

//     // // 将warp的nz起始位置进行压缩
//     // cout << compress_warp_nz_begin_offset(new_template) << endl;

//     // // 将thread块的输出行号进行压缩
//     // compress_global_row_index_of_thread_level_block(new_template);

//     // 对block的warp首个warp索引进行压缩
//     compress_block_begin_thread_index_offset(new_template);
//     // compress_warp_begin_thread_index_offset(new_template);

//     compress_row_offset_in_thread_tmp_result(new_template);

//     // // block第一个非零元的压缩
//     // compress_block_nz_begin_offset(new_template);

//     // // 设置线程块的结构
//     // new_template->tblock_num = new_template->size_of_block_begin_warp_index_offset - 1;
//     new_template->tblock_num = 489;
//     new_template->thread_num_in_block = 1024;
    
//     // compress_thread_block_size_in_warp(new_template, true, BRANCH_COMPRESS);

//     // add_template_to_builder(builder, new_template, DIRECT_ATOM_TEMPLATE, 0);

//     // cout << code_of_template_data_struct(new_template, 0) << endl;

//     // cout << code_of_read_template_data_from_file_func_define(new_template, 0) << endl;

//     // cout << code_of_write_template_data_to_gpu(new_template, 0) << endl;

//     // cout << build_header_file(builder) << endl;
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.hpp", build_header_file(builder));
//     write_string_to_file("/home/duzhen/spmv_builder/cuda_code/template.cu", build_main_file(builder));
//     // cout << code_of_kernal_func_define(builder, 0) << endl;

//     // cout << code_of_main_function(builder, "/home/duzhen/spmv_builder/data_source/857772456") << endl;

//     // 尝试一下压缩
//     // linear_compress_t* compressor = init_linear_compressor(new_template->block_begin_warp_index_offset, new_template->data_type_of_block_begin_warp_index_offset, new_template->size_of_block_begin_warp_index_offset, true);

//     // branch_compress_t* compressor = init_branch_compressor(new_template->thread_block_size_in_warp, new_template->data_type_of_thread_block_size_in_warp, new_template->size_of_thread_block_size_in_warp, true);

//     // cout << code_of_template_kernal(new_template, 0) << endl;

//     return 1;
// }