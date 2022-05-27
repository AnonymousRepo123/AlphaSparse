#include "code_builder.hpp"
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
#include "empty_op.hpp"
#include <unistd.h>
#include <memory.h>

code_builder_t *init_code_builder(operator_manager_t *op_manager)
{
    assert(op_manager != NULL);
    assert(op_manager->matrix->block_coor_table.item_arr.size() > 0);


    // 检查所有压缩快是不是都有7层索引
    for (unsigned int i = 0; i < op_manager->matrix->block_coor_table.item_arr.size(); i++)
    {
        assert(op_manager->matrix->block_coor_table.item_arr[i]->compressed_block_ptr != NULL);
        assert(op_manager->matrix->block_coor_table.item_arr[i]->compressed_block_ptr->read_index.size() == 7);
    }

    code_builder_t *new_builder = new code_builder_t();
    new_builder->op_manager = op_manager;

    sparse_struct_t *matrix = op_manager->matrix;
    // 表格
    dense_block_table_t *block_coor_table = &(matrix->block_coor_table);

    // 查看是不是在全局排序过
    assert((matrix->is_sorted == true && matrix->sorted_row_index != NULL) || (matrix->is_sorted == false && matrix->sorted_row_index == NULL));

    // 遍历所有的密集子矩阵
    for (unsigned int i = 0; i < block_coor_table->item_arr.size(); i++)
    {
        compressed_block_t *compressed_block_view = block_coor_table->item_arr[i]->compressed_block_ptr;
        assert(compressed_block_view != NULL);

        // 查看当前块是不是和其他块共享行
        // 块级别的共享导致了全局层次的归约
        new_builder->is_reduce_in_global_level_vec.push_back(compressed_block_view->share_row_with_other_block);
        // 默认使用的归约方式是原子加的方式，如果warp和block层次的规约都是原子加，可以直通显存
        new_builder->reduce_type_in_global_level_vec.push_back(ATOMIC_ADD);

        // 线程级别的行共享导致了warp层次的归约
        new_builder->is_reduce_in_warp_level_vec.push_back(compressed_block_view->share_row_with_other_thread);
        new_builder->reduce_type_in_warp_level_vec.push_back(ATOMIC_ADD);

        // warp级别的行共享导致了block层次的规约
        new_builder->is_reduce_in_block_level_vec.push_back(compressed_block_view->share_row_with_other_warp);
        new_builder->reduce_type_in_block_level_vec.push_back(ATOMIC_ADD);

        // 分配计算资源
        new_builder->kernal_block_num_vec.push_back(get_config()["DEFAULT_THREAD_BLOCK_NUM"].as_integer());
        new_builder->kernal_thread_num_in_block_vec.push_back(get_config()["DEFAULT_THREAD_NUM_IN_BLOCK"].as_integer());

        // 查看是是不是全局排序
        if (matrix->is_sorted == true)
        {
            // 全局排序了，局部肯定不排序
            assert(compressed_block_view->y_write_index.size() == 0);
            new_builder->sub_block_sort_type_vec.push_back(GLOBAL_SORT);
        }
        else if (compressed_block_view->y_write_index.size() > 0)
        {
            // 有局部的排序
            new_builder->sub_block_sort_type_vec.push_back(SUB_BLOCK_SORT);
        }
        else
        {
            new_builder->sub_block_sort_type_vec.push_back(NO_SORT);
        }
    }

    assert(new_builder->sub_block_sort_type_vec.size() == block_coor_table->item_arr.size());

    // 为每一个block在reduce_help_csr中添加一个数组来记录每个thread块结果的全局行索引的CSR压缩
    for (unsigned long dense_block_index = 0; dense_block_index < block_coor_table->item_arr.size(); dense_block_index++)
    {

        compressed_block_t *compressed_block_view = block_coor_table->item_arr[dense_block_index]->compressed_block_ptr;
        // 删除归约的相关数组，然后为每个密集矩阵加入一个reduce_help_csr数组
        // 删除
        assert(compressed_block_view->read_index.size() == 7);
        for (unsigned long index_of_read_index = 0; index_of_read_index < compressed_block_view->read_index.size(); index_of_read_index++)
        {
            index_of_compress_block_t *read_index = compressed_block_view->read_index[index_of_read_index];

            if (read_index->child_tmp_row_csr_index_arr != NULL)
            {
                delete_arr_with_data_type(read_index->child_tmp_row_csr_index_arr, read_index->data_type_of_child_tmp_row_csr_index);
                read_index->child_tmp_row_csr_index_arr = NULL;
            }

            if (read_index->begin_index_in_tmp_row_csr_arr_of_block != NULL)
            {
                delete_arr_with_data_type(read_index->begin_index_in_tmp_row_csr_arr_of_block, read_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block);
                read_index->begin_index_in_tmp_row_csr_arr_of_block = NULL;
            }
        }

        // 只有全局同步的模板中需要这些

        // index_of_compress_block_t* thread_block_index = compressed_block_view->read_index[4];
        // assert(thread_block_index->level_of_this_index == THREAD_LEVEL);

        // // 因为线程内部没有归约，一个线程最多一行，所以用线程粒度索引的首行号，来确定归约索引
        // // vector<unsigned long> get_nnz_of_each_row_in_spec_range(void* row_index_arr, data_type data_type_of_row_index_arr, unsigned long begin_row_bound, unsigned long end_row_bound, unsigned long global_coo_start, unsigned long global_coo_end)

        // // 分别遍历三个层次的索引，获取每个thread的全局行号，大小为这个子块行的数量
        // vector<unsigned long> tmp_result_number_of_each_row(thread_block_index->max_row_index - thread_block_index->min_row_index + 1);

        // memset(&(tmp_result_number_of_each_row[0]), 0, sizeof(unsigned long) * tmp_result_number_of_each_row.size());

        // index_of_compress_block_t* warp_block_index = compressed_block_view->read_index[3];
        // assert(warp_block_index->level_of_this_index == WRAP_LEVEL);

        // index_of_compress_block_t* block_block_index = compressed_block_view->read_index[2];
        // assert(block_block_index->level_of_this_index == TBLOCK_LEVEL);

        // // 遍历线程块粒度的所有块
        // for(unsigned long index_of_block_level_block = 0; index_of_block_level_block < block_block_index->block_num; index_of_block_level_block++){
        //     // 当前线程块的首行行号
        //     unsigned long block_first_row_index = read_from_array_with_data_type(block_block_index->index_of_the_first_row_arr, block_block_index->data_type_of_index_of_the_first_row_arr, index_of_block_level_block);
        //     // 遍历warp粒度的所有块
        //     unsigned long warp_index_begin = read_from_array_with_data_type(block_block_index->index_arr, block_block_index->index_data_type, index_of_block_level_block);
        //     unsigned long warp_index_end = read_from_array_with_data_type(block_block_index->index_arr, block_block_index->index_data_type, index_of_block_level_block + 1);

        //     for(unsigned long index_of_warp_level_block_index = warp_index_begin; index_of_warp_level_block_index < warp_index_end; index_of_warp_level_block_index++){
        //         // if(index_of_warp_level_block_index >= warp_block_index->block_num){
        //         //     cout << "index_of_warp_level_block_index:" << index_of_warp_level_block_index << ", " << "warp_block_index->block_num:" << warp_block_index->block_num << endl;
        //         // }
        //         assert(index_of_warp_level_block_index < warp_block_index->block_num);
        //         unsigned long warp_first_row_index = read_from_array_with_data_type(warp_block_index->index_of_the_first_row_arr, warp_block_index->data_type_of_index_of_the_first_row_arr, index_of_warp_level_block_index);
        //         // 遍历thread粒度的所有块
        //         unsigned long thread_index_begin = read_from_array_with_data_type(warp_block_index->index_arr, warp_block_index->index_data_type, index_of_warp_level_block_index);
        //         unsigned long thread_index_end = read_from_array_with_data_type(warp_block_index->index_arr, warp_block_index->index_data_type, index_of_warp_level_block_index + 1);
        //         for(unsigned long index_of_thread_warp = thread_index_begin; index_of_thread_warp < thread_index_end; index_of_thread_warp++){
        //             // 当前thread块的第一行
        //             assert(index_of_thread_warp < thread_block_index->block_num);
        //             unsigned long thread_first_row_index = read_from_array_with_data_type(thread_block_index->index_of_the_first_row_arr, thread_block_index->data_type_of_index_of_the_first_row_arr, index_of_thread_warp);

        //             // 当前块的全局行号
        //             unsigned long global_row_index_of_thread_block = block_first_row_index + warp_first_row_index + thread_first_row_index;
        //             assert(global_row_index_of_thread_block < tmp_result_number_of_each_row.size());
        //             tmp_result_number_of_each_row[global_row_index_of_thread_block]++;
        //         }
        //     }
        // }

        // 当前块的行数量
        // assert(tmp_result_number_of_each_row.size() == thread_block_index->max_row_index - thread_block_index->min_row_index + 1);

        // vector<unsigned long> new_reduce_help_csr_item_vec;
        // new_reduce_help_csr_item_vec.push_back(0);

        // for(unsigned long row_index_in_this_dense_block = 0; row_index_in_this_dense_block < tmp_result_number_of_each_row.size(); row_index_in_this_dense_block++){
        //     new_reduce_help_csr_item_vec.push_back(new_reduce_help_csr_item_vec[new_reduce_help_csr_item_vec.size()-1] + tmp_result_number_of_each_row[row_index_in_this_dense_block]);
        // }
        // // 线程粒度结果的行索引的CSR版本的最后一个元素的大小，是thread的块数量
        // assert(new_reduce_help_csr_item_vec[new_reduce_help_csr_item_vec.size() - 1] == thread_block_index->block_num);

        // // 创建新的索引
        // index_of_compress_block_t* new_reduce_index = new index_of_compress_block_t();
        // new_reduce_index->level_of_this_index = THREAD_LEVEL;
        // new_reduce_index->index_compressed_type = CSR;
        // new_reduce_index->index_data_type = find_most_suitable_data_type(thread_block_index->block_num);
        // new_reduce_index->length = new_reduce_help_csr_item_vec.size();

        // new_reduce_index->index_arr = malloc_arr(new_reduce_index->length, new_reduce_index->index_data_type);
        // // 拷贝
        // copy_unsigned_long_arr_to_others(&(new_reduce_help_csr_item_vec[0]), new_reduce_index->index_arr, new_reduce_index->index_data_type, new_reduce_index->length);

        // compressed_block_view->reduce_help_csr.push_back(new_reduce_index);

        // print_arr_to_file_with_data_type(new_reduce_index->index_arr, new_reduce_index->index_data_type, new_reduce_index->length, "/home/duzhen/spmv_builder/data_source/test7.log");

        // 给模板初始化为空指针
        new_builder->template_vec.push_back(NULL);
        new_builder->template_type_vec.push_back(NONE_TEMPLATE);
    }

    return new_builder;
}

code_builder_t *init_code_builder(operator_manager_t *op_manager, vector<int> sub_matrix_id_vec)
{
    // 将一部分的子块初始化到代码生成器中
    assert(op_manager != NULL);
    assert(op_manager->matrix != NULL);
    assert(op_manager->matrix->block_coor_table.item_arr.size() > 0);

    // 检查输入是不是正确，是不是每一个
    for (int i = 0; i < sub_matrix_id_vec.size(); i++)
    {
        int sub_matrix_id = sub_matrix_id_vec[i];

        assert(sub_matrix_id < op_manager->matrix->block_coor_table.item_arr.size());

        // 经过压缩
        assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id] != NULL);
        assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
        assert(op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);
    }

    // 针对大矩阵的几个子块创建代码生成器
    code_builder_t* new_builder = new code_builder_t();
    new_builder->op_manager = op_manager;

    sparse_struct_t *matrix = op_manager->matrix;

    dense_block_table_t *block_coor_table = &(matrix->block_coor_table);

    // 在builder中的有些元数据是空的，先全部初始化，把位置占住
    for (unsigned int i = 0; i < block_coor_table->item_arr.size(); i++)
    {
        new_builder->is_reduce_in_global_level_vec.push_back(false);
        new_builder->reduce_type_in_global_level_vec.push_back(ATOMIC_ADD);

        // 线程级别的行共享导致了warp层次的归约
        new_builder->is_reduce_in_warp_level_vec.push_back(false);
        new_builder->reduce_type_in_warp_level_vec.push_back(ATOMIC_ADD);

        new_builder->is_reduce_in_block_level_vec.push_back(false);
        new_builder->reduce_type_in_block_level_vec.push_back(ATOMIC_ADD);

        new_builder->kernal_block_num_vec.push_back(get_config()["DEFAULT_THREAD_BLOCK_NUM"].as_integer());
        new_builder->kernal_thread_num_in_block_vec.push_back(get_config()["DEFAULT_THREAD_NUM_IN_BLOCK"].as_integer());

        new_builder->sub_block_sort_type_vec.push_back(NO_SORT);
    }

    assert(new_builder->sub_block_sort_type_vec.size() == block_coor_table->item_arr.size());

    // 遍历需要生成代码的密集子矩阵
    for (unsigned int i = 0; i < sub_matrix_id_vec.size(); i++)
    {
        int sub_matrix_id = sub_matrix_id_vec[i];
        
        // 密集子矩阵
        compressed_block_t* compressed_block_ptr = block_coor_table->item_arr[sub_matrix_id]->compressed_block_ptr;
        assert(compressed_block_ptr != NULL);

        new_builder->is_reduce_in_global_level_vec[sub_matrix_id] = compressed_block_ptr->share_row_with_other_block;
        new_builder->reduce_type_in_global_level_vec[sub_matrix_id] = ATOMIC_ADD;
        
        new_builder->is_reduce_in_warp_level_vec[sub_matrix_id] = compressed_block_ptr->share_row_with_other_thread;
        new_builder->reduce_type_in_warp_level_vec[sub_matrix_id] = ATOMIC_ADD;

        new_builder->is_reduce_in_block_level_vec[sub_matrix_id] = compressed_block_ptr->share_row_with_other_warp;
        new_builder->reduce_type_in_block_level_vec[sub_matrix_id] = ATOMIC_ADD;

        new_builder->kernal_block_num_vec[sub_matrix_id] = get_config()["DEFAULT_THREAD_BLOCK_NUM"].as_integer();
        new_builder->kernal_thread_num_in_block_vec[sub_matrix_id] = get_config()["DEFAULT_THREAD_NUM_IN_BLOCK"].as_integer();

        // 初始化排序相关的数组
        if (matrix->is_sorted == true)
        {
            // 全局排序了，局部肯定不排序
            assert(compressed_block_ptr->y_write_index.size() == 0);
            new_builder->sub_block_sort_type_vec[sub_matrix_id] = GLOBAL_SORT;
        }
        else if (compressed_block_ptr->y_write_index.size() > 0)
        {
            // 有局部的排序
            new_builder->sub_block_sort_type_vec[sub_matrix_id] = SUB_BLOCK_SORT;
        }
        else
        {
            new_builder->sub_block_sort_type_vec[sub_matrix_id] = NO_SORT;
        }

        // 析构一些不需要的数据
        assert(compressed_block_ptr->read_index.size() == 7);

        for (unsigned long index_of_read_index = 0; index_of_read_index < compressed_block_ptr->read_index.size(); index_of_read_index++)
        {
            index_of_compress_block_t *read_index = compressed_block_ptr->read_index[index_of_read_index];

            if (read_index->child_tmp_row_csr_index_arr != NULL)
            {
                delete_arr_with_data_type(read_index->child_tmp_row_csr_index_arr, read_index->data_type_of_child_tmp_row_csr_index);
                read_index->child_tmp_row_csr_index_arr = NULL;
            }

            if (read_index->begin_index_in_tmp_row_csr_arr_of_block != NULL)
            {
                delete_arr_with_data_type(read_index->begin_index_in_tmp_row_csr_arr_of_block, read_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block);
                read_index->begin_index_in_tmp_row_csr_arr_of_block = NULL;
            }
        }
    }

    assert(new_builder->sub_block_sort_type_vec.size() == block_coor_table->item_arr.size());

    // 并且初始化模板指针
    for (unsigned long dense_block_index = 0; dense_block_index < block_coor_table->item_arr.size(); dense_block_index++)
    {
        // 给模板初始化为空指针
        new_builder->template_vec.push_back(NULL);
        new_builder->template_type_vec.push_back(NONE_TEMPLATE);
    }

    return new_builder;
}



void add_template_to_builder(code_builder_t *builder, void *template_ptr, template_type type, unsigned long dense_block_id)
{
    assert(builder != NULL && template_ptr != NULL);        
    assert(builder->op_manager->matrix->block_coor_table.item_arr.size() > dense_block_id);

    assert(builder->template_vec.size() > dense_block_id && builder->template_type_vec.size() > dense_block_id);

    builder->template_vec[dense_block_id] = template_ptr;
    builder->template_type_vec[dense_block_id] = type;
}

// 将归约类型和排序类型转化为字符串
string convert_sort_type_to_str(sort_type type)
{
    if (type == GLOBAL_SORT)
    {
        return "GLOBAL_SORT";
    }

    if (type == SUB_BLOCK_SORT)
    {
        return "SUB_BLOCK_SORT";
    }

    if (type == NO_SORT)
    {
        return "NO_SORT";
    }

    assert(false);
}

string convert_reduce_type_to_str(reduce_type type)
{
    if (type == ATOMIC_ADD)
    {
        return "ATOMIC_ADD";
    }

    if (type == SYNC_BY_SHARED_MEM)
    {
        return "SYNC_BY_SHARED_MEM";
    }

    assert(false);
}

// 将元数据转化为字符串
string convert_code_builder_to_str(code_builder_t *builder)
{
    assert(builder != NULL);

    string return_str = "builder{\n";

    // 打印7个数组
    // is_reduce_in_warp_level_vec
    return_str = return_str + "is_reduce_in_warp_level_vec[";
    for (int i = 0; i < builder->is_reduce_in_warp_level_vec.size(); i++)
    {
        return_str = return_str + to_string(builder->is_reduce_in_warp_level_vec[i]) + ",";
    }
    return_str = return_str + "],\n";

    // is_reduce_in_block_level_vec
    return_str = return_str + "is_reduce_in_block_level_vec[";
    for (int i = 0; i < builder->is_reduce_in_block_level_vec.size(); i++)
    {
        return_str = return_str + to_string(builder->is_reduce_in_block_level_vec[i]) + ",";
    }
    return_str = return_str + "],\n";

    // is_reduce_in_global_level_vec
    return_str = return_str + "is_reduce_in_global_level_vec[";
    for (int i = 0; i < builder->is_reduce_in_global_level_vec.size(); i++)
    {
        return_str = return_str + to_string(builder->is_reduce_in_global_level_vec[i]) + ",";
    }
    return_str = return_str + "],\n";

    // reduce_type_in_warp_level_vec
    return_str = return_str + "reduce_type_in_warp_level_vec[";
    for (int i = 0; i < builder->reduce_type_in_warp_level_vec.size(); i++)
    {
        return_str = return_str + convert_reduce_type_to_str(builder->reduce_type_in_warp_level_vec[i]) + ",";
    }
    return_str = return_str + "],\n";

    // reduce_type_block_level_vec
    return_str = return_str + "reduce_type_block_level_vec[";
    for (int i = 0; i < builder->reduce_type_in_block_level_vec.size(); i++)
    {
        return_str = return_str + convert_reduce_type_to_str(builder->reduce_type_in_block_level_vec[i]) + ",";
    }
    return_str = return_str + "],\n";

    // reduce_type_in_global_level_vec
    return_str = return_str + "reduce_type_in_global_level_vec[";
    for (int i = 0; i < builder->reduce_type_in_global_level_vec.size(); i++)
    {
        return_str = return_str + convert_reduce_type_to_str(builder->reduce_type_in_global_level_vec[i]) + ",";
    }
    return_str = return_str + "],\n";

    // sub_block_sort_type_vec
    return_str = return_str + "sub_block_sort_type_vec[";
    for (int i = 0; i < builder->sub_block_sort_type_vec.size(); i++)
    {
        return_str = return_str + convert_sort_type_to_str(builder->sub_block_sort_type_vec[i]) + ",";
    }
    return_str = return_str + "],\n";

    return_str = return_str + "}\n";

    return return_str;
}

string readFileIntoString(string filename)
{
    ifstream ifile(filename.c_str());
    //将文件读入到ostringstream对象buf中
    ostringstream buf;
    char ch;
    while (buf && ifile.get(ch))
    {
        buf.put(ch);
    }

    //返回与流对象buf关联的字符串
    return buf.str();
}

string code_of_template_data_struct(void *output_template, template_type type, unsigned long dense_block_id)
{
    assert(output_template != NULL);
    assert(type != NONE_TEMPLATE);

    if (type == DIRECT_ATOM_TEMPLATE)
    {
        direct_atom_template_t *template_ptr = (direct_atom_template_t *)output_template;
        return code_of_template_data_struct(template_ptr, dense_block_id);
    }

    if (type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
    {
        direct_atom_template_warp_compress_t *template_ptr = (direct_atom_template_warp_compress_t *)output_template;
        return code_of_template_data_struct(template_ptr, dense_block_id);
    }

    if (type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
    {
        direct_atom_template_warp_block_compress_t *template_ptr = (direct_atom_template_warp_block_compress_t *)output_template;
        return code_of_template_data_struct(template_ptr, dense_block_id);
    }

    if (type == SHARED_MEMORY_TEMPLATE)
    {
        shared_memory_template_t *template_ptr = (shared_memory_template_t *)output_template;
        return code_of_template_data_struct(template_ptr, dense_block_id);
    }

    if (type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
    {
        shared_memory_template_warp_compress_t *template_ptr = (shared_memory_template_warp_compress_t *)output_template;
        return code_of_template_data_struct(template_ptr, dense_block_id);
    }

    if (type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
    {
        shared_memory_long_row_template_t *template_ptr = (shared_memory_long_row_template_t *)output_template;
        return code_of_template_data_struct(template_ptr, dense_block_id);
    }

    if (type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        shared_memory_total_warp_reduce_template_t *template_ptr = (shared_memory_total_warp_reduce_template_t *)output_template;
        return code_of_template_data_struct(template_ptr, dense_block_id);
    }

    if (type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        direct_atom_total_warp_reduce_template_t *template_ptr = (direct_atom_total_warp_reduce_template_t *)output_template;
        return code_of_template_data_struct(template_ptr, dense_block_id);
    }

    if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
    {
        unaligned_warp_reduce_same_TLB_size_template_t* template_ptr = (unaligned_warp_reduce_same_TLB_size_template_t*)output_template;
        return code_of_template_data_struct(template_ptr, dense_block_id);
    }

    if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
    {
        unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t* template_ptr = (unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *) output_template;
        return code_of_template_data_struct(template_ptr, dense_block_id);
    }

    if (type == EMPTY_TEMPLATE)
    {
        empty_template_t *template_ptr = (empty_template_t *)output_template;
        return code_of_template_data_struct(template_ptr, dense_block_id);
    }

    cout << "code_of_template_data_struct: the template has not support now" << endl;
    assert(false);
}

string code_of_read_template_data_from_file_func_define(void *output_template, template_type type, unsigned long dense_block_id, bool force_not_share_global_sort_index)
{
    assert(output_template != NULL);
    assert(type != NONE_TEMPLATE);

    if (type == DIRECT_ATOM_TEMPLATE)
    {
        direct_atom_template_t *template_ptr = (direct_atom_template_t *)output_template;
        return code_of_read_template_data_from_file_func_define(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
    {
        direct_atom_template_warp_compress_t *template_ptr = (direct_atom_template_warp_compress_t *)output_template;
        return code_of_read_template_data_from_file_func_define(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
    {
        direct_atom_template_warp_block_compress_t *template_ptr = (direct_atom_template_warp_block_compress_t *)output_template;
        return code_of_read_template_data_from_file_func_define(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == SHARED_MEMORY_TEMPLATE)
    {
        shared_memory_template_t *template_ptr = (shared_memory_template_t *)output_template;
        return code_of_read_template_data_from_file_func_define(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
    {
        shared_memory_long_row_template_t *template_ptr = (shared_memory_long_row_template_t *)output_template;
        return code_of_read_template_data_from_file_func_define(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        shared_memory_total_warp_reduce_template_t *template_ptr = (shared_memory_total_warp_reduce_template_t *)output_template;
        return code_of_read_template_data_from_file_func_define(template_ptr, dense_block_id);
    }

    if (type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        direct_atom_total_warp_reduce_template_t *template_ptr = (direct_atom_total_warp_reduce_template_t *)output_template;
        return code_of_read_template_data_from_file_func_define(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
    {
        unaligned_warp_reduce_same_TLB_size_template_t* template_ptr = (unaligned_warp_reduce_same_TLB_size_template_t *)output_template;
        return code_of_read_template_data_from_file_func_define(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
    {
        unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t* template_ptr = (unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *)output_template;
        return code_of_read_template_data_from_file_func_define(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
    {
        shared_memory_template_warp_compress_t *template_ptr = (shared_memory_template_warp_compress_t *)output_template;
        return code_of_read_template_data_from_file_func_define(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == EMPTY_TEMPLATE)
    {
        empty_template_t *template_ptr = (empty_template_t *)output_template;
        return code_of_read_template_data_from_file_func_define(template_ptr, dense_block_id);
    }

    cout << "code_of_read_template_data_from_file_func_define: the template has not support now" << endl;
    assert(false);
}

// 生成新的头文件
string build_header_file(code_builder_t *builder)
{
    assert(builder != NULL);
    string return_str = readFileIntoString(get_config()["spmv_header_file"].as_string());

    // 遍历所有的模板，生成对应的代码，分别是数据结构定义和磁盘读
    return_str = return_str + "\n\n";

    assert(builder->op_manager->matrix->block_coor_table.item_arr.size() == builder->template_vec.size());

    for (unsigned long template_index = 0; template_index < builder->template_vec.size(); template_index++)
    {
        return_str = return_str + code_of_template_data_struct(builder->template_vec[template_index], builder->template_type_vec[template_index], template_index) + "\n";
    }

    // 读文件的函数声明
    for (unsigned long template_index = 0; template_index < builder->template_vec.size(); template_index++)
    {
        return_str = return_str + code_of_read_template_data_from_file_func_define(builder->template_vec[template_index], builder->template_type_vec[template_index], template_index) + "\n";
    }

    return return_str;
}

string build_header_file(code_builder_t *builder, vector<int> sub_matrix_id_vec)
{
    assert(builder != NULL);
    assert(sub_matrix_id_vec.size() > 0);

    // 每一个子块都不超标
    for (int i = 0; i < sub_matrix_id_vec.size(); i++)
    {
        int sub_matrix_id = sub_matrix_id_vec[i];
        
        assert(sub_matrix_id < builder->template_vec.size());
        assert(sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());
    }


    string return_str = readFileIntoString(get_config()["spmv_header_file"].as_string());

    // 遍历所有的模板，生成对应的代码，分别是数据结构定义和磁盘读
    return_str = return_str + "\n\n";

    assert(builder->op_manager->matrix->block_coor_table.item_arr.size() == builder->template_vec.size());
    assert(builder->template_vec.size() == builder->template_type_vec.size());
    
    // 遍历所有要生成模板的子块
    for (int i = 0; i < sub_matrix_id_vec.size(); i++)
    {
        int sub_matrix_id = sub_matrix_id_vec[i];

        return_str = return_str + code_of_template_data_struct(builder->template_vec[sub_matrix_id], builder->template_type_vec[sub_matrix_id], sub_matrix_id) + "\n";
    }

    for (int i = 0; i < sub_matrix_id_vec.size(); i++)
    {
        int sub_matrix_id = sub_matrix_id_vec[i];
        // 
        return_str = return_str + code_of_read_template_data_from_file_func_define(builder->template_vec[sub_matrix_id], builder->template_type_vec[sub_matrix_id], sub_matrix_id, true) + "\n";
    }

    return return_str;
}

// string build_header_file(code_builder_t *builder)
// {
//     assert(builder != NULL);
//     string return_str = readFileIntoString(get_config()["spmv_header_file"].as_string());

//     return_str = return_str + code_of_compressed_matrix_content_define(builder);

//     return_str = return_str + "\n" + code_of_all_compressed_block_define(builder);

//     return_str = return_str + "\n" + code_of_matrix_file_read(builder);

//     return return_str;
// }

string code_of_compressed_matrix_content_define(code_builder_t *code_builder)
{
    assert(code_builder != NULL);

    string return_str = "\n\ntypedef struct compressed_matrix_content\n{\n";
    sparse_struct_t *matrix = code_builder->op_manager->matrix;
    // 遍历所有的子块中的数组
    unsigned long index_of_dense_block;
    for (index_of_dense_block = 0; index_of_dense_block < matrix->block_coor_table.item_arr.size(); index_of_dense_block++)
    {
        compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[index_of_dense_block]->compressed_block_ptr;
        // dense block的值数组
        assert(compressed_block_view != NULL && compressed_block_view->staggered_padding_val_arr != NULL);
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(index_of_dense_block, -1, "staggered_padding_val_arr") + " = " + to_string(compressed_block_view->staggered_padding_val_arr_size) + ";\n";
        return_str = return_str + code_of_data_type(compressed_block_view->val_data_type) + "* " + code_of_arr_var_name(index_of_dense_block, -1, "staggered_padding_val_arr") + ";\n\n";

        // dense block的所有读索引
        unsigned long index_of_read_index;
        assert(compressed_block_view->read_index.size() == 7);
        for (index_of_read_index = 0; index_of_read_index < compressed_block_view->read_index.size(); index_of_read_index++)
        {
            index_of_compress_block_t *read_index = compressed_block_view->read_index[index_of_read_index];

            // 写对应的数组
            if (read_index->index_arr != NULL)
            {
                return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_arr") + "=" + to_string(read_index->length) + ";\n";
                return_str = return_str + code_of_data_type(read_index->index_data_type) + "* " + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_arr") + ";\n";
            }

            if (read_index->index_of_the_first_row_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3 || index_of_read_index == 4);
                return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_of_the_first_row_arr") + "=" + to_string(read_index->block_num) + ";\n";
                return_str = return_str + code_of_data_type(read_index->data_type_of_index_of_the_first_row_arr) + "* " + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_of_the_first_row_arr") + ";\n";
            }

            if (read_index->row_number_of_block_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "row_number_of_block_arr") + "=" + to_string(read_index->block_num) + ";\n";
                return_str = return_str + code_of_data_type(read_index->data_type_of_row_number_of_block_arr) + "* " + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "row_number_of_block_arr") + ";\n";
            }

            if (read_index->tmp_result_write_index_arr != NULL)
            {
                // 只有warp层次的分块需要
                assert(index_of_read_index == 3);
                return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "tmp_result_write_index_arr") + "=" + to_string(read_index->block_num) + ";\n";
                return_str = return_str + code_of_data_type(read_index->data_type_of_tmp_result_write_index_arr) + "* " + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "tmp_result_write_index_arr") + ";\n";
            }

            if (read_index->coo_begin_index_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                unsigned long size;

                if (index_of_read_index == 2)
                {
                    size = read_index->length;
                }
                else
                {
                    size = read_index->block_num;
                }
                return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_begin_index_arr") + "=" + to_string(size) + ";\n";
                return_str = return_str + code_of_data_type(read_index->data_type_of_coo_begin_index_arr) + "* " + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_begin_index_arr") + ";\n";
            }

            if (read_index->coo_block_size_arr != NULL)
            {
                assert(index_of_read_index == 3 || index_of_read_index == 4);
                return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_block_size_arr") + "=" + to_string(compressed_block_view->read_index[3]->block_num) + ";\n";
                return_str = return_str + code_of_data_type(read_index->data_type_of_coo_block_size_arr) + "* " + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_block_size_arr") + ";\n";
            }

            // 归约信息
            if (read_index->child_tmp_row_csr_index_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "child_tmp_row_csr_index_arr") + "=" + to_string(read_index->size_of_child_tmp_row_csr_index) + ";\n";
                return_str = return_str + code_of_data_type(read_index->data_type_of_child_tmp_row_csr_index) + "* " + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "child_tmp_row_csr_index_arr") + ";\n";
            }

            // 归约信息的索引
            if (read_index->begin_index_in_tmp_row_csr_arr_of_block != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "begin_index_in_tmp_row_csr_arr_of_block") + "=" + to_string(read_index->block_num) + ";\n";
                return_str = return_str + code_of_data_type(read_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block) + "* " + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "begin_index_in_tmp_row_csr_arr_of_block") + ";\n";
            }

            return_str = return_str + "\n";
        }

        // return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" +
        // 遍历所有输入索引
        unsigned long index_of_y_write_index;
        for (index_of_y_write_index = 0; index_of_y_write_index < compressed_block_view->y_write_index.size(); index_of_y_write_index++)
        {
            index_of_compress_block_t *y_write_index = compressed_block_view->y_write_index[index_of_y_write_index];
            assert(y_write_index != NULL && y_write_index->index_arr != NULL);

            return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_y_write_arr_var_name(index_of_dense_block, index_of_y_write_index, "index_arr") + "=" + to_string(y_write_index->length) + ";\n";
            return_str = return_str + code_of_data_type(y_write_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block) + "* " + code_of_y_write_arr_var_name(index_of_dense_block, index_of_y_write_index, "index_arr") + ";\n";
        }
    }

    return_str = return_str + "}compressed_matrix_content_t;\n";

    return return_str;
}

// 对部分子块的结构体的定义，部分子块定义矩阵结构
string code_of_compressed_matrix_content_define(code_builder_t *code_builder, vector<int> sub_matrix_id_vec)
{
    assert(code_builder != NULL && sub_matrix_id_vec.size() > 0);

    string return_str = "\n\ntypedef struct compressed_matrix_content\n{\n";

    sparse_struct_t *matrix = code_builder->op_manager->matrix;
    assert(matrix != NULL);

    // 遍历所有需要进一步执行子矩阵元数据结构体定义的子矩阵
    for (int i = 0; i < sub_matrix_id_vec.size(); i++)
    {
        int sub_matrix_id = sub_matrix_id_vec[i];

        assert(sub_matrix_id < code_builder->template_vec.size());
        assert(sub_matrix_id < code_builder->op_manager->matrix->block_coor_table.item_arr.size());

        assert(code_builder->template_vec[sub_matrix_id] != NULL);
        assert(code_builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr != NULL);
        assert(code_builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);

        compressed_block_t* compressed_block_ptr = code_builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

        // dense block的值数组
        assert(compressed_block_ptr != NULL && compressed_block_ptr->staggered_padding_val_arr != NULL);
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(sub_matrix_id, -1, "staggered_padding_val_arr") + " = " + to_string(compressed_block_ptr->staggered_padding_val_arr_size) + ";\n";
        return_str = return_str + code_of_data_type(compressed_block_ptr->val_data_type) + "* " + code_of_arr_var_name(sub_matrix_id, -1, "staggered_padding_val_arr") + ";\n\n";
    
        // 遍历当前子块的所有读索引
        for (unsigned long index_of_read_index = 0; index_of_read_index < compressed_block_ptr->read_index.size(); index_of_read_index++)
        {
            index_of_compress_block_t *read_index = compressed_block_ptr->read_index[index_of_read_index];

            // 写对应的数组
            if (read_index->index_arr != NULL)
            {
                return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "index_arr") + "=" + to_string(read_index->length) + ";\n";
                return_str = return_str + code_of_data_type(read_index->index_data_type) + "* " + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "index_arr") + ";\n";
            }

            if (read_index->index_of_the_first_row_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3 || index_of_read_index == 4);
                return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "index_of_the_first_row_arr") + "=" + to_string(read_index->block_num) + ";\n";
                return_str = return_str + code_of_data_type(read_index->data_type_of_index_of_the_first_row_arr) + "* " + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "index_of_the_first_row_arr") + ";\n";
            }

            if (read_index->row_number_of_block_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "row_number_of_block_arr") + "=" + to_string(read_index->block_num) + ";\n";
                return_str = return_str + code_of_data_type(read_index->data_type_of_row_number_of_block_arr) + "* " + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "row_number_of_block_arr") + ";\n";
            }

            if (read_index->tmp_result_write_index_arr != NULL)
            {
                // 只有warp层次的分块需要
                assert(index_of_read_index == 3);
                return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "tmp_result_write_index_arr") + "=" + to_string(read_index->block_num) + ";\n";
                return_str = return_str + code_of_data_type(read_index->data_type_of_tmp_result_write_index_arr) + "* " + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "tmp_result_write_index_arr") + ";\n";
            }

            if (read_index->coo_begin_index_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                unsigned long size;

                if (index_of_read_index == 2)
                {
                    size = read_index->length;
                }
                else
                {
                    size = read_index->block_num;
                }
                return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "coo_begin_index_arr") + "=" + to_string(size) + ";\n";
                return_str = return_str + code_of_data_type(read_index->data_type_of_coo_begin_index_arr) + "* " + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "coo_begin_index_arr") + ";\n";
            }

            if (read_index->coo_block_size_arr != NULL)
            {
                assert(index_of_read_index == 3 || index_of_read_index == 4);
                return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "coo_block_size_arr") + "=" + to_string(compressed_block_ptr->read_index[3]->block_num) + ";\n";
                return_str = return_str + code_of_data_type(read_index->data_type_of_coo_block_size_arr) + "* " + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "coo_block_size_arr") + ";\n";
            }

            // 归约信息
            if (read_index->child_tmp_row_csr_index_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "child_tmp_row_csr_index_arr") + "=" + to_string(read_index->size_of_child_tmp_row_csr_index) + ";\n";
                return_str = return_str + code_of_data_type(read_index->data_type_of_child_tmp_row_csr_index) + "* " + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "child_tmp_row_csr_index_arr") + ";\n";
            }

            // 归约信息的索引
            if (read_index->begin_index_in_tmp_row_csr_arr_of_block != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "begin_index_in_tmp_row_csr_arr_of_block") + "=" + to_string(read_index->block_num) + ";\n";
                return_str = return_str + code_of_data_type(read_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block) + "* " + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "begin_index_in_tmp_row_csr_arr_of_block") + ";\n";
            }

            return_str = return_str + "\n";
        }

        // 遍历所有写y的索引
        for (unsigned long index_of_y_write_index = 0; index_of_y_write_index < compressed_block_ptr->y_write_index.size(); index_of_y_write_index++)
        {
            index_of_compress_block_t *y_write_index = compressed_block_ptr->y_write_index[index_of_y_write_index];
            assert(y_write_index != NULL && y_write_index->index_arr != NULL);

            return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_y_write_arr_var_name(sub_matrix_id, index_of_y_write_index, "index_arr") + "=" + to_string(y_write_index->length) + ";\n";
            return_str = return_str + code_of_data_type(y_write_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block) + "* " + code_of_y_write_arr_var_name(sub_matrix_id, index_of_y_write_index, "index_arr") + ";\n";
        }
    }

    return_str = return_str + "}compressed_matrix_content_t;\n";

    return return_str;
}

string code_of_all_compressed_block_define(code_builder_t *code_builder)
{
    assert(code_builder != NULL);

    string return_str = "\ntypedef struct all_compressed_block\n{\ncompressed_matrix_content_t * all_compressed_matrix_info;\n";

    // 查看是否有全局的排序
    if (code_builder->op_manager->matrix->sorted_row_index != NULL)
    {
        assert(code_builder->sub_block_sort_type_vec[0] == GLOBAL_SORT);
        return_str = return_str + code_of_data_type(UNSIGNED_LONG) + " size_of_" + code_of_arr_var_name(-1, -1, "sorted_row_index") + "=" + to_string(code_builder->op_manager->matrix->dense_row_number) + ";\n";
        return_str = return_str + code_of_data_type(code_builder->op_manager->matrix->data_type_of_sorted_row_index) + "* " + code_of_arr_var_name(-1, -1, "sorted_row_index") + ";\n";
    }

    return_str = return_str + "}all_compressed_block_t;\n";

    return return_str;
}

string code_of_data_type(data_type type)
{
    if (type == CHAR)
    {
        return "char";
    }
    else if (type == UNSIGNED_CHAR)
    {
        return "unsigned char";
    }
    else if (type == SHORT)
    {
        return "short";
    }
    else if (type == UNSIGNED_SHORT)
    {
        return "unsigned short";
    }
    else if (type == INT)
    {
        return "int";
    }
    else if (type == UNSIGNED_INT)
    {
        return "unsigned int";
    }
    else if (type == LONG)
    {
        return "long";
    }
    else if (type == UNSIGNED_LONG)
    {
        return "unsigned long";
    }
    else if (type == LONG_LONG)
    {
        return "long long";
    }
    else if (type == UNSIGNED_LONG_LONG)
    {
        return "unsigned long long";
    }
    else if (type == FLOAT)
    {
        return "float";
    }
    else if (type == DOUBLE)
    {
        return "double";
    }
    else if (type == BOOL)
    {
        return "bool";
    }

    assert(false);
}

// 某一个稠密子块、某一个read_index的某一个数组的变量名，传入-1表示没有则个层次的索引
string code_of_arr_var_name(int index_of_dense_block, int index_of_read_index, string arr_name)
{
    if (index_of_dense_block == -1 && index_of_read_index != -1)
    {
        return "read_index_" + to_string(index_of_read_index) + "_" + arr_name;
    }
    else if (index_of_dense_block != -1 && index_of_read_index == -1)
    {
        return "dense_" + to_string(index_of_dense_block) + "_" + arr_name;
    }
    else if (index_of_dense_block == -1 && index_of_dense_block == -1)
    {
        return arr_name;
    }
    else
    {
        return "dense_" + to_string(index_of_dense_block) + "_read_index_" + to_string(index_of_read_index) + "_" + arr_name;
    }
}

string code_of_y_write_arr_var_name(int index_of_dense_block, int index_of_y_write_index, string arr_name)
{
    assert(index_of_dense_block >= -1 && index_of_y_write_index >= 0);

    // -1省略
    if (index_of_dense_block == -1)
    {
        return "y_write_index_" + to_string(index_of_y_write_index) + "_" + arr_name;
    }
    else
    {
        return "dense_" + to_string(index_of_dense_block) + "_y_write_index_" + to_string(index_of_y_write_index) + "_" + arr_name;
    }
}

string code_of_matrix_file_read(code_builder_t *code_builder)
{
    assert(code_builder != NULL && code_builder->op_manager != NULL && code_builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = code_builder->op_manager->matrix;

    string return_str = "all_compressed_block_t *read_matrix_from_file(string file_name_prefix)\n{\nall_compressed_block_t* total_matrix = new all_compressed_block_t();\n";

    return_str = return_str + "unsigned long num_of_compressed_matrix = " + to_string(matrix->block_coor_table.item_arr.size()) + ";\n";

    // 读全局排序之后的索引
    if (matrix->is_sorted == true)
    {
        assert(matrix->sorted_row_index != NULL);
        return_str = return_str + "total_matrix->" + code_of_arr_var_name(-1, -1, "sorted_row_index") + " = (" + code_of_data_type(matrix->data_type_of_sorted_row_index) + "*)read_arr_from_file_with_data_type(total_matrix->size_of_" + code_of_arr_var_name(-1, -1, "sorted_row_index") + ", " + convert_data_type_to_string(matrix->data_type_of_sorted_row_index) + ", file_name_prefix + \"/" + code_of_arr_var_name(-1, -1, "sorted_row_index\"") + ");\n";
    }

    return_str = return_str + "compressed_matrix_content_t* compressed_block = new compressed_matrix_content_t();\n\n";

    unsigned long index_of_dense_block;
    for (index_of_dense_block = 0; index_of_dense_block < matrix->block_coor_table.item_arr.size(); index_of_dense_block++)
    {
        return_str = return_str + "string sub_block_file_prefix = file_name_prefix + \"/dense_block_" + to_string(index_of_dense_block) + "\";\n";

        compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[index_of_dense_block]->compressed_block_ptr;

        // 所有稠密视图子块的一系列的数据的代码
        assert(compressed_block_view != NULL && compressed_block_view->staggered_padding_val_arr != NULL);

        return_str = return_str + "compressed_block->" + code_of_arr_var_name(index_of_dense_block, -1, "staggered_padding_val_arr") + " = (" + code_of_data_type(compressed_block_view->val_data_type) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, -1, "staggered_padding_val_arr") + ", " + convert_data_type_to_string(compressed_block_view->val_data_type) + ", sub_block_file_prefix + " + "\"/staggered_padding_val_arr\");\n\n";

        unsigned long index_of_read_index;
        assert(compressed_block_view->read_index.size() == 7);
        for (index_of_read_index = 0; index_of_read_index < compressed_block_view->read_index.size(); index_of_read_index++)
        {
            index_of_compress_block_t *read_index = compressed_block_view->read_index[index_of_read_index];

            if (read_index->index_arr != NULL)
            {
                return_str = return_str + "compressed_block->" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_arr") + " = (" + code_of_data_type(read_index->index_data_type) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_arr") + ", " + convert_data_type_to_string(read_index->index_data_type) + ", sub_block_file_prefix + \"/read_index_" + to_string(index_of_read_index) + "/index_arr\");\n";
            }

            if (read_index->index_of_the_first_row_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3 || index_of_read_index == 4);
                return_str = return_str + "compressed_block->" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_of_the_first_row_arr") + " = (" + code_of_data_type(read_index->data_type_of_index_of_the_first_row_arr) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_of_the_first_row_arr") + ", " + convert_data_type_to_string(read_index->data_type_of_index_of_the_first_row_arr) + ", sub_block_file_prefix + \"/read_index_" + to_string(index_of_read_index) + "/index_of_the_first_row_arr\");\n";
            }

            if (read_index->row_number_of_block_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                return_str = return_str + "compressed_block->" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "row_number_of_block_arr") + " = (" + code_of_data_type(read_index->data_type_of_row_number_of_block_arr) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "row_number_of_block_arr") + ", " + convert_data_type_to_string(read_index->data_type_of_row_number_of_block_arr) + ", sub_block_file_prefix + \"/read_index_" + to_string(index_of_read_index) + "/row_number_of_block_arr\");\n";
            }

            if (read_index->tmp_result_write_index_arr != NULL)
            {
                // 只有warp层次的分块需要
                assert(index_of_read_index == 3);
                return_str = return_str + "compressed_block->" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "tmp_result_write_index_arr") + " = (" + code_of_data_type(read_index->data_type_of_tmp_result_write_index_arr) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "tmp_result_write_index_arr") + ", " + convert_data_type_to_string(read_index->data_type_of_tmp_result_write_index_arr) + ", sub_block_file_prefix + \"/read_index_" + to_string(index_of_read_index) + "/tmp_result_write_index_arr\");\n";
            }

            if (read_index->coo_begin_index_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                return_str = return_str + "compressed_block->" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_begin_index_arr") + " = (" + code_of_data_type(read_index->data_type_of_coo_begin_index_arr) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_begin_index_arr") + ", " + convert_data_type_to_string(read_index->data_type_of_coo_begin_index_arr) + ", sub_block_file_prefix + \"/read_index_" + to_string(index_of_read_index) + "/coo_begin_index_arr\");\n";
            }

            if (read_index->coo_block_size_arr != NULL)
            {
                assert(index_of_read_index == 3 || index_of_read_index == 4);
                return_str = return_str + "compressed_block->" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_block_size_arr") + " = (" + code_of_data_type(read_index->data_type_of_coo_block_size_arr) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_block_size_arr") + ", " + convert_data_type_to_string(read_index->data_type_of_coo_block_size_arr) + ", sub_block_file_prefix + \"/read_index_" + to_string(index_of_read_index) + "/coo_block_size_arr\");\n";
            }

            if (read_index->child_tmp_row_csr_index_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                return_str = return_str + "compressed_block->" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "child_tmp_row_csr_index_arr") + " = (" + code_of_data_type(read_index->data_type_of_child_tmp_row_csr_index) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "child_tmp_row_csr_index_arr") + ", " + convert_data_type_to_string(read_index->data_type_of_child_tmp_row_csr_index) + ", sub_block_file_prefix + \"/read_index_" + to_string(index_of_read_index) + "/child_tmp_row_csr_index_arr\");\n";
            }

            if (read_index->begin_index_in_tmp_row_csr_arr_of_block != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                return_str = return_str + "compressed_block->" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "begin_index_in_tmp_row_csr_arr_of_block") + " = (" + code_of_data_type(read_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "begin_index_in_tmp_row_csr_arr_of_block") + ", " + convert_data_type_to_string(read_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block) + ", sub_block_file_prefix + \"/read_index_" + to_string(index_of_read_index) + "/begin_index_in_tmp_row_csr_arr_of_block\");\n";
            }
            return_str = return_str + "\n";
        }

        // 遍历所有的输出索引
        unsigned long index_of_y_write_index;
        for (index_of_y_write_index = 0; index_of_y_write_index < compressed_block_view->y_write_index.size(); index_of_y_write_index++)
        {
            index_of_compress_block_t *y_write_index = compressed_block_view->y_write_index[index_of_y_write_index];
            assert(y_write_index != NULL && y_write_index->index_arr != NULL);

            return_str = return_str + "compressed_block->" + code_of_y_write_arr_var_name(index_of_dense_block, index_of_y_write_index, "index_arr") + " = (" + code_of_data_type(y_write_index->index_data_type) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_y_write_arr_var_name(index_of_dense_block, index_of_y_write_index, "index_arr") + ", " + convert_data_type_to_string(y_write_index->index_data_type) + ", sub_block_file_prefix + \"/y_write_arr_index_" + to_string(index_of_read_index) + "/index_arr\");\n";
            return_str = return_str + "\n";
        }
    }

    return_str = return_str + "\ntotal_matrix->all_compressed_matrix_info = compressed_block;\nreturn total_matrix;\n";

    return_str = return_str + "}\n";

    return return_str;
}

// 对矩阵的特定子块执行从disk的读操作，
string code_of_matrix_file_read(code_builder_t *code_builder, vector<int> sub_matrix_id_vec)
{
    assert(code_builder != NULL && code_builder->op_manager != NULL && code_builder->op_manager->matrix != NULL);
    
    sparse_struct_t *matrix = code_builder->op_manager->matrix;

    string return_str = "all_compressed_block_t *read_matrix_from_file(string file_name_prefix)\n{\nall_compressed_block_t* total_matrix = new all_compressed_block_t();\n";

    return_str = return_str + "unsigned long num_of_compressed_matrix = " + to_string(matrix->block_coor_table.item_arr.size()) + ";\n";

    // 读全局排序之后的索引
    if (matrix->is_sorted == true)
    {
        assert(matrix->sorted_row_index != NULL);
        return_str = return_str + "total_matrix->" + code_of_arr_var_name(-1, -1, "sorted_row_index") + " = (" + code_of_data_type(matrix->data_type_of_sorted_row_index) + "*)read_arr_from_file_with_data_type(total_matrix->size_of_" + code_of_arr_var_name(-1, -1, "sorted_row_index") + ", " + convert_data_type_to_string(matrix->data_type_of_sorted_row_index) + ", file_name_prefix + \"/" + code_of_arr_var_name(-1, -1, "sorted_row_index\"") + ");\n";
    }

    return_str = return_str + "compressed_matrix_content_t* compressed_block = new compressed_matrix_content_t();\n\n";

    for (int i = 0; i < sub_matrix_id_vec.size(); i++)
    {
        int sub_matrix_id = sub_matrix_id_vec[i];

        // 满足要求
        assert(sub_matrix_id < code_builder->op_manager->matrix->block_coor_table.item_arr.size());
        assert(code_builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id] != NULL);

        return_str = return_str + "string sub_block_file_prefix = file_name_prefix + \"/dense_block_" + to_string(sub_matrix_id) + "\";\n";

        compressed_block_t* compressed_block_ptr = matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr;

        assert(compressed_block_ptr != NULL && compressed_block_ptr->staggered_padding_val_arr != NULL);

        return_str = return_str + "compressed_block->" + code_of_arr_var_name(sub_matrix_id, -1, "staggered_padding_val_arr") + " = (" + code_of_data_type(compressed_block_ptr->val_data_type) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(sub_matrix_id, -1, "staggered_padding_val_arr") + ", " + convert_data_type_to_string(compressed_block_ptr->val_data_type) + ", sub_block_file_prefix + " + "\"/staggered_padding_val_arr\");\n\n";

        unsigned long index_of_read_index;
        assert(compressed_block_ptr->read_index.size() == 7);
        for (index_of_read_index = 0; index_of_read_index < compressed_block_ptr->read_index.size(); index_of_read_index++)
        {
            index_of_compress_block_t *read_index = compressed_block_ptr->read_index[index_of_read_index];

            if (read_index->index_arr != NULL)
            {
                return_str = return_str + "compressed_block->" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "index_arr") + " = (" + code_of_data_type(read_index->index_data_type) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "index_arr") + ", " + convert_data_type_to_string(read_index->index_data_type) + ", sub_block_file_prefix + \"/read_index_" + to_string(index_of_read_index) + "/index_arr\");\n";
            }

            if (read_index->index_of_the_first_row_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3 || index_of_read_index == 4);
                return_str = return_str + "compressed_block->" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "index_of_the_first_row_arr") + " = (" + code_of_data_type(read_index->data_type_of_index_of_the_first_row_arr) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "index_of_the_first_row_arr") + ", " + convert_data_type_to_string(read_index->data_type_of_index_of_the_first_row_arr) + ", sub_block_file_prefix + \"/read_index_" + to_string(index_of_read_index) + "/index_of_the_first_row_arr\");\n";
            }

            if (read_index->row_number_of_block_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                return_str = return_str + "compressed_block->" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "row_number_of_block_arr") + " = (" + code_of_data_type(read_index->data_type_of_row_number_of_block_arr) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "row_number_of_block_arr") + ", " + convert_data_type_to_string(read_index->data_type_of_row_number_of_block_arr) + ", sub_block_file_prefix + \"/read_index_" + to_string(index_of_read_index) + "/row_number_of_block_arr\");\n";
            }

            if (read_index->tmp_result_write_index_arr != NULL)
            {
                // 只有warp层次的分块需要
                assert(index_of_read_index == 3);
                return_str = return_str + "compressed_block->" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "tmp_result_write_index_arr") + " = (" + code_of_data_type(read_index->data_type_of_tmp_result_write_index_arr) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "tmp_result_write_index_arr") + ", " + convert_data_type_to_string(read_index->data_type_of_tmp_result_write_index_arr) + ", sub_block_file_prefix + \"/read_index_" + to_string(index_of_read_index) + "/tmp_result_write_index_arr\");\n";
            }

            if (read_index->coo_begin_index_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                return_str = return_str + "compressed_block->" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "coo_begin_index_arr") + " = (" + code_of_data_type(read_index->data_type_of_coo_begin_index_arr) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "coo_begin_index_arr") + ", " + convert_data_type_to_string(read_index->data_type_of_coo_begin_index_arr) + ", sub_block_file_prefix + \"/read_index_" + to_string(index_of_read_index) + "/coo_begin_index_arr\");\n";
            }

            if (read_index->coo_block_size_arr != NULL)
            {
                assert(index_of_read_index == 3 || index_of_read_index == 4);
                return_str = return_str + "compressed_block->" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "coo_block_size_arr") + " = (" + code_of_data_type(read_index->data_type_of_coo_block_size_arr) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "coo_block_size_arr") + ", " + convert_data_type_to_string(read_index->data_type_of_coo_block_size_arr) + ", sub_block_file_prefix + \"/read_index_" + to_string(index_of_read_index) + "/coo_block_size_arr\");\n";
            }

            if (read_index->child_tmp_row_csr_index_arr != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                return_str = return_str + "compressed_block->" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "child_tmp_row_csr_index_arr") + " = (" + code_of_data_type(read_index->data_type_of_child_tmp_row_csr_index) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "child_tmp_row_csr_index_arr") + ", " + convert_data_type_to_string(read_index->data_type_of_child_tmp_row_csr_index) + ", sub_block_file_prefix + \"/read_index_" + to_string(index_of_read_index) + "/child_tmp_row_csr_index_arr\");\n";
            }

            if (read_index->begin_index_in_tmp_row_csr_arr_of_block != NULL)
            {
                assert(index_of_read_index == 2 || index_of_read_index == 3);
                return_str = return_str + "compressed_block->" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "begin_index_in_tmp_row_csr_arr_of_block") + " = (" + code_of_data_type(read_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_arr_var_name(sub_matrix_id, index_of_read_index, "begin_index_in_tmp_row_csr_arr_of_block") + ", " + convert_data_type_to_string(read_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block) + ", sub_block_file_prefix + \"/read_index_" + to_string(index_of_read_index) + "/begin_index_in_tmp_row_csr_arr_of_block\");\n";
            }
            return_str = return_str + "\n";
        }

        // 遍历所有的输出索引
        unsigned long index_of_y_write_index;
        for (index_of_y_write_index = 0; index_of_y_write_index < compressed_block_ptr->y_write_index.size(); index_of_y_write_index++)
        {
            index_of_compress_block_t *y_write_index = compressed_block_ptr->y_write_index[index_of_y_write_index];
            assert(y_write_index != NULL && y_write_index->index_arr != NULL);

            return_str = return_str + "compressed_block->" + code_of_y_write_arr_var_name(sub_matrix_id, index_of_y_write_index, "index_arr") + " = (" + code_of_data_type(y_write_index->index_data_type) + "*)read_arr_from_file_with_data_type(compressed_block->size_of_" + code_of_y_write_arr_var_name(sub_matrix_id, index_of_y_write_index, "index_arr") + ", " + convert_data_type_to_string(y_write_index->index_data_type) + ", sub_block_file_prefix + \"/y_write_arr_index_" + to_string(index_of_read_index) + "/index_arr\");\n";
            return_str = return_str + "\n";
        }
    }

    return_str = return_str + "\ntotal_matrix->all_compressed_matrix_info = compressed_block;\nreturn total_matrix;\n";

    return_str = return_str + "}\n";

    return return_str;
}

void write_string_to_file(string file_name, string output_str)
{
    ofstream outfile(file_name, ios::trunc);
    outfile << output_str << endl;
}

// 最后一个参数用来记录全局排序行索引是不是需要禁止共享
string code_of_write_template_data_to_gpu(void *output_template, template_type type, unsigned long dense_block_id, bool force_not_share_global_sort_index)
{
    assert(output_template != NULL && type != NONE_TEMPLATE);

    if (type == DIRECT_ATOM_TEMPLATE)
    {
        direct_atom_template_t *template_ptr = (direct_atom_template_t *)output_template;
        return code_of_write_template_data_to_gpu(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
    {
        direct_atom_template_warp_compress_t *template_ptr = (direct_atom_template_warp_compress_t *)output_template;
        return code_of_write_template_data_to_gpu(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
    {
        direct_atom_template_warp_block_compress_t *template_ptr = (direct_atom_template_warp_block_compress_t *)output_template;
        return code_of_write_template_data_to_gpu(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == SHARED_MEMORY_TEMPLATE)
    {
        shared_memory_template_t *template_ptr = (shared_memory_template_t *)output_template;
        return code_of_write_template_data_to_gpu(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
    {
        shared_memory_long_row_template_t *template_ptr = (shared_memory_long_row_template_t *)output_template;
        return code_of_write_template_data_to_gpu(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        shared_memory_total_warp_reduce_template_t *template_ptr = (shared_memory_total_warp_reduce_template_t *)output_template;
        return code_of_write_template_data_to_gpu(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        direct_atom_total_warp_reduce_template_t *template_ptr = (direct_atom_total_warp_reduce_template_t *)output_template;
        return code_of_write_template_data_to_gpu(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
    {
        unaligned_warp_reduce_same_TLB_size_template_t* template_ptr = (unaligned_warp_reduce_same_TLB_size_template_t *)output_template;
        return code_of_write_template_data_to_gpu(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
    {
        unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t* template_ptr = (unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *)output_template;
        return code_of_write_template_data_to_gpu(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
    {
        shared_memory_template_warp_compress_t *template_ptr = (shared_memory_template_warp_compress_t *)output_template;
        return code_of_write_template_data_to_gpu(template_ptr, dense_block_id, force_not_share_global_sort_index);
    }

    if (type == EMPTY_TEMPLATE)
    {
        empty_template_t *template_ptr = (empty_template_t *)output_template;
        return code_of_write_template_data_to_gpu(template_ptr, dense_block_id);
    }

    cout << "code_of_write_template_data_to_gpu: has not support this template type" << endl;
    assert(false);
}

string code_of_template_kernal(void *output_template, template_type type, unsigned long dense_block_id)
{
    assert(output_template != NULL && type != NONE_TEMPLATE);

    if (type == DIRECT_ATOM_TEMPLATE)
    {
        direct_atom_template_t *template_ptr = (direct_atom_template_t *)output_template;
        return code_of_template_kernal(template_ptr, dense_block_id);
    }

    if (type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
    {
        direct_atom_template_warp_compress_t *template_ptr = (direct_atom_template_warp_compress_t *)output_template;
        return code_of_template_kernal(template_ptr, dense_block_id);
    }

    if (type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
    {
        direct_atom_template_warp_block_compress_t *template_ptr = (direct_atom_template_warp_block_compress_t *)output_template;
        return code_of_template_kernal(template_ptr, dense_block_id);
    }

    if (type == SHARED_MEMORY_TEMPLATE)
    {
        shared_memory_template_t *template_ptr = (shared_memory_template_t *)output_template;
        return code_of_template_kernal(template_ptr, dense_block_id);
    }

    if (type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
    {
        shared_memory_long_row_template_t *template_ptr = (shared_memory_long_row_template_t *)output_template;
        return code_of_template_kernal(template_ptr, dense_block_id);
    }

    if (type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        shared_memory_total_warp_reduce_template_t *template_ptr = (shared_memory_total_warp_reduce_template_t *)output_template;
        return code_of_template_kernal(template_ptr, dense_block_id);
    }

    if (type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        direct_atom_total_warp_reduce_template_t *template_ptr = (direct_atom_total_warp_reduce_template_t *)output_template;
        return code_of_template_kernal(template_ptr, dense_block_id);
    }

    if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
    {
        unaligned_warp_reduce_same_TLB_size_template_t* template_ptr = (unaligned_warp_reduce_same_TLB_size_template_t*)output_template;
        return code_of_template_kernal(template_ptr, dense_block_id);
    }

    if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
    {
        unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t* template_ptr = (unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *)output_template;
        return code_of_template_kernal(template_ptr, dense_block_id);
    }

    if (type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
    {
        shared_memory_template_warp_compress_t *template_ptr = (shared_memory_template_warp_compress_t *)output_template;
        return code_of_template_kernal(template_ptr, dense_block_id);
    }

    if (type == EMPTY_TEMPLATE)
    {
        empty_template_t *template_ptr = (empty_template_t *)output_template;
        return code_of_template_kernal(template_ptr, dense_block_id);
    }

    cout << "code_of_template_kernal: has not support this template type" << endl;
    assert(false);
}

string code_of_kernal_function_call(void *output_template, template_type type, unsigned long dense_block_id)
{
    assert(output_template != NULL && type != NONE_TEMPLATE);

    if (type == DIRECT_ATOM_TEMPLATE)
    {
        direct_atom_template_t *template_ptr = (direct_atom_template_t *)output_template;
        return code_of_kernal_function_call(template_ptr, dense_block_id);
    }

    if (type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
    {
        direct_atom_template_warp_compress_t *template_ptr = (direct_atom_template_warp_compress_t *)output_template;
        return code_of_kernal_function_call(template_ptr, dense_block_id);
    }

    if (type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
    {
        direct_atom_template_warp_block_compress_t *template_ptr = (direct_atom_template_warp_block_compress_t *)output_template;
        return code_of_kernal_function_call(template_ptr, dense_block_id);
    }

    if (type == SHARED_MEMORY_TEMPLATE)
    {
        shared_memory_template_t *template_ptr = (shared_memory_template_t *)output_template;
        return code_of_kernal_function_call(template_ptr, dense_block_id);
    }

    if (type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
    {
        shared_memory_long_row_template_t *template_ptr = (shared_memory_long_row_template_t *)output_template;
        return code_of_kernal_function_call(template_ptr, dense_block_id);
    }

    if (type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        shared_memory_total_warp_reduce_template_t *template_ptr = (shared_memory_total_warp_reduce_template_t *)output_template;
        return code_of_kernal_function_call(template_ptr, dense_block_id);
    }

    if (type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        direct_atom_total_warp_reduce_template_t *template_ptr = (direct_atom_total_warp_reduce_template_t *)output_template;
        return code_of_kernal_function_call(template_ptr, dense_block_id);
    }

    if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
    {
        unaligned_warp_reduce_same_TLB_size_template_t* template_ptr = (unaligned_warp_reduce_same_TLB_size_template_t *)output_template;
        return code_of_kernal_function_call(template_ptr, dense_block_id);
    }

    if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
    {
        unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t* template_ptr = (unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *)output_template;
        return code_of_kernal_function_call(template_ptr, dense_block_id);
    }

    if (type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
    {
        shared_memory_template_warp_compress_t *template_ptr = (shared_memory_template_warp_compress_t *)output_template;
        return code_of_kernal_function_call(template_ptr, dense_block_id);
    }

    if (type == EMPTY_TEMPLATE)
    {
        empty_template_t *template_ptr = (empty_template_t *)output_template;
        return code_of_kernal_function_call(template_ptr, dense_block_id);
    }

    cout << "code_of_kernal_function_call: has not support this template type" << endl;
    assert(false);
}

// 这里处理main函数
string code_of_main_function(code_builder_t *code_builder, unsigned long kernal_repeat, bool perf_test)
{
    assert(code_builder != NULL);

    string return_str = "int main()\n{\n";

    // 查看对于设备的设定是不是正确的
    assert(get_config()["DEFAULT_DEVICE_ID"].as_integer() >= 0);

    cout << "code_of_main_function: device_id:" << get_config()["DEFAULT_DEVICE_ID"].as_integer() << endl;

    // 查看GPU设备的数量
    return_str = return_str + "int gpuDeviceCount = 0;\n";

    return_str = return_str + "cudaGetDeviceCount(&gpuDeviceCount);\n";

    return_str = return_str + "assert(" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer()) + " <= (gpuDeviceCount - 1));\n\n";

    return_str = return_str + "cudaSetDevice(" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer()) + ");\n\n";

    // 遍历所有的子矩阵，将所有显存拷贝的代码导出
    for (unsigned long dense_block_id = 0; dense_block_id < code_builder->template_vec.size(); dense_block_id++)
    {
        return_str = return_str + code_of_write_template_data_to_gpu(code_builder->template_vec[dense_block_id], code_builder->template_type_vec[dense_block_id], dense_block_id);
    }

    // 申请固定的x和y数组
    return_str = return_str + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + " *host_y_arr = (" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + " *)malloc(sizeof(" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + ") * " + to_string(code_builder->op_manager->matrix->dense_row_number) + ");\n";
    return_str = return_str + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + " *host_x_arr = (" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + " *)malloc(sizeof(" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + ") * " + to_string(code_builder->op_manager->matrix->dense_col_number) + ");\n\n";

    return_str = return_str + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + " *device_y_arr = NULL;\n";
    return_str = return_str + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + " *device_x_arr = NULL;\n\n";

    return_str = return_str + "cudaMalloc(&device_y_arr, sizeof(" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + ") * " + to_string(code_builder->op_manager->matrix->dense_row_number) + ");\n";
    return_str = return_str + "cudaMalloc(&device_x_arr, sizeof(" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + ") * " + to_string(code_builder->op_manager->matrix->dense_col_number) + ");\n\n";

    // 用定值初始化两个数组
    return_str = return_str + "for (unsigned long i = 0; i < " + to_string(code_builder->op_manager->matrix->dense_row_number) + "; i++)\n{\nhost_y_arr[i] = 0;\n}\n\n";
    return_str = return_str + "for (unsigned long i = 0; i < " + to_string(code_builder->op_manager->matrix->dense_col_number) + "; i++)\n{\nhost_x_arr[i] = 100;\n}\n\n";

    // 拷贝两个数组到显存
    return_str = return_str + "cudaMemcpy(device_y_arr, host_y_arr, sizeof(" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + ") * " + to_string(code_builder->op_manager->matrix->dense_row_number) + ", cudaMemcpyHostToDevice);\n";
    return_str = return_str + "cudaMemcpy(device_x_arr, host_x_arr, sizeof(" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + ") * " + to_string(code_builder->op_manager->matrix->dense_col_number) + ", cudaMemcpyHostToDevice);\n\n";

    return_str = return_str + "cudaStream_t stream_arr[" + to_string(code_builder->template_vec.size()) + "];\n";

    // 声明对应数量的流
    return_str = return_str + "for(unsigned long i = 0; i < " + to_string(code_builder->template_vec.size()) + "; i++)\n{\ncudaStreamCreate(&(stream_arr[i]));\n}\n\n";

    // 加入一个同步函数
    return_str = return_str + "cudaDeviceSynchronize();\n";

    // 这里看情况加一个计时函数
    if (perf_test == true)
    {
        return_str = return_str + "struct timeval start,end;\n";
        return_str = return_str + "gettimeofday(&start, NULL);\n";
    }

    assert(kernal_repeat > 0);

    if (kernal_repeat != 1)
    {
        return_str = return_str + "for (int i = 0; i < " + to_string(kernal_repeat) + "; i++)\n{\n";
    }

    // 遍历所有的模板，生成对应核函数调用
    for (unsigned long dense_block_id = 0; dense_block_id < code_builder->template_vec.size(); dense_block_id++)
    {
        return_str = return_str + "\n";
        return_str = return_str + code_of_kernal_function_call(code_builder->template_vec[dense_block_id], code_builder->template_type_vec[dense_block_id], dense_block_id);
    }

    // 加入一个同步函数
    return_str = return_str + "\ncudaDeviceSynchronize();\n";

    if (kernal_repeat != 1)
    {
        return_str = return_str + "}\n";
    }

    // 这里看情况加一个计时函数
    if (perf_test == true)
    {
        return_str = return_str + "gettimeofday(&end, NULL);\n\n";
        return_str = return_str + "long timeuse = 1000000 * (end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;\n";
        return_str = return_str + "double gflops = ((double)2.0 * " + to_string(code_builder->op_manager->matrix->origin_nnz) + " * " + to_string(kernal_repeat) + " / ((double)timeuse / 1000000)) / 1000000000;\n\n";

        return_str = return_str + "printf(\"time=%fms, gflops=%f\\n\",timeuse /1000.0, gflops);\n";
    }

    return_str = return_str + "cudaMemcpy(host_y_arr, device_y_arr, sizeof(" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + ") * " + to_string(code_builder->op_manager->matrix->dense_row_number) + ", cudaMemcpyDeviceToHost);\n";

    // 加入打印结果的函数
    return_str = return_str + "print_arr_to_file_with_data_type(host_y_arr, " + convert_data_type_to_string(code_builder->op_manager->matrix->val_data_type) + ", " + to_string(code_builder->op_manager->matrix->dense_row_number) + ", \"" + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/test_result_3\");\n";

    // 打印性能
    return_str = return_str + "ofstream resultWrite(\"" + get_config()["ROOT_PATH_STR"].as_string() + "/cuda_code/perf_result\", ios::out | ios::trunc);\n";
    return_str = return_str + "resultWrite << timeuse /1000.0 << endl << gflops << endl;\n";
    return_str = return_str + "resultWrite.close();\n";

    return_str = return_str + "\nreturn 0;\n}\n";

    return return_str;
}

string code_of_main_function(code_builder_t *code_builder, vector<int> sub_matrix_id_vec, unsigned long kernal_repeat, bool perf_test)
{
    // 将一部分模板放到main函数中
    assert(code_builder != NULL);

    string return_str = "int main()\n{\n";

    cout << "code_of_main_function: device_id:" << get_config()["DEFAULT_DEVICE_ID"].as_integer() << endl;

    // 查看GPU设备的数量
    return_str = return_str + "int gpuDeviceCount = 0;\n";

    return_str = return_str + "cudaGetDeviceCount(&gpuDeviceCount);\n";

    return_str = return_str + "assert(" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer()) + " <= (gpuDeviceCount - 1));\n\n";

    return_str = return_str + "cudaSetDevice(" + to_string(get_config()["DEFAULT_DEVICE_ID"].as_integer()) + ");\n\n";

    // 对需要子块执行显存拷贝的代码
    for (int i = 0; i < sub_matrix_id_vec.size(); i++)
    {
        int sub_matrix_id = sub_matrix_id_vec[i];
        
        assert(sub_matrix_id < code_builder->template_vec.size());
        assert(sub_matrix_id < code_builder->op_manager->matrix->block_coor_table.item_arr.size());

        assert(code_builder->template_vec[sub_matrix_id] != NULL);
        assert(code_builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);

        // 执行对应显存拷贝，并且禁止全局排序数组的共享
        return_str = return_str + code_of_write_template_data_to_gpu(code_builder->template_vec[sub_matrix_id], code_builder->template_type_vec[sub_matrix_id], sub_matrix_id, true);
    }

    // 申请固定的x和y数组
    return_str = return_str + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + " *host_y_arr = (" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + " *)malloc(sizeof(" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + ") * " + to_string(code_builder->op_manager->matrix->dense_row_number) + ");\n";
    return_str = return_str + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + " *host_x_arr = (" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + " *)malloc(sizeof(" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + ") * " + to_string(code_builder->op_manager->matrix->dense_col_number) + ");\n\n";

    return_str = return_str + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + " *device_y_arr = NULL;\n";
    return_str = return_str + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + " *device_x_arr = NULL;\n\n";

    return_str = return_str + "cudaMalloc(&device_y_arr, sizeof(" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + ") * " + to_string(code_builder->op_manager->matrix->dense_row_number) + ");\n";
    return_str = return_str + "cudaMalloc(&device_x_arr, sizeof(" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + ") * " + to_string(code_builder->op_manager->matrix->dense_col_number) + ");\n\n";

    // 用定值初始化两个数组
    return_str = return_str + "for (unsigned long i = 0; i < " + to_string(code_builder->op_manager->matrix->dense_row_number) + "; i++)\n{\nhost_y_arr[i] = 0;\n}\n\n";
    return_str = return_str + "for (unsigned long i = 0; i < " + to_string(code_builder->op_manager->matrix->dense_col_number) + "; i++)\n{\nhost_x_arr[i] = 100;\n}\n\n";

    // 拷贝两个数组到显存
    return_str = return_str + "cudaMemcpy(device_y_arr, host_y_arr, sizeof(" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + ") * " + to_string(code_builder->op_manager->matrix->dense_row_number) + ", cudaMemcpyHostToDevice);\n";
    return_str = return_str + "cudaMemcpy(device_x_arr, host_x_arr, sizeof(" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + ") * " + to_string(code_builder->op_manager->matrix->dense_col_number) + ", cudaMemcpyHostToDevice);\n\n";

    // 申请流
    return_str = return_str + "cudaStream_t stream_arr[" + to_string(code_builder->template_vec.size()) + "];\n";

    // 声明对应数量的流
    return_str = return_str + "for(unsigned long i = 0; i < " + to_string(code_builder->template_vec.size()) + "; i++)\n{\ncudaStreamCreate(&(stream_arr[i]));\n}\n\n";

    // 加入一个同步函数
    return_str = return_str + "cudaDeviceSynchronize();\n";

    // 这里看情况加一个计时函数
    if (perf_test == true)
    {
        return_str = return_str + "struct timeval start,end;\n";
        return_str = return_str + "gettimeofday(&start, NULL);\n";
    }

    assert(kernal_repeat > 0);

    if (kernal_repeat != 1)
    {
        return_str = return_str + "for (int i = 0; i < " + to_string(kernal_repeat) + "; i++)\n{\n";
    }

    // 遍历部分子块，执行核函数调用
    for (int i = 0; i < sub_matrix_id_vec.size(); i++)
    {
        int sub_matrix_id = sub_matrix_id_vec[i];

        return_str = return_str + "\n";
        
        // 这里的子块编号和子块对应的流号不是统一的
        return_str = return_str + code_of_kernal_function_call(code_builder->template_vec[sub_matrix_id], code_builder->template_type_vec[sub_matrix_id], sub_matrix_id);
    }

    // 加入一个同步函数
    return_str = return_str + "\ncudaDeviceSynchronize();\n";

    if (kernal_repeat != 1)
    {
        return_str = return_str + "}\n";
    }

    // 这里看情况加一个计时函数
    if (perf_test == true)
    {
        return_str = return_str + "gettimeofday(&end, NULL);\n\n";
        return_str = return_str + "long timeuse = 1000000 * (end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;\n";

        unsigned long origin_nnz = 0;
        
        // 这里计算的gflop是不准确的，因为没有办法考虑到子块原有的nnz数量
        for (int i = 0; i < sub_matrix_id_vec.size(); i++)
        {
            int sub_matrix_id = sub_matrix_id_vec[i];

            origin_nnz = origin_nnz + code_builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->end_coo_index - code_builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->begin_coo_index + 1;
        }

        return_str = return_str + "double gflops = ((double)2.0 * " + to_string(origin_nnz) + " * " + to_string(kernal_repeat) + " / ((double)timeuse / 1000000)) / 1000000000;\n\n";

        return_str = return_str + "printf(\"time=%fms, gflops=%f\\n\",timeuse /1000.0, gflops);\n";
    }

    return_str = return_str + "cudaMemcpy(host_y_arr, device_y_arr, sizeof(" + code_of_data_type(code_builder->op_manager->matrix->val_data_type) + ") * " + to_string(code_builder->op_manager->matrix->dense_row_number) + ", cudaMemcpyDeviceToHost);\n";

    return_str = return_str + "// print_arr_to_file_with_data_type(host_y_arr, " + convert_data_type_to_string(code_builder->op_manager->matrix->val_data_type) + ", " + to_string(code_builder->op_manager->matrix->dense_row_number) + ", \"" + get_config()["ROOT_PATH_STR"].as_string() + "/data_source/test_result_3\");\n";

    // 向一个文件中输出结果
    // 打印性能
    return_str = return_str + "ofstream resultWrite(\"" + get_config()["ROOT_PATH_STR"].as_string() + "/cuda_code/perf_result\", ios::out | ios::trunc);\n";
    return_str = return_str + "resultWrite << timeuse /1000.0 << endl << gflops << endl;\n";
    return_str = return_str + "resultWrite.close();\n";

    return_str = return_str + "\nreturn 0;\n}\n";

    return return_str;
}

// 这里打印所有的内核函数
string code_of_kernal_define(code_builder_t *builder)
{
    assert(builder != NULL);

    string return_str = "\n";
    // 遍历子矩阵，分别打印每个子矩阵的核函数
    for (unsigned long dense_block_id = 0; dense_block_id < builder->template_vec.size(); dense_block_id++)
    {
        return_str = return_str + code_of_template_kernal(builder->template_vec[dense_block_id], builder->template_type_vec[dense_block_id], dense_block_id);
        return_str = return_str + "\n";
    }

    return return_str;
}

// 打印一部分的内核函数
string code_of_kernal_define(code_builder_t *builder, vector<int> sub_matrix_id_vec)
{
    assert(builder != NULL);
    
    string return_str = "\n";

    // 遍历所有的子块编号
    for (int i = 0; i < sub_matrix_id_vec.size(); i++)
    {
        int sub_matrix_id = sub_matrix_id_vec[i];

        assert(sub_matrix_id < builder->template_vec.size());
        assert(sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());
        
        assert(builder->template_vec[sub_matrix_id] != NULL);
        assert(builder->op_manager->matrix->block_coor_table.item_arr[sub_matrix_id]->compressed_block_ptr->read_index.size() == 7);

        return_str = return_str + code_of_template_kernal(builder->template_vec[sub_matrix_id], builder->template_type_vec[sub_matrix_id], sub_matrix_id);
        return_str = return_str + "\n";
    }

    return return_str;
}

// 包含内核函数文件
string build_main_file(code_builder_t *builder, unsigned long kernal_repeat)
{
    assert(builder != NULL);
    assert(builder->template_vec.size() == builder->op_manager->matrix->block_coor_table.item_arr.size());

    string return_str = "#include <cuda_runtime.h>\n#include <cstdlib>\n#include <stdio.h>\n#include <iostream>\n#include <stdio.h>\n#include <fstream>\n#include <stdlib.h>\n#include <vector>\n#include <string.h>\n#include \"template.h\"\n#include <cuda.h>";

    return_str = return_str + "\n\n";

    return_str = return_str + code_of_kernal_define(builder);

    return_str = return_str + code_of_main_function(builder, kernal_repeat);

    return return_str;
}


string build_main_file(code_builder_t *builder, vector<int> sub_matrix_id_vec, unsigned long kernal_repeat)
{
    assert(builder != NULL);
    assert(sub_matrix_id_vec.size() > 0);

    // 遍历所有需要生成代码的块号
    for (int i = 0; i < sub_matrix_id_vec.size(); i++)
    {
        int sub_matrix_id = sub_matrix_id_vec[i];
        
        assert(builder->template_vec[sub_matrix_id] != NULL);
    }

    string return_str = "#include <cuda_runtime.h>\n#include <cstdlib>\n#include <stdio.h>\n#include <iostream>\n#include <stdio.h>\n#include <fstream>\n#include <stdlib.h>\n#include <vector>\n#include <string.h>\n#include \"template.h\"\n#include <cuda.h>";

    return_str = return_str + "\n\n";

    return_str = return_str + code_of_kernal_define(builder, sub_matrix_id_vec);

    return_str = return_str + code_of_main_function(builder, sub_matrix_id_vec, kernal_repeat);

    return return_str;
}


// 传入要处理的稀疏矩阵在磁盘中的位置
// string code_of_main_function(code_builder_t *code_builder, string matrix_file_name)
// {
//     assert(code_builder != NULL);

//     sparse_struct_t *matrix = code_builder->op_manager->matrix;

//     assert(matrix != NULL);

//     string return_str = "int main\n{\n";

//     return_str = return_str + "all_compressed_block_t *total_matrix = read_matrix_from_file(\"" + matrix_file_name + "\");\n";

//     return_str = return_str + "compressed_matrix_content_t *compressed_block = total_matrix->all_compressed_matrix_info;\n";

//     if (matrix->is_sorted == true)
//     {
//         assert(matrix->sorted_row_index != NULL);
//         return_str = return_str + code_line_of_pointer_define(matrix->data_type_of_sorted_row_index, "device_" + code_of_arr_var_name(-1, -1, "sorted_row_index"));
//     }

//     // return_str = return_str + "unsigned int *device_sorted_row_index;\n";

//     return_str = return_str + "\n";
//     // 遍历所有的密集子块，对所有显存上的指针进行声明
//     for (unsigned long index_of_dense_block = 0; index_of_dense_block < matrix->block_coor_table.item_arr.size(); index_of_dense_block++)
//     {
//         compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[index_of_dense_block]->compressed_block_ptr;

//         assert(compressed_block_view != NULL && compressed_block_view->staggered_padding_val_arr != NULL);
//         // 值数组的声明
//         return_str = return_str + code_line_of_pointer_define(compressed_block_view->val_data_type, "device_" + code_of_arr_var_name(index_of_dense_block, -1, "staggered_padding_val_arr"));

//         return_str = return_str + "\n";

//         // 遍历读索引
//         for (unsigned long index_of_read_index = 0; index_of_read_index < compressed_block_view->read_index.size(); index_of_read_index++)
//         {
//             index_of_compress_block_t *read_index = compressed_block_view->read_index[index_of_read_index];

//             if (read_index->index_arr != NULL)
//             {
//                 return_str = return_str + code_line_of_pointer_define(read_index->index_data_type, "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_arr"));
//             }

//             if (read_index->index_of_the_first_row_arr != NULL)
//             {
//                 assert(index_of_read_index == 2 || index_of_read_index == 3 || index_of_read_index == 4);
//                 return_str = return_str + code_line_of_pointer_define(read_index->data_type_of_index_of_the_first_row_arr, "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_of_the_first_row_arr"));
//             }

//             if (read_index->row_number_of_block_arr != NULL)
//             {
//                 assert(index_of_read_index == 2 || index_of_read_index == 3);
//                 return_str = return_str + code_line_of_pointer_define(read_index->data_type_of_row_number_of_block_arr, "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "row_number_of_block_arr"));
//             }

//             if (read_index->tmp_result_write_index_arr != NULL)
//             {
//                 assert(index_of_read_index == 3);
//                 return_str = return_str + code_line_of_pointer_define(read_index->data_type_of_tmp_result_write_index_arr, "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "tmp_result_write_index_arr"));
//             }

//             if (read_index->coo_begin_index_arr != NULL)
//             {
//                 assert(index_of_read_index == 2 || index_of_read_index == 3);
//                 return_str = return_str + code_line_of_pointer_define(read_index->data_type_of_coo_begin_index_arr, "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_begin_index_arr"));
//             }

//             if (read_index->coo_block_size_arr != NULL)
//             {
//                 assert(index_of_read_index == 3 || index_of_read_index == 4);
//                 return_str = return_str + code_line_of_pointer_define(read_index->data_type_of_coo_block_size_arr, "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_block_size_arr"));
//             }

//             if (read_index->child_tmp_row_csr_index_arr != NULL)
//             {
//                 assert(index_of_read_index == 2 || index_of_read_index == 3);
//                 return_str = return_str + code_line_of_pointer_define(read_index->data_type_of_child_tmp_row_csr_index, "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "child_tmp_row_csr_index_arr"));
//             }

//             if (read_index->begin_index_in_tmp_row_csr_arr_of_block != NULL)
//             {
//                 assert(index_of_read_index == 2 || index_of_read_index == 3);
//                 return_str = return_str + code_line_of_pointer_define(read_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block, "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "begin_index_in_tmp_row_csr_arr_of_block"));
//             }

//             return_str = return_str + "\n";
//         }

//         for (unsigned long index_of_y_write_index = 0; index_of_y_write_index < compressed_block_view->y_write_index.size(); index_of_y_write_index++)
//         {
//             index_of_compress_block_t *y_write_index = compressed_block_view->y_write_index[index_of_y_write_index];
//             assert(y_write_index != NULL);
//             if (y_write_index->index_arr != NULL)
//             {
//                 return_str = return_str + code_line_of_pointer_define(y_write_index->index_data_type, "device_" + code_of_y_write_arr_var_name(index_of_dense_block, index_of_y_write_index, "index_arr"));
//             }

//             return_str = return_str + "\n";
//         }

//         return_str = return_str + "\n";
//     }

//     // 多层循环，执行申请显存的代码，和上面的循环结构一样，
//     if (matrix->is_sorted == true)
//     {
//         assert(matrix->sorted_row_index != NULL);
//         return_str = return_str + code_line_of_cuda_malloc(matrix->data_type_of_sorted_row_index, "total_matrix->size_of_" + code_of_arr_var_name(-1, -1, "sorted_row_index"), "device_" + code_of_arr_var_name(-1, -1, "sorted_row_index"));
//     }

//     for (unsigned long index_of_dense_block = 0; index_of_dense_block < matrix->block_coor_table.item_arr.size(); index_of_dense_block++)
//     {
//         compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[index_of_dense_block]->compressed_block_ptr;

//         assert(compressed_block_view != NULL && compressed_block_view->staggered_padding_val_arr != NULL);
//         // 值数组的声明
//         return_str = return_str + code_line_of_cuda_malloc(compressed_block_view->val_data_type, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, -1, "staggered_padding_val_arr"), "device_" + code_of_arr_var_name(index_of_dense_block, -1, "staggered_padding_val_arr"));

//         return_str = return_str + "\n";

//         // 遍历读索引
//         for (unsigned long index_of_read_index = 0; index_of_read_index < compressed_block_view->read_index.size(); index_of_read_index++)
//         {
//             index_of_compress_block_t *read_index = compressed_block_view->read_index[index_of_read_index];

//             if (read_index->index_arr != NULL)
//             {
//                 return_str = return_str + code_line_of_cuda_malloc(read_index->index_data_type, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_arr"), "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_arr"));
//             }

//             if (read_index->index_of_the_first_row_arr != NULL)
//             {
//                 assert(index_of_read_index == 2 || index_of_read_index == 3 || index_of_read_index == 4);
//                 return_str = return_str + code_line_of_cuda_malloc(read_index->data_type_of_index_of_the_first_row_arr, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_of_the_first_row_arr"), "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_of_the_first_row_arr"));
//             }

//             if (read_index->row_number_of_block_arr != NULL)
//             {
//                 assert(index_of_read_index == 2 || index_of_read_index == 3);
//                 return_str = return_str + code_line_of_cuda_malloc(read_index->data_type_of_row_number_of_block_arr, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "row_number_of_block_arr"), "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "row_number_of_block_arr"));
//             }

//             if (read_index->tmp_result_write_index_arr != NULL)
//             {
//                 assert(index_of_read_index == 3);
//                 return_str = return_str + code_line_of_cuda_malloc(read_index->data_type_of_tmp_result_write_index_arr, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "tmp_result_write_index_arr"), "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "tmp_result_write_index_arr"));
//             }

//             if (read_index->coo_begin_index_arr != NULL)
//             {
//                 assert(index_of_read_index == 2 || index_of_read_index == 3);
//                 return_str = return_str + code_line_of_cuda_malloc(read_index->data_type_of_coo_begin_index_arr, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_begin_index_arr"), "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_begin_index_arr"));
//             }

//             if (read_index->coo_block_size_arr != NULL)
//             {
//                 assert(index_of_read_index == 3 || index_of_read_index == 4);
//                 return_str = return_str + code_line_of_cuda_malloc(read_index->data_type_of_coo_block_size_arr, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_block_size_arr"), "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_block_size_arr"));
//             }

//             if (read_index->child_tmp_row_csr_index_arr != NULL)
//             {
//                 assert(index_of_read_index == 2 || index_of_read_index == 3);
//                 return_str = return_str + code_line_of_cuda_malloc(read_index->data_type_of_child_tmp_row_csr_index, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "child_tmp_row_csr_index_arr"), "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "child_tmp_row_csr_index_arr"));
//             }

//             if (read_index->begin_index_in_tmp_row_csr_arr_of_block != NULL)
//             {
//                 assert(index_of_read_index == 2 || index_of_read_index == 3);
//                 return_str = return_str + code_line_of_cuda_malloc(read_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "begin_index_in_tmp_row_csr_arr_of_block"), "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "begin_index_in_tmp_row_csr_arr_of_block"));
//             }

//             return_str = return_str + "\n";
//         }

//         for (unsigned long index_of_y_write_index = 0; index_of_y_write_index < compressed_block_view->y_write_index.size(); index_of_y_write_index++)
//         {
//             index_of_compress_block_t *y_write_index = compressed_block_view->y_write_index[index_of_y_write_index];
//             assert(y_write_index != NULL);
//             if (y_write_index->index_arr != NULL)
//             {
//                 return_str = return_str + code_line_of_cuda_malloc(y_write_index->index_data_type, "compressed_block->size_of_" + code_of_y_write_arr_var_name(index_of_dense_block, index_of_y_write_index, "index_arr"), "device_" + code_of_y_write_arr_var_name(index_of_dense_block, index_of_y_write_index, "index_arr"));
//             }

//             return_str = return_str + "\n";
//         }

//         return_str = return_str + "\n";
//     }

//     // 将所有内存拷贝到显存中
//     if (matrix->is_sorted == true)
//     {
//         assert(matrix->sorted_row_index != NULL);
//         return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(-1, -1, "sorted_row_index"), "total_matrix->" + code_of_arr_var_name(-1, -1, "sorted_row_index"), matrix->data_type_of_sorted_row_index, "total_matrix->size_of_" + code_of_arr_var_name(-1, -1, "sorted_row_index"), "cudaMemcpyHostToDevice");
//     }

//     // 遍历所有密集矩阵视图
//     for (unsigned long index_of_dense_block = 0; index_of_dense_block < matrix->block_coor_table.item_arr.size(); index_of_dense_block++)
//     {
//         compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[index_of_dense_block]->compressed_block_ptr;

//         assert(compressed_block_view != NULL && compressed_block_view->staggered_padding_val_arr != NULL);
//         // 值数组的声明
//         return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(index_of_dense_block, -1, "staggered_padding_val_arr"), "compressed_block->" + code_of_arr_var_name(index_of_dense_block, -1, "staggered_padding_val_arr"), compressed_block_view->val_data_type, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, -1, "staggered_padding_val_arr"), "cudaMemcpyHostToDevice");

//         return_str = return_str + "\n";

//         // 遍历读索引
//         for (unsigned long index_of_read_index = 0; index_of_read_index < compressed_block_view->read_index.size(); index_of_read_index++)
//         {
//             index_of_compress_block_t *read_index = compressed_block_view->read_index[index_of_read_index];

//             if (read_index->index_arr != NULL)
//             {
//                 return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_arr"), "compressed_block->" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_arr"), read_index->index_data_type, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_arr"), "cudaMemcpyHostToDevice");
//             }

//             if (read_index->index_of_the_first_row_arr != NULL)
//             {
//                 assert(index_of_read_index == 2 || index_of_read_index == 3 || index_of_read_index == 4);
//                 return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_of_the_first_row_arr"), "compressed_block->" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_of_the_first_row_arr"), read_index->data_type_of_index_of_the_first_row_arr, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_of_the_first_row_arr"), "cudaMemcpyHostToDevice");
//             }

//             if (read_index->row_number_of_block_arr != NULL)
//             {
//                 assert(index_of_read_index == 2 || index_of_read_index == 3);
//                 return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "row_number_of_block_arr"), "compressed_block->" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "row_number_of_block_arr"), read_index->data_type_of_row_number_of_block_arr, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "row_number_of_block_arr"), "cudaMemcpyHostToDevice");
//             }

//             if (read_index->tmp_result_write_index_arr != NULL)
//             {
//                 assert(index_of_read_index == 3);
//                 return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "tmp_result_write_index_arr"), "compressed_block->" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "tmp_result_write_index_arr"), read_index->data_type_of_tmp_result_write_index_arr, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "tmp_result_write_index_arr"), "cudaMemcpyHostToDevice");
//             }

//             if (read_index->coo_begin_index_arr != NULL)
//             {
//                 assert(index_of_read_index == 2 || index_of_read_index == 3);
//                 return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_begin_index_arr"), "compressed_block->" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_begin_index_arr"), read_index->data_type_of_coo_begin_index_arr, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_begin_index_arr"), "cudaMemcpyHostToDevice");
//             }

//             if (read_index->coo_block_size_arr != NULL)
//             {
//                 assert(index_of_read_index == 3 || index_of_read_index == 4);
//                 return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_block_size_arr"), "compressed_block->" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_block_size_arr"), read_index->data_type_of_coo_block_size_arr, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_block_size_arr"), "cudaMemcpyHostToDevice");
//             }

//             if (read_index->child_tmp_row_csr_index_arr != NULL)
//             {
//                 assert(index_of_read_index == 2 || index_of_read_index == 3);
//                 return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "child_tmp_row_csr_index_arr"), "compressed_block->" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "child_tmp_row_csr_index_arr"), read_index->data_type_of_child_tmp_row_csr_index, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "child_tmp_row_csr_index_arr"), "cudaMemcpyHostToDevice");
//             }

//             if (read_index->begin_index_in_tmp_row_csr_arr_of_block != NULL)
//             {
//                 assert(index_of_read_index == 2 || index_of_read_index == 3);
//                 return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "begin_index_in_tmp_row_csr_arr_of_block"), "compressed_block->" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "begin_index_in_tmp_row_csr_arr_of_block"), read_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block, "compressed_block->size_of_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "begin_index_in_tmp_row_csr_arr_of_block"), "cudaMemcpyHostToDevice");
//             }

//             return_str = return_str + "\n";
//         }

//         for (unsigned long index_of_y_write_index = 0; index_of_y_write_index < compressed_block_view->y_write_index.size(); index_of_y_write_index++)
//         {
//             index_of_compress_block_t *y_write_index = compressed_block_view->y_write_index[index_of_y_write_index];
//             assert(y_write_index != NULL);
//             if (y_write_index->index_arr != NULL)
//             {
//                 return_str = return_str + code_line_of_cuda_memcpy("device_" + code_of_y_write_arr_var_name(index_of_dense_block, index_of_y_write_index, "index_arr"), "compressed_block->" + code_of_y_write_arr_var_name(index_of_dense_block, index_of_y_write_index, "index_arr"), y_write_index->index_data_type, "compressed_block->size_of_" + code_of_y_write_arr_var_name(index_of_dense_block, index_of_y_write_index, "index_arr"), "cudaMemcpyHostToDevice");
//             }

//             return_str = return_str + "\n";
//         }

//         return_str = return_str + "\n";
//     }

//     // double *host_y_arr = (double *)malloc(sizeof(double) * 929901);
//     // // bool*
//     // double *host_x_arr = (double *)malloc(sizeof(double) * 303645);
//     // 用自己的x数组，先申请两个主机数组
//     return_str = return_str + code_of_data_type(matrix->val_data_type) + "* host_y_arr = (" + code_of_data_type(matrix->val_data_type) + " *)malloc(sizeof(" + code_of_data_type(matrix->val_data_type) + ") *" + to_string(matrix->dense_row_number) + ");\n";
//     return_str = return_str + code_of_data_type(matrix->val_data_type) + "* host_x_arr = (" + code_of_data_type(matrix->val_data_type) + " *)malloc(sizeof(" + code_of_data_type(matrix->val_data_type) + ") *" + to_string(matrix->dense_col_number) + ");\n";

//     // 两个主机数组
//     return_str = return_str + "for (unsigned long i = 0; i < " + to_string(matrix->dense_row_number) + "; i++)\n{\nhost_y_arr[i] = 0;\n}\n";
//     return_str = return_str + "for (unsigned long i = 0; i < " + to_string(matrix->dense_col_number) + "; i++)\n{\nhost_x_arr[i] = 100;\n}\n";

//     // 两个设备数组
//     return_str = return_str + "double *device_y_arr = NULL;\ndouble *device_x_arr = NULL;\n";

//     return_str = return_str + code_line_of_cuda_malloc(matrix->val_data_type, to_string(matrix->dense_row_number), code_of_arr_var_name(-1, -1, "device_y_arr"));
//     return_str = return_str + code_line_of_cuda_malloc(matrix->val_data_type, to_string(matrix->dense_col_number), code_of_arr_var_name(-1, -1, "device_x_arr"));

//     return_str = return_str + "\n";
//     // 数组的拷贝
//     return_str = return_str + code_line_of_cuda_memcpy("device_y_arr", "host_y_arr", matrix->val_data_type, to_string(matrix->dense_row_number), "cudaMemcpyHostToDevice");
//     return_str = return_str + code_line_of_cuda_memcpy("device_x_arr", "host_x_arr", matrix->val_data_type, to_string(matrix->dense_col_number), "cudaMemcpyHostToDevice");

//     // 声明并初始化所有的流
//     return_str = return_str + "cudaStream_t stream_arr[" + to_string(matrix->block_coor_table.item_arr.size()) + "]\n";
//     return_str = return_str + "for(unsigned long i = 0; i < " + to_string(matrix->block_coor_table.item_arr.size()) + "; i++)\n{\ncudaStreamCreate(&(stream_arr[i]));\n}\n";

//     // 遍历所有密集子块，调用不同的核函数
//     for (unsigned long index_of_dense_block = 0; index_of_dense_block < matrix->block_coor_table.item_arr.size(); index_of_dense_block++)
//     {
//         return_str = return_str + code_of_kernal_func_call(code_builder, index_of_dense_block);
//     }

//     // 同步函数
//     return_str = return_str + "cudaDeviceSynchronize();\n";

//     // 将数据拷贝回来的代码
//     return_str = return_str + code_line_of_cuda_memcpy("host_y_arr", "device_y_arr", matrix->val_data_type, to_string(matrix->dense_row_number), "cudaMemcpyDeviceToHost");

//     return_str = return_str + "\n}\n";

//     return return_str;
// }

string code_line_of_pointer_define(data_type type, string var_name)
{
    return code_of_data_type(type) + "* " + var_name + ";\n";
}

// 申请数组的代码
string code_line_of_cuda_malloc(data_type type, string code_of_size, string arr_name)
{
    return "cudaMalloc(&" + arr_name + ", sizeof(" + code_of_data_type(type) + ") * " + code_of_size + ");\n";
}

// 数组拷贝的代码
string code_line_of_cuda_memcpy(string var_name_of_dest_arr, string var_name_of_source_arr, data_type type, string size_var_str, string copy_direct_str)
{
    return "cudaMemcpy(" + var_name_of_dest_arr + ", " + var_name_of_source_arr + ", sizeof(" + code_of_data_type(type) + ") * " + size_var_str + ", " + copy_direct_str + ");\n";
}

// 对应稠密子矩阵的spmv调用
string code_of_kernal_func_call(code_builder_t *code_builder, unsigned long index_of_dense_block)
{
    assert(code_builder != NULL && code_builder->op_manager != NULL && code_builder->op_manager->matrix != NULL);
    assert(index_of_dense_block < code_builder->kernal_block_num_vec.size());

    sparse_struct_t *matrix = code_builder->op_manager->matrix;
    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[index_of_dense_block]->compressed_block_ptr;

    string return_str = "spmv_" + to_string(index_of_dense_block) + "<<<" + to_string(code_builder->kernal_block_num_vec[index_of_dense_block]) + ", " + to_string(code_builder->kernal_thread_num_in_block_vec[index_of_dense_block]) + ", 0, stream_arr[" + to_string(index_of_dense_block) + "]>>>(";

    // 全局排序和值数组，查看是不是空的
    if (matrix->is_sorted == true)
    {
        assert(matrix->sorted_row_index != NULL);
        return_str = return_str + "device_sorted_row_index, ";
    }

    assert(compressed_block_view->staggered_padding_val_arr != NULL);
    return_str = return_str + "device_" + code_of_arr_var_name(index_of_dense_block, -1, "staggered_padding_val_arr") + ", ";

    for (unsigned long index_of_read_index = 0; index_of_read_index < compressed_block_view->read_index.size(); index_of_read_index++)
    {
        index_of_compress_block_t *read_index = compressed_block_view->read_index[index_of_read_index];
        assert(read_index != NULL);

        // 打印所有的数组
        if (read_index->index_arr != NULL)
        {
            return_str = return_str + "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_arr") + ", ";
        }

        if (read_index->index_of_the_first_row_arr != NULL)
        {
            assert(index_of_read_index == 2 || index_of_read_index == 3 || index_of_read_index == 4);
            return_str = return_str + "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "index_of_the_first_row_arr") + ", ";
        }

        if (read_index->row_number_of_block_arr != NULL)
        {
            assert(index_of_read_index == 2 || index_of_read_index == 3);
            return_str = return_str + "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "row_number_of_block_arr") + ", ";
        }

        if (read_index->tmp_result_write_index_arr != NULL)
        {
            assert(index_of_read_index == 3);
            return_str = return_str + "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "tmp_result_write_index_arr") + ", ";
        }

        if (read_index->coo_begin_index_arr != NULL)
        {
            assert(index_of_read_index == 2 || index_of_read_index == 3);
            return_str = return_str + "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_begin_index_arr") + ", ";
        }

        if (read_index->coo_block_size_arr != NULL)
        {
            assert(index_of_read_index == 3 || index_of_read_index == 4);
            return_str = return_str + "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "coo_block_size_arr") + ", ";
        }

        if (read_index->child_tmp_row_csr_index_arr != NULL)
        {
            assert(index_of_read_index == 2 || index_of_read_index == 3);
            return_str = return_str + "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "child_tmp_row_csr_index_arr") + ", ";
        }

        if (read_index->begin_index_in_tmp_row_csr_arr_of_block != NULL)
        {
            assert(index_of_read_index == 2 || index_of_read_index == 3);
            return_str = return_str + "device_" + code_of_arr_var_name(index_of_dense_block, index_of_read_index, "begin_index_in_tmp_row_csr_arr_of_block") + ", ";
        }
    }

    for (unsigned long index_of_y_write_index = 0; index_of_y_write_index < compressed_block_view->y_write_index.size(); index_of_y_write_index++)
    {
        index_of_compress_block_t *y_write_index = compressed_block_view->y_write_index[index_of_y_write_index];
        assert(y_write_index != NULL);
        if (y_write_index->index_arr != NULL)
        {
            return_str = return_str + "device_" + code_of_y_write_arr_var_name(index_of_dense_block, index_of_y_write_index, "index_arr");
        }
    }

    // y和x两个矩阵
    return_str = return_str + "device_y_arr, device_x_arr";

    return_str = return_str + ");\n";

    return return_str;
}

string code_of_a_formal_param_declare(data_type type, string var_name)
{
    return code_of_data_type(type) + " " + var_name;
}

// 构造核函数
string code_of_kernal_func_define(code_builder_t *code_builder, unsigned long index_of_dense_block)
{
    assert(code_builder != NULL && code_builder->op_manager != NULL);

    sparse_struct_t *matrix = code_builder->op_manager->matrix;

    assert(matrix != NULL);
    assert(index_of_dense_block < matrix->block_coor_table.item_arr.size());
    compressed_block_t *compressed_block_view = matrix->block_coor_table.item_arr[index_of_dense_block]->compressed_block_ptr;
    assert(compressed_block_view != NULL);

    // 函数的声明
    string return_str = "void spmv_" + to_string(index_of_dense_block) + "(";
    // 所有形参的声明
    // 全局排序和值数组，查看是不是空的
    if (matrix->is_sorted == true)
    {
        assert(matrix->sorted_row_index != NULL);
        return_str = return_str + code_of_a_formal_param_declare(matrix->data_type_of_sorted_row_index, " *sorted_row_index") + ", ";
    }

    assert(compressed_block_view->staggered_padding_val_arr != NULL);
    return_str = return_str + code_of_a_formal_param_declare(compressed_block_view->val_data_type, " *staggered_padding_val_arr") + ", ";

    for (unsigned long index_of_read_index = 0; index_of_read_index < compressed_block_view->read_index.size(); index_of_read_index++)
    {
        index_of_compress_block_t *read_index = compressed_block_view->read_index[index_of_read_index];
        assert(read_index != NULL);

        // 打印所有的数组
        if (read_index->index_arr != NULL)
        {
            return_str = return_str + code_of_a_formal_param_declare(read_index->index_data_type, " *" + code_of_arr_var_name(-1, index_of_read_index, "index_arr")) + ", ";
        }

        if (read_index->index_of_the_first_row_arr != NULL)
        {
            assert(index_of_read_index == 2 || index_of_read_index == 3 || index_of_read_index == 4);
            return_str = return_str + code_of_a_formal_param_declare(read_index->data_type_of_index_of_the_first_row_arr, " *" + code_of_arr_var_name(-1, index_of_read_index, "index_of_the_first_row_arr")) + ", ";
        }

        if (read_index->row_number_of_block_arr != NULL)
        {
            assert(index_of_read_index == 2 || index_of_read_index == 3);
            return_str = return_str + code_of_a_formal_param_declare(read_index->data_type_of_row_number_of_block_arr, " *" + code_of_arr_var_name(-1, index_of_read_index, "row_number_of_block_arr")) + ", ";
        }

        if (read_index->tmp_result_write_index_arr != NULL)
        {
            assert(index_of_read_index == 3);
            return_str = return_str + code_of_a_formal_param_declare(read_index->data_type_of_tmp_result_write_index_arr, " *" + code_of_arr_var_name(-1, index_of_read_index, "tmp_result_write_index_arr")) + ", ";
        }

        if (read_index->coo_begin_index_arr != NULL)
        {
            assert(index_of_read_index == 2 || index_of_read_index == 3);
            return_str = return_str + code_of_a_formal_param_declare(read_index->data_type_of_coo_begin_index_arr, " *" + code_of_arr_var_name(-1, index_of_read_index, "coo_begin_index_arr")) + ", ";
        }

        if (read_index->coo_block_size_arr != NULL)
        {
            assert(index_of_read_index == 3 || index_of_read_index == 4);
            return_str = return_str + code_of_a_formal_param_declare(read_index->data_type_of_coo_block_size_arr, " *" + code_of_arr_var_name(-1, index_of_read_index, "coo_block_size_arr")) + ", ";
        }

        if (read_index->child_tmp_row_csr_index_arr != NULL)
        {
            assert(index_of_read_index == 2 || index_of_read_index == 3);
            return_str = return_str + code_of_a_formal_param_declare(read_index->data_type_of_child_tmp_row_csr_index, " *" + code_of_arr_var_name(-1, index_of_read_index, "child_tmp_row_csr_index_arr")) + ", ";
        }

        if (read_index->begin_index_in_tmp_row_csr_arr_of_block != NULL)
        {
            assert(index_of_read_index == 2 || index_of_read_index == 3);
            return_str = return_str + code_of_a_formal_param_declare(read_index->data_type_of_begin_index_in_tmp_row_csr_arr_of_block, " *" + code_of_arr_var_name(-1, index_of_read_index, "begin_index_in_tmp_row_csr_arr_of_block")) + ", ";
        }
    }

    for (unsigned long index_of_y_write_index = 0; index_of_y_write_index < compressed_block_view->y_write_index.size(); index_of_y_write_index++)
    {
        index_of_compress_block_t *y_write_index = compressed_block_view->y_write_index[index_of_y_write_index];
        assert(y_write_index != NULL);
        if (y_write_index->index_arr != NULL)
        {
            return_str = return_str + code_of_a_formal_param_declare(y_write_index->index_data_type, " *" + code_of_y_write_arr_var_name(-1, index_of_y_write_index, "index_arr")) + ", ";
        }
    }

    // 剩下的形参
    return_str = return_str + code_of_a_formal_param_declare(compressed_block_view->val_data_type, " *y_arr") + ", " + code_of_a_formal_param_declare(compressed_block_view->val_data_type, " *x_arr") + ")\n{\n";

    return return_str;
}

bool compress_template_in_builder(code_builder_t *builder, template_type type, unsigned long dense_block_id)
{
    cout << "compress_template_in_builder: not suggest in auto-tunning" << endl;
    assert(builder != NULL && builder->template_vec.size() > dense_block_id);
    // 针对某一个模板的压缩
    template_type old_type = builder->template_type_vec[dense_block_id];
    void *old_template = builder->template_vec[dense_block_id];

    assert(old_template != NULL);

    // 不成功的压缩返回false
    if (old_type == DIRECT_ATOM_TEMPLATE)
    {
        direct_atom_template_t *old_direct_atom_template = (direct_atom_template_t *)old_template;
        if (type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
        {
            direct_atom_template_warp_compress_t *new_template = init_direct_atom_template_warp_compress(old_direct_atom_template);
            // 修改对应位置的指针
            builder->template_vec[dense_block_id] = new_template;
            // 修改对应位置的类型
            builder->template_type_vec[dense_block_id] = DIRECT_ATOM_TEMPLATE_WARP_COMPRESS;
            return true;
        }

        if (type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
        {
            direct_atom_template_warp_block_compress_t *new_template = init_direct_atom_template_warp_block_compress(old_direct_atom_template);
            // 修改对应位置的指针
            builder->template_vec[dense_block_id] = new_template;
            // 修改对应位置的类型
            builder->template_type_vec[dense_block_id] = DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS;
            return true;
        }

        cout << "compress_template_in_builder: dest template compress is not support" << endl;
        return false;
    }

    if (old_type == SHARED_MEMORY_TEMPLATE)
    {
        shared_memory_template_t *old_shared_memory_template = (shared_memory_template_t *)old_template;
        if (type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
        {
            shared_memory_template_warp_compress_t *new_template = init_shared_memory_template_warp_compress(old_shared_memory_template);
            builder->template_vec[dense_block_id] = new_template;
            builder->template_type_vec[dense_block_id] = SHARED_MEMORY_TEMPLATE_WARP_COMPRESS;
            return true;
        }

        cout << "compress_template_in_builder: dest template compress is not support" << endl;
        return false;
    }

    if (old_type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
    {
        shared_memory_long_row_template_t *old_shared_memory_template = (shared_memory_long_row_template_t *)old_template;

        cout << "compress_template_in_builder: dest template compress is not support" << endl;
        return false;
    }

    // 存在warp reduce的
    if (old_type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        shared_memory_total_warp_reduce_template_t *old_shared_memory_template = (shared_memory_total_warp_reduce_template_t *)old_template;

        cout << "compress_template_in_builder: dest template compress is not support" << endl;
        return false;
    }

    if (old_type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        direct_atom_total_warp_reduce_template_t *old_direct_atom_template = (direct_atom_total_warp_reduce_template_t *)old_template;

        cout << "compress_template_in_builder: dest template compress is not support" << endl;
        return false;
    }

    if (old_type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
    {
        unaligned_warp_reduce_same_TLB_size_template_t* old_direct_atom_template = (unaligned_warp_reduce_same_TLB_size_template_t*)old_template;
        
        cout << "compress_template_in_builder: dest template compress is not support" << endl;
        return false;
    }

    if (old_type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
    {
        unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t* old_direct_atom_template = (unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t*)old_template;

        cout << "compress_template_in_builder: dest template compress is not support" << endl;
        return false;
    }

    if (old_type == EMPTY_TEMPLATE)
    {
        empty_template_t *old_shared_memory_template = (empty_template_t *)old_template;

        cout << "compress_template_in_builder: dest template compress is not support" << endl;
        return false;
    }

    cout << "compress_template_in_builder: source template compress is not support" << endl;
    return false;
}

void store_code_builder_data(code_builder_t *builder, string output_dir)
{
    assert(builder != NULL);
    assert(builder->template_type_vec.size() > 0 && builder->template_vec.size() > 0);

    for (unsigned long i = 0; i < builder->template_type_vec.size(); i++)
    {
        assert(builder->template_vec[i] != NULL && builder->template_type_vec[i] != NONE_TEMPLATE);
    }

    for (unsigned long i = 0; i < builder->template_type_vec.size(); i++)
    {
        // 睡一下防止相同的随机数种子
        sleep(1);
        void *cur_template_ptr = builder->template_vec[i];
        template_type cur_template_type = builder->template_type_vec[i];

        cout << "store_code_builder_data: template i:" << i << " " << convert_template_type_to_string(cur_template_type) << endl;

        if (cur_template_type == DIRECT_ATOM_TEMPLATE)
        {
            store_template_data((direct_atom_template_t *)cur_template_ptr, output_dir);
            continue;
        }

        if (cur_template_type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
        {
            store_template_data((direct_atom_template_warp_compress_t *)cur_template_ptr, output_dir);
            continue;
        }

        if (cur_template_type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
        {
            store_template_data((direct_atom_template_warp_block_compress_t *)cur_template_ptr, output_dir);
            continue;
        }

        if (cur_template_type == SHARED_MEMORY_TEMPLATE)
        {
            store_template_data((shared_memory_template_t *)cur_template_ptr, output_dir);
            continue;
        }

        if (cur_template_type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
        {
            store_template_data((shared_memory_long_row_template_t *)cur_template_ptr, output_dir);
            continue;
        }

        if (cur_template_type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
        {
            store_template_data((shared_memory_total_warp_reduce_template_t *)cur_template_ptr, output_dir);
            continue;
        }

        if (cur_template_type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
        {
            store_template_data((direct_atom_total_warp_reduce_template_t *)cur_template_ptr, output_dir);
            continue;
        }

        if (cur_template_type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
        {
            store_template_data((unaligned_warp_reduce_same_TLB_size_template_t *)cur_template_ptr, output_dir);
            continue;
        }

        if (cur_template_type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
        {
            store_template_data((unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *)cur_template_ptr, output_dir);
            continue;
        }
        
        if (cur_template_type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
        {
            store_template_data((shared_memory_template_warp_compress_t *)cur_template_ptr, output_dir);
            continue;
        }

        if (cur_template_type == EMPTY_TEMPLATE)
        {
            store_template_data((empty_template_t *)cur_template_ptr, output_dir);
            continue;
        }

        cout << "store_code_builder_data: error, output template is not supported" << endl;
        assert(false);
    }
}

void store_code_builder_data(code_builder_t *builder, vector<int> sub_matrix_id_vec, string output_dir)
{
    // 将一部分的数据写到磁盘中
    assert(builder != NULL);
    assert(builder->template_type_vec.size() > 0 && builder->template_vec.size() > 0);

    // 遍历所有的需要存起来的子块
    for (unsigned long i = 0; i < sub_matrix_id_vec.size(); i++)
    {
        int sub_matrix_id = sub_matrix_id_vec[i];
        
        assert(sub_matrix_id < builder->template_vec.size() && sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());

        assert(builder->template_vec[sub_matrix_id] != NULL && builder->template_type_vec[sub_matrix_id] != NONE_TEMPLATE);
    }

    // 因为所有子矩阵的全局行排序数组都是一样的，所以在生成整个矩阵的排序索引中，只有第一个矩阵才需要需要真正写和读取
    // 这直接导致了存储全局排序索引和读全局排序索引这两个过程需要重新实现。在这一点上采取新的策略。让每一个子块都必须自己处理全局排序行数组
    // 禁止由第一个块来共享，这一点上修改所有的store_template_data的实现
    // 这里调用对应模板数据存储函数
    for (unsigned long i = 0; i < sub_matrix_id_vec.size(); i++)
    {
        int sub_matrix_id = sub_matrix_id_vec[i];
        
        assert(sub_matrix_id < builder->template_vec.size() && sub_matrix_id < builder->op_manager->matrix->block_coor_table.item_arr.size());

        // 睡一下防止相同的随机数种子
        sleep(1);
        void *cur_template_ptr = builder->template_vec[sub_matrix_id];
        template_type cur_template_type = builder->template_type_vec[sub_matrix_id];
        
        cout << "template sub_matrix_id:" << sub_matrix_id << " " << convert_template_type_to_string(cur_template_type) << endl;

        if (cur_template_type == DIRECT_ATOM_TEMPLATE)
        {
            store_template_data((direct_atom_template_t *)cur_template_ptr, output_dir, true);
            continue;
        }

        if (cur_template_type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
        {
            store_template_data((direct_atom_template_warp_compress_t *)cur_template_ptr, output_dir, true);
            continue;
        }

        if (cur_template_type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
        {
            store_template_data((direct_atom_template_warp_block_compress_t *)cur_template_ptr, output_dir, true);
            continue;
        }

        if (cur_template_type == SHARED_MEMORY_TEMPLATE)
        {
            store_template_data((shared_memory_template_t *)cur_template_ptr, output_dir, true);
            continue;
        }

        if (cur_template_type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
        {
            store_template_data((shared_memory_long_row_template_t *)cur_template_ptr, output_dir, true);
            continue;
        }

        if (cur_template_type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
        {
            store_template_data((shared_memory_total_warp_reduce_template_t *)cur_template_ptr, output_dir, true);
            continue;
        }

        if (cur_template_type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
        {
            store_template_data((direct_atom_total_warp_reduce_template_t *)cur_template_ptr, output_dir, true);
            continue;
        }

        if (cur_template_type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
        {
            store_template_data((unaligned_warp_reduce_same_TLB_size_template_t *)cur_template_ptr, output_dir, true);
            continue;
        }

        if (cur_template_type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
        {
            store_template_data((unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t *)cur_template_ptr, output_dir, true);
            continue;
        }
        
        if (cur_template_type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
        {
            store_template_data((shared_memory_template_warp_compress_t *)cur_template_ptr, output_dir, true);
            continue;
        }

        if (cur_template_type == EMPTY_TEMPLATE)
        {
            store_template_data((empty_template_t *)cur_template_ptr, output_dir);
            continue;
        }

        cout << "store_code_builder_data: error, output template is not supported" << endl;
        assert(false);
    }
}

string convert_template_type_to_string(template_type type)
{
    if (type == DIRECT_ATOM_TEMPLATE)
    {
        return "DIRECT_ATOM_TEMPLATE";
    }

    if (type == DIRECT_ATOM_TEMPLATE_WARP_COMPRESS)
    {
        return "DIRECT_ATOM_TEMPLATE_WARP_COMPRESS";
    }

    if (type == DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS)
    {
        return "DIRECT_ATOM_TEMPLATE_WARP_BLOCK_COMPRESS";
    }

    if (type == SHARED_MEMORY_TEMPLATE)
    {
        return "SHARED_MEMORY_TEMPLATE";
    }

    if (type == SHARED_MEMORY_TEMPLATE_WARP_COMPRESS)
    {
        return "SHARED_MEMORY_TEMPLATE_WARP_COMPRESS";
    }

    if (type == SHARED_MEMORY_LONG_ROW_TEMPLATE)
    {
        return "SHARED_MEMORY_LONG_ROW_TEMPLATE";
    }

    if (type == SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        return "SHARED_MEMORY_TOTAL_WARP_REDUCE_TEMPLATE";
    }

    if (type == DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE)
    {
        return "DIRECT_ATOM_TOTAL_WARP_REDUCE_TEMPLATE";
    }

    if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE)
    {
        return "UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE";
    }

    if (type == UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE)
    {
        return "UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE_WITH_WARP_REDUCE";
    }

    if (type == EMPTY_TEMPLATE)
    {
        return "EMPTY_TEMPLATE";
    }

    if (type == NONE_TEMPLATE)
    {
        return "NONE_TEMPLATE";
    }

    cout << "convert_template_type_to_string: template type is not supported, type:" << type << endl;
    assert(false);
}