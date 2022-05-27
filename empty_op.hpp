#ifndef EMPTY_TEMPLATE_H
#define EMPTY_TEMPLATE_H

#include "struct.hpp"
#include "config.hpp"
#include "arr_optimization.hpp"
#include "code_builder.hpp"

typedef struct empty_template
{
    // 模板对应的稠密矩阵号
    unsigned long dense_block_index;
    // 对应的密集矩阵
    sparse_struct_t* matrix = NULL;
}empty_template_t;

empty_template_t* init_empty_template(code_builder_t* builder, unsigned long dense_block_id);

void store_template_data(empty_template_t *output_template, string output_dir);

string code_of_template_data_struct(empty_template_t *output_template, unsigned long dense_block_id);

string code_of_read_template_data_from_file_func_define(empty_template_t *output_template, unsigned long dense_block_id);

string code_of_template_kernal(empty_template_t *output_template, unsigned long dense_block_id);

string code_of_kernal_function_call(empty_template_t *output_template, unsigned long dense_block_id);

string code_of_write_template_data_to_gpu(empty_template_t *output_template, unsigned long dense_block_id);

#endif