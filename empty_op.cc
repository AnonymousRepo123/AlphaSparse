#include "empty_op.hpp"

empty_template_t* init_empty_template(code_builder_t* builder, unsigned long dense_block_id)
{
    assert(builder != NULL);
    assert(builder->op_manager != NULL);
    assert(builder->op_manager->matrix != NULL);

    sparse_struct_t *matrix = builder->op_manager->matrix;

    dense_block_table_item_t* item = matrix->block_coor_table.item_arr[dense_block_id];

    if (item->begin_coo_index != item->end_coo_index)
    {
        cout << "this sub dense block is not empty" << endl;
        assert(false);
    }

    // 需要验证非零元的数量是不是0
    empty_template_t* new_template = new empty_template_t();

    new_template->matrix = matrix;
    new_template->dense_block_index = dense_block_id;

    return new_template;
}

// 模板的所有输出都是回车
void store_template_data(empty_template_t *output_template, string output_dir)
{
    // 空的
    return;
}

string code_of_template_data_struct(empty_template_t *output_template, unsigned long dense_block_id)
{
    assert(output_template != NULL);
    return "\n";
}

string code_of_read_template_data_from_file_func_define(empty_template_t *output_template, unsigned long dense_block_id)
{
    assert(output_template != NULL);
    return "\n";
}

string code_of_template_kernal(empty_template_t *output_template, unsigned long dense_block_id)
{
    assert(output_template != NULL);
    return "\n";
}

string code_of_kernal_function_call(empty_template_t *output_template, unsigned long dense_block_id)
{
    assert(output_template != NULL);
    return "\n";
}

string code_of_write_template_data_to_gpu(empty_template_t *output_template, unsigned long dense_block_id)
{
    assert(output_template != NULL);
    return "\n";
}