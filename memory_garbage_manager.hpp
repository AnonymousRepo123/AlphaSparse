#ifndef MEMORY_GARBAGE_MANAGER_H
#define MEMORY_GARBAGE_MANAGER_H

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
#include <set>

using namespace std;

typedef struct memory_garbage_manager
{
    // 用一个set存储已经被析构的指针地址
    set<void *> ptr_set;
}memory_garbage_manager_t;

// 析构所有必要的结构，首先析构矩阵
void delete_sparse_struct_t(memory_garbage_manager_t* mem_manager, sparse_struct_t* matrix);

// 析构direct_atom_template
void delete_direct_atom_template(memory_garbage_manager_t* mem_manager, direct_atom_template_t* del_template);

// 析构direct_atom_template_warp_compress
void delete_direct_atom_template_warp_compress(memory_garbage_manager_t* mem_manager, direct_atom_template_warp_compress_t* del_template);

// 析构direct_atom_template_warp_block_compress
void delete_direct_atom_template_warp_block_compress(memory_garbage_manager_t* mem_manager, direct_atom_template_warp_block_compress_t* del_template);

// 析构direct_atom_total_warp_reduce_template
void delete_direct_atom_total_warp_reduce_template(memory_garbage_manager_t* mem_manager, direct_atom_total_warp_reduce_template_t* del_template);

// 析构shared_memory_long_row_template
void delete_shared_memory_long_row_template(memory_garbage_manager_t* mem_manager, shared_memory_long_row_template_t* del_template);

// 析构shared_memory_template_warp_compress
void delete_shared_memory_template_warp_compress(memory_garbage_manager_t* mem_manager, shared_memory_template_warp_compress_t* del_template);

// 析构shared_memory_template
void delete_shared_memory_template(memory_garbage_manager_t* mem_manager, shared_memory_template_t* del_template);

// 析构shared_memory_total_warp_reduce_template
void delete_shared_memory_total_warp_reduce_template(memory_garbage_manager_t* mem_manager, shared_memory_total_warp_reduce_template_t* del_template);

// 析构unaligned_warp_reduce_same_TLB_size_template
void delete_unaligned_warp_reduce_same_TLB_size_template(memory_garbage_manager_t* mem_manager, unaligned_warp_reduce_same_TLB_size_template_t* del_template);

void delete_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce(memory_garbage_manager_t* mem_manager, unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t* del_template);

// 按照一定类型析构对应模板
void delete_template_with_type(memory_garbage_manager_t* mem_manager, void* template_ptr, template_type type);

// 析构一个操作管理器
void delete_op_manager(memory_garbage_manager_t* mem_manager, operator_manager_t* op_manager);

// 析构一个代码生成器
void delete_code_builder(memory_garbage_manager_t* mem_manager, code_builder_t* builder);

// 析构一个代码生成器除了矩阵的部分
void delete_code_builder_without_operator_manager(memory_garbage_manager* mem_manager, code_builder_t *builder);

// 判断一个指针是不是已经被析构了
bool is_deleted(memory_garbage_manager_t* mem_manager, void* del_ptr);

// 登记一个新的被析构的指针
void register_del_ptr(memory_garbage_manager_t* mem_manager, void* del_ptr);

// 打印所有的登记过的指针
void print_all_register_ptr(memory_garbage_manager_t* mem_manager);

// 下面这个函数为析构某一个或几个压缩子
// 获得一个矩阵除了某几个压缩视图之外所有的指针，用来在模板和代码生成器中的部分析构一些内容，而不该被析构的内容需要通过
// 登记到析构器中保护起来，从而避免被析构。
set<void *> get_all_mem_ptr_from_matrix_dense_view_and_some_compressed_sub_block(sparse_struct_t* matrix, vector<int> not_need_to_del_compressed_block_id);

// 析构一个模板，但是不析构矩阵的信息
void delete_template_without_matrix_with_type(memory_garbage_manager_t* mem_manager, void* template_ptr, template_type type);

void delete_template_without_matrix_with_type(memory_garbage_manager_t* mem_manager, code_builder_t* builder, int template_id_in_code_builder);

// 析构direct_atom_template
void delete_direct_atom_template_without_matrix(memory_garbage_manager_t* mem_manager, direct_atom_template_t* del_template);

// 析构direct_atom_template_warp_compress
void delete_direct_atom_template_warp_compress_without_matrix(memory_garbage_manager_t* mem_manager, direct_atom_template_warp_compress_t* del_template);

// 析构direct_atom_template_warp_block_compress
void delete_direct_atom_template_warp_block_compress_without_matrix(memory_garbage_manager_t* mem_manager, direct_atom_template_warp_block_compress_t* del_template);

void delete_direct_atom_total_warp_reduce_template_without_matrix(memory_garbage_manager_t* mem_manager, direct_atom_total_warp_reduce_template_t* del_template);

void delete_shared_memory_long_row_template_without_matrix(memory_garbage_manager_t* mem_manager, shared_memory_long_row_template_t* del_template);

// 析构使用shared mem的但是在WLB层次压缩的模板
void delete_shared_memory_template_warp_compress_without_matrix(memory_garbage_manager_t* mem_manager, shared_memory_template_warp_compress_t* del_template);

void delete_shared_memory_template_without_matrix(memory_garbage_manager_t *mem_manager, shared_memory_template_t *del_template);

void delete_shared_memory_total_warp_reduce_template_without_matrix(memory_garbage_manager_t *mem_manager, shared_memory_total_warp_reduce_template_t *del_template);

// 类似于CSR5的模板
void delete_unaligned_warp_reduce_same_TLB_size_template_without_matrix(memory_garbage_manager_t* mem_manager, unaligned_warp_reduce_same_TLB_size_template_t* del_template);

void delete_unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_without_matrix(memory_garbage_manager_t* mem_manager, unaligned_warp_reduce_same_TLB_size_template_with_warp_reduce_t* del_template);

#endif