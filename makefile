all: main

CXX := g++
LD := ${CXX}

# Flags to enable link-time optimization and GDB
LTO := -fno-lto
ENABLE_DGB :=

BUILDER_HOME := .

INC	:= -I ${BUILDER_HOME}

# DEBUG := -DNDEBUG
CPPFLAGS := ${ENABLE_DGB} ${LTO} -O2 ${DEBUG} -std=c++11 ${INC} \
	-Wno-unused-result -Wno-unused-value -Wno-unused-function \
	# -Winline

LDFLAGS := ${ENABLE_DGB} ${LTO} -pthread

src :=${BUILDER_HOME}/config.o ${BUILDER_HOME}/op_manager.o ${BUILDER_HOME}/struct.o ${BUILDER_HOME}/code_builder.o ${BUILDER_HOME}/arr_optimization.o \
${BUILDER_HOME}/direct_atom_op.o ${BUILDER_HOME}/direct_atom_op_warp_compress.o \
${BUILDER_HOME}/direct_atom_op_warp_block_compress.o ${BUILDER_HOME}/shared_memory_op.o ${BUILDER_HOME}/shared_memory_op_warp_compress.o\
${BUILDER_HOME}/multi_thread_function.o ${BUILDER_HOME}/shared_memory_long_row_op.o ${BUILDER_HOME}/empty_op.o  ${BUILDER_HOME}/shared_memory_total_warp_reduce_op.o\
${BUILDER_HOME}/dataset_builder.o ${BUILDER_HOME}/direct_atom_total_warp_reduce_op.o ${BUILDER_HOME}/memory_garbage_manager.o \
${BUILDER_HOME}/exe_graph.o ${BUILDER_HOME}/user_pruning_strategy.o ${BUILDER_HOME}/graph_enumerate.o ${BUILDER_HOME}/unaligned_warp_reduce_same_TLB_size_op.o \
${BUILDER_HOME}/unaligned_warp_reduce_same_TLB_size_op_with_warp_reduce.o ${BUILDER_HOME}/default_auto_tuner.o \
${BUILDER_HOME}/matrix_info.o ${BUILDER_HOME}/param_enumerater.o ${BUILDER_HOME}/executor.o ${BUILDER_HOME}/parameter_set_strategy.o \
${BUILDER_HOME}/whilte_list_path_impl.o ${BUILDER_HOME}/search_strategy.o ${BUILDER_HOME}/code_source_data.o ${BUILDER_HOME}/machine_learning_data_set_collector.o \
${BUILDER_HOME}/main.o


main: ${src}
	${LD} -o $@ $^ ${LDFLAGS}

PHONY: clean
clean:
	rm -f *.o main ${src}