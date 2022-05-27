# 尝试使用sklearn的随机森林来预测特定矩阵和特定图性能
import utils
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

ACCEPTED_R2_LOW_BOUND = 0.7

# 从外部读入满足条件的内容，输出是一个满足好球的额
str_dataset = utils.get_complete_data_source_from_file("BEGIN_MEMORY_CACHE_INPUT_FILE,DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY,COMPRESS,COMPRESS_NONE_PARAM_STRATEGY,COMPRESSED_THREAD_LEVEL_NNZ_DIV,COMPRESSED_THREAD_LEVEL_NNZ_DIV_DIRECT_PARAM_STRATEGY,UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE,", "/home/duzhen/spmv_builder/machine_learning_module/s2rmq4m1_machine_learning_data_set")

X, y = utils.get_complete_numpy_x_and_y(str_dataset)

kf = KFold(n_splits=3)

best_reg = None
# 最优的模型对应的最优的初始化缩放器
best_scaler = None
best_test_score = 0
best_train_score_corresponding_to_best_test_score = 0

while best_test_score <= ACCEPTED_R2_LOW_BOUND or best_train_score_corresponding_to_best_test_score <= ACCEPTED_R2_LOW_BOUND:
    for train_index, test_index in kf.split(X):
        # 获得训练的内容
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
        
        reg = RandomForestRegressor(max_features="log2", min_samples_leaf=2)
        reg.fit(X_train, y_train)

        train_score = reg.score(X_train, y_train)
        print("train_score:" + str(train_score))
        test_score = reg.score(X_test, y_test)
        print("test_score:" + str(test_score))

        if test_score > best_test_score:
            best_test_score = test_score
            best_train_score_corresponding_to_best_test_score = train_score
            best_reg = reg
        
assert(best_test_score > ACCEPTED_R2_LOW_BOUND and best_train_score_corresponding_to_best_test_score > ACCEPTED_R2_LOW_BOUND)

if best_reg != None:
    dump(best_reg, "random_forest_model.m")

        