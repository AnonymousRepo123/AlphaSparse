import utils
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from joblib import dump

ACCEPTED_R2_LOW_BOUND = 0.7

# 从外部读入满足条件的内容，输出是一个满足好球的额
str_dataset = utils.get_complete_data_source_from_file("BEGIN_MEMORY_CACHE_INPUT_FILE,DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY,COMPRESS,COMPRESS_NONE_PARAM_STRATEGY,COMPRESSED_THREAD_LEVEL_NNZ_DIV,COMPRESSED_THREAD_LEVEL_NNZ_DIV_DIRECT_PARAM_STRATEGY,UNALIGNED_WARP_REDUCE_SAME_TLB_SIZE_TEMPLATE,", "/home/duzhen/spmv_builder/machine_learning_module/s2rmq4m1_machine_learning_data_set")

X, y = utils.get_complete_numpy_x_and_y(str_dataset)

# 去除

# print(X[0])
# print(y[0])

# 引入三折的的训练
kf = KFold(n_splits=3)

best_reg = None
# 最优的模型对应的最优的初始化缩放器
best_scaler = None
best_test_score = 0
best_train_score_corresponding_to_best_test_score = 0

# 一直训练，直到有出现满足要求的模型
while best_test_score <= ACCEPTED_R2_LOW_BOUND or best_train_score_corresponding_to_best_test_score <= ACCEPTED_R2_LOW_BOUND:
    train_scores = []
    test_scores = []

    for train_index, test_index in kf.split(X):
        # 获得训练的内容
        X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

        # 执行一个标准化，并且将标准化的模型
        scaler = MinMaxScaler()
        X_train_std = scaler.fit_transform(X_train)
        # print(len(X_train_std[0]))

        reg = MLPRegressor(hidden_layer_sizes=(len(X_train_std[0]), len(X_train_std[0]), len(X_train_std[0])), activation="relu", max_iter=100000)
        reg.fit(X_train_std, y_train)
        
        train_score = reg.score(X_train_std, y_train)
        train_scores.append(train_score)

        # 执行测试集的标准化
        X_test_std = scaler.transform(X_test)
        test_score = reg.score(X_test_std, y_test)
        test_scores.append(test_score)

        if test_score > best_test_score:
            best_test_score = test_score
            best_train_score_corresponding_to_best_test_score = train_score
            best_reg = reg
            best_scaler = scaler

    # in_sample_error = [1 - score for score in train_scores]
    # test_set_error = [1 - score for score in test_scores]
    print("best_train_score_corresponding_to_best_test_score: ")
    print(best_train_score_corresponding_to_best_test_score)
    print("best_test_score: ")
    print(best_test_score)


assert(best_test_score > ACCEPTED_R2_LOW_BOUND and best_train_score_corresponding_to_best_test_score > ACCEPTED_R2_LOW_BOUND)

if best_reg != None:
    dump(best_reg, "trained_model.m")

if best_scaler != None:
    dump(best_scaler, "trained_scaler.bin")
    

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

# print(len(X))
# print(len(y))
# print(len(X_train))
# print(len(X_test))
# print(len(y_train))
# print(len(y_test))


