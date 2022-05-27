# 使用xgboost的测试
import utils
import numpy as np
from sklearn.model_selection import KFold
from joblib import dump
import xgboost as xgb

ACCEPTED_R2_LOW_BOUND = 0.7

# 从外部读入满足条件的内容，输出是一个满足好球的额
str_dataset = utils.get_complete_data_source_from_file("BEGIN_MEMORY_CACHE_INPUT_FILE,DENSE_BEGIN_MEMORY_CACHE_INPUT_FILE_DIRECT_PARAM_STRATEGY,DENSE_ROW_COARSE_SORT,DENSE_ROW_COARSE_SORT_FIXED_PARAM_STRATEGY,DENSE_ROW_DIV,DENSE_ROW_DIV_ACC_TO_EXPONENTIAL_INCREASE_ROW_NNZ_PARAM_STRATEGY,COMPRESS,COMPRESS_NONE_PARAM_STRATEGY,COMPRESSED_ROW_PADDING,COMPRESSED_ROW_PADDING_DIRECT_PARAM_STRATEGY,COMPRESSED_TBLOCK_LEVEL_ROW_DIV,COMPRESSED_TBLOCK_LEVEL_ROW_DIV_EVENLY_PARAM_STRATEGY,COMPRESSED_THREAD_LEVEL_COL_DIV,COMPRESSED_THREAD_LEVEL_COL_DIV_FIXED_PARAM_STRATEGY,SHARED_MEMORY_TEMPLATE_WARP_COMPRESS,", "/home/duzhen/spmv_builder/machine_learning_module/s2rmq4m1_machine_learning_data_set")

X, y = utils.get_complete_numpy_x_and_y(str_dataset)

kf = KFold(n_splits=3)

best_reg = None
# 最优的模型对应的最优的初始化缩放器
best_scaler = None
best_test_score = 0
best_train_score_corresponding_to_best_test_score = 0

print(len(y))

# while best_test_score <= ACCEPTED_R2_LOW_BOUND or best_train_score_corresponding_to_best_test_score <= ACCEPTED_R2_LOW_BOUND:
for train_index, test_index in kf.split(X):
    # 获得训练的内容
    X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
    
    model = xgb.XGBRegressor()

    model.fit(X_train, y_train, eval_metric="mae", eval_set=[(X_train, y_train), (X_test, y_test)])
    ans = model.predict(X_test)
    
    print(ans)
    print(y_test)

    

        
        
        
        