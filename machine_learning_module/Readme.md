# 一些基本的资料

怎么交叉验证：https://www.cnblogs.com/liuxin0430/p/12130346.html

怎么用多层感知机来回归：
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
https://blog.csdn.net/clever_wr/article/details/88097802

训练结果的随机性：
https://www.zhihu.com/question/294058968

决定系数的含义：
https://www.sohu.com/a/158761950_655168

预处理的方法：
https://www.codenong.com/cs106402843/

树形算法，除了神经网络之外的流行机器学习类别，用线性边界构造非线性模型：
https://zhuanlan.zhihu.com/p/82054400

集成学习的思路，bagging（并行的模型集成）， boosting（串行的模型集成）：
https://zhuanlan.zhihu.com/p/37730184

随机深林，有部分特征和数据训练不同的决策树，并且获得综合所有树的训练结果：
https://zhuanlan.zhihu.com/p/51165358

xgboost：
文档
https://xgboost.readthedocs.io/en/stable/python/python_api.html
学习目标等重要参数
https://www.cnblogs.com/TimVerion/p/11436001.html
例子
https://blog.csdn.net/rosefun96/article/details/78876569
一个推导的过程，xgboost不断产生决策树。训练新的决策树和老的决策树加到一起，弥补老的决策树的不足。新的决策树也是以输入数据集X作为输入，然后用一个小的学习率作为权重加到老的决策树后面。学习率的存在是为了方式过拟合，降低新的树的权重
https://zhuanlan.zhihu.com/p/29765582


