#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/8 10:26
# @Author  : wutianxiong
# @File    : postgradu_admit.py
# @Software: PyCharm
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# 读取数据集
train_data = pd.read_csv("dataset/lr-binary.csv")

# train_data.hist()
# pl.show()

dummy_ranks = pd.get_dummies(train_data['rank'], prefix='rank')
# print dummy_ranks.head()
# 为逻辑回归创建所需的data frame
# 除admit、gre、gpa外，加入了上面常见的虚拟变量（注意，引入的虚拟变量列数应为虚拟变量总列数减1，减去的1列作为基准）
cols_to_keep = ['admit', 'gre', 'gpa']
data = train_data[cols_to_keep].join(dummy_ranks.ix[:, 'rank_2':])

# 需要自行添加逻辑回归所需的intercept变量
data['intercept'] = 1.0

train_X = data[data.columns[1:]]

train_Y = data[data.columns[0:1]]

lr = LogisticRegression()
lr.fit(train_X, np.ravel(train_Y))
# 读取测试集数据
test_data = pd.read_csv('dataset/lr-binary-test.csv')

dummy_test_ranks = pd.get_dummies(test_data['rank'], prefix='rank')
# 为逻辑回归创建所需的data frame
# 除gre、gpa外，加入了上面常见的虚拟变量（注意，引入的虚拟变量列数应为虚拟变量总列数减1，减去的1列作为基准）
cols_to_keep_test = ['gre', 'gpa']
data_test = test_data[cols_to_keep_test].join(dummy_ranks.ix[:, 'rank_2':])
data_test['intercept'] = 1.0
test_X = data_test[data_test.columns[0:]]

test_Y = test_data['admit']
predict_test_Y = lr.predict(test_X)

print(confusion_matrix(test_Y, predict_test_Y, labels=[0, 1]))
# 利用逻辑斯蒂回归自带的评分函数score获得模型在测试集上的准确定结果
print '精确率为：', lr.score(test_X, test_Y)

print classification_report(test_Y, predict_test_Y, labels=[0, 1], target_names=['not_admit', 'admit'])
