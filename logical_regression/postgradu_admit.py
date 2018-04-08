#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/8 10:26
# @Author  : wutianxiong
# @File    : postgradu_admit.py
# @Software: PyCharm
import pandas as pd
import statsmodels.api as sm
import matplotlib as plt
import pylab as pl
import numpy as np
from sklearn.linear_model import LogisticRegression

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

test_Y = lr.predict(test_X)

print(test_Y)
