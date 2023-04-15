#  -*-  coding  =  utf-8  -*-
#  @Time  :2023/3/23  21:05
#  @Author:Su
#  @File  :  L_R_test.py
#  @Software:  PyCharm
# import torch
# import random
#
#
# def synthetic_data(W, b, num_example):
#     X = torch.normal(0, 1, (num_example, len(W)))
#     y = torch.matmul(X, W) + b
#     y += torch.normal(0,0.01, y.shape)
#     return X, y.reshape(-1,1)
#
#
# def data_iter(batch_size, X, y):
