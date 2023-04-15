#  -*-  coding  =  utf-8  -*-
#  @Time  :2023/3/12  10:45
#  @Author:Su
#  @File  :  test.py
#  @Software:  PyCharm

import torch
import matplotlib.pyplot as plt
import collections
import d_l_util as dt
import numpy
import time
#
# # 生成一个随机的 3x256x256 张量作为示例
# img_tensor = torch.rand(3, 256, 256)
#
# # 将张量转换为 numpy 数组
# img_np = img_tensor.numpy()
#
# # 显示彩色图像
# plt.imshow(img_np.transpose(1, 2, 0))
# plt.show()

#
# s = ['aa', 'xax', 'aa', 'aa', 'xscac', 'a', 'a', 'a', 'a']
# count = collections.Counter(s)
# print(count.items())
# b= sorted(count.items(), key=lambda x : x[1], reverse=True)
# print(b)
# b = list(enumerate(s))
# print(b)
# for idx, item in enumerate(s):
#     print(idx, item)

#
# d = {'s':1, 'a':2, 'c':3}
# d['e'] = 5
# print(d.get('s',0))
# print(d.get('xascasc',0))
#
# class Vocab:
#
#     def __init__(self, dic):
#         self.token_to_idx = dic
#
#     def __getitem__(self, items):
#         if not isinstance(items, (list, tuple)):
#             return self.token_to_idx.get(items, 0)
#         return [self.__getitem__(item) for item in items]
#
#
# item = {'a':1, 'b':2, 'c':3}
# vocab = Vocab(item)
# corpus = [vocab[item] for item in item]
# print(corpus)
# X = numpy.arange(0, 4, 0.1)
# Y = [X, X]
# print(X,Y)
# dt.plot(X, Y,'x', 'y', legend=['x1', 'x2'])
# print(len(X),len(Y))
# X = torch.normal(0, 1, (3, 1))
# W_xh = torch.normal(0, 1, (1, 4))
# H = torch.normal(0, 1, (3, 4))
# W_hh = torch.normal(0, 1, (4, 4))
# X_1 = torch.cat((X, H), dim=1)
# print(X_1)
# W = torch.cat((W_xh, W_hh), dim=0)
# print(W)
# print(torch.matmul(X_1, W))
# se = set(['a', 'b', 'c', 'd', 'd']) # 第一种初始化方法
# print(se)
# # 输出为{'a', 'b', 'c', 'd'}，自动去重，顺序是乱序
# se.add('x')   # 无法添加list add(['a','b'])会报错
# se.remove('a')
# print(se)
# # 输出为{'b','x','c','d'}
# se1 = set('bcdefx') # 第二种初始化方法
# se2 = se - se1      # 集合差
# print(se2)
# # 输出为set()
# se3 = se | se1		# 集合并
# print(se3)
# se4 = se & se1     # 集合交
# print(se4)
# se5 = se ^ se1    # 集合并减去集合交
# print(se5)
vocab = ['a','b','c']
res  =[0] * len(vocab)
print(res)