#  -*-  coding  =  utf-8  -*-
#  @Time  :2023/3/25  8:31
#  @Author:Su
#  @File  :  10.Conv_Layer.py
#  @Software:  PyCharm
import torch
from torch import nn
from d2l import torch as d2l

#
def corr2d(X, K):
    h = K.shape[0]
    w = K.shape[1]
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i][j] = (X[i:i+h, j:j+w] * K).sum()
    return Y
#
#
# X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# # print(corr2d(X, K))
#
#
# class Conv2D(nn.Module):
#     def __init__(self, kernel_size):
#         super().__init__()
#         self.weight = nn.Parameter(torch.rand(kernel_size))
#         self.bias = nn.Parameter(torch.zeros(1))
#
#     def forward(self, X):
#         return corr2d(X, self.weight) + self.bias
#
#
# X = torch.ones((6, 8))
# X[:, 2:6] = 0
# # print(X)
# K = torch.tensor([[1.0, -1.0]])
# Y = corr2d(X, K)
# # print(corr2d(X, K))
#
# conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
# X = X.reshape(1, 1, 6, 8)
# Y = Y.reshape(1, 1, 6, 7)
#
# for i in range(10):
#     Y_hat = conv2d(X)
#     loss = (Y_hat - Y) ** 2
#     conv2d.zero_grad()
#     loss.sum().backward()
#     conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
#     if (i+1) % 2 == 0:
#         print(f'batch{i+1}, loss{loss.sum():.3f}')
#
# # print(conv2d.weight.data.reshape(1, 2))


# # 输入X是2维矩阵
# def comp_conv2d(conv2d, X):
#     # 将X扩为4维
#     X = X.reshape((1, 1) + X.shape)
#     Y = conv2d(X)
#     return Y.reshape(Y.shape[2:])
#
#
# #
# # conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
# X = torch.rand(size=(8, 8))
# # print(comp_conv2d(conv2d, X).shape)
#
#
# conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
# print(comp_conv2d(conv2d, X).shape)
#
# conv2d = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=1, stride=2)
#


def corr2d_multi_in(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))


X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
#
# print(corr2d_multi_in(X, K))


def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


def corr2d_multi_in_out_detail(X, K):
    z = []
    for k in K:
        print('k', k)
        y = corr2d_multi_in(X, k)
        print('y ', y)
        z.append(y)
    z = torch.stack(z, 0)
    return z


K = torch.stack((K, K + 1, K + 2), 0)
print(K.shape)
print(corr2d_multi_in_out_detail(X, K))
# print(corr2d_muti_out(X, K))

# 矩阵乘法实现全连接
def corr2d_multi_in_out_1X1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1X1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6