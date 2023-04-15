#  -*-  coding  =  utf-8  -*-
#  @Time  :2023/3/8  15:35
#  @Author:Su
#  @File  :  1.DataPretrain.py
#  @Software:  PyCharm
import torch
import os
import pandas as pd
# x = torch.arange(12)
#
# print(x)
# print(x.shape)
# print(x.numel())
#
# X = x.reshape(3,4)
# print(X)
# z = torch.zeros(2,3,4)
# print(z)
# one = torch.ones((2,3,4))
# print(one)
# x = torch.tensor([[1,2,3,4],[2,5,6,4],[5,6,2,8]])
# print(x)
# x = torch.tensor([1,2.3,6,8,9])
# y = torch.tensor([2,3,6,5,9])
# print(x+y,x-y,x*y,x/y,x**y)
#
# X = torch.arange(12,dtype=torch.float32).reshape((3,4))
# Y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[5,6,3,2]])
# Z = torch.cat((X,Y),dim=0)
# S = torch.cat((X,Y),dim=1)
# print(X,Y,Z,S)
# print(X==Y)
# sum = X.sum()
# print(sum)
#广播机制
# a = torch.arange(3).reshape(3,1)
# b = torch.arange(2).reshape(1,2)
# print(a,b,a+b)
# c = a + b
# print(c)
# print(c[-1,:],c[1:3,:]) #左闭右开 1:3 指的第一行到第二行所有的数

# os.makedirs(os.path.join('..','data'),exist_ok=True)
# data_file = os.path.join('..','data','house_tiny.csv')
# with open(data_file,'w') as f:
#     f.write('NumRooms,Alley,Price\n')
#     f.write('NA,Pave,127500\n')
#     f.write('2,NA,12377500\n')
#     f.write('4,BA,18375400\n')
# data = pd.read_csv(data_file)
# print(data)
#
# inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]
#
# # print(inputs)
#
# inputs = inputs.fillna(inputs.mean())
# # print(inputs)
#
# inputs = pd.get_dummies(inputs,dummy_na=True)
# print(inputs)
#
# print(outputs)
#
# x,y = torch.tensor(inputs.values),torch.tensor(outputs.values)
#
# print(x,y)

# x = torch.arange(4.0)
# x.requires_grad_(True) #梯度存起来
# print(x.grad)
# y = 2 * torch.dot(x, x)
# print(y)
# y.backward() #反向传播函数自动计算y关于x每个分量的梯度
# print(x.grad)
# print(x.grad == 4 * x)
#
# x.grad.zero_()
# y = x.sum() #sum函数相当与x1+x2+..xn
# y.backward()
# print(x.grad)
#
# x.grad.zero_()
# y = x * x
# print(y)
# print(y.sum())
# y.sum().backward()
# x.grad
# print(x.grad)

# y = x * x
# print(y)
# u = y.detach() #表示将y作为一个常数，而不是x的函数赋值给u,u为一个常数
# print(u)
# z = u * x
# print(z)
# print(z.sum())
# z.sum().backward()
# print(x.grad == u)
#
# x.grad.zero_()
# y.sum().backward()
# print(x.grad)
# print(x.grad == 2 * x)
# def f(a):
#     b = 2 * a
#     while b.norm() < 1000:
#         b = b * 2
#     if b.sum() > 0:
#         c = b
#     else:
#         c = 100 * b
#     return c
# a = torch.randn(size=(),requires_grad=True)
# print(a)
# d = f(a)
# d.backward()
# print(a.grad)
# print(d / a)h.randn(5,2,2)
a = torc
print(a)