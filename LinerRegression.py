#  -*-  coding  =  utf-8  -*-
#  @Time  :2023/3/9  16:04
#  @Author:Su
#  @File  :  LinerRegression.py
#  @Software:  PyCharm
import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt


def synthetic_data(w, b, num_example):
    """生成y = Xw + b + 噪声。"""
    # 返回一个张量，该张量规模为num_example×len(w)大小的张量，其中每个数据服从均值为0，标准差为1的正态分布
    X = torch.normal(0, 1, (num_example,len(w)))
    y = torch.matmul(X,w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1,1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w,true_b,100)
# print('feature:',features[0],'\nlabel:',labels[0])
# d2l.set_figsize()
# d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# plt.show()

#输出batch_size大小的特征和标签
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    #取随机样本下标
    indics = list(range(num_examples))
    random.shuffle(indics) #随机原地打乱indics的数据
    for i in range(0, num_examples, batch_size):
        batch_indics = torch.tensor(indics[i:min(i + batch_size, num_examples)])
        yield features[batch_indics],labels[batch_indics]

batch_size = 10
# for X, y in data_iter(batch_size, features,labels):
#     print(X, '\n',y)
#     break

#定义初始模型参数
w = torch.normal(0, 0.01, size=(2, 1),requires_grad=True)
b = torch.zeros(1, requires_grad=True) #标量，偏差为0

#定义模型
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

#定义损失函数
def squared_loss(y_hat, y):
    """均方误差"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2
#定义优化算法
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad(): #torch.no_grad() 计算过程中不需要计算梯度
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.05
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch + 1}, loss{float(train_l.mean()):f}')



