#  -*-  coding  =  utf-8  -*-
#  @Time  :2023/3/12  16:51
#  @Author:Su
#  @File  :  3.2SoftMaxEasyImp.py
#  @Software:  PyCharm
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std = 0.01)

net.apply(init_weight)
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(),  lr = 0.1)
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()