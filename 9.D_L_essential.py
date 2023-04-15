#  -*-  coding  =  utf-8  -*-
#  @Time  :2023/3/24  13:59
#  @Author:Su
#  @File  :  9.D_L_essential.py
#  @Software:  PyCharm
import torch
from torch import nn
from torch.nn import functional as F

# 关于构建一些基本的继承类的用法
# net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
#
# X = torch.rand(2, 20)
# print(X)
# print(net(X))
#
#
# class MLP(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(20, 256)
#         self.out = nn.Linear(256, 10)
#
#     def forward(self, X):
#         return self.out(F.relu(self.hidden(X)))
#
#
# net = MLP()
# print(net(X))
#
#
# class MySequential(nn.Module):
#
#     def __init__(self, *args):
#         super().__init__()
#         for block in args:
#             self._modules[block] = block
#
#     def forward(self, X):
#         for block in self._modules.values():
#             X = block(X)
#         return X
#
#
# net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
#
# print(net(X))
#
#
#
# class FixedHiddenMLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 不计算梯度的随机权重参数。因此其在训练期间保持不变
#         self.rand_weight = torch.rand((20, 20), requires_grad=False)
#         self.linear = nn.Linear(20, 20)
#
#     def forward(self, X):
#         X = self.linear(X)
#         # 使用创建的常量参数以及relu和mm函数
#         X = F.relu(torch.mm(X, self.rand_weight) + 1)
#         # 复用全连接层。这相当于两个全连接层共享参数
#         X = self.linear(X)
#         # 控制流
#         while X.abs().sum() > 1:
#             X /= 2
#         return X.sum()

# # 关于参数管理的用法
#
# net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
# X = torch.rand(size=(2, 4))
# print(net(X))
#
# print(net[2].state_dict())
#
# print(type(net[2].bias))
# print(net[2])
# print(net[2].weight)
# print(net[2].weight.data)
#
# print(net[2].weight.grad == None)
#
# # 一次性访问所有参数
# print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# print(*[(name, param.shape) for name, param in net.named_parameters()])
# print(net.state_dict()['2.weight'].data)
#
#
# # 嵌套收集参数
# def block1():
#     return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())
#
# def block2():
#     net = nn.Sequential()
#     for i in range(4):
#         net.add_module(f'block{i}',block1())
#     return net
#
#
# rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
# rgnet(X)
# print(rgnet(X))
# print(rgnet)
#
#
# def init_normal(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight, mean=0, std=0.01)
#         nn.init.zeros_(m.bias)
#
#
# net.apply(init_normal)
# print(net[0].weight.data[0])
# print(net[0].bias.data[0])
#
#
# # 自定义初始化
# def init_constant(m):
#     if type(m) == nn.Linear:
#         nn.init.constant_(m.weight, 1)
#         nn.init.zeros_(m.bias)
#
# def my_init(m):
#     if type(m) == nn.Linear:
#         print("Init", *[(name, param.shape)
#                         for name, param in m.named_parameters()][0])
#         nn.init.uniform_(m.weight, -10, 10)
#         m.weight.data *= m.weight.data.abs() >= 5
#
# net.apply(my_init)
# net[0].weight[:2]
#
#
# # 参数绑定
# # 我们需要给共享层一个名称，以便可以引用它的参数
# shared = nn.Linear(8, 8)
# net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
#                     shared, nn.ReLU(),
#                     shared, nn.ReLU(),
#                     nn.Linear(8, 1))
# net(X)
# # 检查参数是否相同
# print(net[2].weight.data[0] == net[4].weight.data[0])
# net[2].weight.data[0, 0] = 100
# # 确保它们实际上是同一个对象，而不只是有相同的值
# print(net[2].weight.data[0] == net[4].weight.data[0])


# 自定义层
# class CenteredLayer(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, X):
#         return X - X.mean()
#
#
# layer = CenteredLayer()
# print(layer(torch.Floattensor([1, 2, 3, 6, 4])))
# net = nn.Sequential(nn.Linear(8, 128), layer)
# Y = net(torch.rand(4, 8))


# # 带参数的层
# class MyLinear(nn.Module):
#     def __init__(self, in_units, units):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn(in_units, units))
#         self.bias = nn.Parameter(torch.randn(units))
#
#     def forward(self, X):
#         linear = torch.matmul(self.weight, X) + self.bias.data
#         return F.relu(linear)
#
#
# dense = MyLinear(5, 3)
# print(dense.weight)

#
# # 读写文件
# x = torch.arange(4)
# torch.save(x, 'x-file')
#
# x2 = torch.load('x-file')
# print(x2)
#
# y = torch.zeros(4)
# torch.save([x, y], 'x-files')
# x2, y2 = torch.load('x-files')
# print(x2,y2)
#
# mydict = {'x': x, 'y': y}
# torch.save(mydict, 'mydict')
# mydict2 = torch.load('mydict')
# print(mydict2)
#
# # 加载和保存模型
# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(20, 256)
#         self.out = nn.Linear(256, 10)
#
#     def forward(self, X):
#         return self.out(F.relu(self.hidden(X)))
#
#
# net = MLP()
# X = torch.randn(size=(2, 20))
# Y = net(X)
# # 保存模型参数
# torch.save(net.state_dict(),'mlp.params')
# # 会随机初始化MLP参数
# clone = MLP()
# # 从mlp.params中读取数据并覆盖到clone的参数
# clone.load_state_dict(torch.load('mlp.params'))
# clone.eval()
#
# clone_Y = clone(X)
# print(clone_Y == Y)

# print(torch.device('cpu'))
# print(torch.cuda.device('cuda'))
# print(torch.cuda.device_count())

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

# print(try_gpu())
# print(try_gpu(10))
# print(try_all_gpus())
# x = torch.tensor([1, 2, 3])
# print(x.device)


x = torch.ones( 2, 3, device=try_gpu())
print(x)

y = torch.rand(2, 3,device=try_gpu())
print(y)

z = x.cuda(0)
print(x)
print(z)
print(x+y)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())

print(net(x))