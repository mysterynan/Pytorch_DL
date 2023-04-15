#  -*-  coding  =  utf-8  -*-
#  @Time  :2023/4/1  20:57
#  @Author:Su
#  @File  :  d_l_util.py
#  @Software:  PyCharm

import numpy as np
import math
import time
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt
import torch


#使用svg格式绘制图片
def use_svg_display():
    backend_inline.set_matplotlib_formats('svg')


# 设置图表的大小
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


# 设置坐标系的各个轴的参数，以及legend（图表曲线的名称）
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    # x刻度与y刻度
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    # 设置x与y的视图大小
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    # 设置网格线
    axes.grid()


# 绘制多条曲线 fmts：定义基本格式的便捷方法
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
         xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):

    if legend is None:
        legend = []

    set_figsize(figsize)
    # ???
    axes = axes if axes else plt.gca()

    # 如X只有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        # ???
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        # 将X维数与Y维数变成一样的
        X = X * len(Y)
    # ???
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.show()


# def f(x):
#     return 3 * x ** 2 - 4 * x


# x = np.arange(0, 3, 0.1)
# print(x)
# print(f(x).shape, x.shape,  [f(x), 2 * x - 3])
# plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])


# 计时器,记录多次运行时间
class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times)/len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()



















