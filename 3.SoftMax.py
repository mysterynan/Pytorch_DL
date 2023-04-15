#  -*-  coding  =  utf-8  -*-
#  @Time  :2023/3/9  20:39
#  @Author:Su
#  @File  :  3.SoftMax.py
#  @Software:  PyCharm
from matplotlib import pyplot as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms #对数据进行操作
from d2l import torch as d2l

d2l.use_svg_display()

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../data",train=True,
                                               transform=trans, download=False)
mnist_test = torchvision.datasets.FashionMNIST(root="../data",train=False,
                                               transform=trans, download=False)
# print(len(mnist_train))
# print(len(mnist_test))

# 输出为torch.Size([1, 28, 28])
# print(mnist_train[0][0].shape)
def get_fashion_mnist_labels(labels):

    # 返回Fashion-MINST数据集的文本标签

     text_labels = ['t_shirt','trouser', 'pullover', 'dress', 'coat', 'sandal',
                    'shirt', 'sneaker', 'bag', 'ankle boot']
     return [text_labels[int(i)] for i in labels]



def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    #使用matplotlib来画出图片
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))#shape是（18，28，28） 分成2行 9列
# plt.show()   #或者d2l.plt.show()  pycharm必用
batch_size = 256

def get_dataloader_workers():

    return 2

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
# timer = d2l.Timer()
# for X,y in train_iter:
#     continue
# print(f'{timer.stop():.2f} sec')


def load_data_fashion_mnist(batch_size, resize=None):  # @save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]  # 图片转成张量
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

# 测试resize是否可以重置图片大小
# train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
# for X, y in train_iter:
#     print(X.shape, X.dtype, y.shape, y.dtype)
#     break