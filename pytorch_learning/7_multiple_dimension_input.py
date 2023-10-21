#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/21 19:21
# @Author  : xuxiaojun
# @File    : 7_multiple_dimension_input.py
# @Description :

"""
一般希望将方程计算转换为矩阵运行，变换为矩阵运算可以使用到CPU的运算能力/GPU的运算能力。
矩阵的本质是将一个N维空间映射到M维空间的一个线性转换， Y=AX Y为Mx1 A为MxN X为N维， 可以将矩阵看为一个空间变化的函数
在神经网络中，一般使用32位浮点数， 这是因为在在模型运算中，使用的游戏显卡比较多比如1080， 2080 等， 他们只支持32位浮点数。
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1]) # 第一个‘：’是指读取所有行，第二个‘：’是指从第一列开始，最后一列不要
y_data = torch.from_numpy(xy[:, [-1]]) # [-1] 最后得到的是个矩阵

# design model using class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 输入数据x的特征是8维，x有8个特征
        # 代表的是可以将一个任意8维空间的向量映射到6维空间的一个线性变化
        # 但是在做空间变化的时候， 不一定用的是线性变换，可能是非线性的，
        # 所以希望使用用多个线性变化层（找到组合的权重）模拟出非线性的变换
        # 神经网络的本质是找一个非线性的空间变化函数
        # 将维度分步降低的过程中， 一定需要过一个非线性的函数， 因为linear是不做非线性的
        # 今天这个问题的本质是需要找一个8维到1维空间的非线性变化函数。
        # 引入的sigmoid 函数在神经网络里面叫激活函数
        # 神经网络的设计，网络输入输出层的数值取啥， 是一个典型的超参数搜索的问题，
        # 一一般来说隐层越多，神经元越多，对非线性变化的学习能力越强。
        # 并非学习能力越强越好， 因为学习能力强会将输入样本中的噪声规律学习进去，这不是我们想要的，学习能力必须要有泛化能力。

        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        # sigmoid 就是激活函数，他是一个模块， 将其看作是网络的一层，而不是简单的函数使用，
        # 他没有参数，也没有需要训练的东西
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))  # y hat
        return x


model = Model()

# construct loss and optimizer
# criterion = torch.nn.BCELoss(size_average = True)
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch_list = []
loss_list = []
# training cycle forward, backward, update
def training():
    for epoch in range(100):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())
        epoch_list.append(epoch)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


if __name__ == '__main__':
    training()
    plt.plot(epoch_list, loss_list)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()