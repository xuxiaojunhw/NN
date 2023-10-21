#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/21 11:33
# @Author  : xu_xiao_jun
# @File    : 6_logistic_regression.py
# @Description : logistic regression 虽然叫回归，其实是一个分类问题，
# 如果输出的数值为实数空间的那就叫做回归问题，如果结果是一个离散的集合，就叫做分类问题了。


import torch
import numpy as np
import matplotlib.pyplot as plt

# prepare data_set
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


# design model using class
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 代表输入为一维， 输出也是一维度

    def forward(self, x):
        """
        该函数必须定义，
        @param x:
        @return:
        """
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()

# construct loss and optimizer
# 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加。
# BCELoss 是CrossEntropyLoss的一个特例，只用于二分类问题，而CrossEntropyLoss可以用于二分类，也可以用于多分类。
criterion = torch.nn.BCELoss(size_average=False)
# 随机梯度下降优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# training cycle forward, backward, update
def training():
    """
    训练代码，训练次数为1000
    @return:
    """
    for epoch in range(1000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print(epoch, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('w = ', model.linear.weight.item())
    print('b = ', model.linear.bias.item())


def test1():
    """
    测试对学习小时数为4.0 的时候， 学习成绩合格的概率，数值越大合格的概率越大，取值在0-1之间
    @return:
    """
    training()
    x_test = torch.Tensor([[4.0]])
    y_test = model(x_test)
    print('y_pred = ', y_test.data)


def test2():
    """
    使用100个数值进行测试，并保存测试结果
    @return:
    """
    training()
    x = np.linspace(0, 10, 200)  # 在0 -10 之间获取200个数值
    x_t = torch.Tensor(x).view((200, 1))
    y_t = model(x_t)
    y = y_t.data.numpy()
    plt.plot(x, y)
    plt.plot([0, 10], [0.5, 0.5], c='r')
    plt.xlabel("Hours")
    plt.ylabel("Probability of pass")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    test2()


