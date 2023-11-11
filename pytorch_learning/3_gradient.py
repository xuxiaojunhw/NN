#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/7 22:25
# @Author  : xuxiaojun
# @File    : 3_gradient.py
# @Description : 
import matplotlib.pyplot as plt

# prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# initial guess of weight
w = 1.0

# 梯度下降法只能找到局部最优点， 无法找到全局最优点
# 但是深度学习中， 损失函数不会有很多局部最优点，事实证明，局部最优点是很少的，训练过程中，一般不会陷入到局部最优点
# 但是会存在鞍点， 鞍点位置的梯度为0, 梯度为0 的地方没有局部最优点，鞍点附近无法在进行迭代。
# 一般来说训练后的损失函数都是下降的，但是如果发现损失函数有上升的情况说明训练发散了， 有可能是学习率太大导致的。
# 实际使用过程中，更多的是使用随机梯度下降，正常的梯度下降是使用样本的平均损失作为更新的依据
# 随机梯度下降就是对提供的N个数据中，随机获取一个数据，计算其梯度，并更新其权重
# 随机梯度下降的原因是因为有鞍点（对所有样本的损失求平均获取的），如果只取其中的一个样本， 由于样本有噪声，随机噪声可以将权重向前推动
# 事实证明，随机梯度下降对跨越鞍点是非常有效的。

# define the model linear model y = w*x
def forward(x):
    return x * w


# define the cost function MSE
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


# define the gradient function  gd
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


epoch_list = []
cost_list = []

print('predict (before training)', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val  # 0.01 learning rate
    print('epoch:', epoch, 'w=', w, 'loss=', cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)

print('predict (after training)', 4, forward(4))
plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()