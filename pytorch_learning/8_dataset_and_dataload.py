#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/21 20:54
# @Author  : xuxiaojun
# @File    : 8_dataset_and_dataload.py
# @Description :

import torch
import numpy as np
from torch.utils.data import Dataset  # 是一个抽象类， 不可以直接实例化
from torch.utils.data import  DataLoader  # 可以被直接实例化


# prepare data_set

class DiabetesDataset(Dataset):
    """
     DiabetesDataset 自 Dataset, 必须要实现如下3个魔法方法。
    在进行梯度下降的时候， 有两个选择一个是使用全量数据（batch）, 其
    优点是可以各个数据样本间没有依赖可以充分使用向量计算优势，提升计算速度。
    另外一个是使用随机梯度下降 SDG, 随机梯度下降使用的是一个样本，
    只用一个样本会有一个比较好的随机性能可以帮助跨越鞍点， 但是训练的时间会比较长。
    因此推出了一个mini_batch 的办法进行训练性能和训练时间熵的均衡。
    """
    def __init__(self, filepath):
        """
        在构造数据集时有两种方法：
            方法一： 在__init__ 中读取所有的数据， 加载到内存中， 适合于小数据集的情况。
            方法二： 如果数据级太大（图片文件）， 在__init__里面只定义一个列表， 给出每一条数据的文件名。
                    在__getitem__函数中，读取具体的数值。
        @param filepath:
        """
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]  # shape(多少行，多少列) 取 0 其实取的就是数据集有多少个
        self.x_data = torch.from_numpy(xy[:, :-1])  # 将数据直接保存到内存中
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('diabetes.csv')
# DataLoader 是pytorch 提供的一个加载器，
train_loader = DataLoader(dataset=dataset,  # 传递的数据集
                          batch_size=32,  # 一个小批量的容量是多少
                          shuffle=True,  # 是否需要进行数据打乱，提高随机性
                          num_workers=0)  # 读取数据的时候， 是否使用多线程， num_workers 为多线程的数量


# design model using class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training cycle forward, backward, update
if __name__ == '__main__':
    for epoch in range(100):
        """
        将整批数据全都跑完一次叫epoch
        the epoch means one forward pass and one backward pass of all the training examples
        """
        for i, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch
            # batch size means the number of training examples in one forward pass
            # lteration: the number of passes, each pass using [batch-size] number of examples
            # 一个有10000 个样本， 一共分为1000个mini-batch , 那么batch-size 为1000, lteration 就 为10
            inputs, labels = data  # train_loader 会自动将数据转换为tensor格式， 此时input, labels 都是张量
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
