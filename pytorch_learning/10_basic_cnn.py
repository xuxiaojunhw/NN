#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/22 20:22
# @Author  : xuxiaojun
# @File    : 10_basic_cnn.py
# @Description : 


import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


"""
卷积： 是卷积核和图片的一个区域进行数乘， 也就是下标一致的数值进行相乘。
对于多个通道， 各个通道的卷积核可以不一样， 输入通道的通道数和卷积核的通道数需要保持一致。
卷积运算： 各个通道分别与卷积核数乘， 并将结果相加的运算, 这样得到的结果只有一个通道
如果要求输出有M 个通道， 那么可以使用M个卷积核与输入进行卷积运算。

"""
# prepare dataset

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# design model using class


class Net(torch.nn.Module):
    # 0、前一部分叫做Feature
    # Extraction，后一部分叫做classification
    # 1、每一个卷积核它的通道数量要求和输入通道是一样的。这种卷积核的总数有多少个和你输出通道的数量是一样的。
    # 2、卷积(convolution)后，C(Channels)变，W(width)和H(Height) 可变可不变，
    # 取决于是否padding（在原始图像四周加一圈像素）。subsampling(或pooling ) 后，C不变，W和H变。
    # 3、卷积层：保留图像的空间信息。
    # 4、卷积层要求输入输出是四维张量(B, C, W, H)，全连接层的输入与输出都是二维张量(B, Input_feature)。 B为batch-size
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)   # 10 必须与输入通道数相同， 1 决定了输出通道数 stride 代表卷积核移动的像素点数
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)  # max Poll2d 就是一个下采样的方法，将数据使用2X2 的网格划分，去网格类的最大值， MaxPool2d 不能改变通道数量。
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # flatten data from (n,1,28,28) to (n, 784)

        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # -1 此处自动算出的是320
        # print("x.shape",x.shape)
        x = self.fc(x)

        return x


model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 获取设备
model.to(device)  # convert parameters and buffers of all the modules to CUDA Tensor

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)  # 需要将模型和数据放到同一个显卡上
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def model_test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))
    return correct / total


if __name__ == '__main__':
    epoch_list = []
    acc_list = []

    for epoch in range(10):
        train(epoch)
        acc = model_test()
        epoch_list.append(epoch)
        acc_list.append(acc)

    plt.plot(epoch_list, acc_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

