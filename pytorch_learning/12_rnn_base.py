#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/1 21:49
# @Author  : xuxiaojun
# @File    : 12_rnn_base.py
# @Description : 
# 输入维度input_size,隐藏层维度hidden_size

import torch

batch_size = 1
seq_len = 3  # 序列长度， X1, X2, X3 的个数
input_size = 4  # 输入x1,x2 的维度
hidden_size = 2  # 输入序列通过转换后的维度


def cell_test():
    cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
    # 维度最重要
    dataset = torch.randn(seq_len, batch_size, input_size)
    # [[[-0.4225, 1.2800, 0.3116, -2.3824]],
    #  [[-2.8636, 0.6819, -0.0600, 0.3516]],
    #  [[-0.1530, -0.0433, -0.8542, 1.3860]]]
    # 初始化时设为零向量
    hidden = torch.zeros(batch_size, hidden_size)

    for idx, input in enumerate(dataset):
        # 这个循环的以上是获取当前时刻t 的张量
        print('=' * 20, idx, '=' * 20)
        print('Input size:', input.shape)
        # 输入的input 的维度（B*input_size）, hidden的维度（B*hidden_size）
        # 输出的hidden维度（B*hidden_size）
        hidden = cell(input, hidden)  # 当前时刻的输入iz
        print('outputs size: ', hidden.shape)
        print(hidden)


def rnn_test():
    # 说明input维度，hidden维度，以及RNN层数
    # RNN计算耗时大，不建议层数过深

    # inputs指的是X1……Xn的整个输入序列(seq_len, batch_size, input_size)
    # hidden指的是前置条件H0 (num_layers, batch_size, hidden_size)
    # out指的是每一次迭代的H1……Hn隐藏层序列
    # hidden_out指的是最后一次迭代得到输出Hn
    # out, hidden_out = cell(inputs, hidden)

    batch_size = 1
    seq_len = 5
    input_size = 4
    hidden_size = 2
    num_layers = 3
    # 其他参数
    # batch_first=True 维度从(SeqLen*Batch*input_size)变为（Batch*SeqLen*input_size）
    cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    inputs = torch.randn(seq_len, batch_size, input_size)
    hidden = torch.zeros(num_layers, batch_size, hidden_size)
    out, hidden = cell(inputs, hidden)
    print("Output size: ", out.shape)
    print("Output: ", out)
    print("Hidden size: ", hidden.shape)
    print("Hidden: ", hidden)


def seq2seq_rnn_cell_example_test():
    input_size = 4
    hidden_size = 3
    batch_size = 1
    # 构建输入输出字典
    idx2char_1 = ['e', 'h', 'l', 'o']
    idx2char_2 = ['h', 'l', 'o']
    x_data = [1, 0, 2, 2, 3]
    y_data = [2, 0, 1, 2, 1]
    # y_data = [3, 1, 2, 2, 3]
    one_hot_lookup = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]
    # 构造独热向量，此时向量维度为(SeqLen*InputSize)
    x_one_hot = [one_hot_lookup[x] for x in x_data]
    # view(-1……)保留原始SeqLen，并添加batch_size,input_size两个维度
    inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
    # 将labels转换为（SeqLen*1）的维度
    labels = torch.LongTensor(y_data).view(-1, 1)

    class Model(torch.nn.Module):
        def __init__(self, input_size, hidden_size, batch_size):
            super(Model, self).__init__()
            self.batch_size = batch_size
            self.input_size = input_size
            self.hidden_size = hidden_size

            self.rnncell = torch.nn.RNNCell(input_size=self.input_size,
                                            hidden_size=self.hidden_size)

        def forward(self, input, hidden):
            # RNNCell input = (batchsize*inputsize)
            # RNNCell hidden = (batchsize*hiddensize)
            hidden = self.rnncell(input, hidden)
            return hidden

        # 初始化零向量作为h0，只有此处用到batch_size
        def init_hidden(self):
            return torch.zeros(self.batch_size, self.hidden_size)

    net = Model(input_size, hidden_size, batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

    for epoch in range(15):
        # 损失及梯度置0，创建前置条件h0
        loss = 0
        optimizer.zero_grad()
        hidden = net.init_hidden()

        print("Predicted string: ", end="")
        # inputs=（seqLen*batchsize*input_size） labels = (seqLen*1)
        # input是按序列取的inputs元素（batchsize*inputsize）
        # label是按序列去的labels元素（1）
        for input, label in zip(inputs, labels):
            hidden = net(input, hidden)
            # 序列的每一项损失都需要累加
            loss += criterion(hidden, label)
            # 多分类取最大
            _, idx = hidden.max(dim=1)
            print(idx2char_2[idx.item()], end='')

        loss.backward()
        optimizer.step()

        print(", Epoch [%d/15] loss = %.4f" % (epoch + 1, loss.item()))


def seq2seq_rnn_example():
    input_size = 4
    hidden_size = 3
    batch_size = 1
    num_layers = 1
    seq_len = 5
    # 构建输入输出字典
    idx2char_1 = ['e', 'h', 'l', 'o']
    idx2char_2 = ['h', 'l', 'o']
    x_data = [1, 0, 2, 2, 3]
    y_data = [2, 0, 1, 2, 1]
    # y_data = [3, 1, 2, 2, 3]
    one_hot_lookup = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]

    x_one_hot = [one_hot_lookup[x] for x in x_data]

    inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
    # labels（seqLen*batchSize,1）为了之后进行矩阵运算，计算交叉熵
    labels = torch.LongTensor(y_data)

    class Model(torch.nn.Module):
        def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
            super(Model, self).__init__()
            self.batch_size = batch_size  # 构造H0
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.rnn = torch.nn.RNN(input_size=self.input_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=num_layers)

        def forward(self, input):
            hidden = torch.zeros(self.num_layers,
                                 self.batch_size,
                                 self.hidden_size)
            out, _ = self.rnn(input, hidden)
            # reshape成（SeqLen*batchsize,hiddensize）便于在进行交叉熵计算时可以以矩阵进行。
            return out.view(-1, self.hidden_size)

    net = Model(input_size, hidden_size, batch_size, num_layers)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

    # RNN中的输入（SeqLen*batchsize*inputsize）
    # RNN中的输出（SeqLen*batchsize*hiddensize）
    # labels维度 hiddensize*1
    for epoch in range(15):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, idx = outputs.max(dim=1)
        idx = idx.data.numpy()
        print('Predicted string: ', ''.join([idx2char_2[x] for x in idx]), end='')
        print(", Epoch [%d/15] loss = %.3f" % (epoch + 1, loss.item()))


def rnn_embedding_test():
    input_size = 4
    num_class = 4
    hidden_size = 8
    embedding_size = 10
    batch_size = 1
    num_layers = 2
    seq_len = 5

    idx2char_1 = ['e', 'h', 'l', 'o']
    idx2char_2 = ['h', 'l', 'o']

    x_data = [[1, 0, 2, 2, 3]]
    y_data = [3, 1, 2, 2, 3]

    # inputs 作为交叉熵中的Inputs，维度为（batchsize，seqLen）
    inputs = torch.LongTensor(x_data)
    # labels 作为交叉熵中的Target，维度为（batchsize*seqLen）
    labels = torch.LongTensor(y_data)

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.emb = torch.nn.Embedding(input_size, embedding_size)

            self.rnn = torch.nn.RNN(input_size=embedding_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    batch_first=True)

            self.fc = torch.nn.Linear(hidden_size, num_class)

        def forward(self, x):
            hidden = torch.zeros(num_layers, x.size(0), hidden_size)
            x = self.emb(x)
            x, _ = self.rnn(x, hidden)
            x = self.fc(x)
            return x.view(-1, num_class)

    net = Model()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

    for epoch in range(15):
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, idx = outputs.max(dim=1)
        idx = idx.data.numpy()
        print('Predicted string: ', ''.join([idx2char_1[x] for x in idx]), end='')
        print(", Epoch [%d/15] loss = %.3f" % (epoch + 1, loss.item()))


if __name__ == '__main__':
    rnn_embedding_test()
