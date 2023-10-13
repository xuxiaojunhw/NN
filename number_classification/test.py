import torch
import random

import numpy as np
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tqdm import tqdm
from torchvision import datasets, transforms

transform=transforms.Compose([
    transforms.ToTensor(),
    ])

dataset1 = datasets.MNIST(
    '/home/featurize/data',
    train=True,
    download=True,
    transform=transform
)
dataset2 = datasets.MNIST(
    '/home/featurize/data',
    train=False,
    transform=transform
)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=512)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=512)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


model = Model()

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


amp = True


def strat_training():
    losses = []
    epochs = 10
    print('Start training...')
    running_loss = 0.0
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            if amp == False:
                scaler = torch.cuda.amp.GradScaler()
                autocast = torch.cuda.amp.autocast
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        print(f'Epoch: {epoch + 1} Loss: {loss.item()}')

    print('Training Finished.')
    f, ax = plt.subplots()
    sns.lineplot(x=np.linspace(0, 0 + epochs - 1, epochs), y=losses).set_title('Training Loss');
    plt.show()


def start_test():
    for img, label in test_loader:
        pred = model(img.cuda())
        label = torch.argmax(pred, axis=1)
        break
    f, axs = plt.subplots(2, 10, figsize=(16, 4))
    random_index = [random.randint(0, 128 - 1) for i in range(20)]
    for i in range(2):
        for j in range(10):
            axs[i][j].set_axis_off()
            axs[i][j].set_title(f'Label {label[random_index[i * 10 + j]]}')
            axs[i][j].imshow(img[random_index[i * 10 + j]].permute(1, 2, 0), cmap=cm.gray)
    plt.show()


if __name__ == '__main__':
    strat_training()
    start_test()
