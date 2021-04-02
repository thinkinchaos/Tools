import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

batch_size = 100
mean = 0.1307
std = 0.3081

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


def show_some_imgs_in_batch(batch_data, img_num):
    assert img_num <= batch_size

    assert img_num % 2 == 0
    batch_data = batch_data[:img_num]

    batch_data = torchvision.utils.make_grid(batch_data, nrow=img_num // 2, padding=0)

    batch_data = np.array(batch_data * std + mean)
    batch_data = np.transpose(batch_data, (1, 2, 0))

    plt.imshow(batch_data)
    plt.axis('off')
    plt.pause(0.5)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_channels = 1

        self.conv2d_1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2d_2 = nn.Conv2d(32, 32, 3, 1)
        self.conv2d_3 = nn.Conv2d(32, 64, 3, 1)
        self.conv2d_4 = nn.Conv2d(64, 64, 3, 1)

        self.dense_1 = nn.Linear(4 * 4 * 64, 200)
        self.dense_2 = nn.Linear(200, 200)
        self.dense_3 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2d_3(x))
        x = F.relu(self.conv2d_4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.permute((0, 2, 3, 1))

        x = x.contiguous().view(-1, 4 * 4 * 64)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)
        return F.log_softmax(x, dim=1)


net = Net().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
epochs = 10
for epoch in range(epochs):
    net.train()
    for i, data in enumerate(trainloader, 0):
        datas, labels = data
        datas = datas.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = net(datas)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print('Epoch %d [%5d/%5d] \tloss %.3f' % (epoch + 1, batch_size * (i + 1), len(trainset), loss.item()))

    net.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            datas, labels = data
            datas = datas.cuda()
            labels = labels.cuda()

            outputs = net(datas)
            _, predicted = torch.max(outputs, 1)

            show_num = 10
            predicted = predicted[:show_num]
            show_some_imgs_in_batch(datas.cpu(), img_num=show_num)

            print(predicted)
