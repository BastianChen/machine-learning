import torch
from torch import nn


# 用MLP实现
class Net_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, 10)
        )
        self.crossentropyloss = nn.CrossEntropyLoss()

    def forward(self, data):
        data = data.reshape(data.shape[0], -1)
        y = self.linear(data)
        return y

    def getLoss(self, outputs, labels):
        return self.crossentropyloss(outputs, labels)


# 在MLP的基础上加入Conv算子
class Net_Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 32, 3),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(9 * 9 * 32, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, 10),
            nn.BatchNorm1d(10)
        )
        self.crossentropyloss = nn.CrossEntropyLoss()

    def forward(self, data):
        y = self.conv(data)
        y = y.reshape(y.shape[0], -1)
        y = self.linear(y)
        return y

    def getLoss(self, outputs, labels):
        return self.crossentropyloss(outputs, labels)


if __name__ == '__main__':
    x = torch.Tensor(2, 1, 28, 28)
    # net = Net_Linear()
    # output = net(x)
    # print(output.shape)
    net = Net_Conv()
    output = net(x)
    print(output.shape)
