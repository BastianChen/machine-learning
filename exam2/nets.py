import torch
from torch import nn
import torchvision


# 封装卷积层
class ConvolutionalLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


# 残差层
class ResidualLayer(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.layer = nn.Sequential(
            ConvolutionalLayer(input_channels, input_channels // 2, 1, 1, 0),
            ConvolutionalLayer(input_channels // 2, input_channels // 2, 3, 1, 1),
            ConvolutionalLayer(input_channels // 2, input_channels, 1, 1, 0)
        )

    def forward(self, x):
        return x + self.layer(x)


# 下采样层
class DownSamplingLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layer = ConvolutionalLayer(input_channels, output_channels, 3, 2, 1)

    def forward(self, data):
        return self.layer(data)


class Net1(nn.Module):
    def __init__(self, num, channels=3):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(channels, 16, 3, 2),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 64, 3),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 32, 3),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 16, 3),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Flatten()
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(2240, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, num)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layer(x)
        # print(x.shape)
        x = self.linear_layer(x)
        x = self.sigmoid(x)
        return x


class Net2(nn.Module):
    def __init__(self, num, channels=3):
        super().__init__()
        self.conv_layer = nn.Sequential(
            ConvolutionalLayer(channels, 64, 3, 1),
            nn.MaxPool2d(3, 2),
            ResidualLayer(64),
            DownSamplingLayer(64, 128),
            ResidualLayer(128),
            DownSamplingLayer(128, 256),
            ResidualLayer(256),
            ResidualLayer(256),
            DownSamplingLayer(256, 512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            DownSamplingLayer(512, 256),
            ResidualLayer(256),
            ResidualLayer(256),
            ConvolutionalLayer(256, 128, 3),
            ResidualLayer(128),
            ConvolutionalLayer(128, 64, 3),
            ResidualLayer(64),
            nn.Flatten()
        )
        self.linear_layer = nn.Linear(384, num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layer(x)
        # print(x.shape)
        x = self.linear_layer(x)
        x = self.sigmoid(x)
        return x


class Net3(nn.Module):
    def __init__(self, num, type='18', channels=3):
        super().__init__()
        if type == '18':
            print(f"using resnet18,channels is {channels}")
            self.model = torchvision.models.resnet18(weights=True)
            self.model.fc = nn.Linear(512, num, bias=True)
        elif type == '34':
            print(f"using resnet34,channels is {channels}")
            self.model = torchvision.models.resnet34(weights=True)
            self.model.fc = nn.Linear(512, num, bias=True)
        else:
            print(f"using resnet50,channels is {channels}")
            self.model = torchvision.models.resnet50(weights=True)
            self.model.fc = nn.Linear(2048, num, bias=True)

        if channels == 1:
            self.model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

        self.sigmoid = nn.Sigmoid()
        # print(self.model)

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    data = torch.randn((2, 3, 210, 180))
    net = Net2(2)
    params = sum([param.numel() for param in net.parameters()])
    print(params)
    output = net(data)
    print(output.shape)
