import torch
import os
from torch import nn
from nets import Net1, Net2, Net3
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from gen_datasets import Datasets
import argparse


class Trainer:
    def __init__(self, args=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert args.model in (1, 2, 3), 'model type must be 1 or 2 or 3'
        if args.model == 1:
            self.net = Net1(args.num_classes, args.channels).to(self.device)
            log_name = f"{args.pth_name}-lr{args.lr_rate}-epoch{args.epoch}-channels{args.channels}"
            self.save_path = f"models/{args.pth_name}-lr{args.lr_rate}-epoch{args.epoch}-channels{args.channels}"
        elif args.model == 2:
            self.net = Net2(args.num_classes, args.channels).to(self.device)
            log_name = f"{args.pth_name}-lr{args.lr_rate}-epoch{args.epoch}-channels{args.channels}"
            self.save_path = f"models/{args.pth_name}-lr{args.lr_rate}-epoch{args.epoch}-channels{args.channels}"
        elif args.model == 3:
            self.net = Net3(args.num_classes, args.type, args.channels).to(self.device)
            log_name = f"{args.pth_name}-lr{args.lr_rate}-epoch{args.epoch}-channels{args.channels}"
            self.save_path = f"models/{args.pth_name}-lr{args.lr_rate}-epoch{args.epoch}-channels{args.channels}"

        self.train_data = Datasets(f"{args.datasets_path}/train", True, args.channels)
        self.test_data = Datasets(f"{args.datasets_path}/test", False, args.channels)
        self.train_dataset = DataLoader(self.train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.test_dataset = DataLoader(self.test_data, batch_size=1, shuffle=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.gamma)
        self.loss = nn.BCELoss()
        # self.loss = nn.CrossEntropyLoss()
        # self.loss = nn.MSELoss()
        self.writer = SummaryWriter(f"runs/{log_name}")
        self.pre_loss = 1000000
        if not os.path.exists("models"):
            os.mkdir("models")

    def train(self):
        index = 0
        for epoch in range(args.epoch):
            self.net.train()
            for i, (data, label) in enumerate(self.train_dataset):
                data, label = data.to(self.device), label.to(self.device)
                outputs = self.net(data)
                loss = self.get_loss(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar("loss", loss, index)
                index += 1
            self.validate(epoch)
            self.scheduler.step()

    def validate(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, label, _ in self.test_dataset:
                data, label = data.to(self.device), label.to(self.device)
                outputs = self.net(data)
                test_loss += self.get_loss(outputs, label)
                pred = outputs.argmax(dim=1, keepdim=True)
                label = label.argmax(dim=1, keepdim=True)
                correct += pred.eq(label).sum().item()

        test_loss /= len(self.test_dataset.dataset)
        if test_loss < self.pre_loss:
            torch.save(self.net.state_dict(), f"{self.save_path}.pth")
            self.pre_loss = test_loss
        acc = 100. * correct / len(self.test_dataset.dataset)

        print(
            f'epoch: {epoch},loss: {test_loss},lr: {self.scheduler.get_last_lr()[0]}, accuracy: {correct}/{len(self.test_dataset.dataset)} ({acc:.3f}%)')
        self.writer.add_scalar("accuracy", acc, epoch)
        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], epoch)

    def get_loss(self, outputs, label):
        return self.loss(outputs, label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default=3, type=int, help="model type,1:net9layers,2:net44layers,3:resnet")
    parser.add_argument('-c', '--channels', default=3, type=int, help="channels of images")
    parser.add_argument('-t', '--type', default='34', type=str,
                        help="type of resnet,18:resnet18,34:resnet34,50:resnet50")
    parser.add_argument('-l', '--lr_rate', default=8e-05, type=float, help="learning rate")
    parser.add_argument('-e', '--epoch', default=30, type=int, help="epoch number")
    parser.add_argument('-p', '--pth_name', default="net34_without_data_augmentation", type=str,
                        help="name of pth file")
    parser.add_argument('-bs', '--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('-gm', '--gamma', default=0.99, type=int, help="gamma of the ExponentialLR")
    parser.add_argument('-dp', '--datasets_path', default="datasets", type=str, help="path of Datasets")
    parser.add_argument('-n', '--num_classes', default=2, type=int, help="num of class")

    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
