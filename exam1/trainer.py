import torch
import os
from torchvision import datasets, transforms
from net import Net_Linear, Net_Conv
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse


class Trainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_path = "model"
        if args.model:
            self.save_path = f"{self.net_path}/model_with_conv.pth"
            log_name = f"runs/Layer5_with_conv_lr{args.lr_rate}_epoch{args.epoch}"
            self.net = Net_Conv().to(self.device)
        else:
            self.save_path = f"{self.net_path}/model_without_conv.pth"
            log_name = f"runs/Layer5_without_conv_lr{args.lr_rate}_epoch{args.epoch}"
            self.net = Net_Linear().to(self.device)
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.train_data = DataLoader(datasets.MNIST("datasets/", train=True, download=False, transform=self.trans),
                                     batch_size=500, shuffle=True)
        self.test_data = DataLoader(datasets.MNIST("datasets/", train=False, download=False, transform=self.trans),
                                    batch_size=100, shuffle=True, drop_last=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr_rate)
        self.writer = SummaryWriter(log_name)
        if not os.path.exists(self.net_path):
            os.mkdir(self.net_path)

    def train(self):
        index = 0
        for epoch in range(args.epoch):
            self.net.train()
            for _, (data, label) in enumerate(self.train_data):
                data, label = data.to(self.device), label.to(self.device)
                output = self.net(data)
                loss = self.net.getLoss(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar("loss", loss, index)
                index += 1
            self.validate(epoch)
            torch.save(self.net.state_dict(), self.save_path)

    def validate(self, epoch):
        self.net.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, label in self.test_data:
                data, label = data.to(self.device), label.to(self.device)
                output = self.net(data)
                test_loss += self.net.getLoss(output, label)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

        test_loss /= len(self.test_data.dataset)
        acc = 100. * correct / len(self.test_data.dataset)

        print(f'Test:epoch: {epoch}, average loss: {test_loss:.8f}, '
              f'accuracy: {correct}/{len(self.test_data.dataset)} ({acc:.3f}%)')
        self.writer.add_scalar("acc", acc, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default=1, type=int, help="model type,0:without conv layer,1:with conv layer")
    parser.add_argument('-l', '--lr_rate', default=5e-04, type=float, help="learning rate")
    parser.add_argument('-e', '--epoch', default=20, type=int, help="epoch number")

    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
