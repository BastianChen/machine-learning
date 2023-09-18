import time
import torch
from torchvision import datasets, transforms
from net import Net_Linear
from torch.utils.data import DataLoader


class Detector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net_Linear().to(self.device)
        self.net.load_state_dict((torch.load("model/model_without_conv.pth")))
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.test_dataset = DataLoader(datasets.MNIST("datasets/", train=False, download=False, transform=trans),
                                       batch_size=1, shuffle=False)

    def detect(self):
        self.net.eval()
        correct = 0
        with torch.no_grad():
            for index, (data, label) in enumerate(self.test_dataset):
                data, label = data.to(self.device), label.to(self.device)
                outputs = self.net(data)
                pred = torch.argmax(outputs)
                gt = label.item()

                num = pred == gt
                correct += num

        acc = 100. * correct / len(self.test_dataset.dataset)

        print(
            f'accuracy: {correct}/{len(self.test_dataset.dataset)} ({acc:.3f}%)')


if __name__ == '__main__':
    start = time.time()
    detector = Detector()
    detector.detect()
    end = time.time()
    print(f"latency:{end - start}")
