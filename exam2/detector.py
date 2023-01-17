import torch
from nets import Net1, Net2, Net3
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from gen_datasets import Datasets
import argparse
import time


class Detector:
    def __init__(self, args=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert args.model in (1, 2, 3), 'model type must be 1 or 2 or 3'
        if args.model == 1:
            self.net = Net1(args.num_classes, args.channels).to(self.device)
        elif args.model == 2:
            self.net = Net2(args.num_classes, args.channels).to(self.device)
        elif args.model == 3:
            self.net = Net3(args.num_classes, args.type, args.channels).to(self.device)

        self.save_path = f"models/{args.pth_name}"
        self.net.load_state_dict((torch.load(self.save_path)))
        self.test_data = Datasets(f"{args.datasets_path}/test", False, args.channels)
        self.test_dataset = DataLoader(self.test_data, batch_size=1, shuffle=True)
        self.writer = SummaryWriter(f"runs/error_images/")

    def detect(self):
        self.net.eval()
        correct = 0
        step = 1
        start_time, end_time = 0, 0
        time_list = []
        with torch.no_grad():
            for index, (data, label, image) in enumerate(self.test_dataset):
                data, label = data.to(self.device), label.to(self.device)
                start_time = time.time()
                outputs = self.net(data)
                end_time = time.time()
                inference_time = end_time - start_time
                if index > 0:
                    time_list.append(inference_time)
                pred = outputs.argmax(dim=1, keepdim=True)
                label = label.argmax(dim=1, keepdim=True)
                num = pred.eq(label).sum().item()
                if num == 0:
                    image = image.swapaxes(2, 3)
                    image = image.swapaxes(1, 2)
                    self.writer.add_images(f"error_image_pred_{pred.item()}", image, step)
                    step += 1
                correct += num

        acc = 100. * correct / len(self.test_dataset.dataset)

        print(
            f'accuracy: {correct}/{len(self.test_dataset.dataset)} ({acc:.3f}%) ,mean_latency:{sum(time_list) / len(time_list)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default=3, type=int, help="model type,1:net-9layers,2:net-40layers,3:resnet")
    parser.add_argument('-c', '--channels', default=3, type=int, help="channels of images")
    parser.add_argument('-t', '--type', default='34', type=str,
                        help="type of resnet,18:resnet18,34:resnet34,50:resnet50")
    parser.add_argument('-p', '--pth_name', default="resnet34_without_data_augmentation-lr8e-05-epoch30-channels3.pth",
                        type=str,
                        help="name of pth file")
    parser.add_argument('-dp', '--datasets_path', default="datasets", type=str, help="path of Datasets")
    parser.add_argument('-n', '--num_classes', default=2, type=int, help="num of class")

    args = parser.parse_args()
    detector = Detector(args)
    detector.detect()
