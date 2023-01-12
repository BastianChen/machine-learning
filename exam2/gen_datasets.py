import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class Datasets(Dataset):
    def __init__(self, root_path, isTrain=True, channels=3):
        data_files = os.listdir(root_path)
        self.data_list = []
        self.label_list = []
        self.isTrain = isTrain
        self.channels = channels
        self.trans_train_c1 = transforms.Compose([
            # transforms.RandomVerticalFlip(0.5),  # 依据概率p对PIL图片进行垂直翻转 参数： p- 概率，默认值为0.5
            # transforms.RandomHorizontalFlip(0.5),  # 依据概率p对PIL图片进行水平翻转 参数： p- 概率，默认值为0.5
            # transforms.RandomAffine(30, translate=(0, 0.01), scale=(0.95, 1.05), shear=None, interpolation=False,
            #                         fill=0),
            transforms.Resize((210, 180)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.trans_test_c1 = transforms.Compose([
            transforms.Resize((210, 180)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.trans_train_c3 = transforms.Compose([
            # transforms.RandomVerticalFlip(0.5),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomAffine(30, translate=(0, 0.01), scale=(0.95, 1.05), shear=None, interpolation=False,
            #                         fill=0),
            transforms.Resize((210, 180)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.trans_test_c3 = transforms.Compose([
            transforms.Resize((210, 180)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        # 获取数据跟标签以及相应的文件路径
        for data_file in data_files:
            pic_path = os.path.join(root_path, data_file)
            for image_name in os.listdir(pic_path):
                self.data_list.append(os.path.join(pic_path, image_name))
                self.label_list.append(data_file)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path = self.data_list[index]
        image = Image.open(image_path)
        if self.channels == 3:
            data = self.trans_train_c3(image) if self.isTrain else self.trans_test_c3(image)
        else:
            image = image.convert('L')
            data = self.trans_train_c1(image) if self.isTrain else self.trans_test_c1(image)
        label = torch.zeros(2)
        label[int(self.label_list[index])] = 1
        if self.isTrain:
            return data, label
        else:
            return data, label, np.array(image)


if __name__ == '__main__':
    datasets = Datasets(r"xxx\exam2\data\train")
    datasets_len = datasets.__len__()
    print(datasets_len)
    # data, label = datasets[1006]
    # print(data.shape, label)
    sum_x, sum_y = 0, 0
    for item in datasets:
        sum_x += item[0].shape[1]
        sum_y += item[0].shape[2]

    print(f"mean_x:{sum_x / datasets_len},mean_y:{sum_y / datasets_len}")
