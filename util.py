import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from kornia.metrics import psnr as kornia_psnr
from kornia.metrics import ssim as kornia_ssim
from config import DefaultConfig

config = DefaultConfig()


class Div2kDataset(Dataset):
    def __init__(self, lr_path, hr_path, mean=None, std=None):
        super().__init__()
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.lr_list = os.listdir(self.lr_path)
        self.hr_list = os.listdir(self.hr_path)
        if mean:
            self.mean = mean
            self.std = std

    def __len__(self):
        return len(self.lr_list)

    def __getitem__(self, index):
        lr = Image.open(os.path.join(self.lr_path, self.lr_list[index]))
        hr = Image.open(os.path.join(self.hr_path, self.hr_list[index]))
        # (C, H, W)
        lr = transforms.ToTensor()(lr)
        hr = transforms.ToTensor()(hr)

        if self.mean:
            lr = transforms.Normalize(self.mean, self.std)(lr)
            hr = transforms.Normalize(self.mean, self.std)(hr)
        else:
            # normalize to [-1, 1]
            lr = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(lr)
            hr = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(hr)

        return lr, hr


def get_mean_std(train_dataset):
    lr_list = []
    hr_list = []
    for i in range(len(train_dataset)):
        lr_list.append(train_dataset[i][0])
        hr_list.append(train_dataset[i][1])
    lr_torch = torch.stack(lr_list, dim=0)
    hr_torch = torch.stack(hr_list, dim=0)
    lr_mean = lr_torch.mean(dim=(0, 2, 3))
    lr_std = lr_torch.std(dim=(0, 2, 3))
    hr_mean = hr_torch.mean(dim=(0, 2, 3))
    hr_std = hr_torch.std(dim=(0, 2, 3))
    return lr_mean, hr_mean, lr_std, hr_std


def load_data():
    train_dataset = Div2kDataset(config.train_val_lr_path, config.train_val_hr_path)
    test_dataset = Div2kDataset(config.test_lr_path, config.test_hr_path)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    return train_loader, test_loader


def load_train_val_data():
    all_train_dataset = Div2kDataset(config.train_val_lr_path, config.train_val_hr_path)
    train_dataset, val_dataset = torch.utils.data.random_split(all_train_dataset, [550, 250])

    # lr_mean, hr_mean, lr_std, hr_std = get_mean_std(train_dataset)
    # temp_dataset = Div2kDataset(config.train_val_lr_path, config.train_val_hr_path,
    #                             lr_mean, lr_std)
    # train_dataset, val_dataset = torch.utils.data.random_split(temp_dataset, [550, 250])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    return train_loader, val_loader


def visualize_lr_hr(lr, hr):
    lr = transforms.ToPILImage()(lr)
    hr = transforms.ToPILImage()(hr)
    lr.show()
    hr.show()


def compute_psnr(img1, img2):
    return kornia_psnr(img1, img2, max_val=1.)


def compute_ssim(img1, img2):
    return kornia_ssim(img1, img2, window_size=11, max_val=1.)

