import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from kornia.metrics import psnr as kornia_psnr
from kornia.metrics import ssim as kornia_ssim
from config import DefaultConfig

config = DefaultConfig()


class Div2kDataset(Dataset):
    def __init__(self, lr_path, hr_path, mean=None, std=None, val_status=0, scale=2):
        super().__init__()
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.lr_list = os.listdir(self.lr_path)
        self.hr_list = os.listdir(self.hr_path)
        # sort the list by the name of the file
        self.lr_list.sort()
        self.hr_list.sort()
        self.scale = scale
        # val_status: 0 w.o val, 1 w. val and return train data, 2 for val and return val data
        if val_status == 1:
            self.lr_list = self.lr_list[:550]
            self.hr_list = self.hr_list[:550]
        elif val_status == 2:
            self.lr_list = self.lr_list[550:]
            self.hr_list = self.hr_list[550:]
        if mean:
            self.mean = mean
            self.std = std

    def __len__(self):
        return len(self.lr_list)

    # # keep the center of the lr image and hr image as the same
    def get_random_patch(self, lr, hr):
        # lr and hr are both (C, H, W) tensors
        lr_height, lr_width = lr.shape[1:]
        lr_patch_size = config.patch_size
        hr_patch_size = self.scale * lr_patch_size
        # [0, lr_height - config.patch_size]
        lr_y = random.randint(0, lr_height - lr_patch_size)
        # [0, lr_width - config.patch_size]
        lr_x = random.randint(0, lr_width - lr_patch_size)
        hr_y, hr_x = lr_y * self.scale, lr_x * self.scale

        lr_patch = lr[:, lr_y:lr_y + lr_patch_size, lr_x:lr_x + lr_patch_size]
        hr_patch = hr[:, hr_y:hr_y + hr_patch_size, hr_x:hr_x + hr_patch_size]

        return lr_patch, hr_patch

    def __getitem__(self, index):
        lr = Image.open(os.path.join(self.lr_path, self.lr_list[index]))
        hr = Image.open(os.path.join(self.hr_path, self.hr_list[index]))
        # (C, H, W)
        lr = transforms.ToTensor()(lr)
        hr = transforms.ToTensor()(hr)

        # if there is property self.mean, then normalize the image
        if hasattr(self, "mean"):
            lr = transforms.Normalize(self.mean, self.std)(lr)
            hr = transforms.Normalize(self.mean, self.std)(hr)
        else:
            # normalize to [-1, 1]
            lr = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(lr)
            hr = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(hr)

        lr, hr = self.get_random_patch(lr, hr)

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

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, pin_memory=True)

    return train_loader, test_loader


def load_train_val_data():
    train_dataset = Div2kDataset(config.train_val_lr_path, config.train_val_hr_path, val_status=1)
    val_dataset = Div2kDataset(config.train_val_lr_path, config.train_val_hr_path, val_status=2)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)

    return train_loader, val_loader


def visualize_lr_hr_sr(lr, hr, sr):
    # lr, hr, sr are all (C, H, W) normalized tensors
    # (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    lr = lr * 0.5 + 0.5
    hr = hr * 0.5 + 0.5
    sr = sr * 0.5 + 0.5

    transform = transforms.ToPILImage()
    lr = transform(lr)
    hr = transform(hr)
    sr = transform(sr)

    # show the image and set the title
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lr)
    axes[0].set_title("LR")
    axes[1].imshow(hr)
    axes[1].set_title("HR")
    axes[2].imshow(sr)
    axes[2].set_title("SR")
    plt.show()


def compute_psnr(img1, img2):
    return kornia_psnr(img1, img2, max_val=1.)


def compute_ssim(img1, img2):
    ssim_map = kornia_ssim(img1, img2, window_size=11, max_val=1.)
    return ssim_map.mean()
