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
    def __init__(self, lr_path, hr_path, val_status=0, scale=2, is_train=True, need_crop=True):
        super().__init__()
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.lr_list = os.listdir(self.lr_path)
        self.hr_list = os.listdir(self.hr_path)
        # remove non-png files
        self.lr_list = [x for x in self.lr_list if x.endswith('.png')]
        self.hr_list = [x for x in self.hr_list if x.endswith('.png')]
        # sort the list by the name of the file
        self.lr_list.sort()
        self.hr_list.sort()

        self.scale = scale
        self.val_status = val_status
        # val_status: 0 w.o val, 1 w. val and return train data, 2 for val and return val data
        if self.val_status == 1:
            self.lr_list = self.lr_list[:550]
            self.hr_list = self.hr_list[:550]
        elif self.val_status == 2:
            self.lr_list = self.lr_list[550:]
            self.hr_list = self.hr_list[550:]

        self.is_train = is_train
        # for testing, in case we need to crop the image
        self.need_crop = need_crop

    def __len__(self):
        return len(self.lr_list)

    # keep the center of the lr image and hr image as the same
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
        # [0, 1]
        lr = transforms.ToTensor()(lr)
        hr = transforms.ToTensor()(hr)

        if self.val_status == 0:
            # train_val_lr_mean:  tensor([0.4479, 0.4354, 0.4026])
            # train_val_hr_mean:  tensor([0.4479, 0.4354, 0.4026])
            # train_val_lr_std:  tensor([0.2415, 0.2318, 0.2424])
            # train_val_hr_std:  tensor([0.2455, 0.2359, 0.2459])
            lr = transforms.Normalize(mean=[0.4479, 0.4354, 0.4026],
                                      std=[0.2415, 0.2318, 0.2424])(lr)
            hr = transforms.Normalize(mean=[0.4479, 0.4354, 0.4026],
                                      std=[0.2455, 0.2359, 0.2459])(hr)

        else:
            # all_train_lr_mean: tensor([0.4485, 0.4375, 0.4046])
            # all_train_hr_mean: tensor([0.4485, 0.4375, 0.4045])
            # all_train_lr_std: tensor([0.2397, 0.2290, 0.2389])
            # all_train_hr_std: tensor([0.2436, 0.2330, 0.2424])
            lr = transforms.Normalize(mean=[0.4485, 0.4375, 0.4046],
                                      std=[0.2397, 0.2290, 0.2389])(lr)
            hr = transforms.Normalize(mean=[0.4485, 0.4375, 0.4045],
                                      std=[0.2436, 0.2330, 0.2424])(hr)

        if self.is_train or self.need_crop:
            lr, hr = self.get_random_patch(lr, hr)

        return lr, hr


def load_data(need_crop=True):
    train_dataset = Div2kDataset(config.train_val_lr_path, config.train_val_hr_path, is_train=True)
    test_dataset = Div2kDataset(config.test_lr_path, config.test_hr_path,
                                is_train=False, need_crop=need_crop)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, pin_memory=True)

    return train_loader, test_loader


def load_train_val_data(need_crop=True):
    train_dataset = Div2kDataset(config.train_val_lr_path, config.train_val_hr_path,
                                 val_status=1, is_train=True)
    val_dataset = Div2kDataset(config.train_val_lr_path, config.train_val_hr_path,
                               val_status=2, is_train=False, need_crop=need_crop)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, pin_memory=True)

    return train_loader, val_loader


def denormalize_lr(image, is_val=False, on_gpu=True):
    device = config.device if on_gpu else torch.device('cpu')
    if is_val:
        lr_mean = torch.tensor([0.4479, 0.4354, 0.4026]).view(3, 1, 1).to(device)
        lr_std = torch.tensor([0.2415, 0.2318, 0.2424]).view(3, 1, 1).to(device)
    else:
        lr_mean = torch.tensor([0.4485, 0.4375, 0.4046]).view(3, 1, 1).to(device)
        lr_std = torch.tensor([0.2397, 0.2290, 0.2389]).view(3, 1, 1).to(device)
    image = image * lr_std + lr_mean
    return image


def denormalize_hr(image, is_val=False, on_gpu=True):
    device = config.device if on_gpu else torch.device('cpu')
    if is_val:
        hr_mean = torch.tensor([0.4479, 0.4354, 0.4026]).view(3, 1, 1).to(device)
        hr_std = torch.tensor([0.2455, 0.2359, 0.2459]).view(3, 1, 1).to(device)
    else:
        hr_mean = torch.tensor([0.4485, 0.4375, 0.4045]).view(3, 1, 1).to(device)
        hr_std = torch.tensor([0.2436, 0.2330, 0.2424]).view(3, 1, 1).to(device)
    image = image * hr_std + hr_mean
    return image


def visualize_lr_hr_sr(lr, hr, sr, is_val=False, download=False, index=0, on_gpu=True):
    lr = denormalize_lr(lr, is_val, on_gpu)
    hr = denormalize_hr(hr, is_val, on_gpu)
    sr = denormalize_hr(sr, is_val, on_gpu)

    transform = transforms.ToPILImage(mode='RGB')
    lr = transform(lr)
    hr = transform(hr)
    sr = transform(sr)

    # show the image and set the title
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lr)
    axes[0].axis('off')
    axes[0].set_title("LR")
    axes[1].imshow(hr)
    axes[1].axis('off')
    axes[1].set_title("HR")
    axes[2].imshow(sr)
    axes[2].axis('off')
    axes[2].set_title("SR")
    if download:
        fig.savefig(f"result_{index}.pdf", bbox_inches='tight')
    plt.show()


def compute_psnr(image1, image2, is_val=False, on_gpu=True):
    image1 = denormalize_hr(image1, is_val, on_gpu)
    image2 = denormalize_hr(image2, is_val, on_gpu)
    return kornia_psnr(image1, image2, max_val=1.)


def compute_ssim(image1, image2, is_val=False, on_gpu=True):
    image1 = denormalize_hr(image1, is_val, on_gpu)
    image2 = denormalize_hr(image2, is_val, on_gpu)
    return kornia_ssim(image1, image2, window_size=11, max_val=1.).mean()


# load a particular image
def load_image_visualize(model, index):
    test_dataset = Div2kDataset(config.test_lr_path, config.test_hr_path,
                                is_train=False, need_crop=True)
    lr, hr = test_dataset[index]
    model.to('cpu')
    sr = model(lr.unsqueeze(0))
    sr = sr.squeeze(0)
    visualize_lr_hr_sr(lr, hr, sr, is_val=True, download=True, index=index, on_gpu=False)
    lr, hr, sr = lr.unsqueeze(0), hr.unsqueeze(0), sr.unsqueeze(0)
    psnr = compute_psnr(hr, sr, is_val=False, on_gpu=False)
    ssim = compute_ssim(hr, sr, is_val=False, on_gpu=False)
    print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")



