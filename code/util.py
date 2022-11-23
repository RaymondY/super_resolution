import numpy as np
from datasets import load_dataset
from PIL import Image
import random
import torch
from torch.utils.data import Dataset, DataLoader

from config import DefaultConfig

config = DefaultConfig()


class Div2kDataset(Dataset):
    def __init__(self, image_pairs, lr_mu, lr_std, hr_mu, hr_std):
        self.data = image_pairs
        self.lr_mu = lr_mu
        self.lr_std = lr_std
        self.hr_mu = hr_mu
        self.hr_std = hr_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_pair = self.data[index]
        lr_image = np.array(Image.open(image_pair['lr']).convert("RGB"))
        hr_image = np.array(Image.open(image_pair['hr']).convert("RGB"))

        # normalize
        lr_image = (lr_image - self.lr_mu) / self.lr_std
        hr_image = (hr_image - self.hr_mu) / self.hr_std

        lr_image = lr_image / 255
        hr_image = hr_image / 255

        # to tensor
        lr_image = torch.from_numpy(lr_image)
        hr_image = torch.from_numpy(hr_image)

        return lr_image, hr_image


def get_mu_std(train_data):
    lr_mu = 0
    lr_std = 0
    hr_mu = 0
    hr_std = 0
    for i in range(len(train_data)):
        lr_image = np.array(Image.open(train_data[i]['lr']).convert("RGB"))
        hr_image = np.array(Image.open(train_data[i]['hr']).convert("RGB"))
        lr_mu += lr_image.mean()
        lr_std += lr_image.std()
        hr_mu += hr_image.mean()
        hr_std += hr_image.std()
    lr_mu /= len(train_data)
    lr_std /= len(train_data)
    hr_mu /= len(train_data)
    hr_std /= len(train_data)
    return lr_mu, lr_std, hr_mu, hr_std


def load_data():
    dataset = load_dataset('eugenesiow/Div2k', 'bicubic_x2')
    temp_data = dataset['train']
    # since the test data is unavailable, we consider the validation set as the new test set
    test_data = dataset['validation']

    # randomly select 250 samples from the training set as the new validation set
    random.seed(config.seed)
    random.shuffle(temp_data)
    val_data = temp_data[:250]
    train_data = temp_data[250:]

    # data has "lr" and "hr" keys
    # RGB images are 3 channels
    # normalize the images based on the mean and std of the training set
    lr_mu, lr_std, hr_mu, hr_std = get_mu_std(train_data)
    train_data = Div2kDataset(train_data, lr_mu, lr_std, hr_mu, hr_std)
    val_data = Div2kDataset(val_data, lr_mu, lr_std, hr_mu, hr_std)
    test_data = Div2kDataset(test_data, lr_mu, lr_std, hr_mu, hr_std)

    # *************************************
    # random_patch? scale? crop?
    # *************************************

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
