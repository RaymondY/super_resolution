import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from config import DefaultConfig

config = DefaultConfig()


def calculate_mean_std(train_lr_list, train_hr_list):
    lr_path = config.train_val_lr_path
    hr_path = config.train_val_hr_path
    train_lr_mean = torch.zeros(3)
    train_hr_mean = torch.zeros(3)
    train_lr_std = torch.zeros(3)
    train_hr_std = torch.zeros(3)
    for i in range(len(train_lr_list)):
        lr_img = Image.open(os.path.join(lr_path, train_lr_list[i]))
        hr_img = Image.open(os.path.join(hr_path, train_hr_list[i]))
        lr_img = transforms.ToTensor()(lr_img)
        hr_img = transforms.ToTensor()(hr_img)
        train_lr_mean += lr_img.mean(dim=(1, 2))
        train_hr_mean += hr_img.mean(dim=(1, 2))
        train_lr_std += lr_img.std(dim=(1, 2))
        train_hr_std += hr_img.std(dim=(1, 2))
    train_lr_mean /= len(train_lr_list)
    train_hr_mean /= len(train_hr_list)
    train_lr_std /= len(train_lr_list)
    train_hr_std /= len(train_hr_list)

    return train_lr_mean, train_hr_mean, train_lr_std, train_hr_std


# calculate the mean and std of the train_val dataset and all_train dataset
# train_val dataset: the first 550 images are used for training
# all_train dataset: all the images are used for training
def main():
    lr_path = config.train_val_lr_path
    hr_path = config.train_val_hr_path
    all_train_lr_list = os.listdir(lr_path)
    all_train_hr_list = os.listdir(hr_path)
    # remove non-png files
    all_train_lr_list = [x for x in all_train_lr_list if x.endswith('.png')]
    all_train_hr_list = [x for x in all_train_hr_list if x.endswith('.png')]
    # sort the list by the name of the file
    all_train_lr_list.sort()
    all_train_hr_list.sort()

    # train_val dataset
    train_val_lr_list = all_train_lr_list[:550]
    train_val_hr_list = all_train_hr_list[:550]

    # all_train dataset
    all_train_lr_list = all_train_lr_list
    all_train_hr_list = all_train_hr_list

    # calculate the mean and std of the train_val dataset
    train_val_lr_mean, train_val_hr_mean, \
    train_val_lr_std, train_val_hr_std = \
        calculate_mean_std(train_val_lr_list, train_val_hr_list)
    print('train_val_lr_mean: ', train_val_lr_mean)
    print('train_val_hr_mean: ', train_val_hr_mean)
    print('train_val_lr_std: ', train_val_lr_std)
    print('train_val_hr_std: ', train_val_hr_std)

    # calculate the mean and std of the all_train dataset
    all_train_lr_mean, all_train_hr_mean, \
    all_train_lr_std, all_train_hr_std = \
        calculate_mean_std(all_train_lr_list, all_train_hr_list)
    print('all_train_lr_mean: ', all_train_lr_mean)
    print('all_train_hr_mean: ', all_train_hr_mean)
    print('all_train_lr_std: ', all_train_lr_std)
    print('all_train_hr_std: ', all_train_hr_std)


if __name__ == '__main__':
    main()
    # train_val_lr_mean:  tensor([0.4479, 0.4354, 0.4026])
    # train_val_hr_mean:  tensor([0.4479, 0.4354, 0.4026])
    # train_val_lr_std:  tensor([0.2415, 0.2318, 0.2424])
    # train_val_hr_std:  tensor([0.2455, 0.2359, 0.2459])
    # all_train_lr_mean:  tensor([0.4485, 0.4375, 0.4046])
    # all_train_hr_mean:  tensor([0.4485, 0.4375, 0.4045])
    # all_train_lr_std:  tensor([0.2397, 0.2290, 0.2389])
    # all_train_hr_std:  tensor([0.2436, 0.2330, 0.2424])
