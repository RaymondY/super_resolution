import torch
import torch.nn as nn
from utils import load_data, load_train_val_data, visualize_lr_hr_sr
from RDSR import RDSR
from train_test import train, test
from config import DefaultConfig

config = DefaultConfig()


def main():
    # train_loader, test_loader = load_data()
    train_loader, val_loader = load_train_val_data()
    model = RDSR(block_num=8).to(config.device)
    train(train_loader, val_loader, model)
    # test(val_loader, model)


def read_model_and_test():
    model = RDSR(block_num=8).to(config.device)
    model.load_state_dict(torch.load('model/xxx.pth'))
    _, test_loader = load_data()
    test(test_loader, model)


if __name__ == '__main__':
    main()
    # read_model_and_test()

