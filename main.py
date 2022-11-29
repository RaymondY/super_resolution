import torch
import torch.nn as nn
from util import load_data, load_train_val_data
from resSR import Generator
from train_test import train, test
from config import DefaultConfig

config = DefaultConfig()


def main():
    # train_loader, test_loader = load_data()
    train_loader, val_loader = load_train_val_data()
    model = Generator(block_nums=3).to(config.device)
    train(train_loader, val_loader, model)
    # test(val_loader, model)


if __name__ == '__main__':
    main()
