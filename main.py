import torch
from utils import load_data, load_image_visualize
from model import ResModel
from train_test import train, test
from config import DefaultConfig

config = DefaultConfig()


def main():
    train_loader, test_loader = load_data()
    # train_loader, test_loader = load_train_val_data()
    model = ResModel(block_num=30).to(config.device)
    train(train_loader, test_loader, model)
    # test(test_loader, model)


def read_model_and_test(block_num=30):
    model = ResModel(block_num=block_num).to(config.device)
    # load block_num.pth
    print("Loading model...")
    model.load_state_dict(torch.load(f"model/{block_num}.pth"))
    # print("Testing...")
    # _, test_loader = load_data()
    # test(model, test_loader)
    print("Visualizing...")
    # load index=0 image and visualize
    load_image_visualize(model, 29)


if __name__ == '__main__':
    main()
    # read_model_and_test()
