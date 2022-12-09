import os
import torch
from datetime import datetime


class DefaultConfig(object):
    # general
    seed = 44
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)

    # dir path
    current_path = os.path.dirname(os.path.abspath(__file__))
    train_val_lr_path = os.path.join(current_path, 'data', 'DIV2K_train_LR_bicubic', 'X2')
    train_val_hr_path = os.path.join(current_path, 'data', 'DIV2K_train_HR')
    test_lr_path = os.path.join(current_path, 'data', 'DIV2K_valid_LR_bicubic', 'X2')
    test_hr_path = os.path.join(current_path, 'data', 'DIV2K_valid_HR')
    data_path = os.path.join(current_path, 'data')

    # the path of the model
    model_dir = os.path.join(current_path, 'model')
    model_path = os.path.join(model_dir, f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pth')

    # data
    patch_size = 64

    epoch_num = 100
    lr = 2e-4
    batch_size = 16

    epsilon = 1e-8

