import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import compute_psnr, compute_ssim, visualize_lr_hr_sr
from config import DefaultConfig

config = DefaultConfig()


def train(train_loader, test_loader, model):
    device = config.device
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    # use l1 loss
    criterion = nn.L1Loss()
    model.train()
    for epoch in range(config.epoch_num):
        with tqdm(train_loader, desc=f"Epoch: {epoch + 1}", unit="batch") as tepoch:
            for batch, (lr_images, hr_images) in enumerate(tepoch):
                # process data
                lr_images, hr_images = lr_images.to(device), hr_images.to(device)
                # forward
                sr_images = model(lr_images)
                # backward
                loss = criterion(sr_images, hr_images)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        # test model
        if (epoch + 1) % 10 == 0:
            test(model, test_loader)
            model.train()
        scheduler.step()

    # save model
    torch.save(model.state_dict(), config.model_path)


def test(model, test_loader):
    device = config.device
    model.eval()
    psnr_sum = 0
    ssim_sum = 0
    with torch.no_grad():
        for batch, (lr_images, hr_images) in enumerate(test_loader):
            # process data
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            # forward
            sr_images = model(lr_images)
            psnr = compute_psnr(sr_images, hr_images)
            ssim = compute_ssim(sr_images, hr_images)
            psnr_sum += psnr
            ssim_sum += ssim

    print(f"PSNR: {psnr_sum / len(test_loader):.4f}, SSIM: {ssim_sum / len(test_loader):.4f}")
