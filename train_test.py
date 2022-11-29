import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from util import compute_psnr, compute_ssim, visualize_lr_hr_sr
from config import DefaultConfig

config = DefaultConfig()


def train(train_loader, test_loader, model):
    device = config.device
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # use l1 loss
    criterion = nn.L1Loss()
    model.train()
    for epoch in range(config.epoch_num):
        loss_sum = 0
        # use tqdm to show the progress bar
        for batch, (lr_images, hr_images) in tqdm(enumerate(train_loader)):
            # process data
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            # forward
            sr_images = model(lr_images)

            loss = criterion(sr_images, hr_images)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            # print loss
            if (batch + 1) % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss.item():.4f}")
                loss_sum = 0

        # test model
        if (epoch + 1) % 10 == 0:
            test(model=model, test_loader=test_loader)
            # visualize the result
            # get the first image of the first batch of the test_loader
            lr_image, hr_image = test_loader.dataset[0]
            lr_image = lr_image.unsqueeze(0).to(device)
            sr_image = model(lr_image)
            # load the image to cpu
            lr_image = lr_image.cpu()
            sr_image = sr_image.cpu()
            hr_image = hr_image.cpu()
            visualize_lr_hr_sr(lr_image, hr_image, sr_image)

            model.train()

    # save model
    torch.save(model.state_dict(), config.model_path)


def test(model, test_loader):
    device = config.device
    model.eval()
    psnr_sum = 0
    ssim_sum = 0
    with torch.no_grad():
        for batch, (lr_images, hr_images) in tqdm(enumerate(test_loader)):
            # process data
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            # forward
            sr_images = model(lr_images)

            psnr = compute_psnr(sr_images, hr_images)
            ssim = compute_ssim(sr_images, hr_images)
            psnr_sum += psnr
            ssim_sum += ssim

    print(f"PSNR: {psnr_sum / len(test_loader):.4f}, SSIM: {ssim_sum / len(test_loader):.4f}")
