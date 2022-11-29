import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from util import compute_psnr, compute_ssim, visualize_lr_hr_sr
from config import DefaultConfig

config = DefaultConfig()


def train(train_loader, test_loader, model):
    device = config.device
    # optimizer = optim.Adam(model.parameters(), lr=config.lr)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    # step lr
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # cosine lr
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
    # use l1 loss
    criterion = nn.L1Loss()
    # use smooth l1 loss / charbonnier loss
    # criterion = nn.SmoothL1Loss(beta=config.epsilon)
    model.train()
    for epoch in range(config.epoch_num):
        # use tqdm to show the progress bar
        # for batch, (lr_images, hr_images) in enumerate(train_loader):
        with tqdm(train_loader, desc=f"Epoch: {epoch}", unit="batch") as tepoch:
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

                tepoch.set_postfix(loss=loss.item(), psnr=compute_psnr(sr_images, hr_images).item(),
                                   ssim=compute_ssim(sr_images, hr_images).item())

        # test model
        if (epoch + 1) % 32 == 0:
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
            visualize_lr_hr_sr(lr_image[0], hr_image[0], sr_image[0])

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
    # print(f"PSNR: {psnr_sum / len(test_loader):.4f}")
