import os

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torch
from glob2 import glob
from natsort import natsorted
from pytorch_diffusion import Diffusion

from constants import CLEAN_DIR, SIZE, CHANNEL_NUM, LEVELS, NOISY_DIR, GAUSSIAN_NOISE, DEVICE, MODEL
from util import calc_psnr_hvsm, calc_ssim

image_num = 5
diffusion = Diffusion.from_pretrained(MODEL)

# Image for the introduction
# clean = io.imread(f"./samples/clean/0.jpg").astype(float) / 255
# noisy = io.imread(f"./samples/noisy/0-0.2-g.jpg").astype(float) / 255
# ddpm_denoised = io.imread(
#     f"./samples/ddpm-denoised/0-0.2-g.png").astype(float) / 255
# mprnet_denoised = io.imread(
#     f"./samples/mprnet-denoised/0-0.2-g.png").astype(float) / 255
#
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
# axs[0, 0].imshow(clean)
# axs[0, 0].set_title("Ground Truth")
# axs[0, 0].set_axis_off()
# axs[0, 1].imshow(noisy)
# axs[0, 1].set_title("Noisy Image with sigma=0.2")
# axs[0, 1].set_axis_off()
# axs[1, 0].imshow(ddpm_denoised)
# axs[1, 0].set_title("Denoised by DDPM")
# axs[1, 0].set_axis_off()
# axs[1, 1].imshow(mprnet_denoised)
# axs[1, 1].set_title("Denoised by MPRNet")
# axs[1, 1].set_axis_off()
# plt.savefig("introduction.png", bbox_inches='tight')


# Image for PSNR and SSIM on each step
clean_image_paths = natsorted(glob(os.path.join(CLEAN_DIR, '*.jpg')))
image_num = len(clean_image_paths)
clean_images = np.zeros((image_num, SIZE, SIZE, CHANNEL_NUM))

for i in range(image_num):
    clean_images[i] = io.imread(clean_image_paths[i]).astype(float) / 255

levels = [0.05, 0.1, 0.2, 0.4]
level_to_curr_step = {0.05: 15, 0.1: 33, 0.2: 69, 0.4: 120}
psnr_curves = {}
ssim_curves = {}

for level in levels:
    psnr_curves[level] = []
    ssim_curves[level] = []
    x = np.zeros_like(clean_images)
    for i in range(image_num):
        x[i] = io.imread(
            os.path.join(NOISY_DIR, f"{i}-{round(level, 2)}-{'g' if GAUSSIAN_NOISE else 'p'}.jpg")
        ).astype(float) / 255
    x = torch.Tensor(x.transpose([0, 3, 1, 2])).to(DEVICE)
    curr_step = level_to_curr_step[level]
    while curr_step > 0:
        x = diffusion.denoise(
            image_num,
            x=x,
            curr_step=curr_step,
            n_steps=1
        ).cpu().numpy().transpose([0, 2, 3, 1])
        # Calculate and update the maximal PSNR
        psnr = 0
        ssim = 0
        for i in range(image_num):
            psnr += calc_psnr_hvsm(x[i], clean_images[i])
            ssim += calc_ssim(x[i], clean_images[i])
        psnr_curves[level].append(psnr / image_num)
        ssim_curves[level].append(ssim / image_num)
        curr_step -= 1
        x = torch.Tensor(x.transpose([0, 3, 1, 2])).to(DEVICE)
print(psnr_curves)
print(ssim_curves)


fig, ax = plt.subplots()
for level in levels:
    ax.plot(range(level_to_curr_step[level], 0, -1), ssim_curves[level], label=f"sigma={level}")
    # ax.plot(range(level_to_curr_step[level], 0, -1), psnr_curves[level], label=f"sigma={level}")
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Denoising Step Number")
ax.set_ylabel("SSIM")
ax.grid(True)
plt.show()