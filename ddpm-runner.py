import cv2
import torch
from pytorch_diffusion import Diffusion
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from util import calc_psnr_hvsm, calc_ssim, calc_psnr
from constants import DEVICE, DDPM_DENOISED_DIR
import time

diffusion = Diffusion.from_pretrained("lsun_church")
total_time = 0
noise_type = 'g'
levels = [0.05, 0.1, 0.2, 0.4]
level_to_curr_step = {0.05: 15, 0.1: 33, 0.2: 69, 0.4: 120}

# levels = [0.4]
# level_to_curr_step = {0.4: 300}
# for level in levels:
#     psnr = 0
#     phm = 0
#     ssim = 0
#     noisy_psnr = 0
#     noisy_phm = 0
#     noisy_ssim = 0
#     for img_id in range(8, 11):
#         clean = io.imread(f"./samples/clean/validation/{img_id}.jpg").astype(float) / 255
#         noisy_img = io.imread(f"./samples/noisy/validation/{img_id}-{level}-{noise_type}.jpg").astype(float) / 255
#         noisy_psnr += calc_psnr(noisy_img, clean)
#         noisy_phm += calc_psnr_hvsm(noisy_img, clean)
#         noisy_ssim += calc_ssim(noisy_img, clean)
#         x = torch.Tensor([noisy_img.transpose([2, 0, 1])]).to(DEVICE)
#         curr_step = level_to_curr_step[level]
#         # denoised = diffusion.denoise(1, x=x, curr_step=curr_step, n_steps=curr_step)[0, ...].cpu().detach().numpy().transpose([1,2,0])
#         denoised = io.imread(f"./samples/bilateral-denoised/validation/{img_id}-{level}-{noise_type}.jpg").astype(float) / 255
#         # plt.imshow(denoised)
#         # plt.show()
#         # print(denoised.shape, clean.shape)
#         # # total_time += time.time()-start_time
#         psnr += calc_psnr(denoised, clean)
#         phm += calc_psnr_hvsm(denoised, clean)
#         ssim += calc_ssim(denoised, clean)
#         # plt.imsave(f"{DDPM_DENOISED_DIR}/{img_id}-{level}-{noise_type}.jpg", np.clip(denoised, a_min=0., a_max=1.))
#     print(level)
#     print("noisy", noisy_psnr / 3, noisy_phm / 3, noisy_ssim / 3)
#     print("denoised", psnr / 3, phm / 3, ssim / 3)
#     print(f"sigma-{level}, time is {total_time / 5}")
