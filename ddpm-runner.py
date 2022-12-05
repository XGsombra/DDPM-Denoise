import torch
from pytorch_diffusion import Diffusion
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from util import calc_psnr_hvsm, calc_ssim
from constants import DEVICE, DDPM_DENOISED_DIR
import time

diffusion = Diffusion.from_pretrained("lsun_church")
total_time = 0
noise_type = 'g'
levels = [0.05, 0.1, 0.2, 0.4]
level_to_curr_step = {0.05: 15, 0.1: 33, 0.2: 69, 0.4: 120}
for level in levels:
    psnr = 0
    ssim = 0
    noisy_psnr = 0
    noisy_ssim = 0
    for img_id in range(14, 15):
        clean = io.imread(f"./samples/clean/ood/{img_id}.jpg").astype(float) / 255
        noisy_img = io.imread(f"./samples/noisy/ood/{img_id}-{level}-{noise_type}.jpg").astype(float) / 255
        noisy_psnr += calc_psnr_hvsm(noisy_img, clean)
        noisy_ssim += calc_ssim(noisy_img, clean)
        # actual_noise = noisy_img - clean
        # print(f"std is {np.std(actual_noise)}")
        # print(f"mean is {np.mean(actual_noise)}")
        x = torch.Tensor([noisy_img.transpose([2, 0, 1])]).to(DEVICE)
        curr_step = level_to_curr_step[level]
        start_time = time.time()
        denoised = diffusion.denoise(1, x=x, curr_step=curr_step, n_steps=curr_step)[0, ...].cpu().detach().numpy().transpose([1,2,0])
        total_time += time.time()-start_time
        psnr += calc_psnr_hvsm(denoised, clean)
        ssim += calc_ssim(denoised, clean)
        plt.imsave(f"{DDPM_DENOISED_DIR}/ood/{img_id}-{level}-{noise_type}.png", np.clip(denoised, a_min=0., a_max=1.))
    print(level)
    print("noisy", noisy_psnr)
    print("noisy", noisy_ssim)
    print("denoised", psnr)
    print("denoised", ssim)
    print(f"sigma-{level}, time is {total_time / 5}")

# sigma-0.05, time is 1.6719985008239746
# sigma-0.1, time is 3.3238061904907226
# sigma-0.2, time is 6.579318428039551
# sigma-0.4, time is 11.229995155334473






