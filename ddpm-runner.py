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
sigma = 0.05
noise_type = 'g'
for img_id in range(5):
    clean = io.imread(f"./samples/clean/{img_id}.jpg").astype(float) / 255
    noisy_img = io.imread(f"./samples/noisy/{img_id}-{sigma}-{noise_type}.jpg").astype(float) / 255
    x = torch.Tensor([noisy_img.transpose([2, 0, 1])]).to(DEVICE)
    curr_step = 15
    start_time = time.time()
    denoised = diffusion.denoise(1, x=x, curr_step=curr_step, n_steps=curr_step)[0, ...].cpu().detach().numpy().transpose([1,2,0])
    total_time += time.time()-start_time
    print(curr_step)
    print(calc_psnr_hvsm(denoised, clean))
    print(calc_ssim(denoised, clean))

    # psnr-hvs-m and ssim for noisy [image 0.jpg]
    # 0.05 - 33.1531862025744      0.864376505561547
    # 0.1  - 26.597715048852017    0.6506967503999341
    # 0.2  - 20.50806272172695     0.39812667053699524
    # 0.4  - 14.455494322395255    0.2073370738246524
    # 0.8  - 10.334581997747138    0.10917886070032523

    plt.imsave(f"{DDPM_DENOISED_DIR}/{img_id}-{sigma}-{noise_type}.png", np.clip(denoised, a_min=0., a_max=1.))
print(f"sigma-{sigma}, time is {total_time / 5}")

# sigma-0.05, time is 1.6719985008239746
# sigma-0.1, time is 3.3238061904907226
# sigma-0.2, time is 6.579318428039551
# sigma-0.4, time is 11.229995155334473






