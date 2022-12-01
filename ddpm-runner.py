import torch
from pytorch_diffusion import Diffusion
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from util import calc_psnr_hvsm, calc_ssim
from constants import DEVICE, DDPM_DENOISED_DIR

diffusion = Diffusion.from_pretrained("lsun_church")
# for img_id in range(5):
for img_id in [0]:
    sigma = 0.05
    noise_type = 'g'
    clean = io.imread(f"./samples/clean/{img_id}.jpg").astype(float) / 255
    noisy_img = io.imread(f"./samples/noisy/{img_id}-{sigma}-{noise_type}.jpg").astype(float) / 255

    x = torch.Tensor([noisy_img.transpose([2, 0, 1])]).to(DEVICE)
    curr_step = 12
    denoised = diffusion.denoise(1, x=x, curr_step=curr_step, n_steps=1000)[0, ...].cpu().detach().numpy().transpose([1,2,0])
    print(curr_step)
    print(calc_psnr_hvsm(denoised, clean))
    print(calc_ssim(denoised, clean))

    # psnr-hvs-m and ssim for noisy [image 0.jpg]
    # 0.05 - 33.1531862025744      0.864376505561547
    # 0.1  - 26.597715048852017    0.6506967503999341
    # 0.2  - 20.50806272172695     0.39812667053699524
    # 0.4  - 14.455494322395255    0.2073370738246524
    # 0.8  - 10.334581997747138    0.10917886070032523

    # plt.imsave(f"{DDPM_DENOISED_DIR}/{img_id}-{sigma}-{noise_type}.png", np.clip(denoised, a_min=0., a_max=1.))

# img_id = 0
# sigma = 0.1
# noise_type = 'g'
# clean = io.imread(f"./samples/clean/{img_id}.jpg").astype(float) / 255
# noisy_img = io.imread(f"./samples/noisy/{img_id}-{sigma}-{noise_type}.jpg").astype(float) / 255
# x = torch.Tensor([noisy_img.transpose([2, 0, 1])]).to(DEVICE)
# curr_step = 35
# total_step_count = 0
#
# while curr_step > 0:
#     total_step_count += 1
#     x = diffusion.denoise(1, x=x, curr_step=curr_step, n_steps=1)[0, ...].cpu().detach().numpy().transpose([1,2,0])
#     print(curr_step)
#     print(calc_psnr_hvsm(x, clean))
#     print(calc_ssim(x, clean))
#     # plt.imshow(np.clip(x, a_min=0., a_max=1.))
#     # plt.show()
#     curr_step -= 1 # max(1, int(np.sqrt(curr_step)) // 2)
#     # curr_step = int(curr_step / 1.1)
#     x = torch.Tensor([x.transpose([2, 0, 1])])
# print(f"total step is {total_step_count}")
# plt.imshow(np.clip(x, a_min=0., a_max=1.)[0, ...].cpu().detach().numpy().transpose([1,2,0]))
# plt.show()
#
# #175 11.98347038387504
# 0.5649341





