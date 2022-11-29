import torch
from pytorch_diffusion import Diffusion
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from util import calc_psnr_hvs_hvsm

img_id = 0
sigma = 0.05
noise_type = 'g'
clean = io.imread(f"./samples/clean/{img_id}.jpg").astype(float) / 255
noisy_img = io.imread(f"./samples/noisy/{img_id}-{sigma}-{noise_type}.jpg").astype(float) / 255

x = torch.Tensor([noisy_img.transpose([2, 0, 1])]).to("cuda:0")
diffusion = Diffusion.from_pretrained("lsun_church")
x = diffusion.denoise(1, x=x, curr_step=10, n_steps=11)[0, ...].cpu().detach().numpy().transpose([1,2,0])
print(calc_psnr_hvs_hvsm(clean, x))

# plt.imshow(noisy_img)
# plt.show()

plt.imshow(x)
plt.show()