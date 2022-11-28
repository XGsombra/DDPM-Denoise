import torch
from pytorch_diffusion import Diffusion
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

def calc_psnr(x, gt):
    out = 10 * np.log10(1 / ((x - gt)**2).mean().item())
    return out

img = io.imread('samples/church.jpg').astype(float)/255
height, width, _ = img.shape

noisy_img = np.zeros(np.shape(img))

sigma_noise = 0.5
for it in range(3):
    noisy_img[:, :, it] = sigma_noise * np.random.randn(img.shape[0], img.shape[1]) + img[:, :, it]
print(calc_psnr(img, noisy_img))
plt.imsave("noisy_church.jpg", np.clip(noisy_img, a_min=0.,a_max=1.))

use_ddpm = False

x = torch.Tensor([noisy_img.transpose([2, 0, 1])]).to("cuda:0")
diffusion = Diffusion.from_pretrained("lsun_church")
x = diffusion.denoise(1, x=x, curr_step=100, n_steps=101)


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
axs[0].imshow(img)
axs[0].set_axis_off()
axs[0].set_title("Ground Truth")
axs[1].imshow(noisy_img)
axs[1].set_axis_off()
axs[1].set_title("Gaussian Noise with sigma=0.5")
axs[2].imshow(x[0, ...].cpu().detach().numpy().transpose([1,2,0]))
axs[2].set_axis_off()
axs[2].set_title("Denoised by DDPM")
fig.tight_layout()
plt.savefig("ddpm_church.png", bbox_inches='tight')