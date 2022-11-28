# from scipy.signal import convolve2d
# # import packages
# import numpy as np
# from numpy.fft import fft2, ifft2
#
# import skimage.io as io
# from skimage.filters import gaussian
#
# from pypher.pypher import psf2otf
# import matplotlib.pyplot as plt
# import cv2 as cv
#
# from PIL import Image
#
# # helper function for computing a 2D Gaussian convolution kernel
import torch
#
# def pca(data,k):
#     n_samples,n_features = data.shape
#     mean = np.array([np.mean(data[:,i]) for i in range(n_features)])
#     normal_data = data - mean
#     matrix_ = np.dot(np.transpose(normal_data),normal_data)
#
#     eig_val,eig_vec = np.linalg.eig(matrix_)
#
#     eigIndex = np.argsort(eig_val)
#     eigVecIndex = eigIndex[:-(k+1):-1]
#     feature = eig_vec[:,eigVecIndex]
#     new_data = np.dot(normal_data,feature)
#     rec_data = np.dot(new_data,np.transpose(feature))+ mean
#     return rec_data
#
#
# # select target image and load it
# name = 'bear'
# img = io.imread(f'{name}.jpg').astype(float)/255
#
# noisy_img = np.zeros(np.shape(img))
# blur = np.zeros(np.shape(img))
# denoised = np.zeros(np.shape(img))
#
# sigma_noise = 0.2
# for it in range(3):
#     noisy_img[:, :, it] = img[:, :, it] + sigma_noise * np.random.randn(img.shape[0], img.shape[1])
#     # noisy_img[:, :, it] = pca(noisy_img[:, :, it], 50)
# print(calc_psnr(img, noisy_img))
# plt.imshow(np.clip(noisy_img, a_min=0.,a_max=1.))
# plt.show()
#
#
# sigma_blur = 9
# decay = 0.7
# sigma_blur_filtSize = 20
# gaussian_weight = 0.1
#
# for _ in range(100):
#     for i in range(3):
#         lp_blur = fspecial_gaussian_2d((sigma_blur_filtSize, sigma_blur_filtSize), sigma_blur)
#         otf = psf2otf(lp_blur, img[..., i].shape)
#
#         fft_img = fft2(noisy_img[..., i])
#         blur[..., i] = ifft2(fft_img * otf).real
#
#         denoised[..., i] = blur[..., i] * gaussian_weight + noisy_img[..., i] * (1-gaussian_weight)
#         noisy_img[..., i] = denoised[..., i]
#         sigma_blur *= decay
#         sigma_blur_filtSize = max(3, int(sigma_blur_filtSize * decay))
#         gaussian_weight = 1 - (1 - gaussian_weight) * decay
#
# # sigma_blurs = np.arange(1, 10, 2)
# # decays = np.arange(1, 10, 2) * 0.1
# # sigma_blur_filtSizes = np.arange(1, 10, 2) * 10
# # gaussian_weights = np.arange(1, 10, 2) * 0.1
# # max_its = [2, 5, 10, 50, 100]
# #
# #
# # max_psnr = calc_psnr(img, noisy_img)
# # for sigma_blur in sigma_blurs:
# #     for decay in decays:
# #         for sigma_blur_filtSize in sigma_blur_filtSizes:
# #             for gaussian_weight in gaussian_weights:
# #                 for max_it in max_its:
# #                     sigma = sigma_blur
# #                     size = sigma_blur_filtSize
# #                     weight = gaussian_weight
# #                     print(sigma_blur, decay, sigma_blur_filtSize, gaussian_weight, max_it)
# #                     for it in range(3):
# #                         noisy_img[:, :, it] = img[:, :, it] + sigma_noise * np.random.randn(img.shape[0], img.shape[1])
# #                     for _ in range(max_it):
# #                         for i in range(3):
# #                             lp_blur = fspecial_gaussian_2d((size, size), sigma)
# #                             otf = psf2otf(lp_blur, img[..., i].shape)
# #
# #                             fft_img = fft2(noisy_img[..., i])
# #                             blur[..., i] = ifft2(fft_img * otf).real
# #
# #                             denoised[..., i] = blur[..., i] * weight + noisy_img[..., i] * (1-weight)
# #                             noisy_img[..., i] = denoised[..., i]
# #                             sigma *= decay
# #                             size = max(3, int(size * decay))
# #                             weight = 1 - (1 - weight) * decay
# #                     psnr = calc_psnr(img, denoised)
# #                     if psnr > max_psnr:
# #                         max_psnr = psnr
# #                         print(f"max psnr={max_psnr} when sigma_blur={sigma_blur} decay={decay} sigma_blur_filtsize={sigma_blur_filtSize} gaussian_weight={gaussian_weight} max_it={max_it}")
#
#
# # plt.imshow(blur)
# # # plt.show()
#
# plt.imshow(denoised)
# plt.show()
# for i in range(3):
#     lp_blur = fspecial_gaussian_2d((200, 200), sigma_blur)
#     otf = psf2otf(lp_blur, denoised[..., i].shape)
#     denoised[:, :, i] = ifft2(fft2(denoised[..., i]) * otf).real
#
# print(calc_psnr(img, denoised))
# plt.imshow(denoised)
# plt.show()

from pytorch_diffusion import Diffusion
import matplotlib.pyplot as plt
import skimage.io as io
from scipy.signal import convolve2d
# import packages
import numpy as np
from numpy.fft import fft2, ifft2
import os

import skimage.io as io
from skimage.filters import gaussian

from pypher.pypher import psf2otf
import matplotlib.pyplot as plt

from PIL import Image

def fspecial_gaussian_2d(size, sigma):
    kernel = np.zeros(tuple(size))
    kernel[size[0]//2, size[1]//2] = 1
    kernel = gaussian(kernel, sigma)
    return kernel/np.sum(kernel)

def calc_psnr(x, gt):
    out = 10 * np.log10(1 / ((x - gt)**2).mean().item())
    return out

# im = Image.open('church.jpg')
# # print(im.format, im.size, im.mode)
# size = (256, 256)
# # print(size)
# name = "church_compressed.jpg"
# im.thumbnail(size)
# im.save(name, 'JPEG')

img = io.imread('church.jpg').astype(float)/255
height, width, _ = img.shape

noisy_img = np.zeros(np.shape(img))

sigma_noise = 0.5
for it in range(3):
    noisy_img[:, :, it] = sigma_noise * np.random.randn(img.shape[0], img.shape[1]) + img[:, :, it]
print(calc_psnr(img, noisy_img))
plt.imsave("noisy_church.jpg", np.clip(noisy_img, a_min=0.,a_max=1.))

use_ddpm = False

if use_ddpm:
    x = torch.Tensor([noisy_img.transpose([2, 0, 1])]).to("cuda:0")
    diffusion = Diffusion.from_pretrained("lsun_church")
    x = diffusion.denoise(1, x=x, curr_step=300, n_steps=301)
    # diffusion.save(samples, "lsun_church_sample_{:02}.png")


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
else:
    import torch
    from pytorch_pretrained_gans import make_gan

    # Sample a class-conditional image from BigGAN with default resolution 256
    G = make_gan(gan_type='biggan')  # -> nn.Module
    y = G.sample_class(batch_size=1)  # -> torch.Size([1, 1000])
    z = G.sample_latent(batch_size=1)  # -> torch.Size([1, 128])
    x = G(z=z, y=y)  # -> torch.Size([1, 3, 256, 256])

    plt.imshow(x[0, ...].cpu().detach().numpy().transpose([1, 2, 0]))
    plt.show()