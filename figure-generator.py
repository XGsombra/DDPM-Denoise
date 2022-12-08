import os
import torch
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from glob2 import glob
from scipy import interpolate
from natsort import natsorted
from pytorch_diffusion import Diffusion
from util import calc_psnr_hvsm, calc_ssim, calc_psnr
from constants import CLEAN_DIR, SIZE, CHANNEL_NUM, LEVELS, NOISY_DIR, GAUSSIAN_NOISE, DEVICE, MODEL, DDPM_DENOISED_DIR

# ---------------------Comment out any part to exclude it from being executed-------------

# Load the pretrained DDPM model
diffusion = Diffusion.from_pretrained(MODEL)

# ----------------------------- Image for the introduction -----------------------------
clean = io.imread(f"./samples/clean/0.jpg").astype(float) / 255
noisy = io.imread(f"./samples/noisy/0-0.2-g.jpg").astype(float) / 255
ddpm_denoised = io.imread(
    f"./samples/ddpm-denoised/0-0.2-g.png").astype(float) / 255
mprnet_denoised = io.imread(
    f"./samples/mprnet-denoised/0-0.2-g.png").astype(float) / 255

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
axs[0, 0].imshow(clean)
axs[0, 0].set_title("Ground Truth")
axs[0, 0].set_axis_off()
axs[0, 1].imshow(noisy)
axs[0, 1].set_title("Noisy Image with sigma=0.2")
axs[0, 1].set_axis_off()
axs[1, 0].imshow(ddpm_denoised)
axs[1, 0].set_title("Denoised by DDPM")
axs[1, 0].set_axis_off()
axs[1, 1].imshow(mprnet_denoised)
axs[1, 1].set_title("Denoised by MPRNet")
axs[1, 1].set_axis_off()
plt.savefig("introduction.png", bbox_inches='tight')

# ----------------------------- Image for PSNR and SSIM on each step -----------------------------
image_num = 5
plot_psnr_hvs_m = True  # modify this to plot the curve for SSIM
clean_image_paths = natsorted(glob(os.path.join(CLEAN_DIR, '*.jpg')))
clean_images = np.zeros((image_num, SIZE, SIZE, CHANNEL_NUM))
image_num = len(clean_image_paths)

# Load the clean images
for i in range(image_num):
    clean_images[i] = io.imread(clean_image_paths[i]).astype(float) / 255

levels = [0.05, 0.1, 0.2, 0.4]
level_to_curr_step = {0.05: 15, 0.1: 33, 0.2: 69, 0.4: 120}
psnr_curves = {}
ssim_curves = {}

for level in levels:
    psnr_curves[level] = []
    ssim_curves[level] = []

    # Load the noisy image
    x = np.zeros_like(clean_images)
    for i in range(image_num):
        x[i] = io.imread(
            os.path.join(NOISY_DIR, f"{i}-{round(level, 2)}-{'g' if GAUSSIAN_NOISE else 'p'}.jpg")
        ).astype(float) / 255
    x = torch.Tensor(x.transpose([0, 3, 1, 2])).to(DEVICE)

    # start to denoise
    curr_step = level_to_curr_step[level]

    # take 1 step of denoising for each iteration
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

# Plot the curves
fig, ax = plt.subplots()
for level in levels:
    if plot_psnr_hvs_m:
        ax.plot(range(level_to_curr_step[level], 0, -1), psnr_curves[level], label=f"sigma={level}")
    else:
        ax.plot(range(level_to_curr_step[level], 0, -1), ssim_curves[level], label=f"sigma={level}")
ax.invert_xaxis()
ax.legend()
ax.set_xlabel("Denoising Step Number")
if plot_psnr_hvs_m:
    ax.set_ylabel("PSNR-HVS-M (dB)")
else:
    ax.set_ylabel("SSIM")
ax.grid(True)
plt.show()

# ----------------------------- high/low frequency images -----------------------------
noise_type = 'g'
image_num = 3
get_noisy_metrics = True
levels = [0.05, 0.1, 0.2, 0.4]
level_to_curr_step = {0.05: 15, 0.1: 33, 0.2: 69, 0.4: 120}

for level in levels:
    psnr = 0
    psnr_hvs_m = 0
    ssim = 0
    for img_id in range(5, 8):
        clean = io.imread(f"./samples/clean/validation/{img_id}.jpg").astype(float) / 255
        noisy_img = io.imread(f"./samples/noisy/validation/{img_id}-{level}-{noise_type}.jpg").astype(float) / 255
        if get_noisy_metrics:
            denoised = noisy_img
        else:
            x = torch.Tensor([noisy_img.transpose([2, 0, 1])]).to(DEVICE)
            curr_step = level_to_curr_step[level]
            denoised = diffusion.denoise(1, x=x, curr_step=curr_step, n_steps=curr_step)[
                0, ...].cpu().detach().numpy().transpose([1, 2, 0])
        psnr += calc_psnr(denoised, clean)
        psnr_hvs_m += calc_psnr_hvsm(denoised, clean)
        ssim += calc_ssim(denoised, clean)
        plt.imsave(f"{DDPM_DENOISED_DIR}/{img_id}-{level}-{noise_type}.png", np.clip(denoised, a_min=0., a_max=1.))
    print(psnr / image_num)
    print(psnr_hvs_m / image_num)
    print(ssim / image_num)

for level in levels:
    psnr = 0
    psnr_hvs_m = 0
    ssim = 0
    for img_id in range(8, 11):
        clean = io.imread(f"./samples/clean/validation/{img_id}.jpg").astype(float) / 255
        noisy_img = io.imread(f"./samples/mprnet-denoised/{img_id}-{level}-{noise_type}.png").astype(float) / 255
        if get_noisy_metrics:
            denoised = noisy_img
        else:
            x = torch.Tensor([noisy_img.transpose([2, 0, 1])]).to(DEVICE)
            curr_step = level_to_curr_step[level]
            denoised = diffusion.denoise(1, x=x, curr_step=curr_step, n_steps=curr_step)[
                0, ...].cpu().detach().numpy().transpose([1, 2, 0])
        psnr += calc_psnr(denoised, clean)
        psnr_hvs_m += calc_psnr_hvsm(denoised, clean)
        ssim += calc_ssim(denoised, clean)
        plt.imsave(f"{DDPM_DENOISED_DIR}/{img_id}-{level}-{noise_type}.png", np.clip(denoised, a_min=0., a_max=1.))
    print(psnr / image_num)
    print(psnr_hvs_m / image_num)
    print(ssim / image_num)

# ----------------------------- Graph for Tuning -----------------------------
psnr_result = {0.05: [], 0.1: [], 0.2: [], 0.4: []}
ssim_result = {0.05: [], 0.1: [], 0.2: [], 0.4: []}
tune_range = range(120, 1001, 10)
image_num = 5
for curr_step in tune_range:
    for level in levels:
        print(level, curr_step)
        psnr = 0
        ssim = 0
        for img_id in range(image_num):
            clean = io.imread(f"./samples/clean/{img_id}.jpg").astype(float) / 255
            noisy_img = io.imread(f"./samples/noisy/{img_id}-{level}-{noise_type}.jpg").astype(float) / 255
            x = torch.Tensor([noisy_img.transpose([2, 0, 1])]).to(DEVICE)
            denoised = diffusion.denoise(1, x=x, curr_step=curr_step, n_steps=curr_step)[
                0, ...].cpu().detach().numpy().transpose([1, 2, 0])
            psnr += calc_psnr_hvsm(denoised, clean)
            ssim += calc_ssim(denoised, clean)
        psnr_result[level].append(psnr / image_num)
        ssim_result[level].append(ssim / image_num)

    # Store the plot for every iteration, so the latest plots could be saved when this program stops.
    fig, ax = plt.subplots()
    for level in levels:
        ax.plot(range(10, curr_step + 1, 10), psnr_result[level], label=f"sigma={level}")
    ax.legend()
    ax.set_xlabel("Starting Step Number")
    ax.set_ylabel("PSNR-HVS-M (dB)")
    ax.grid(True)
    plt.savefig("psnr-curr-tune.png")

    fig, ax = plt.subplots()
    for level in levels:
        ax.plot(range(10, curr_step + 1, 10), ssim_result[level], label=f"sigma={level}")
    ax.legend()
    ax.set_xlabel("Starting Step Number")
    ax.set_ylabel("SSIM")
    ax.grid(True)
    plt.savefig("ssim-curr-tune.png")

# ----------------------------- Tuning for one sigma -----------------------------
psnr_result = []
ssim_result = []
level = 2
start = 150
end = 300
tune_step = 5
tune_range = range(start, end+1, tune_step)
for curr_step in tune_range:
    psnr = 0
    ssim = 0
    for img_id in range(5):
        clean = io.imread(f"./samples/clean/{img_id}.jpg").astype(float) / 255
        # noisy_img = io.imread(f"./samples/noisy/{img_id}-{level}-{noise_type}.jpg").astype(float) / 255
        noisy_img = np.clip(clean + level * np.random.randn(256, 256, 3), a_min=0., a_max=1.)
        actual_noise = noisy_img - clean
        print(np.std(actual_noise))
        print(np.mean(actual_noise))
        x = torch.Tensor([noisy_img.transpose([2, 0, 1])]).to(DEVICE)
        denoised = diffusion.denoise(1, x=x, curr_step=curr_step, n_steps=curr_step)[0, ...].cpu().detach().numpy().transpose([1,2,0])
        psnr += calc_psnr_hvsm(denoised, clean)
        ssim += calc_ssim(denoised, clean)
    psnr_result.append(psnr / 5)
    ssim_result.append(ssim / 5)

    fig, ax = plt.subplots()
    ax.plot(range(start, curr_step+1, tune_step), psnr_result, label=f"sigma={level}")
    ax.legend()
    ax.set_xlabel("Starting Step Number")
    ax.set_ylabel("PSNR-HVS-M (dB)")
    ax.grid(True)
    plt.savefig("psnr-curr-tune-0.3.png")

    fig, ax = plt.subplots()
    ax.plot(range(start, curr_step+1, tune_step), ssim_result, label=f"sigma={level}")
    ax.legend()
    ax.set_xlabel("Starting Step Number")
    ax.set_ylabel("SSIM")
    ax.grid(True)
    plt.savefig("ssim-curr-tune-0.3.png")


# ----------------------------- Crop images to show details -----------------------------
for level in levels:
    i = 5
    image_path = f"./samples/mprnet-denoised/{i}-{level}-g.png"
    clean_images = np.zeros((image_num, SIZE, SIZE, CHANNEL_NUM))
    clean_images = io.imread(image_path).astype(float) / 255
    plt.imsave(f"{image_path[:-4]}-cropped.png", clean_images[120:184, 70:132])

i = 5
image_path = f"./samples/clean/validation/{i}.jpg"
clean_images = np.zeros((image_num, SIZE, SIZE, CHANNEL_NUM))
clean_images = io.imread(image_path).astype(float) / 255
plt.imsave(f"{image_path[:-4]}-cropped.jpg", clean_images[120:184, 70:132])

# ----------------------------- Interpolation vs closed form -----------------------------

betas = np.linspace(0.0001, 0.02, 1000)
alphas = 1.0 - betas
alphas_bar = np.cumprod(alphas, axis=0)
vars = np.sqrt(1-alphas_bar)

sigmas = [0.05, 0.1, 0.2, 0.4]
curr_steps = [15, 33, 69, 120]
sigma2curr_step = interpolate.interp1d(sigmas, curr_steps)
sigma_space = np.linspace(0.05, 0.4, 100)
fig, ax = plt.subplots()
ax.plot(sigma_space, sigma2curr_step(sigma_space), label="Interpoaltion")
ax.plot(vars, np.linspace(1, 1000, 1000), label="Closed Form")
ax.legend()
ax.grid(True)
ax.set_xlabel("Standard Deviation of Gaussian Noise")
ax.set_ylabel("Optimal Starting Step Number for Denoising")
plt.savefig("interpolation.png")


# ----------------------------- Denoise image with realistic noises -----------------------------
clean = io.imread(f"./samples/clean/4.jpg").astype(float) / 255
noisy_img = io.imread(f"./samples/noisy/realistic/unprocess.jpg").astype(float) / 255
print(np.std(noisy_img - clean))
print(calc_psnr(noisy_img, clean))  # 16.87135728981658
print(calc_psnr_hvsm(noisy_img, clean))   # 17.714713168601556
print(calc_ssim(noisy_img, clean))  # 0.4436537359259245
x = torch.Tensor([noisy_img.transpose([2, 0, 1])]).to(DEVICE)
curr_step = 150
denoised = diffusion.denoise(1, x=x, curr_step=curr_step, n_steps=curr_step)[0, ...].cpu().detach().numpy().transpose([1,2,0])
plt.imsave("realistic_denoised.jpg", np.clip(denoised, a_min=0., a_max=1.))
denoised = io.imread(f"./samples/bilateral-denoised/unprocess.jpg").astype(float) / 255
print(calc_psnr(denoised, clean))
print(calc_psnr_hvsm(denoised, clean))
print(calc_ssim(denoised, clean))
