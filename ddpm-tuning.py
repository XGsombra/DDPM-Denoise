import torch
from pytorch_diffusion import Diffusion
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from natsort import natsorted
from glob import glob
import os

# define the constants
CHANNEL_NUM = 3
SIZE = 256
CLEAN_DIR = './samples/clean'
NOISY_DIR = './samples/noisy'
DENOISED_DIR = 'samples/ddpm-denoised'
MODEL = 'lsun_church'
DEVICE = 'cuda:0' if torch.cuda.is_available() else "cpu"
SAVE_NOISY = False  # True when saving noisy images, False when denoising
GAUSSIAN_NOISE = True  # True if noises are Gaussian-distributed, False if Poisson-distributed


def calc_psnr(x, gt):
    out = 10 * np.log10(1 / ((x - gt) ** 2).mean().item())
    return out


# Load the DDPM model
print("Loading Model...")
diffusion = Diffusion.from_pretrained(MODEL)

# Load images
print("Loading Images...")
clean_image_paths = natsorted(glob(os.path.join(CLEAN_DIR, '*.jpg')))
image_num = len(clean_image_paths)
clean_images = np.zeros((image_num, SIZE, SIZE, CHANNEL_NUM))
for i in range(image_num):
    clean_images[i] = io.imread(clean_image_paths[i]).astype(float) / 255

# Initialize noise levels
print("Initiating Noise Levels...")
levels = np.arange(1, 10) * 0.1

opt_curr_steps = []
opt_step_nums = []

for level in levels:
    opt_sig_curr_steps = []
    opt_sig_step_nums = []
    if SAVE_NOISY:
        # Generate noisy images and save them to NOISY_DIR
        noisy_images = np.copy(clean_images)
        if GAUSSIAN_NOISE:
            noisy_images += level * np.random.randn(image_num, SIZE, SIZE, CHANNEL_NUM)
        else:
            noisy_images += np.random.poisson(level, (image_num, SIZE, SIZE, CHANNEL_NUM))
        # Save noisy images
        for i in range(image_num):
            plt.imsave(
                os.path.join(NOISY_DIR, f"{i}-{round(level, 1)}-{'g' if GAUSSIAN_NOISE else 'p'}.jpg"),
                np.clip(noisy_images[i], a_min=0., a_max=1.)
            )
    else:
        # Load noising image previously saved and convert to tensor
        noisy_images = np.zeros_like(clean_images)
        for i in range(image_num):
            noisy_images[i] = io.imread(
                os.path.join(NOISY_DIR, f"{i}-{round(level, 1)}-{'g' if GAUSSIAN_NOISE else 'p'}.jpg")
            ).astype(float) / 255
            plt.imshow(noisy_images[i])
            plt.show()
        noisy_images = torch.Tensor(noisy_images.transpose([0, 3, 1, 2])).to(DEVICE)
        # Initialize the range
        opt_max_psnr = 0
        opt_curr_step = 0
        opt_step_num = 0
        start = int((1 - level) * 100)
        curr_steps = range(start, start + 100, 5)
        # Loop and tune
        for curr_step in curr_steps:
            max_psnr = 0
            step_num = 0
            while True:
                # Denoise the noisy image
                denoised_images = diffusion.denoise(
                    image_num,
                    x=noisy_images,
                    curr_step=curr_step,
                    n_steps=curr_step + 1
                ).cpu().numpy().transpose([0, 2, 3, 1])
                # Calculate and update the maximal PSNR
                psnr = calc_psnr(clean_images, denoised_images)
                print(psnr)
                if psnr > max_psnr:
                    max_psnr = psnr
                    step_num += 1
                else:
                    # Not doing better, break the loop
                    break
            if max_psnr > opt_max_psnr:
                opt_max_psnr = max_psnr
                opt_curr_step = curr_step
                opt_step_num = step_num
        print(f"------------PSNR={max_psnr}------------")
        opt_sig_curr_steps.append(opt_curr_step)
        opt_sig_step_nums.append(opt_step_num)
        opt_curr_steps.append(opt_sig_curr_steps)
        opt_step_nums.append(opt_sig_step_nums)
# Save the parameters
np.savetxt("curr_steps.csv", np.array(opt_curr_steps), delimiter=',')
np.savetxt("step_nums.csv", np.array(opt_step_nums), delimiter=',')

