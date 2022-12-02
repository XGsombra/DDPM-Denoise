import torch
from pytorch_diffusion import Diffusion
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from natsort import natsorted
from glob import glob
import os
from util import calc_psnr_hvsm, calc_ssim
from constants import CHANNEL_NUM, SIZE, CLEAN_DIR, NOISY_DIR, MODEL, DEVICE, SAVE_NOISY, GAUSSIAN_NOISE, LEVELS


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

opt_curr_steps = []

LEVELS = [0.4] # TODO: REMOVE THIS LINE
manual_tuning = True

for level in LEVELS:
    opt_sig_curr_steps = []
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
                os.path.join(NOISY_DIR, f"{i}-{round(level, 2)}-{'g' if GAUSSIAN_NOISE else 'p'}.jpg"),
                np.clip(noisy_images[i], a_min=0., a_max=1.)
            )
    else:
        # Load noising image previously saved and convert to tensor
        noisy_images = np.zeros_like(clean_images)
        print("Loading noisy images...")
        for i in range(image_num):
            noisy_images[i] = io.imread(
                os.path.join(NOISY_DIR, f"{i}-{round(level, 2)}-{'g' if GAUSSIAN_NOISE else 'p'}.jpg")
            ).astype(float) / 255
        noisy_images = torch.Tensor(noisy_images.transpose([0, 3, 1, 2])).to(DEVICE)
        # Initialize the range
        opt_max_psnr = 0
        opt_curr_step = 0
        opt_step_num = 0
        start = 10
        curr_steps = range(10, 1000, 5)
        curr_steps = [125]  # todo: remove this line
        # Loop and tune
        for curr_step in curr_steps:
            max_psnr = 0
            step_num = 0
            # Denoise the noisy image
            while True:
                denoised_images = diffusion.denoise(
                    image_num,
                    x=noisy_images,
                    curr_step=curr_step,
                    n_steps=curr_step
                ).cpu().numpy().transpose([0, 2, 3, 1])
                # Calculate and update the maximal PSNR
                psnr = 0
                ssim = 0
                for i in range(image_num):
                    psnr += calc_psnr_hvsm(denoised_images[i], clean_images[i])
                    ssim += calc_ssim(denoised_images[i], clean_images[i])
                psnr /= image_num
                ssim /= image_num
                if manual_tuning:
                    print(curr_step, level)
                    print(f"psnr is {psnr}")
                    print(f"ssim is {ssim}")
                    exit()
                if psnr > max_psnr:
                    max_psnr = psnr
                    curr_step -= 1
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
        opt_curr_steps.append(opt_sig_curr_steps)
# Save the parameters
np.savetxt("curr_steps.csv", np.array(opt_curr_steps), delimiter=',')
